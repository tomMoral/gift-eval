"""Mantis solver for time series classification on UCR datasets.

Uses the official ``mantis-tsfm`` API to load a pretrained Mantis checkpoint,
extract embeddings with ``MantisTrainer.transform``, and train a Random Forest
classifier on top.

References:
    https://huggingface.co/paris-noah/Mantis-8M
    https://github.com/vfeofanov/mantis
"""

import numpy as np
import torch
from benchopt import BaseSolver
from sklearn.ensemble import RandomForestClassifier


class Solver(BaseSolver):
    """Mantis time series classification solver with Random Forest.

    The model is loaded once in ``set_objective`` (not timed). Training
    embeddings are extracted and a Random Forest classifier is trained.
    During ``run`` the predictions are generated for the test set.
    """

    name = "Mantis-RandomForest"

    # mantis-tsfm and torch are required to load the model and run inference.
    requirements = [
        "pip::mantis-tsfm>=1.0.0",
        "pip::torch>=2.0.0",
        "pip::scikit-learn>=1.0.0",
    ]

    parameters = {
        "checkpoint": ["paris-noah/Mantis-8M"],
        "batch_size": [32],
        "dtype": ["float32"],
        "n_estimators": [100],
        "interpolate_to": [512],
    }

    def set_objective(self, ucr_dataset):
        """Prepare the solver for a given dataset configuration.

        Model loading is done here (not inside ``run``) so that the
        checkpoint download/loading time is excluded from the benchmark
        timing.
        """
        try:
            from mantis.trainer import MantisTrainer
        except ImportError:
            raise ImportError(
                "mantis-tsfm is required. Install with: pip install mantis-tsfm"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model and trainer only on the first call for this checkpoint.
        should_reload = (
            not hasattr(self, "_network")
            or not hasattr(self, "_loaded_checkpoint")
            or self._loaded_checkpoint != self.checkpoint
        )
        if should_reload:
            try:
                if "MantisV2" in self.checkpoint:
                    from mantis.architecture import MantisV2 as MantisBackbone
                else:
                    from mantis.architecture import MantisV1 as MantisBackbone

                network = MantisBackbone(device=device)
                network = network.from_pretrained(self.checkpoint)

                self._network = network
                self._trainer = MantisTrainer(device=device, network=self._network)
                self._loaded_checkpoint = self.checkpoint
                print(
                    f"✓ Mantis checkpoint loaded: {self.checkpoint} on device: {device}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Mantis checkpoint '{self.checkpoint}' from Hugging Face: {e}. "
                    "Make sure you have internet access and the model is available."
                )

        self._device = device
        self.ucr_dataset = ucr_dataset
        self._dtype_torch = (
            torch.float32 if self.dtype == "float32" else torch.float64
        )

    def run(self, n_iter):
        """Extract embeddings and train Random Forest classifier (this is the timed section)."""
        X_train = self.ucr_dataset.X_train
        y_train = self.ucr_dataset.y_train
        X_test = self.ucr_dataset.X_test
        y_test = self.ucr_dataset.y_test
        
        batch_size = self.batch_size
        
        # Extract embeddings for training data
        print("\n[Mantis-RandomForest] Extracting training embeddings...")
        train_embeddings = self._extract_embeddings(X_train, batch_size)
        print(f"  Training embeddings shape: {train_embeddings.shape}")
        
        # Extract embeddings for test data
        print("[Mantis-RandomForest] Extracting test embeddings...")
        test_embeddings = self._extract_embeddings(X_test, batch_size)
        print(f"  Test embeddings shape: {test_embeddings.shape}")
        
        # Train Random Forest classifier
        print(f"[Mantis-RandomForest] Training Random Forest (n_estimators={self.n_estimators})...")
        self._classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        self._classifier.fit(train_embeddings, y_train)
        print("  Random Forest training complete")
        
        # Make predictions
        print("[Mantis-RandomForest] Making predictions on test set...")
        y_pred = self._classifier.predict(test_embeddings)
        y_pred_proba = self._classifier.predict_proba(test_embeddings)
        
        # Compute accuracy for debugging
        accuracy = np.mean(y_pred == y_test)
        print(f"\n[DEBUG] Accuracy: {accuracy:.4f}")
        
        self._y_pred = y_pred
        self._y_pred_proba = y_pred_proba
    
    def _extract_embeddings(self, X, batch_size):
        """Extract embeddings for a batch of time series.
        
        Parameters
        ----------
        X : np.ndarray
            Input time series of shape (n_samples, num_variates, sequence_length)
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        np.ndarray
            Embeddings of shape (n_samples, embedding_dim)
        """
        n_samples = len(X)
        all_embeddings = []
        
        for batch_idx in range(0, n_samples, batch_size):
            batch_end = min(batch_idx + batch_size, n_samples)
            X_batch = X[batch_idx:batch_end].astype(np.float32)
            X_batch_processed = self._prepare_inputs(X_batch)
            
            try:
                with torch.no_grad():
                    embeddings_np = self._trainer.transform(X_batch_processed)
                all_embeddings.append(np.asarray(embeddings_np))
                
            except Exception as e:
                print(f"  Warning: Failed to process batch {batch_idx}: {e}")
                if all_embeddings:
                    embedding_dim = all_embeddings[0].shape[1]
                else:
                    embedding_dim = 128
                all_embeddings.append(np.zeros((batch_end - batch_idx, embedding_dim), dtype=np.float32))
        
        # Concatenate all embeddings
        if all_embeddings:
            embeddings_all = np.vstack(all_embeddings)
        else:
            embeddings_all = np.zeros((n_samples, 768))
        
        return embeddings_all

    def _prepare_inputs(self, X_batch):
        """Ensure Mantis-compatible shape and sequence length.

        Mantis expects arrays of shape (n_samples, n_channels, seq_len), and the
        sequence length should be divisible by 32. Following official guidance,
        we interpolate to ``interpolate_to`` (default 512).
        """
        X_in = X_batch
        if X_in.ndim == 2:
            X_in = X_in[:, None, :]

        if X_in.ndim != 3:
            raise ValueError(
                f"Expected a 3D array (n_samples, n_channels, seq_len), got shape={X_in.shape}"
            )

        current_len = X_in.shape[-1]
        target_len = int(self.interpolate_to)

        if current_len != target_len:
            tensor = torch.tensor(X_in, dtype=torch.float32)
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=target_len,
                mode="linear",
                align_corners=False,
            )
            X_in = tensor.numpy()

        if X_in.shape[-1] % 32 != 0:
            raise ValueError(
                f"Sequence length must be divisible by 32 for Mantis, got {X_in.shape[-1]}"
            )

        return X_in

    def get_result(self):
        """Return the classification predictions and probabilities."""
        return {
            "y_pred": self._y_pred,
            "y_pred_proba": self._y_pred_proba,
        }

