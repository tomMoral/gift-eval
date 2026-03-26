"""
Chronos-2 solver for the GIFT-Eval benchopt benchmark.

Wraps the Amazon Chronos-2 pipeline using the GluonTS-compatible interface
demonstrated in ``notebooks/chronos-2.ipynb``.

Install:
    pip install "chronos-forecasting>=2.1"

Reference:
    https://github.com/amazon-science/chronos-forecasting
"""

import logging

import numpy as np
import torch
from benchopt import BaseSolver
from gluonts.model.forecast import QuantileForecast

logger = logging.getLogger(__name__)

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Solver(BaseSolver):
    """Chronos-2 zero-shot forecasting solver.

    The model is loaded once in ``set_objective`` (not timed).  During
    ``run`` the predictions are generated window-by-window to prevent
    in-context learning leakage between rolling evaluation windows of the
    same time series, following the approach in the official GIFT-Eval
    notebook.
    """

    name = "Chronos-2"

    # chronos-forecasting is required to load the model and run inference.
    requirements = ["pip::chronos-forecasting>=2.1"]

    parameters = {
        "batch_size": [100],
        "dtype": ["float32"],
    }

    def set_objective(self, gift_eval_dataset):
        """Prepare the solver for a given dataset configuration.

        Model loading is done here (not inside ``run``) so that the
        checkpoint download/loading time is excluded from the benchmark
        timing.
        """
        from chronos import BaseChronosPipeline, Chronos2Pipeline

        device_map = "auto"  # Automatically use GPU if available.

        # Load the model only on the first call; reuse across dataset configs.
        if not hasattr(self, "_pipeline"):
            self._pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2", device_map=device_map, dtype=self.dtype,
            )
            if not isinstance(self._pipeline, Chronos2Pipeline):
                raise RuntimeError(
                    f"Expected Chronos2Pipeline for '{self.model_name}'. "
                    "Use the chronos.ipynb notebook for Chronos / Chronos-Bolt."
                )

        self._prediction_length = gift_eval_dataset.prediction_length
        n_windows = gift_eval_dataset.windows

        # Materialise all test inputs once and group them by window index so
        # that each window's series can be predicted jointly (no leakage
        # between windows of the same time series through in-context learning).
        all_inputs = list(gift_eval_dataset.test_data.input)
        # test_data.input order: [ts1_w0, ts1_w1, ..., ts2_w0, ts2_w1, ...]
        # all_inputs[w::n_windows] gives all series for window w.
        self._windowed_inputs = [
            all_inputs[w::n_windows] for w in range(n_windows)
        ]

    def run(self, n_iter):
        """Generate Chronos-2 forecasts (this is the timed section)."""
        model_batch_size = self.batch_size
        forecast_windows = []

        for window_inputs in self._windowed_inputs:
            packed = [{"target": entry["target"]} for entry in window_inputs]
            is_univariate = packed[0]["target"].ndim == 1

            while True:
                try:
                    quantiles, _ = self._pipeline.predict_quantiles(
                        inputs=packed,
                        prediction_length=self._prediction_length,
                        batch_size=model_batch_size,
                        quantile_levels=QUANTILE_LEVELS,
                        cross_learning=True,  # allow all predictions at once
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    model_batch_size //= 2
                    logger.warning(
                        f"CUDA OOM – reducing batch size to {model_batch_size}"
                    )

            # quantiles: list of tensors → stack to [batch, variates, T, Q]
            q = torch.stack(quantiles).permute(0, 3, 2, 1).cpu().numpy()
            # → shape [batch, Q, T, variates]
            if is_univariate:
                q = q.squeeze(-1)  # → [batch, Q, T]

            window_forecasts = []
            for arr, ts in zip(q, window_inputs):
                # If the model emits invalid values, fall back to a simple
                # last-value forecast to keep evaluation stable.
                if not np.isfinite(arr).all():
                    target = np.asarray(ts["target"])
                    if target.ndim == 1:
                        last_val = target[-1]
                        arr = np.full(
                            (len(QUANTILE_LEVELS), self._prediction_length),
                            last_val,
                            dtype=np.float32,
                        )
                    else:
                        last_val = target[:, -1]
                        arr = np.tile(
                            last_val[None, None, :],
                            (len(QUANTILE_LEVELS), self._prediction_length, 1),
                        ).astype(np.float32)

                start_date = ts["start"] + len(ts["target"])
                window_forecasts.append(
                    QuantileForecast(
                        forecast_arrays=arr,
                        forecast_keys=[str(lv) for lv in QUANTILE_LEVELS],
                        start_date=start_date,
                    )
                )
            forecast_windows.append(window_forecasts)

        # Re-interleave windows into the original test_data.input order:
        # [ts1_w0, ts1_w1, ts2_w0, ts2_w1, ...]
        self._forecasts = [
            item for items in zip(*forecast_windows) for item in items
        ]

    def get_result(self):
        return {"forecasts": self._forecasts}
