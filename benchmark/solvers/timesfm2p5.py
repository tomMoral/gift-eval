"""
TimesFM-2.5 solver for the GIFT-Eval benchopt benchmark.

Wraps the Google TimesFM-2.5 (200M parameter) model using the interface
demonstrated in ``notebooks/timesfm2p5.ipynb``.

Install (requires Python 3.11):
    git clone https://github.com/google-research/timesfm.git
    cd timesfm && pip install -e .

Reference:
    https://github.com/google-research/timesfm
"""

import numpy as np
from benchopt import BaseSolver
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast

# Quantile levels output by TimesFM (indices 1-9 of its output, skipping the
# mean at index 0).
_QUANTILE_LEVELS = list(np.arange(1, 10) / 10.0)  # [0.1, ..., 0.9]


class Solver(BaseSolver):
    """TimesFM-2.5 (200 M) zero-shot forecasting solver.

    The model checkpoint is loaded once in ``set_objective`` (not timed).
    Forecasts are produced in ``run`` (timed) using batched inference.
    """

    name = "TimesFM-2.5"
    sampling_strategy = "run_once"

    # timesfm must be installed from source (see module docstring).
    # No conda/pip package is available on PyPI yet.
    install_cmd = "conda"
    requirements = []  # install manually: pip install -e path/to/timesfm

    parameters = {
        "batch_size": [1024],
    }

    def set_objective(self, gift_eval_dataset):
        """Prepare the solver for a given dataset configuration.

        The TimesFM checkpoint is loaded here (not inside ``run``) to
        exclude one-time loading time from the benchmark timing.
        """
        # Load the model only once; reuse across dataset configurations.
        if not hasattr(self, "_tfm"):
            from timesfm.timesfm_2p5 import timesfm_2p5_torch

            self._tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
            self._tfm.load_checkpoint()

        self._prediction_length = gift_eval_dataset.prediction_length
        # Materialise inputs now so that ``run`` is not slowed by iterator
        # overhead and the same data is used for both timing and evaluation.
        self._test_data_input = list(gift_eval_dataset.test_data.input)

    def run(self, n_iter):
        """Generate TimesFM-2.5 forecasts (this is the timed section)."""
        from timesfm import configs

        pred_len = self._prediction_length
        forecast_outputs = []

        for batch in batcher(self._test_data_input, batch_size=self.batch_size):
            context = []
            max_context = 0
            for entry in batch:
                arr = np.array(entry["target"])
                if arr.shape[0] > max_context:
                    max_context = arr.shape[0]
                context.append(arr)

            # Round max_context up to the nearest multiple of model patch size.
            p = self._tfm.model.p
            max_context = ((max_context + p - 1) // p) * p

            self._tfm.compile(
                forecast_config=configs.ForecastConfig(
                    max_context=min(15360, max_context),
                    max_horizon=1024,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=128,
                ),
            )

            _, full_preds = self._tfm.forecast(
                horizon=pred_len,
                inputs=context,
            )
            # full_preds: (batch, horizon, n_outputs)
            # Index 0 is the mean; indices 1-9 are the nine quantile levels.
            quantile_preds = full_preds[:, :pred_len, 1:]  # (batch, T, 9)
            forecast_outputs.append(quantile_preds.transpose((0, 2, 1)))  # (batch, 9, T)

        forecast_arrays = np.concatenate(forecast_outputs)  # (N, 9, T)

        forecasts = []
        for arr, ts in zip(forecast_arrays, self._test_data_input):
            start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=arr,
                    forecast_keys=[str(q) for q in _QUANTILE_LEVELS],
                    start_date=start_date,
                )
            )
        self._forecasts = forecasts

    def get_result(self):
        return {"forecasts": self._forecasts}
