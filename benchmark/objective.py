"""
GIFT-Eval Forecasting Benchmark Objective.

Evaluates time series forecasting models using the full set of GIFT-Eval
metrics via the GluonTS evaluation framework.
"""

import logging

from benchopt import BaseObjective

from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality


# Suppress the gluonts warning when QuantileForecast has no explicit mean key.
class _WarningFilter(logging.Filter):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def filter(self, record):
        return self.text not in record.getMessage()


logging.getLogger("gluonts.model.forecast").addFilter(
    _WarningFilter("The mean prediction is not stored in the forecast data")
)

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Objective(BaseObjective):
    """GIFT-Eval time series forecasting objective.

    Solvers receive the GiftEval Dataset object and must return a list of
    GluonTS Forecast objects (e.g. QuantileForecast or SampleForecast).
    """

    name = "GIFT-Eval Forecasting"
    url = "https://github.com/SalesforceAIResearch/gift-eval"
    min_benchopt_version = "1.9"
    requirements = [
        # for now, use a patched version of gift-eval to have debug datasets
        "pip::git+https://github.com/tommoral/gift-eval.git@INI_benchopt_runner"
    ]

    sampling_strategy = "run_once"

    test_dataset_name = "gift-eval"
    test_config = {
        "dataset": {'debug': True},  # use debug subset for fast checks
    }

    def set_data(self, gift_eval_dataset, domain, num_variates):
        self.gift_eval_dataset = gift_eval_dataset
        self.domain = domain
        self.num_variates = num_variates

        self._metrics = [
            MSE(forecast_type="mean"),
            MSE(forecast_type=0.5),
            MAE(),
            MASE(),
            MAPE(),
            SMAPE(),
            MSIS(),
            RMSE(),
            NRMSE(),
            ND(),
            MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
        ]

    def get_objective(self):
        """Pass the Dataset object to the solver.

        Solvers get the full Dataset so they can access prediction_length,
        frequency, windows, and the test split input.
        """
        return dict(gift_eval_dataset=self.gift_eval_dataset)

    def evaluate_result(self, forecasts):
        """Compute GIFT-Eval metrics from the solver's forecast list."""
        season_length = get_seasonality(self.gift_eval_dataset.freq)
        res = (
            evaluate_forecasts(
                forecasts,
                test_data=self.gift_eval_dataset.test_data,
                metrics=self._metrics,
                batch_size=1024,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")[0]
        )

        # Expose all GIFT-Eval metrics with the same naming as CSV output.
        return {
            "MSE[mean]": res.get("MSE[mean]"),
            "MSE[0.5]": res.get("MSE[0.5]"),
            "MAE[0.5]": res.get("MAE[0.5]"),
            "MASE[0.5]": res.get("MASE[0.5]"),
            "MAPE[0.5]": res.get("MAPE[0.5]"),
            "sMAPE[0.5]": res.get("sMAPE[0.5]"),
            "MSIS": res.get("MSIS"),
            "RMSE[mean]": res.get("RMSE[mean]"),
            "NRMSE[mean]": res.get("NRMSE[mean]"),
            "ND[0.5]": res.get("ND[0.5]"),
            "mean_weighted_sum_quantile_loss": res.get(
                "mean_weighted_sum_quantile_loss"
            ),
        }

    def get_one_result(self):
        """Return a trivial (constant) forecast for benchopt self-checks."""
        import numpy as np
        from gluonts.model.forecast import QuantileForecast

        pred_len = self.gift_eval_dataset.prediction_length
        forecasts = []
        for entry in self.gift_eval_dataset.test_data.input:
            target = np.array(entry["target"])
            start_date = entry["start"] + len(target)
            # Constant last-value forecast for all quantile levels.
            last_val = target[-1] if target.ndim == 1 else target[:, -1]
            forecast_arrays = np.tile(
                last_val, (len(QUANTILE_LEVELS), pred_len)
            )
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=forecast_arrays,
                    forecast_keys=[str(q) for q in QUANTILE_LEVELS],
                    start_date=start_date,
                )
            )
        return {"forecasts": forecasts}
