"""Utilities to repair invalid forecast values.

These helpers implement a deterministic last-value fallback policy for
quantile forecasts, operating directly on Forecast objects.
"""

from __future__ import annotations

import numpy as np
from gluonts.model.forecast import QuantileForecast


def repair_quantile_forecast_with_last_value(input, forecast):
    """Repair invalid values in one QuantileForecast using last-value fallback.

    Non-QuantileForecast objects are returned unchanged.
    """
    if not isinstance(forecast, QuantileForecast):
        return forecast, False

    arr = np.asarray(forecast.forecast_array)
    if np.isfinite(arr).all():
        return forecast, False

    inp = np.asarray(input)
    pred_len = arr.shape[1] if arr.ndim >= 2 else 1
    n_quantiles = len(forecast.forecast_keys)

    if arr.ndim == 2:
        if inp.ndim == 1:
            last_val = inp[-1]
        else:
            last_val = inp[0, -1]
        repaired_arr = np.full(
            (n_quantiles, pred_len),
            last_val,
            dtype=np.float32,
        )
    elif arr.ndim == 3:
        if inp.ndim == 1:
            last_vec = np.full(arr.shape[2], inp[-1], dtype=np.float32)
        else:
            last_vec = inp[:, -1].astype(np.float32, copy=False)
        repaired_arr = np.tile(
            last_vec[None, None, :], (n_quantiles, pred_len, 1)
        )
        repaired_arr = repaired_arr.astype(np.float32, copy=False)
    else:
        last_val = inp[-1] if inp.ndim == 1 else inp[0, -1]
        repaired_arr = np.full(
            (n_quantiles, pred_len), last_val, dtype=np.float32
        )

    repaired_forecast = QuantileForecast(
        forecast_arrays=repaired_arr,
        forecast_keys=list(forecast.forecast_keys),
        start_date=forecast.start_date,
    )
    return repaired_forecast, True


def repair_forecast_ffill(
    forecasts,
    test_inputs,
) -> tuple[list, int]:
    """Repair invalid values in QuantileForecast objects.

    Only QuantileForecast instances are repaired. Other forecast types are
    returned untouched.
    """
    repaired_count = 0
    repaired_forecasts = []

    for forecast, ts in zip(forecasts, test_inputs):
        repaired_forecast, repaired = repair_quantile_forecast_with_last_value(
            input=np.asarray(ts["target"]),
            forecast=forecast,
        )
        if repaired:
            repaired_count += 1
        repaired_forecasts.append(repaired_forecast)

    return repaired_forecasts, repaired_count


def build_quantile_forecasts(
    forecast_arrays,
    test_inputs,
    forecast_keys,
):
    """Build QuantileForecast objects from raw arrays.

    Parameters
    ----------
    forecast_arrays : iterable
        Arrays with shape [Q, T] or [Q, T, D].
    test_inputs : iterable
        Matching GluonTS inputs containing ``target`` and ``start``.
    forecast_keys : list[str]
        Quantile labels.
    """
    forecasts = []

    for arr, ts in zip(forecast_arrays, test_inputs):
        forecasts.append(
            QuantileForecast(
                forecast_arrays=arr,
                forecast_keys=[str(k) for k in forecast_keys],
                start_date=ts["start"] + len(ts["target"]),
            )
        )

    return forecasts
