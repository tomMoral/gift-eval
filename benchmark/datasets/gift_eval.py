"""
GIFT-Eval dataset loader for benchopt.

Loads a GIFT-Eval dataset split using the ``gift_eval.data.Dataset`` class.
The default configuration targets ``m4_weekly / short`` which is the smallest
dataset in the benchmark (359 weekly univariate time series, pred_len = 8).
"""
import os
import json
from pathlib import Path

from benchopt import BaseDataset
from benchopt.config import get_data_path

from gift_eval.data import Dataset as GiftEvalDataset


# Mapping from raw dataset names used by the HuggingFace loader to the
# canonical pretty names used in result files.
_PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

_DEBUG_MAX_SERIES = 10


def _load_dataset_properties() -> dict:
    root = Path(__file__).resolve().parents[2]
    properties_path = root / "notebooks" / "dataset_properties.json"

    with properties_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class Dataset(BaseDataset):
    """GIFT-Eval dataset wrapper for benchopt.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset as accepted by ``gift_eval.data.Dataset``.
        Datasets with per-frequency sub-splits are specified as
        ``<name>/<freq>`` (e.g. ``"electricity/H"``).
    term : str
        Evaluation horizon.  One of ``"short"``, ``"medium"``, ``"long"``.
        Medium and long horizons are only available for a subset of datasets.
    """

    name = "GiftEval"

    # Default to the smallest dataset: m4_weekly with short horizon.
    parameters = {
        "dataset_name": ["m4_weekly"],
        "term": ["short"],
        "debug": [False],  # if True, use only 10 series for fast iteration
    }

    # Prepare does not depends on the specific dataset configuration, but
    # simply downloads the full GIFT-Eval dataset if not already present.
    ignore_parameters_prepare = ["dataset_name", "term", "debug"]

    def prepare(self):
        """Prepare data artifacts for this dataset configuration.

        Current benchopt versions do not call this automatically yet; we also
        call it from get_data for backward compatibility.
        """
        storage_path = get_data_path("GiftEval")
        if storage_path.exists():
            return

        # Download the full GIFT-Eval dataset repo if local artifacts are
        # missing. This matches the layout expected by gift_eval.data.Dataset.
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Salesforce/GiftEval",
            repo_type="dataset",
            local_dir=str(storage_path),
        )

        if not storage_path.exists():
            raise FileNotFoundError(
                f"GiftEval was not found under {storage_path} after download"
            )

    def get_data(self):
        # Keep compatibility with current benchopt versions where prepare() is
        # not orchestrated yet.
        self.prepare()
        dataset_properties = _load_dataset_properties()

        # Determine the canonical key used in the properties map.
        base_name = self.dataset_name.split("/")[0]
        props_key = _PRETTY_NAMES.get(base_name, base_name).lower()

        properties = dataset_properties.get(
            props_key, {"domain": "unknown", "num_variates": 1}
        )

        # With gluonts, need to pass the dataset path using a env var
        os.environ["GIFT_EVAL"] = str(get_data_path("GiftEval"))
        dataset_probe = GiftEvalDataset(
            name=self.dataset_name,
            term=self.term,
            to_univariate=False,
        )
        to_univariate = dataset_probe.target_dim > 1

        dataset = GiftEvalDataset(
            name=self.dataset_name,
            term=self.term,
            to_univariate=to_univariate,
            debug=self.debug,
        )

        return dict(
            gift_eval_dataset=dataset,
            domain=properties["domain"],
            num_variates=properties["num_variates"],
        )
