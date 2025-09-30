# tests/test_data.py
"""
Unittests for basic data utilities in the repo.

These tests are designed to be lightweight and not depend on the full Kaggle dataset.
They create small synthetic CSV files in a temporary directory and exercise:
- src.data_prep.load_raw
- src.data_prep.build_preprocessor
- src.data_prep.get_feature_target
- src.sampling.create_samples (by temporarily pointing RAW_DATA_PATH)

Uses unittest (compatible with CI in .github/workflows/ci.yml).
"""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Import modules under test
from src import data_prep
import src.sampling as sampling_module


class TestDataPrepAndSampling(unittest.TestCase):
    def setUp(self):
        # Create temporary directory and small synthetic dataframe
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ccfraud_test_"))
        self.raw_path = self.tmpdir / "creditcard_small.csv"

        # Build a tiny DataFrame with the necessary columns
        # Include Time, Amount, a few V* features, and Class
        self.df = pd.DataFrame(
            {
                "Time": [0, 1000, 2000, 3000, 4000],
                "V1": [0.1, -1.2, 0.5, 0.3, -0.4],
                "V2": [1.1, 0.2, -0.3, 0.4, 0.5],
                "Amount": [10.0, 50.5, 3.0, 500.0, 7.25],
                "Class": [0, 0, 1, 0, 1],
            }
        )
        self.df.to_csv(self.raw_path, index=False)

        # Keep copies of original module paths so we can restore
        self._orig_raw_path = getattr(sampling_module, "RAW_DATA_PATH", None)
        self._orig_sample_raw = getattr(sampling_module, "SAMPLE_RAW_PATH", None)
        self._orig_sample_proc = getattr(sampling_module, "SAMPLE_PROCESSED_PATH", None)

    def tearDown(self):
        # Remove temp dir and restore sampling module globals
        try:
            for f in self.tmpdir.iterdir():
                f.unlink()
            self.tmpdir.rmdir()
        except Exception:
            pass

        # restore sampling module paths
        if self._orig_raw_path is not None:
            sampling_module.RAW_DATA_PATH = self._orig_raw_path
        if self._orig_sample_raw is not None:
            sampling_module.SAMPLE_RAW_PATH = self._orig_sample_raw
        if self._orig_sample_proc is not None:
            sampling_module.SAMPLE_PROCESSED_PATH = self._orig_sample_proc

    def test_load_raw_and_split(self):
        # Test load_raw reads csv and get_feature_target splits correctly
        df_loaded = data_prep.load_raw(self.raw_path)
        self.assertIsInstance(df_loaded, pd.DataFrame)
        self.assertIn("Class", df_loaded.columns)

        X, y = data_prep.get_feature_target(df_loaded)
        self.assertNotIn("Class", X.columns)
        self.assertEqual(len(X), len(y))
        self.assertTrue(all(y.isin([0, 1])))

    def test_build_preprocessor_output(self):
        # Preprocessor should be a scikit-learn ColumnTransformer-like object
        pre = data_prep.build_preprocessor()
        # Ensure it has a transform method and can fit/transform a small DF
        self.assertTrue(hasattr(pre, "fit"))
        self.assertTrue(hasattr(pre, "transform"))

        # Fit transformer on dataframe (only Time and Amount expected to be scaled)
        X, _ = data_prep.get_feature_target(self.df)
        pre.fit(X)
        Xt = pre.transform(X)
        # transformed output should be numpy-like and have same row count
        self.assertEqual(Xt.shape[0], X.shape[0])

    def test_create_samples_uses_temp_raw(self):
        # Point sampling module to our temporary CSV and create samples
        sampling_module.RAW_DATA_PATH = self.raw_path
        sampling_module.SAMPLE_RAW_PATH = self.tmpdir / "sample_raw.csv"
        sampling_module.SAMPLE_PROCESSED_PATH = self.tmpdir / "sample_processed.csv"

        # Ensure previous sample files (if any) are removed
        if sampling_module.SAMPLE_RAW_PATH.exists():
            sampling_module.SAMPLE_RAW_PATH.unlink()
        if sampling_module.SAMPLE_PROCESSED_PATH.exists():
            sampling_module.SAMPLE_PROCESSED_PATH.unlink()

        # Create small sample
        sampling_module.create_samples(n=3)

        # Check that sample files were created
        self.assertTrue(sampling_module.SAMPLE_RAW_PATH.exists())
        self.assertTrue(sampling_module.SAMPLE_PROCESSED_PATH.exists())

        # Verify sample files contain expected columns
        df_raw_sample = pd.read_csv(sampling_module.SAMPLE_RAW_PATH)
        self.assertIn("Class", df_raw_sample.columns)
        self.assertLessEqual(len(df_raw_sample), 3 + 1)  # allow small rounding in stratified sampling

        df_proc_sample = pd.read_csv(sampling_module.SAMPLE_PROCESSED_PATH)
        self.assertIn("Amount", df_proc_sample.columns)


if __name__ == "__main__":
    unittest.main()