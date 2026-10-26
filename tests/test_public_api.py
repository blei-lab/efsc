from importlib.metadata import version as distribution_version

import efsc


EXPECTED_PUBLIC_API = {
    "CausalResult",
    "EFSC",
    "EFSCResult",
    "InferenceConfig",
    "__version__",
}


def test_public_version_matches_installed_distribution():
    assert efsc.__version__ == distribution_version("efsc")
    assert "__version__" in efsc.__all__


def test_top_level_api_is_deliberately_small():
    assert set(efsc.__all__) == EXPECTED_PUBLIC_API
