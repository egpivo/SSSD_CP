import numpy as np
import pytest

from sssd.data.generator import ArDataGenerator, SeasonalityGenerator


def test_generate():
    coefficients = [0.1, 0.2, 0.3]
    n_sample = 3
    seed = 1
    expected_output = np.array([1.62434536, -0.44932188, -0.24823487])

    generator = ArDataGenerator(
        coefficients=coefficients, series_length=n_sample, seed=seed
    )
    output = generator.generate()

    try:
        np.testing.assert_almost_equal(output, expected_output, decimal=8)
    except AssertionError:
        print("Actual output:", output)
        print("Expected output:", expected_output)
        raise


def test_generate_invalid_coefficients():
    coefficients = [[1.2]]
    n_sample = 5
    seed = 1

    with pytest.raises(ValueError):
        _ = ArDataGenerator(coefficients, n_sample, seed)


def test_generate_with_different_std():
    coefficients = [0.1, 0.2, 0.3]
    n_sample = 5
    seed = 1
    std = 0.5

    generator = ArDataGenerator(coefficients, n_sample, std=std, seed=seed)
    output = generator.generate()

    assert len(output) == n_sample


def test_generate_sine_seasonality():
    generator = SeasonalityGenerator(series_length=100, season_period=12, seed=42)
    seasonality = generator.generate_sine_seasonality()

    assert isinstance(seasonality, np.ndarray)
    assert len(seasonality) == 100
    assert np.allclose(seasonality, np.sin(2 * np.pi * np.arange(100) / 12))


def test_generate_cosine_seasonality():
    generator = SeasonalityGenerator(series_length=120, season_period=24, seed=42)
    seasonality = generator.generate_cosine_seasonality()

    assert isinstance(seasonality, np.ndarray)
    assert len(seasonality) == 120
    assert np.allclose(seasonality, np.cos(2 * np.pi * np.arange(120) / 24))
