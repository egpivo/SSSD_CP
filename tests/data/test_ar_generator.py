import numpy as np
import pytest

from sssd.data.ar_generator import ArDataGenerator


def test_generate():
    coefficients = [0.1, 0.2, 0.3]
    n_sample = 3
    seed = 1
    expected_output = np.array([1.62434536, -0.44932188, -0.24823487])

    generator = ArDataGenerator(coefficients=coefficients, n_sample=n_sample, seed=seed)
    output = generator.generate()

    try:
        np.testing.assert_almost_equal(output, expected_output, decimal=8)
    except AssertionError:
        print("Actual output:", output)
        print("Expected output:", expected_output)
        raise


def test_generate_invalid_coefficients():
    coefficients = [1.2, 0.3, 0.4]
    n_sample = 5
    seed = 1

    with pytest.raises(ValueError):
        ArDataGenerator(coefficients, n_sample, seed)


def test_generate_invalid_std():
    coefficients = [0.1, 0.2, 0.3]
    n_sample = 5
    seed = 1
    std = 0

    with pytest.raises(ValueError):
        ArDataGenerator(coefficients, n_sample, std=std, seed=seed)


def test_generate_with_different_std():
    coefficients = [0.1, 0.2, 0.3]
    n_sample = 5
    seed = 1
    std = 0.5

    generator = ArDataGenerator(coefficients, n_sample, std=std, seed=seed)
    output = generator.generate()

    assert len(output) == n_sample
