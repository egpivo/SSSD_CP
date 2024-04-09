import numpy as np

from sssd.data.ar_generator import ArDataGenerator


def test_generate():
    coefficients = [0.1, 0.2, 0.3]
    n_sample = 3
    seed = 1
    expected_output = np.array([1.62434536, -0.44932188, -0.24823487])

    generator = ArDataGenerator(coefficients, n_sample, seed)
    output = generator.generate()

    try:
        np.testing.assert_almost_equal(output, expected_output, decimal=8)
    except AssertionError:
        print("Actual output:", output)
        print("Expected output:", expected_output)
        raise
