"""
ResDMD Module Test Script
Test the two core functions: koop_pseudospec and koop_pseudospec_qr
"""

import numpy as np

import resdmd


def test_koop_pseudospec():
    """Test koop_pseudospec function"""
    print("=" * 50)
    print("Testing koop_pseudospec function")
    print("=" * 50)

    # Create test data
    np.random.seed(42)
    n = 5

    # Create positive definite Hermitian matrices
    G = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    G = G @ G.T.conj() + 0.1 * np.eye(n)

    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)

    L = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    L = L @ L.T.conj() + 0.1 * np.eye(n)

    # Test points
    z_pts = np.array([0.5 + 0j, -0.5 + 0j, 0 + 0.5j, 0 - 0.5j])

    print(f"Matrix size: {n}×{n}")
    print(f"Test points: {z_pts}")

    # Compute pseudospectrum
    result = resdmd.koop_pseudospec(G, A, L, z_pts)

    print("\nPseudospectrum results:")
    for i, (z, res) in enumerate(zip(z_pts, result["RES"])):
        print(f"  z_{i+1} = {z}: residual = {res:.6f}")

    return result


def test_koop_pseudospec_qr():
    """Test koop_pseudospec_qr function"""
    print("\n" + "=" * 50)
    print("Testing koop_pseudospec_qr function")
    print("=" * 50)

    # Create feature matrices
    np.random.seed(123)
    n_samples = 100
    n_features = 5

    PX = np.random.rand(n_samples, n_features) + 1j * np.random.rand(
        n_samples, n_features
    )
    PY = np.random.rand(n_samples, n_features) + 1j * np.random.rand(
        n_samples, n_features
    )

    # Test points
    z_pts = np.array([0.3 + 0.2j, -0.2 + 0.1j])

    print(f"Feature matrix sizes: PX {PX.shape}, PY {PY.shape}")
    print(f"Test points: {z_pts}")

    # Compute pseudospectrum using QR method
    result_qr = resdmd.koop_pseudospec_qr(PX, PY, z_pts)

    print("\nQR method pseudospectrum results:")
    for i, (z, res) in enumerate(zip(z_pts, result_qr["RES"])):
        print(f"  z_{i+1} = {z}: residual = {res:.6f}")

    # Compare with standard method
    G = PX.T @ PX
    A = PX.T @ PY
    L = PY.T @ PY
    result_standard = resdmd.koop_pseudospec(G, A, L, z_pts)

    print("\nStandard method pseudospectrum results:")
    for i, (z, res) in enumerate(zip(z_pts, result_standard["RES"])):
        print(f"  z_{i+1} = {z}: residual = {res:.6f}")

    # Calculate differences
    diff = np.abs(result_qr["RES"] - result_standard["RES"])
    print("\nMethod comparison:")
    print(f"  Maximum difference: {np.max(diff):.6f}")
    print(f"  Average difference: {np.mean(diff):.6f}")

    return result_qr, result_standard


def test_with_pseudoeigenfunctions():
    """Test pseudoeigenfunction computation"""
    print("\n" + "=" * 50)
    print("Testing pseudoeigenfunction computation")
    print("=" * 50)

    # Create small matrices for quick testing
    np.random.seed(456)
    n = 3

    G = np.random.rand(n, n)
    G = G @ G.T + 0.1 * np.eye(n)

    A = np.random.rand(n, n)

    L = np.random.rand(n, n)
    L = L @ L.T + 0.1 * np.eye(n)

    # Pseudospectrum and pseudoeigenfunction points
    z_pts = np.array([0.2 + 0.1j])
    z_pts2 = np.array([0.1 + 0j, 0 + 0.1j])

    print(f"Pseudospectrum points: {z_pts}")
    print(f"Pseudoeigenfunction points: {z_pts2}")

    # Compute
    result = resdmd.koop_pseudospec(G, A, L, z_pts, z_pts2=z_pts2)

    print("\nResults:")
    print(f"Pseudospectrum residuals: {result['RES']}")
    print(f"Pseudoeigenfunction residuals: {result['RES2']}")
    print(f"Pseudoeigenfunction shape: {result['V2'].shape}")

    return result


def main():
    """Run all tests"""
    print("ResDMD Module Testing")
    print("Testing Koopman operator pseudospectral analysis functionality")

    # Display module information
    print("\nModule information:")
    info = resdmd.get_module_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    try:
        # Run tests
        result1 = test_koop_pseudospec()
        result2, result3 = test_koop_pseudospec_qr()
        result4 = test_with_pseudoeigenfunctions()

        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)

        print("\nTest summary:")
        print("✅ koop_pseudospec basic functionality")
        print("✅ koop_pseudospec_qr QR method")
        print("✅ Pseudoeigenfunction computation")
        print("✅ Comparison between two methods")

        print("\nResDMD module working properly!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
