# test the polynomial observable

import ddrv


def test_poly_observ():
    """Test the polynomial observable"""
    # create a polynomial observable
    poly_observ = ddrv.PolyObservable(
        name="poly_observ", description="poly_observ", dim_in=2, degree=2
    )
    print(poly_observ)

    # test the apply method
