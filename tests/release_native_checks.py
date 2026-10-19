"""Release checks for checked native workspace order boundaries."""
import unittest

from gvwzz.polynomial import IntegerPolynomialRing

try:
    from gvwzz._native_reduction import NativeWorkspace
except ImportError:
    NativeWorkspace = None


@unittest.skipIf(NativeWorkspace is None, 'native reduction is unavailable')
class NativeWorkspaceOrderChecks(unittest.TestCase):
    def test_lex_order_is_rejected_instead_of_silently_using_grlex(self):
        backend = IntegerPolynomialRing(['x', 'y'], 'lex')
        x, y = backend.gens
        polynomial = x + y**2
        self.assertEqual(backend.leading(polynomial)[0], (1, 0))
        with self.assertRaisesRegex(ValueError, 'graded monomial order'):
            NativeWorkspace(backend, polynomial)

    def test_supported_orders_keep_the_exact_leading_term(self):
        for order in ('grlex', 'grevlex'):
            with self.subTest(order=order):
                backend = IntegerPolynomialRing(['x', 'y'], order)
                x, y = backend.gens
                polynomial = x + y**2
                workspace = NativeWorkspace(backend, polynomial)
                self.assertEqual(workspace.leading(), backend.leading(polynomial))
                self.assertEqual(workspace.result(), polynomial)


if __name__ == '__main__':
    unittest.main()
