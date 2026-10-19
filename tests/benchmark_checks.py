"""Fast checks for the published benchmark definitions and their CLI wiring."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import sympy as sp
from gvwzz import Options, compute, reconstruct_syzygies, verify_generation_paths
from gvwzz.benchmarks import eco, noon, reimer, henrion, _principal_minor
from gvwzz.cases import CASES, SUITES, get_case, input_metadata
from gvwzz.cli import main, load_case, build_parser
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.verification import verify_result


class BenchmarkChecks(unittest.TestCase):
    def assert_system(self, name, equations):
        case = get_case(name)
        ring = IntegerPolynomialRing(case['variables'], 'grevlex')
        self.assertEqual([ring.parse(p) for p in case['polynomials']],
                         [ring.parse(p) for p in equations])

    def test_01_cyclic_example(self):
        self.assert_system('cyclic3', ['x1+x2+x3', 'x1*x2+x2*x3+x3*x1', 'x1*x2*x3-1'])

    def test_02_katsura_example_and_order(self):
        self.assert_system('katsura3', ['u0+2*u1+2*u2-1',
                                      'u0^2+2*u1^2+2*u2^2-u0', '2*u0*u1+2*u1*u2-u1'])
        for n in range(1, 13):
            original = get_case(f'katsura{n+1}')
            reference = get_case(f'katsura{n}_qfirst')
            self.assertEqual(reference['polynomials'],
                             original['polynomials'][1:] + original['polynomials'][:1])
            self.assertEqual(len(original['variables']), n + 1)

    def test_03_eco_example(self):
        self.assert_system('eco3', ['x1*x3+x1*x2*x3-1', 'x2*x3-2', 'x1+x2+1'])

    def test_04_noon_example(self):
        self.assert_system('noon2', ['10*x1*x2^2-11*x1+10', '10*x2*x1^2-11*x2+10'])

    def test_05_reimer_example(self):
        self.assert_system('reimer2', ['2*x1^2-2*x2^2-1', '2*x1^3-2*x2^3-1'])

    def test_06_henrion_minors_independent_matrix(self):
        for n in (5, 6, 7):
            matrix = sp.diag(*([2] * (n-1) + [sp.Rational(n+1, n)]))
            for j in range(n-1):
                matrix[j, j+1] = matrix[j+1, j] = -1
            for degree in range(1, n+1):
                for subset in combinations(range(1, n+1), degree):
                    indices = [i-1 for i in subset]
                    self.assertEqual(sp.Rational(_principal_minor(n, subset)),
                                     matrix.extract(indices, indices).det())

    def test_07_henrion_coefficients_and_extension(self):
        c = henrion(5)
        self.assertEqual(c['variables'], ['f1', 'f2', 'f3', 'f4', 'f5', 't'])
        self.assertEqual(c['provenance']['denominator_multipliers'], [1,5,5,5,5,1])
        ring = IntegerPolynomialRing(c['variables'], 'grevlex')
        self.assertEqual(ring.parse(c['polynomials'][-1]), ring.parse('f1+2*f2+3*f3+4*f4+5*f5+6*t'))
        self.assertEqual(ring.parse(c['polynomials'][4]), ring.parse('10*f1+10*f2+10*f3+10*f4+6*f5-1075'))
        # A common factor is deliberately retained after clearing denominators.
        self.assertEqual(sp.igcd(*[int(a) for a in ring.parse(c['polynomials'][2]).values()]), 2)
        self.assertEqual(henrion(6)['provenance']['denominator_multipliers'], [1,6,3,6,3,6])
        self.assertEqual(henrion(7)['provenance']['denominator_multipliers'], [1,7,7,7,7,7,7])

    def test_08_catalog_integer_parse_and_degrees(self):
        names = SUITES['classic-paper']
        self.assertEqual(len(names), 18)
        for name in names:
            meta = input_metadata(get_case(name))
            self.assertEqual(meta['coefficient_domain'], 'ZZ')
            self.assertEqual(meta['active_variable_count'], meta['declared_variable_count'])
        for family in ('eco', 'noon', 'reimer', 'henrion', 'katsura-qfirst'):
            for name in SUITES[family]:
                case = get_case(name)
                meta = input_metadata(case)
                n = case['provenance']['parameter']
                degree = {'eco': min(3,n), 'noon': 3, 'reimer': n+1,
                          'henrion': n, 'katsura-qfirst': 2}[family]
                self.assertEqual(meta['max_total_degree'], degree, name)
                self.assertEqual(meta['generator_count'], meta['declared_variable_count'], name)

    def test_09_index_errors(self):
        for generator, low, high in ((eco,2,13), (noon,2,9), (reimer,2,8), (henrion,5,7)):
            for bad in (True, 2.0, '5', low-1, high+1):
                with self.assertRaises(ValueError): generator(bad)

    def test_10_export_without_solving_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'eco3.json'
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(['export-case','eco3','--output',str(path)]), 0)
                self.assertEqual(main(['export-case','eco3','--output',str(path)]), 1)
            data = json.loads(path.read_text())
            self.assertEqual(data['benchmark_input']['input_sha256'], input_metadata(eco(3))['input_sha256'])
            args = build_parser().parse_args(['run','--input',str(path)])
            self.assertEqual(load_case(args)['polynomials'], eco(3)['polynomials'])

    def test_11_small_system_certificates_and_replay(self):
        for name in ('eco3', 'noon2', 'reimer2'):
            case = get_case(name)
            opts = dict(g_first=False, max_seconds=10, max_basis=500, max_candidates=100000)
            online = compute(case['polynomials'], case['variables'], 'grevlex','pot',
                             options=Options(track_representations=True, **opts),
                             show_basis=False, show_u=False)
            self.assertEqual(online.status, 'completed', (name, online.message))
            self.assertTrue(verify_result(online, mode='static', require_certificates=True)['signature_basis_certified'])
            paths = compute(case['polynomials'], case['variables'], 'grevlex','pot',
                            options=Options(record_paths=True, **opts), show_basis=False,show_u=False)
            self.assertEqual(paths.status, 'completed', (name, paths.message))
            self.assertTrue(verify_generation_paths(paths)['signature_basis_certified'])
            self.assertEqual(reconstruct_syzygies(paths, verify=True).syzygies, online.syzygy_representations)

    def test_12_small_default_suites_and_all_names(self):
        self.assertEqual(SUITES['medium'], ['cyclic4','cyclic4_scaled','katsura4','katsura5'])
        for names in SUITES.values():
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(name in CASES for name in names))


if __name__ == '__main__':
    unittest.main(verbosity=2)
