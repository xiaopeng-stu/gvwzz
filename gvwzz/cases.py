"""Built-in examples, Eder benchmark inputs, and input fingerprints.

This module contains data generation only.  It is not imported by the solver
hot path unless a command-line case is requested.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from itertools import permutations

from .core import encode_integer, INTEGER_ENCODING

INPUT_SHA256_FORMAT = "gvwzz-input-sha256-v2"
from .polynomial import IntegerPolynomialRing

RPC70 = '''242*x^3*z+y^2*z-y*z+11*y
y^3*z^2-y^2*z^2+11*y^2*z+484
y^4*z-y^3*z-10648*x^3+11*y^3-44*y^2+44*y
x^3*y*z^2-2
11*x^3*y^2*z+484*x^3+2*y^2-2*y
117128*x^6-121*x^3*y^3+968*x^3*y^2-968*x^3*y+2*y^4-4*y^3+2*y^2
121*x^6*z^3+y*z-z+11
2178*x^2*y^2*z^3-1452*x^2*y*z^3+15972*x^2*y*z^2
1452*x^2*y^3*z^2-1452*x^2*y^2*z^2+7986*x^2*y^2*z
-726*x^3*y^2*z^2+484*x^3*y*z^2+y^4*z^2-5324*x^3*y*z-y^3*z^2+22*y^3*z-11*y^2*z+121*y^2
2904*x^2*y^3*z^2-2178*x^2*y^2*z^2+23958*x^2*y^2*z+351384*x^2
726*x^2*y^4*z-726*x^2*y^3*z+7730448*x^5+31944*x^2*y^2-31944*x^2*y
-968*x^3*y^3*z+726*x^3*y^2*z-2*y^5*z-7986*x^3*y^2+4*y^4*z+21296*x^3*y-22*y^4-2*y^3*z-10648*x^3+110*y^3-132*y^2+44*y
95832*x^2*y^2*z^2-63888*x^2*y*z^2+702768*x^2*y*z
63888*x^2*y^3*z-63888*x^2*y^2*z+351384*x^2*y^2
-5*y^6*z^2+9*y^5*z^2-88*y^5*z-4*y^4*z^2+253*y^4*z-363*y^4-264*y^3*z+968*y^3+88*y^2*z-484*y^2
726*x^5*z^3-6*x^2*y^2*z^3+3*x^2*y*z^3-33*x^2*y*z^2
726*x^5*y*z^2-3*x^2*y^3*z^2+3*x^2*y^2*z^2
-242*x^6*z^2+3*x^3*y^2*z^2-x^3*y*z^2+22*x^3*y*z
-9*x^2*y^3*z^4+6*x^2*y^2*z^4-66*x^2*y^2*z^3
-6*x^2*y^4*z^3+6*x^2*y^3*z^3-33*x^2*y^3*z^2
4*x^3*y^3*z^3-2*x^3*y^2*z^3+33*x^3*y^2*z^2
-12*x^2*y^4*z^3+9*x^2*y^3*z^3-31944*x^5*z^2-99*x^2*y^3*z^2+264*x^2*y^2*z^2-132*x^2*y*z^2
-3*x^2*y^5*z^2+3*x^2*y^4*z^2-63888*x^5*y*z
7*x^3*y^4*z^2-5*x^3*y^3*z^2+66*x^3*y^3*z-176*x^3*y^2*z+88*x^3*y*z
15972*x^5*y*z^2-66*x^2*y^3*z^2+33*x^2*y^2*z^2-363*x^2*y^2*z-15972*x^2
-33*x^2*y^4*z+33*x^2*y^3*z-351384*x^5-1452*x^2*y^2+1452*x^2*y
-5324*x^6*y*z+11*x^3*y^2*z+121*x^3*y^2-968*x^3*y+484*x^3-4*y^3+6*y^2-2*y
-99*x^2*y^4*z^3+66*x^2*y^3*z^3-726*x^2*y^3*z^2-4356*x^2*y^2*z^2+2904*x^2*y*z^2-31944*x^2*y*z
-66*x^2*y^5*z^2+66*x^2*y^4*z^2-363*x^2*y^4*z-2904*x^2*y^3*z+2904*x^2*y^2*z-15972*x^2*y^2
-11*x^3*y^4*z^2+22*x^3*y^3*z^2-8*y^4*z+12*y^3*z-44*y^3-4*y^2*z+22*y^2
-132*x^2*y^5*z^2+99*x^2*y^4*z^2-702768*x^5*y*z-1089*x^2*y^4*z-2904*x^2*y^3*z+2904*x^2*y^2*z-47916*x^2*y^2
-33*x^2*y^6*z+33*x^2*y^5*z-351384*x^5*y^2-1452*x^2*y^4+1452*x^2*y^3
22*x^3*y^5*z-11*x^3*y^4*z+363*x^3*y^4-968*x^3*y^3+484*x^3*y^2-4*y^5+6*y^4-2*y^3
33*x^5*y^2*z^3-1452*x^5*z^2+12*x^2*y^2*z^2-6*x^2*y*z^2
-33*x^5*y^3*z^2-2904*x^5*y*z
-33*x^6*y^2*z^2-8*x^3*y^2*z+4*x^3*y*z
-263538*x^5*y^2*z+726*x^2*y^4*z-363*x^2*y^3*z-7730448*x^5+3993*x^2*y^3-31944*x^2*y^2+31944*x^2*y
-170069856*x^8+87846*x^5*y^3-1405536*x^5*y^2+363*x^2*y^5+1405536*x^5*y-3267*x^2*y^4+5808*x^2*y^3-2904*x^2*y^2
87846*x^6*y^2-468512*x^6*y+363*x^3*y^4+234256*x^6-4235*x^3*y^3+5808*x^3*y^2-8*y^5-1936*x^3*y+20*y^4-16*y^3+4*y^2
-2108304*x^5*y^2*z^2+1089*x^2*y^5*z^2+1405536*x^5*y*z^2-9438*x^2*y^4*z^2-15460896*x^5*y*z+7986*x^2*y^4*z+14520*x^2*y^3*z^2-63888*x^2*y^3*z-5808*x^2*y^2*z^2+63888*x^2*y^2*z
-1405536*x^5*y^3*z+726*x^2*y^6*z+1405536*x^5*y^2*z-6534*x^2*y^5*z-7730448*x^5*y^2+3993*x^2*y^5+11616*x^2*y^4*z-31944*x^2*y^4-5808*x^2*y^3*z+31944*x^2*y^3
726*x^3*y^5*z-4598*x^3*y^4*z+3993*x^3*y^4+5808*x^3*y^3*z-16*y^6*z-21296*x^3*y^3-1936*x^3*y^2*z+40*y^5*z+10648*x^3*y^2-88*y^5-32*y^4*z+132*y^4+8*y^3*z-44*y^3
-2811072*x^5*y^3*z+1452*x^2*y^6*z+2108304*x^5*y^2*z-12705*x^2*y^5*z-11595672*x^5*y^2+11979*x^2*y^5+20328*x^2*y^4*z-127776*x^2*y^4-8712*x^2*y^3*z+111804*x^2*y^3
-702768*x^5*y^4+363*x^2*y^7+702768*x^5*y^3-3267*x^2*y^6+5808*x^2*y^5-2904*x^2*y^4
363*x^3*y^6-2299*x^3*y^5+2904*x^3*y^4-8*y^7-968*x^3*y^3+20*y^6-16*y^5+4*y^4
-702768*x^8*z^2-726*x^5*y^3*z^2+2904*x^5*y^2*z^2+24*x^2*y^4*z^2-36*x^2*y^3*z^2+12*x^2*y^2*z^2
-1405536*x^8*y*z+726*x^5*y^4*z-5808*x^5*y^3*z+5808*x^5*y^2*z
726*x^6*y^3*z-3872*x^6*y^2*z+1936*x^6*y*z-16*x^3*y^4*z+24*x^3*y^3*z-8*x^3*y^2*z
-15460896*x^8*y*z-3993*x^5*y^4*z+31944*x^5*y^2*z+264*x^2*y^5*z-527076*x^5*y^2-396*x^2*y^4*z+1452*x^2*y^4+132*x^2*y^3*z-726*x^2*y^3
-7730448*x^8*y^2+3993*x^5*y^5-31944*x^5*y^4+31944*x^5*y^3
3993*x^6*y^4-21296*x^6*y^3+10648*x^6*y^2-88*x^3*y^5+132*x^3*y^4-44*x^3*y^3
-1452*x^5*y*z^4+726*x^5*z^4-7986*x^5*z^3+726*x^2*z^2
87846*x^8*z^3-726*x^5*y^2*z^3+726*x^5*y*z^3+726*x^2*y*z-726*x^2*z
726*x^6*y*z^3-363*x^6*z^3+3993*x^6*z^2-242*x^3*z+y^2*z-2*y*z+11*y+z-11
-2178*x^5*y^2*z^5+1452*x^5*y*z^5-15972*x^5*y*z^4
-1452*x^5*y^3*z^4+1452*x^5*y^2*z^4-7986*x^5*y^2*z^3
1089*x^6*y^2*z^4-726*x^6*y*z^4+7986*x^6*y*z^3+y^3*z^2-3*y^2*z^2+11*y^2*z+2*y*z^2-22*y*z
-2904*x^5*y^3*z^4+2178*x^5*y^2*z^4-23958*x^5*y^2*z^3+63888*x^5*y*z^3-31944*x^5*z^3-31944*x^2*z
-726*x^5*y^4*z^3+726*x^5*y^3*z^3-11595672*x^8*z^2-31944*x^2*y+31944*x^2
1452*x^6*y^3*z^3-1089*x^6*y^2*z^3+11979*x^6*y^2*z^2-31944*x^6*y*z^2+15972*x^6*z^2+3*y^4*z-6*y^3*z+33*y^3+3*y^2*z-121*y^2+132*y-44
-726*x^8*z^5+3*x^2*y*z^3
-363*x^8*y*z^4+3*x^2*y^2*z^2-3*x^2*y*z^2
363*x^9*z^4-x^3*y*z^2-x^3*z^2
-15972*x^8*y*z^4-2904*x^5*y*z^3+1452*x^5*z^3+33*x^2*y^2*z^2+1452*x^2*z
3993*x^8*y^2*z^3+527076*x^8*z^2+33*x^2*y^3*z-33*x^2*y^2*z+1452*x^2*y-1452*x^2
7986*x^9*y*z^3+1452*x^6*y*z^2-726*x^6*z^2+11*x^3*y^2*z-22*x^3*y*z+4*y^2-6*y+2
263538*x^8*y^2*z^3-1405536*x^8*y*z^3+702768*x^8*z^3-5808*x^5*y^3*z^3+8712*x^5*y^2*z^3-2904*x^5*y*z^3+702768*x^5*z-363*x^2*y^3*z+2904*x^2*y^2*z-2904*x^2*y*z
255104784*x^11*z^2-131769*x^8*y^3*z^2+1054152*x^8*y^2*z^2-1054152*x^8*y*z^2+702768*x^5*y-363*x^2*y^4-702768*x^5+3267*x^2*y^3-5808*x^2*y^2+2904*x^2*y
-131769*x^9*y^2*z^2+702768*x^9*y*z^2-351384*x^9*z^2+2904*x^6*y^3*z^2-4356*x^6*y^2*z^2+1452*x^6*y*z^2-363*x^3*y^3+2299*x^3*y^2-2904*x^3*y+8*y^4+968*x^3-20*y^3+16*y^2-4*y'''.splitlines()
assert len(RPC70) == 70

AUTHOR_URL = 'https://raw.githubusercontent.com/ederc/singular-benchmarks/refs/heads/master/benchs.lib'
AUTHOR_COMMIT = 'dfd8eb9cacbc772cf76a4f2ace49206279c83dd2'
POLYLIB_URL = 'https://raw.githubusercontent.com/Singular/Singular/spielwiese/Singular/LIB/polylib.lib'
PAPER = 'Eder, Pfister, Popescu, Standard Bases over Euclidean Domains, arXiv:1811.05736v1'


def _case(name, variables, polynomials, procedure, lines, label=None, *, variant='author-integer', note='', dependency=None):
    return {
        'name': name, 'variables': list(variables), 'polynomials': list(polynomials),
        'description': f'{label or name}: {variant}. {note}'.strip(),
        'polynomial_order': 'grevlex', 'module_order': 'gvw-g2',
        'provenance': {
            'paper': PAPER, 'paper_table': 4 if label else None,
            'paper_label': label, 'source_url': AUTHOR_URL,
            'source_commit': AUTHOR_COMMIT, 'source_procedure': procedure,
            'source_lines': lines, 'variant': variant, 'note': note,
            'dependency': dependency, 'coefficient_domain': 'ZZ',
            'variable_and_generator_order_preserved': variant != 'conventional-n-plus-one',
            'normalization': 'none; no nonunit content removal',
            'historical_table_driver_reproduced': False,
        },
    }


def _cyclic(n):
    names = [f'x{i}' for i in range(n)]
    polys = ['+'.join('*'.join(names[(start+j) % n] for j in range(d))
                      for start in range(n)) for d in range(1, n)]
    polys.append('*'.join(names) + '-1')
    return _case(f'eder_cyclic{n}', names, polys, f'cyclicnz({n})', '1369-1378', f'Cyclic-{n}',
                 dependency={'url': POLYLIB_URL, 'procedure': 'cyclic', 'lines': '153-172'})


def _katsura(n, conventional=False):
    # The author wrapper declares n+1 variables. Explicit katsura(n) in the
    # inspected polylib uses n variables. Never silently drop the unused one.
    upper = n if conventional else n-1
    names = [f'x{i}' for i in range(n+1)]
    polys = ['x0+2*(' + '+'.join(names[1:upper+1]) + ')-1']
    for i in range(upper):
        terms = [f'x{abs(j)}*x{abs(i-j)}' for j in range(-upper, upper+1) if abs(i-j) <= upper]
        polys.append('+'.join(terms) + f'-x{i}')
    variant = 'conventional-n-plus-one' if conventional else 'author-wrapper-current-polylib'
    note = (f'{n+1} active variables, n+1-variable alternative; not asserted to be the Table 4 input.'
            if conventional else f'Declares {n+1} variables, uses {n}. The 2018 dependency version was not recovered.')
    return _case(f'eder_katsura{n}' + ('_nplus1' if conventional else ''), names, polys,
                 f'katsuranz({n})' if not conventional else None, '1296-1305',
                 None if conventional else f'Katsura-{n}', variant=variant, note=note,
                 dependency={'url': POLYLIB_URL, 'procedure': 'katsura', 'lines': '240-305',
                             'inspected_version': '4.4.0; 2023-11-24'})


def _eco(n):
    last = n-1
    names = [f'x{i}' for i in range(n)]
    polys = []
    for m in range(last-1):
        terms = [f'x{m}*x{last}']
        terms += [f'x{k}*x{j}*x{last}' for k, j in enumerate(range(m+1, last))]
        polys.append('+'.join(terms) + f'-{m+1}')
    polys += [f'x{last-1}*x{last}-{last}', '+'.join(names[:last]) + '+1']
    return _case(f'eder_eco{n}', names, polys, f'econz({n})', '1505-1537', f'Eco-{n}')


def _f633():
    ns = range(2, 7)
    variables = [f'{prefix}{i}' for prefix in ('U', 'u') for i in reversed(ns)]
    lin = ['2*(' + '+'.join(f'{prefix}{i}' for i in reversed(ns)) + ')+1' for prefix in ('u', 'U')]
    bilinear = []
    for left, right in (('u', 'U'), ('U', 'u')):
        terms = [f'({4 if i > j else -4})*{left}{i}*{right}{j}' for i in ns for j in ns if i != j]
        bilinear.append('1+' + '+'.join(terms) + '+2*(' + '+'.join(f'{left}{i}' for i in ns) + ')')
    polys = lin + bilinear + [f'U{i}*u{i}-1' for i in ns]
    return _case('eder_f633', variables, polys, 'f633z()', '2227-2248', 'F-633')


def _f744():
    variables = [f'{prefix}{i}' for prefix in ('U', 'u') for i in range(7, 1, -1)]
    polys = [
        '2*u7+2*u6+2*u5+2*u4+2*u3+2*u2+1',
        '2*U7+2*U6+2*U5+2*U4+2*U3+2*U2+1',
        '8*U6*u7+8*U5*u7+8*U4*u7+8*U3*u7+8*U2*u7+8*U6*u6+8*U5*u6+8*U4*u6+8*U3*u6+8*U2*u6+8*U5*u5+8*U4*u5+8*U3*u5+8*U2*u5+8*U4*u4+8*U3*u4+8*U2*u4+8*U3*u3+8*U2*u3+8*U2*u2-17',
        '8*U7*u6+8*U6*u6+8*U7*u5+8*U6*u5+8*U5*u5+8*U7*u4+8*U6*u4+8*U5*u4+8*U4*u4+8*U7*u3+8*U6*u3+8*U5*u3+8*U4*u3+8*U3*u3+8*U7*u2+8*U6*u2+8*U5*u2+8*U4*u2+8*U3*u2+8*U2*u2-17',
        '16*U5*U3*u4+16*U5*U2*u4+16*U5*U2*u3+16*U4*U2*u3+8*U5*u4+8*U5*u3+8*U4*u3+8*U5*u2+8*U4*u2+8*U3*u2+18*U5+18*U4+18*U3+18*U2+11',
        '16*U4*u5*u3+16*U4*u5*u2+16*U3*u5*u2+16*U3*u4*u2+8*U4*u5+8*U3*u5+8*U2*u5+8*U3*u4+8*U2*u4+8*U2*u3+18*u5+18*u4+18*u3+18*u2+11',
    ] + [f'U{i}*u{i}-1' for i in range(2, 8)]
    return _case('eder_f744', variables, polys, 'f744z()', '2314-2344', 'F-744')


def _noon(n, homogeneous=True):
    names = [f'x{i}' for i in range(1, n+1)]
    sequence = [8,7,6,5,4,3,1,2] if n == 8 and homogeneous else list(range(1, n+1))
    polys = [f'10*x{i}*(' + '+'.join(f'x{j}^2' for j in range(1, n+1) if j != i) + ')'
             + (f'-11*x{i}*h^2+10*h^3' if homogeneous else f'-11*x{i}+10') for i in sequence]
    variant = 'explicit-ZZ-homogeneous' if homogeneous else 'inhomogeneous-coefficients-over-ZZ'
    note = ('Uses the explicit integer noonNzh source. The paper does not state whether its Noon input is homogeneous.'
            if homogeneous else 'Integer expressions of noonN() interpreted over ZZ, not its source field. Alternative, not silently identified with Table 4.')
    return _case(f'eder_noon{n}' + ('' if homogeneous else '_inhom'), names+(['h'] if homogeneous else []),
                 polys, f'noon{n}zh()' if homogeneous else f'noon{n}()',
                 'noon7/noon7zh/noon8/noon8zh procedures', f'Noon-{n}' if homogeneous else None,
                 variant=variant, note=note)


def _reimer(n):
    names = ['x','y','z','t','u'] if n == 5 else [f'x{i}' for i in range(1, n+1)]
    polys = ['+'.join(f'({2 if i % 2 == 0 else -2})*{x}^{degree}' for i, x in enumerate(names)) + '-1'
             for degree in range(2, n+2)]
    return _case(f'eder_reimer{n}', names, polys, f'reimer{n}z()', '6294-6305; 6352-6363', f'Reimer-{n}')


def _lichtblau():
    return _case('eder_lichtblau', ['t','x','y'], [
        'x-110*t^2+495*t^3-1320*t^4+2772*t^5-5082*t^6+7590*t^7-8085*t^8+5555*t^9-2189*t^10+374*t^11',
        'y-22*t+110*t^2-330*t^3+1848*t^5-3696*t^6+3300*t^7-1650*t^8+550*t^9-88*t^10-22*t^11'
    ], 'lichtblauz()', '445-460', 'Lichtblau')


def _bayes():
    return _case('eder_bayes148', [f'x{i}' for i in range(1,33)], [
        '-x23*x32+x24*x31', '-x22*x32+x24*x30', '-x22*x31+x23*x30',
        '-x21*x32+x24*x29', '-x21*x31+x23*x29', '-x21*x30+x22*x29',
        '-x12*x32+x16*x28', '-x19*x28+x20*x27', '-x11*x31+x15*x27',
        '-x18*x28+x20*x26', '-x18*x27+x19*x26', '-x10*x30+x14*x26',
        '-x17*x28+x20*x25', '-x17*x27+x19*x25', '-x17*x26+x18*x25',
        '-x9*x29+x13*x25', 'x20*x8-x24*x4',
        '-x17*x20-x17*x24-2*x17*x28-x17*x32+x18*x19+x18*x23+2*x18*x27+x18*x31+x19*x22+x19*x30-x20*x21-x20*x29-x21*x24-x21*x28-2*x21*x32+x22*x23+x22*x27+2*x22*x31+x23*x26-x24*x25-x25*x28-x25*x32+x26*x27+x26*x31+x27*x30-x28*x29-x29*x32+x30*x31',
        'x19*x7-x23*x3', 'x18*x6-x2*x22', '-x1*x21+x17*x5'
    ], 'bayes148z()', '5950-5963', 'Bayes-148')


def _mayr():
    raw = '''-x10*x51+x4*x49
x3*x48-x51*x9
x2*x47-x51*x8
x1*x46-x51*x7
x4*x44-x49*x9
x3*x43-x48*x8
x2*x42-x47*x7
x1*x41-x46*x6
x39*x4-x49*x9
x3*x38-x48*x8
x2*x37-x47*x7
x1*x36-x46*x6
x34*x9-x49*x9
x34*x4-x5*x51
x33*x8-x48*x8
x3*x33-x4*x51
x32*x7-x47*x7
x2*x32-x3*x51
x31*x6-x46*x6
x1*x31-x2*x51
x14*x39*x9-x29*x44*x9
x13*x38*x8-x28*x43*x8
x12*x37*x7-x27*x42*x7
x11*x36*x6-x26*x41*x6
x26^2*x46*x6-x51^3*x7
x11^2*x46*x6-x2*x51^3
x21^2*x41*x6-x46*x51^2*x6
x16^2*x36*x6-x46*x51^2*x6
x24*x30*x39*x50*x9-x29*x44*x50*x51*x9
x23*x29*x38*x49*x8-x28*x43*x49*x51*x8
x22*x28*x37*x48*x7-x27*x42*x48*x51*x7
x21*x27*x36*x47*x6-x26*x41*x47*x51*x6
x24*x25*x39*x45*x9-x29*x44*x45*x51*x9
x23*x24*x38*x44*x8-x28*x43*x44*x51*x8
x22*x23*x37*x43*x7-x27*x42*x43*x51*x7
x21*x22*x36*x42*x6-x26*x41*x42*x51*x6
x20*x24*x39*x40*x9-x29*x40*x44*x51*x9
x19*x23*x38*x39*x8-x28*x39*x43*x51*x8
x15*x24*x35*x39*x9-x29*x35*x44*x51*x9
x18*x22*x37*x38*x7-x27*x38*x42*x51*x7
x14*x23*x34*x38*x8-x28*x34*x43*x51*x8
x17*x21*x36*x37*x6-x26*x37*x41*x51*x6
x13*x22*x33*x37*x7-x27*x33*x42*x51*x7
x12*x21*x32*x36*x6-x26*x32*x41*x51*x6'''
    return _case('eder_mayr42', [f'x{i}' for i in range(1,52)], raw.splitlines(), 'mayr42z()', '2177-2199', 'Mayr-42')


YANG_ROW_SETS = ((2,3,4,5),(1,3,4,5),(0,3,4,5),(1,2,4,5),(0,2,4,5),
                (0,1,4,5),(1,2,3,5),(0,2,3,5),(0,1,3,5),(0,1,2,5),
                (1,2,3,4),(0,2,3,4),(0,1,3,4),(0,1,2,4),(0,1,2,3))


def _minor(offset, rows):
    terms = []
    for cols in permutations(range(4)):
        sign = -1 if sum(cols[i] > cols[j] for i in range(4) for j in range(i+1,4)) % 2 else 1
        term = '*'.join(f'x{offset+4*row+col+1}' for row, col in zip(rows,cols))
        terms.append(('+' if sign == 1 else '-') + term)
    return ''.join(terms).lstrip('+')


def _yang():
    polys = ['+'.join(f'x{4*a+c+1}*x{24+4*b+c+1}' for c in range(4))
             for b in range(5,-1,-1) for a in range(5,-1,-1)]
    polys += [_minor(offset, rows) for offset in (24,0) for rows in YANG_ROW_SETS]
    return _case('eder_yang1', [f'x{i}' for i in range(1,49)], polys, 'yang1z()', '224-270', 'Yang-1',
                 note='36 dot products followed by 15 minors of the second matrix and 15 of the first, in source order.')


def _jason():
    return _case('eder_jason210', list('abcdefgh'), [
        'a^2*c^4+b^2*d^4+a*b*c^2*e^2+a*b*d^2*f^2+a*b*c*d*e*g+a*b*c*d*f*h', 'b^6', 'a^6'
    ], 'jason210z()', '1887-1900', 'Jason-210')


# The names below deliberately have an eder_ prefix. Old Katsura/Cyclic cases
# and small default suites are not redefined.
_TABLE = ([_cyclic(6), _cyclic(7)] + [_katsura(n) for n in (7,8,9)]
          + [_eco(9), _eco(10), _f633(), _f744(), _noon(7), _noon(8),
             _reimer(5), _reimer(6), _lichtblau(), _bayes(), _mayr(), _yang(), _jason()])
_VARIANTS = [_katsura(n, True) for n in (7,8,9)] + [_noon(n,False) for n in (7,8)]
_EXTRA = [
    _case('eder_example6', ['x'], ['2*x','3*x'], None, None, note='Paper Example 6, not Table 4.'),
    _case('eder_example19', ['x'], ['6*x','8*x'], None, None, note='Paper Example 19, not Table 4.'),
    _case('eder_rpc_small', ['x','y'], ['x+4','x*y+9','x-y+8'], None, None,
          note='Paper Example 21(1), direct integer computation, no rational precheck.'),
]
for _c in _EXTRA:
    _c['provenance']['source_url'] = 'https://arxiv.org/abs/1811.05736v1'
    _c['provenance']['source_commit'] = None

_EXTRA.append(_case('eder_rpc70', ['x','y','z'], RPC70,
                    'rationalPreCheckExample()', '7301-7378',
                    note='Paper Example 21(2), original 70 generators, no precheck and no added constants.'))

EDER_TABLE_CASES = tuple(c['name'] for c in _TABLE)
EDER_VARIANT_CASES = tuple(c['name'] for c in _VARIANTS)
EDER_EXTRA_CASES = tuple(c['name'] for c in _EXTRA)
EDER_CASES = {c['name']: c for c in _TABLE + _VARIANTS + _EXTRA}

MAX_BENCHMARK_INDEX = 10
MAX_KATSURA_INDEX = 13
CYCLIC_SCALING_FACTORS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def cyclic(n: int, scaled: bool = False) -> dict:
    if type(n) is not int or not 2 <= n <= MAX_BENCHMARK_INDEX:
        raise ValueError(f'Choose an integer 2 <= n <= {MAX_BENCHMARK_INDEX}')
    names = [f'x{i+1}' for i in range(n)]
    factors = CYCLIC_SCALING_FACTORS

    # Original Cyclic ordering: cyclic sums of degrees 1,...,n-1,
    # followed by the product equation x1*...*xn - 1.
    polys = []
    for degree in range(1, n):
        f = ' + '.join('*'.join(names[(i+j) % n] for j in range(degree))
                       for i in range(n))
        polys.append(f'{factors[degree-1]}*({f})' if scaled else f)
    product = '*'.join(names) + ' - 1'
    polys.append(f'{factors[n-1]}*({product})' if scaled else product)
    description = 'Cyclic-root equations in ascending cyclic degree, product equation last'
    if scaled:
        description += ', with generator-wise nonunit scaling over ZZ'
    return {'name': f'cyclic{n}' + ('_scaled' if scaled else ''), 'variables': names,
            'polynomials': polys, 'description': description,
            'generator_order': 'ascending-cyclic-degree-product-last'}


def katsura(n: int) -> dict:
    """Singular numbering: n variables/equations; old katsuraN is now katsura(N+1)."""
    if type(n) is not int or not 1 <= n <= MAX_KATSURA_INDEX:
        raise ValueError(f'Choose an integer 1 <= n <= {MAX_KATSURA_INDEX}')
    m = n - 1
    names = [f'u{i}' for i in range(n)]
    polys = ['u0 + 2*(' + '+'.join(names[1:]) + ') - 1'] if m else ['u0 - 1']
    for i in range(m):
        terms = [f'u{abs(j)}*u{abs(i-j)}' for j in range(-m, m+1) if abs(i-j) <= m]
        polys.append(' + '.join(terms) + f' - u{i}')
    return {'name': f'katsura{n}', 'variables': names, 'polynomials': polys,
            'description': f'Katsura-{n}: Singular numbering; variables={n}, equations={n}; linear equation first'}


CASES = {
    'gcd_line': {'variables': ['x'], 'polynomials': ['2*x', '3*x'],
                 'description': 'Minimal nonunit gcd example'},
    'same_monomial_signatures': {'variables': ['x'], 'polynomials': ['2*x', '3*x', '5*x'],
                 'description': 'Different exact signature coefficients on the same module monomial'},
    'content_trap': {'variables': ['x', 'y'], 'polynomials': ['2*x', '2*y'],
                 'description': 'Must never replace 2*x by x by primitive normalization'},
    'zero_and_signs': {'variables': ['x', 'y'], 'polynomials': ['0', '-6*x', '10*x', '15*y'],
                 'description': 'Zero input, signs and three nonunit leading coefficients'},
    'unit_ideal': {'variables': ['x', 'y'], 'polynomials': ['x^2', 'x*y-1'],
                 'description': 'Ideal contains 1, with nonzero reduction steps'},
    'torsion_plane': {'variables': ['x','y'], 'polynomials': ['2*x+3*y', '3*x-2*y', '5*y^2-1'],
                 'description': 'Inhomogeneous integer coefficients'},
    'small_nonlinear': {'variables': ['x','y'], 'polynomials': ['2*x^2-3*y', '3*x*y-2', '5*y^2-x'],
                 'description': 'Small nonlinear integer system'},
    'all_zero': {'variables': ['x','y'], 'polynomials': ['0','0','0'],
                 'description': 'Empty P and empty main queues'},
    'integer_constant': {'variables': ['x'], 'polynomials': ['6'],
                 'description': '(6) over ZZ must not become the unit ideal'},
}
for _n in range(2, MAX_BENCHMARK_INDEX + 1):
    for _scaled in (False, True):
        _d = cyclic(_n, _scaled)
        CASES[_d['name']] = _d
for _n in range(1, MAX_KATSURA_INDEX + 1):
    _d = katsura(_n)
    CASES[_d['name']] = _d

SUITES = {
    'smoke': ['gcd_line', 'same_monomial_signatures', 'content_trap', 'zero_and_signs',
              'unit_ideal', 'all_zero', 'integer_constant'],
    'quick': ['gcd_line', 'same_monomial_signatures', 'content_trap', 'zero_and_signs',
              'unit_ideal', 'torsion_plane', 'small_nonlinear', 'cyclic3', 'cyclic3_scaled', 'katsura3'],
    'medium': ['cyclic4', 'cyclic4_scaled', 'katsura4', 'katsura5'],
    # Large families are opt-in. Keep the original suites and CLI default small.
    'cyclic': [f'cyclic{n}' for n in range(2, MAX_BENCHMARK_INDEX + 1)],
    'cyclic-scaled': [f'cyclic{n}_scaled' for n in range(2, MAX_BENCHMARK_INDEX + 1)],
    'katsura': [f'katsura{n}' for n in range(1, MAX_KATSURA_INDEX + 1)],
    'to10': ([f'cyclic{n}' for n in range(2, MAX_BENCHMARK_INDEX + 1)]
             + [f'katsura{n}' for n in range(2, MAX_BENCHMARK_INDEX + 2)]),
}


# Source-labelled benchmarks are opt-in and never replace the original cases.
CASES.update(EDER_CASES)
SUITES.update({
    'eder': list(EDER_TABLE_CASES),
    'eder-variants': list(EDER_VARIANT_CASES),
    'eder-extra': list(EDER_EXTRA_CASES),
    'eder-all': list(EDER_TABLE_CASES + EDER_VARIANT_CASES + EDER_EXTRA_CASES),
})


# Conventional catalog additions. Existing names, generator orders, and Eder
# variants remain intact. Reference-order Katsura is exposed under a new name.
from .benchmarks import classical_cases, FAMILY_RANGES, REFERENCE_URL, REFERENCE_DATE
CASES.update(classical_cases())
# Keep qfirst reference names, parameters and inputs unchanged.
for _n in range(1, MAX_KATSURA_INDEX):
    _reference = katsura(_n + 1)
    _reference['name'] = f'katsura{_n}_qfirst'
    _reference['polynomials'] = _reference['polynomials'][1:] + _reference['polynomials'][:1]
    _reference['description'] = f'Katsura-{_n}, quadratic equations first, linear equation last'
    _reference['generator_order'] = 'quadratic m=0,...,n-1; linear equation last'
    _reference['provenance'] = {
        'family': 'katsura', 'parameter': _n, 'source_url': REFERENCE_URL,
        'source_accessed': REFERENCE_DATE, 'source_procedure': 'katsuran',
        'variant': 'reference-generator-order', 'coefficient_domain': 'ZZ',
        'note': 'Reference x0,...,xn renamed position-wise to u0,...,un. '
                'The existing katsuraN names keep their linear-first order.',
        'normalization': 'none', 'historical_timing_reproduction': False,
    }
    CASES[_reference['name']] = _reference

SUITES.update({
    family: [f'{family}{n}' for n in range(low, high + 1)]
    for family, (low, high) in FAMILY_RANGES.items()
})
SUITES.update({
    'katsura-qfirst': [f'katsura{n}_qfirst' for n in range(1, MAX_KATSURA_INDEX)],
    'classic-small': ['cyclic3', 'katsura3', 'eco3', 'noon2', 'reimer2'],
    'classic-medium': ['cyclic4', 'katsura5', 'eco5', 'noon4', 'reimer3', 'henrion5'],
    'classic-paper': ([f'cyclic{n}' for n in (7, 8, 9)]
                      + [f'katsura{n}_qfirst' for n in (10, 11, 12)]
                      + [f'eco{n}' for n in (11, 12, 13)]
                      + [f'noon{n}' for n in (7, 8, 9)]
                      + [f'reimer{n}' for n in (6, 7, 8)]
                      + [f'henrion{n}' for n in (5, 6, 7)]),
})


INPUT_ORDER_FORMAT = "gvwzz-input-generator-order-v1"


def reverse_case_generators(case: dict) -> dict:
    """Return a detached input with the supplied generator sequence reversed.

    This is input preparation, not a solver/queue option. Variables, coefficients,
    polynomial/module orders, position priorities and block sizes are unchanged.
    Signatures and certificate coordinates are subsequently created against the
    effective input sequence. Exported inputs are already transformed: loading
    them does not reapply the reversal unless explicitly requested.
    """
    if not isinstance(case, dict) or not isinstance(case.get("polynomials"), list):
        raise ValueError("Generator reversal requires an input dictionary with a polynomials list")
    result = deepcopy(case)
    count = len(result["polynomials"])
    result["polynomials"].reverse()
    previous = result.get("input_order")
    if isinstance(previous, dict) and previous.get("format") == INPUT_ORDER_FORMAT:
        permutation = previous.get("effective_to_source")
        reversals = previous.get("reversals")
        source_metadata = previous.get("source_metadata")
        if (not isinstance(permutation, list)
                or any(type(i) is not int for i in permutation)
                or sorted(permutation) != list(range(1, count + 1))
                or type(reversals) is not int or reversals < 1
                or not isinstance(source_metadata, dict)):
            raise ValueError("Invalid input_order reversal metadata")
        expected = list(range(1, count + 1))
        if reversals % 2:
            expected.reverse()
        if permutation != expected:
            raise ValueError("input_order does not describe a generator reversal")
        permutation = permutation[::-1]
        reversals += 1
        source_name = previous.get("source_name", result.get("name", "custom"))
    else:
        permutation = list(range(count, 0, -1))
        reversals = 1
        source_name = result.get("name", "custom")
        source_metadata = {key: deepcopy(result[key]) for key in
                           ("description", "generator_order", "provenance") if key in result}

    # Restore the source annotations first so that two explicit reversals also
    # restore per-generator metadata, rather than piling up contradictory notes.
    for key in ("description", "generator_order", "provenance"):
        if key in source_metadata:
            result[key] = deepcopy(source_metadata[key])
        else:
            result.pop(key, None)
    reversed_order = bool(reversals % 2)
    if reversed_order:
        result["description"] = f"{source_name}: input generators in reverse order; variables unchanged"
        result["generator_order"] = "full reversal of the source generator sequence"
        provenance = result.get("provenance")
        if isinstance(provenance, dict):
            source_note = provenance.get("note", "")
            provenance["note"] = ("Generator sequence fully reversed; variable order unchanged. "
                                  "Source-order note: " + str(source_note))
            provenance["generator_order"] = result["generator_order"]
            provenance["generator_order_preserved"] = False
            provenance["variable_order_preserved"] = True
            provenance["variable_and_generator_order_preserved"] = False
            provenance["source_generator_indices"] = list(permutation)
            provenance["source_generator_indices_convention"] = "1-based; effective to source position"
            # These are attached to equations, unlike module position priorities.
            multipliers = provenance.get("denominator_multipliers")
            if isinstance(multipliers, list) and len(multipliers) == count:
                provenance["denominator_multipliers"] = multipliers[::-1]

    result["input_order"] = {
        "format": INPUT_ORDER_FORMAT,
        "direction": "reversed" if reversed_order else "original",
        "reversals": reversals,
        "source_name": source_name,
        "effective_to_source": permutation,
        "index_convention": "1-based; effective input i is source input effective_to_source[i-1]",
        "source_metadata": source_metadata,
    }
    # An input export may already have a digest for its previous ordered list.
    # Let the export/run path regenerate it from the actual transformed input.
    result.pop("benchmark_input", None)
    return result


def get_case(name: str, *, reverse_generators: bool = False) -> dict:
    if type(reverse_generators) is not bool:
        raise ValueError("reverse_generators must be a boolean")
    if name not in CASES:
        if name.endswith("_reversed") and name[:-9] in CASES:
            raise ValueError(f"Separate reversed cases are not provided. Use --case {name[:-9]} "
                             "--reverse-generators, or export-case with --reverse-generators")
        raise ValueError(f'Unknown case {name}. Run: python -m gvwzz list')
    case = dict(CASES[name], name=name)
    return reverse_case_generators(case) if reverse_generators else case


def random_case(seed: int, nvars: int = 2, ngens: int = 3, degree: int = 2,
                terms: int = 4, coefficient_bound: int = 5) -> dict:
    if min(nvars, ngens, degree, terms, coefficient_bound) <= 0:
        raise ValueError('Random-case parameters must be positive')
    rng = random.Random(seed)
    names = [f'x{i+1}' for i in range(nvars)]
    polys = []
    for _ in range(ngens):
        parts = []
        for _ in range(terms):
            exps = [0] * nvars
            for _ in range(rng.randint(0, degree)):
                exps[rng.randrange(nvars)] += 1
            c = rng.choice([x for x in range(-coefficient_bound, coefficient_bound+1) if x])
            mon = '*'.join(f'{names[i]}^{e}' for i, e in enumerate(exps) if e) or '1'
            parts.append(f'({c})*{mon}')
        polys.append(' + '.join(parts))
    return {'name': f'random_{seed}', 'variables': names, 'polynomials': polys,
            'description': 'Reproducible random sparse integer input', 'seed': seed}

def input_metadata(case: dict) -> dict:
    b = IntegerPolynomialRing(case['variables'], 'grevlex')
    polys = [b.deserialize(text) for text in case['polynomials']]
    # Canonical MATHEMATICAL terms: ignore display strings, runtime settings,
    # monomial order used by the solver, and all timing/verification fields.
    # Generator and declared-variable sequences are retained without unit
    # normalization. encode_integer has a fixed, versioned convention.
    canonical = {'domain': 'ZZ', 'variables': list(case['variables']),
                 'polynomials': [[[list(m), encode_integer(c)] for m, c in sorted(f.items())] for f in polys]}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    active = [name for i, name in enumerate(case['variables']) if any(any(m[i] for m in f) for f in polys)]
    return {
        'input_sha256': digest, 'input_sha256_format': INPUT_SHA256_FORMAT,
        'integer_encoding': INTEGER_ENCODING, 'coefficient_domain': 'ZZ',
        'declared_variable_count': len(case['variables']), 'active_variable_count': len(active),
        'active_variables': active, 'generator_count': len(polys),
        'term_counts': [len(f) for f in polys],
        'max_total_degree': max((sum(m) for f in polys for m in f), default=-1),
        'max_coefficient_bits': max((abs(int(c)).bit_length() for f in polys for c in f.values()), default=0),
        'homogeneous': all(len({sum(m) for m in f}) <= 1 for f in polys),
        'provenance': case.get('provenance'),
        **({'input_order': deepcopy(case['input_order'])} if 'input_order' in case else {}),
        'fingerprint_convention': ('ZZ; declared variable sequence; ordered polynomial sequence; '
                                   'sorted exponent/coefficient terms; no normalization; '
                                   'fixed decimal <=640 digits, otherwise signed lowercase hex; '
                                   'compact sorted-key JSON with ASCII escaping'),
    }
