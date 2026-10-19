"""Native heap ordering against heapq, including arbitrarily wide exact keys."""
import heapq
import os
import random
import sys
import unittest

try:
    if os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError
    from gvwzz._native_pairs import NativeCandidateHeap
except ImportError:
    NativeCandidateHeap = None


@unittest.skipIf(NativeCandidateHeap is None, 'native candidate heap unavailable')
class NativeCandidateHeapChecks(unittest.TestCase):
    def test_interleaved_operations_match_heapq_layout_and_pop_order(self):
        rng = random.Random(280103)
        for policy in ('fifo', 'small-coeff', 'small-lm'):
            native, reference = NativeCandidateHeap(policy), []
            for token in range(1, 1801):
                bits = rng.choice((0, 1, 63, 64, 65, 127, 128, 129, 1041, 4097))
                # Many ties, plus keys whose first 128 bits agree exactly.
                signature = ((1 << bits) + rng.randrange(5)) if bits else 0
                coefficient = (1 << rng.choice((0, 63, 129, 4097))) + rng.randrange(4)
                leading = rng.randrange(8)
                tie = (() if policy == 'fifo' else
                       (coefficient, leading, coefficient) if policy == 'small-coeff'
                       else (leading, coefficient, coefficient))
                row = signature, token % 2, tie, token
                native.push(row)
                heapq.heappush(reference, row)
                if token % 11 == 0:
                    self.assertEqual(native.pop_token(), heapq.heappop(reference)[3])
                if token % 97 == 0:
                    self.assertEqual(list(native), reference)
            self.assertEqual(list(native), reference)
            while reference:
                self.assertEqual(native.pop_token(), heapq.heappop(reference)[3])
            self.assertEqual(len(native), 0)
            with self.assertRaises(IndexError):
                native.pop_token()

    def test_third_tie_and_arbitrarily_large_serial_are_exact(self):
        native = NativeCandidateHeap('small-coeff')
        rows = [(2**300, 1, (2**200, 3, third), 2**4097 + token)
                for token, third in enumerate((5, 4, 3, 2, 1, 0, 5, 4))]
        for row in rows:
            native.push(row)
        self.assertEqual([native.pop_token() for _ in rows],
                         [row[3] for row in sorted(rows)])

    def test_invalid_key_cannot_mutate_existing_heap(self):
        class IntegerSubclass(int):
            pass
        native = NativeCandidateHeap('small-coeff')
        native.push((5, 1, (7, 2, 7), 1))
        before = list(native)
        for invalid in (-1, -(2**300), '3', IntegerSubclass(3)):
            for position in range(5):
                fields = [4, 5, 6, 7, 8]
                fields[position] = invalid
                with self.assertRaises((TypeError, ValueError, RuntimeError)):
                    native.push_fields(fields[0], 1, *fields[1:])
                self.assertEqual(list(native), before)

    def test_clear_releases_wide_integer_references(self):
        signature, coefficient, token = 2**4097 + 1, 2**4201 + 2, 2**4301 + 3
        native = NativeCandidateHeap('small-coeff')
        before = tuple(sys.getrefcount(value) for value in (signature, coefficient, token))
        for _ in range(20):
            native.push_fields(signature, 1, coefficient, 0, coefficient, token)
        native.clear()
        self.assertEqual(tuple(sys.getrefcount(value) for value in (signature, coefficient, token)), before)
        self.assertEqual(list(native), [])

    def test_pop_and_descriptor_exports_release_owned_references(self):
        # Exercise both prefixed integer storage and raw integer references.
        # Repeated equal entries also force the token comparison and move paths.
        fields = tuple((1 << bits) + offset for offset, bits in
                       enumerate((4097, 4201, 4301, 4401, 4501), 1))
        before = tuple(sys.getrefcount(value) for value in fields)
        native = NativeCandidateHeap('small-coeff')
        for _ in range(10):
            for _ in range(41):
                native.push_fields(fields[0], 1, *fields[1:])
            record = native[0]
            exported = list(native)
            self.assertEqual(record, exported[0])
            del record, exported
            while native:
                token = native.pop_token()
                self.assertIs(token, fields[-1])
                del token
            self.assertEqual(tuple(sys.getrefcount(value) for value in fields), before)
        del native
        self.assertEqual(tuple(sys.getrefcount(value) for value in fields), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
