"""Integer-key heap with exact, arbitrary-width comparisons.

The queue's compact grevlex keys are nonnegative Python integers. Keeping the
common <=128-bit keys in native storage removes millions of tracked heap/tie
tuples. Wider keys retain their full Python integer; a cached prefix only avoids
comparisons whose answer is already certain. No variable-count limit is used.
"""

cdef extern from *:
    """
    #include <Python.h>
    #include <cstdint>
    #include <vector>
    #include <utility>
    #include <stdexcept>

    struct GVWQueueInteger {
        uint64_t low = 0, high = 0;
        size_t bits = 0;
        PyObject* wide = nullptr;

        GVWQueueInteger() = default;
        explicit GVWQueueInteger(PyObject* value) {
            if (!PyLong_CheckExact(value)) {
                PyErr_SetString(PyExc_TypeError, "Native heap keys must be builtin integers");
                throw std::runtime_error("Invalid queue integer type");
            }
            int overflow = 0;
            long long small = PyLong_AsLongLongAndOverflow(value, &overflow);
            if (!overflow && !PyErr_Occurred()) {
                if (small < 0) throw std::runtime_error("Native heap keys must be nonnegative");
                low = static_cast<uint64_t>(small);
                uint64_t rest = low;
                if (rest >> 32) { bits += 32; rest >>= 32; }
                if (rest >> 16) { bits += 16; rest >>= 16; }
                if (rest >> 8) { bits += 8; rest >>= 8; }
                if (rest >> 4) { bits += 4; rest >>= 4; }
                if (rest >> 2) { bits += 2; rest >>= 2; }
                if (rest >> 1) { bits += 1; rest >>= 1; }
                bits += rest;
                return;
            }
            if (overflow < 0 || PyErr_Occurred()) throw std::runtime_error("Invalid queue integer");
            PyObject* count = PyObject_CallMethod(value, "bit_length", nullptr);
            if (!count) throw std::runtime_error("Invalid queue integer");
            bits = PyLong_AsSize_t(count);
            Py_DECREF(count);
            if (PyErr_Occurred()) throw std::runtime_error("Invalid queue integer width");
            PyObject* prefix = value;
            Py_INCREF(prefix);
            if (bits > 128) {
                PyObject* shift = PyLong_FromSize_t(bits - 128);
                PyObject* upper = shift ? PyNumber_Rshift(value, shift) : nullptr;
                Py_XDECREF(shift);
                Py_DECREF(prefix);
                if (!upper) throw std::runtime_error("Cannot construct queue prefix");
                prefix = upper;
            }
            low = PyLong_AsUnsignedLongLongMask(prefix);
            if (bits > 64) {
                PyObject* shift = PyLong_FromLong(64);
                PyObject* upper = shift ? PyNumber_Rshift(prefix, shift) : nullptr;
                Py_XDECREF(shift);
                if (!upper) { Py_DECREF(prefix); throw std::runtime_error("Cannot construct queue prefix"); }
                high = PyLong_AsUnsignedLongLongMask(upper);
                Py_DECREF(upper);
            }
            Py_DECREF(prefix);
            if (PyErr_Occurred()) throw std::runtime_error("Invalid queue integer");
            if (bits > 128) { wide = value; Py_INCREF(wide); }
        }
        GVWQueueInteger(const GVWQueueInteger& other)
            : low(other.low), high(other.high), bits(other.bits), wide(other.wide) { Py_XINCREF(wide); }
        GVWQueueInteger(GVWQueueInteger&& other) noexcept
            : low(other.low), high(other.high), bits(other.bits), wide(other.wide) { other.wide = nullptr; }
        GVWQueueInteger& operator=(GVWQueueInteger other) noexcept {
            std::swap(low, other.low); std::swap(high, other.high);
            std::swap(bits, other.bits); std::swap(wide, other.wide); return *this;
        }
        ~GVWQueueInteger() { Py_XDECREF(wide); }
        int compare(const GVWQueueInteger& other) const {
            if (bits != other.bits) return bits < other.bits ? -1 : 1;
            if (high != other.high) return high < other.high ? -1 : 1;
            if (low != other.low) return low < other.low ? -1 : 1;
            if (bits <= 128 || wide == other.wide) return 0;
            int equal = PyObject_RichCompareBool(wide, other.wide, Py_EQ);
            if (equal < 0) throw std::runtime_error("Cannot compare queue integers");
            if (equal) return 0;
            int less = PyObject_RichCompareBool(wide, other.wide, Py_LT);
            if (less < 0) throw std::runtime_error("Cannot compare queue integers");
            return less ? -1 : 1;
        }
        PyObject* value() const {
            if (wide) { Py_INCREF(wide); return wide; }
            PyObject* lower = PyLong_FromUnsignedLongLong(low);
            if (!lower || !high) return lower;
            PyObject* upper = PyLong_FromUnsignedLongLong(high);
            PyObject* shift = PyLong_FromLong(64);
            PyObject* shifted = upper && shift ? PyNumber_Lshift(upper, shift) : nullptr;
            Py_XDECREF(upper); Py_XDECREF(shift);
            PyObject* answer = shifted ? PyNumber_Or(shifted, lower) : nullptr;
            Py_XDECREF(shifted); Py_DECREF(lower); return answer;
        }
    };

    struct GVWQueueReference {
        PyObject* integer = nullptr;
        GVWQueueReference() = default;
        explicit GVWQueueReference(PyObject* value) {
            if (!PyLong_CheckExact(value)) {
                PyErr_SetString(PyExc_TypeError, "Native heap keys must be builtin integers");
                throw std::runtime_error("Invalid queue integer type");
            }
            int overflow = 0;
            long long small = PyLong_AsLongLongAndOverflow(value, &overflow);
            if (overflow < 0 || (!overflow && small < 0) || PyErr_Occurred())
                throw std::runtime_error("Native heap keys must be nonnegative");
            integer = value; Py_INCREF(integer);
        }
        GVWQueueReference(const GVWQueueReference& other) : integer(other.integer) { Py_XINCREF(integer); }
        GVWQueueReference(GVWQueueReference&& other) noexcept : integer(other.integer) { other.integer = nullptr; }
        GVWQueueReference& operator=(GVWQueueReference other) noexcept {
            std::swap(integer, other.integer); return *this;
        }
        ~GVWQueueReference() { Py_XDECREF(integer); }
        int compare(const GVWQueueReference& other) const {
            if (integer == other.integer) return 0;
            int equal = PyObject_RichCompareBool(integer, other.integer, Py_EQ);
            if (equal < 0) throw std::runtime_error("Cannot compare queue integers");
            if (equal) return 0;
            int less = PyObject_RichCompareBool(integer, other.integer, Py_LT);
            if (less < 0) throw std::runtime_error("Cannot compare queue integers");
            return less ? -1 : 1;
        }
        PyObject* value() const { Py_INCREF(integer); return integer; }
    };

    struct GVWQueueHeapEntry {
        GVWQueueInteger signature, first, second;
        GVWQueueReference third, token;
        int priority = 0;
        unsigned char third_source = 0;
        GVWQueueHeapEntry() = default;
        GVWQueueHeapEntry(PyObject* s, int p, PyObject* a, PyObject* b, PyObject* c, PyObject* t)
            : signature(s), first(a), second(b), third(c), token(t), priority(p),
              third_source(c == a ? 1 : c == b ? 2 : 0) {}
        bool less(const GVWQueueHeapEntry& other) const {
            int cmp = signature.compare(other.signature);
            if (cmp) return cmp < 0;
            if (priority != other.priority) return priority < other.priority;
            cmp = first.compare(other.first); if (cmp) return cmp < 0;
            cmp = second.compare(other.second); if (cmp) return cmp < 0;
            if (!third_source || third_source != other.third_source) {
                cmp = third.compare(other.third); if (cmp) return cmp < 0;
            }
            return token.compare(other.token) < 0;
        }
    };

    class GVWQueueHeap {
        std::vector<GVWQueueHeapEntry> entries;
        void sift_down(size_t start, size_t position, GVWQueueHeapEntry item) {
            while (position > start) {
                size_t parent = (position - 1) >> 1;
                if (!item.less(entries[parent])) break;
                entries[position] = std::move(entries[parent]); position = parent;
            }
            entries[position] = std::move(item);
        }
    public:
        size_t size() const noexcept { return entries.size(); }
        void clear() noexcept { std::vector<GVWQueueHeapEntry>().swap(entries); }
        void push(PyObject* s, int p, PyObject* a, PyObject* b, PyObject* c, PyObject* t) {
            GVWQueueHeapEntry item(s, p, a, b, c, t);
            entries.emplace_back();
            sift_down(0, entries.size() - 1, std::move(item));
        }
        PyObject* pop_token() {
            if (entries.empty()) { PyErr_SetString(PyExc_IndexError, "pop from empty candidate heap"); return nullptr; }
            PyObject* result = entries[0].token.value();
            if (!result) return nullptr;
            GVWQueueHeapEntry last = std::move(entries.back()); entries.pop_back();
            if (!entries.empty()) {
                size_t position = 0, child = 1, end = entries.size();
                while (child < end) {
                    size_t right = child + 1;
                    if (right < end && !entries[child].less(entries[right])) child = right;
                    entries[position] = std::move(entries[child]);
                    position = child; child = (position << 1) + 1;
                }
                sift_down(0, position, std::move(last));
            }
            return result;
        }
        PyObject* item(size_t index, bool has_tie) const {
            const GVWQueueHeapEntry& row = entries.at(index);
            PyObject* result = PyTuple_New(4);
            if (!result) return nullptr;
            PyObject* s = row.signature.value();
            PyObject* p = PyLong_FromLong(row.priority);
            PyObject* tie = PyTuple_New(has_tie ? 3 : 0);
            PyObject* token = row.token.value();
            if (has_tie && tie) {
                PyObject* a = row.first.value(); PyObject* b = row.second.value(); PyObject* c = row.third.value();
                if (!a || !b || !c) {
                    Py_XDECREF(a); Py_XDECREF(b); Py_XDECREF(c); Py_DECREF(tie); tie = nullptr;
                } else {
                    PyTuple_SET_ITEM(tie, 0, a); PyTuple_SET_ITEM(tie, 1, b); PyTuple_SET_ITEM(tie, 2, c);
                }
            }
            if (!s || !p || !tie || !token) {
                Py_XDECREF(s); Py_XDECREF(p); Py_XDECREF(tie); Py_XDECREF(token); Py_DECREF(result); return nullptr;
            }
            PyTuple_SET_ITEM(result, 0, s); PyTuple_SET_ITEM(result, 1, p);
            PyTuple_SET_ITEM(result, 2, tie); PyTuple_SET_ITEM(result, 3, token); return result;
        }
    };
    """
    cdef cppclass GVWQueueHeap:
        GVWQueueHeap() except +
        size_t size() noexcept
        void clear() noexcept
        void push(PyObject*, int, PyObject*, PyObject*, PyObject*, PyObject*) except +
        object pop_token() except +
        object item(size_t, bint) except +


cdef class NativeCandidateHeap:
    """Same binary heap order as heapq, without retained Python heap tuples."""
    cdef GVWQueueHeap _heap
    cdef bint _has_tie

    def __init__(self, tie_policy='fifo'):
        self._has_tie = tie_policy != 'fifo'

    cpdef push_fields(self, object signature, int priority, object first,
                      object second, object third, object token):
        self._heap.push(<PyObject*>signature, priority, <PyObject*>first,
                        <PyObject*>second, <PyObject*>third, <PyObject*>token)

    def push(self, record):
        signature, priority, tie, token = record
        if self._has_tie:
            self.push_fields(signature, priority, tie[0], tie[1], tie[2], token)
        else:
            self.push_fields(signature, priority, 0, 0, 0, token)

    cpdef object pop_token(self):
        return self._heap.pop_token()

    cpdef object pop_candidate(self, object queue):
        cdef dict active = queue.active
        cdef dict pending = queue.pending_j
        # Counter supplies __missing__; an exact-dict Cython declaration would
        # reject that subclass and bypass its missing-key behavior.
        cdef object counts = queue.stats.counters
        cdef bint deduplicate = queue.deduplicate
        cdef object token, candidate, kind
        cdef NativePackedCandidate packed
        while self._heap.size():
            token = self._heap.pop_token()
            counts['native_queue_pops'] += 1
            candidate = active.pop(token, None)
            if candidate is None:
                counts['queue_stale_entries'] += 1
                continue
            if isinstance(candidate, NativePackedCandidate):
                packed = candidate
                kind = packed.kind
                if kind == 'J' and deduplicate:
                    del pending[packed._dedup_key]
                counts['j_popped' if kind == 'J' else kind.lower() + '_popped'] += 1
            else:
                kind = candidate.kind
                if kind == 'J' and deduplicate:
                    del pending[candidate._dedup_key]
                counts[kind.lower() + '_popped'] += 1
            return candidate
        raise IndexError('No active candidate')

    cpdef settle_candidate(self, object queue, object candidate):
        cdef object dkey, key, old, membership
        cdef dict settled
        cdef NativePackedCandidate packed
        if not queue.deduplicate:
            return
        if isinstance(candidate, NativePackedCandidate):
            packed = candidate
            dkey, key = packed._dedup_key, packed._leading_key
        else:
            dkey, key = candidate._dedup_key, candidate._leading_key
        settled = queue.settled_j_bounds
        old = settled.get(dkey)
        if old is None or key < old:
            settled[dkey] = key
            membership = queue._native_j_membership
            if membership is not None:
                membership.add_signature(*dkey)

    def __len__(self):
        return self._heap.size()

    def clear(self):
        self._heap.clear()

    def __iter__(self):
        for i in range(self._heap.size()):
            yield self._heap.item(i, self._has_tie)

    def __getitem__(self, index):
        if index < 0:
            index += self._heap.size()
        if index < 0 or index >= self._heap.size():
            raise IndexError('candidate heap index out of range')
        return self._heap.item(index, self._has_tie)

    def __eq__(self, other):
        if isinstance(other, NativeCandidateHeap):
            return list(self) == list(other)
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented
