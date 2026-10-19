cdef extern from *:
    """
    #include <algorithm>
    #include <list>
    #include "gvwzz_wide_key.hpp"
    #if defined(GVWZZ_HAVE_GMP) && GVWZZ_HAVE_GMP
    // The scalar is borrowed from selector scratch storage. Consume it before
    // the next selection; basis rows and their signature coefficients stay fixed.
    struct GvwzzNativeReducerChoice {
        Py_ssize_t id;
        mpz_srcptr scalar;
        int kind;
        Py_ssize_t candidates, skipped;
    };
    template <class MonomialKey, class RatioKey>
    struct GvwzzReducerRowT {
        RatioKey ratio;
        MonomialKey monomial;
        Py_ssize_t rank, terms, bits;
        mpz_t lc, signature;
        GvwzzReducerRowT(RatioKey v, Py_ssize_t r, PyObject *c, PyObject *s,
                        Py_ssize_t t, Py_ssize_t b, MonomialKey m)
            : ratio(v), monomial(m), rank(r), terms(t), bits(b) {
            mpz_init_set(lc, MPZ(c)); mpz_init_set(signature, MPZ(s));
        }
        GvwzzReducerRowT(const GvwzzReducerRowT &other)
            : ratio(other.ratio), monomial(other.monomial), rank(other.rank), terms(other.terms), bits(other.bits) {
            mpz_init_set(lc, other.lc); mpz_init_set(signature, other.signature);
        }
        GvwzzReducerRowT(GvwzzReducerRowT &&other) noexcept
            : ratio(other.ratio), monomial(other.monomial), rank(other.rank), terms(other.terms), bits(other.bits) {
            mpz_init(lc); mpz_init(signature);
            mpz_swap(lc, other.lc); mpz_swap(signature, other.signature);
        }
        GvwzzReducerRowT &operator=(const GvwzzReducerRowT &) = delete;
        ~GvwzzReducerRowT() { mpz_clear(lc); mpz_clear(signature); }
    };
    template <class MonomialKey, class RatioKey, class Hash>
    class GvwzzReducerRowsT {
        using Row = GvwzzReducerRowT<MonomialKey, RatioKey>;
        std::vector<Row> rows;
        using GeometryRecency = std::list<MonomialKey>;
        struct GeometryEntry {
            size_t count = 0;
            std::vector<Py_ssize_t> ids;
            typename GeometryRecency::iterator recent;
        };
        std::unordered_map<MonomialKey, GeometryEntry, Hash> geometry;
        GeometryRecency geometry_recent;
        size_t geometry_bytes = 0, geometry_evictions = 0;
        // Account vector capacity exactly, plus conservative node/key overhead.
        // This bounds retained cache payload, not total allocator/process RSS.
        static constexpr size_t GEOMETRY_ENTRY_BYTES = sizeof(GeometryEntry)
            + 2 * sizeof(MonomialKey) + 8 * sizeof(void *);
        static constexpr size_t GEOMETRY_MAX_ENTRIES = 65536;
        static constexpr size_t GEOMETRY_MAX_BYTES = size_t(32) * 1024 * 1024;
        void trim_geometry() {
            while (geometry.size() > GEOMETRY_MAX_ENTRIES
                   || geometry_bytes > GEOMETRY_MAX_BYTES) {
                const auto victim = geometry.find(geometry_recent.front());
                geometry_bytes -= GEOMETRY_ENTRY_BYTES
                    + victim->second.ids.capacity() * sizeof(Py_ssize_t);
                geometry.erase(victim);
                geometry_recent.pop_front();
                ++geometry_evictions;
            }
        }
        std::vector<Py_ssize_t> positions;
        MonomialKey guards = MonomialKey(0);
        mpz_t scalar, divisor, half, remainder, product, best_remainder, saved_scalar, best_scalar;
        PyObject *answer(Py_ssize_t id, mpz_srcptr value, int kind,
                         Py_ssize_t candidates, Py_ssize_t skipped) {
            if (id < 0) return Py_BuildValue("(nOinn)", id, Py_None, kind, candidates, skipped);
            MPZ_Object *temporary = GMPy_MPZ_New(NULL);
            if (!temporary) return NULL;
            mpz_set(temporary->z, value);
            PyObject *integer = PyNumber_Long(reinterpret_cast<PyObject *>(temporary));
            Py_DECREF(temporary);
            if (!integer) return NULL;
            PyObject *result = Py_BuildValue("(nOinn)", id, integer, kind, candidates, skipped);
            Py_DECREF(integer);
            return result;
        }
    public:
        GvwzzReducerRowsT() {
            mpz_inits(scalar, divisor, half, remainder, product, best_remainder,
                      saved_scalar, best_scalar, NULL);
        }
        ~GvwzzReducerRowsT() {
            mpz_clears(scalar, divisor, half, remainder, product, best_remainder,
                       saved_scalar, best_scalar, NULL);
        }
        PyObject *geometry_cache_info() const {
            return Py_BuildValue("(KKKKK)",
                static_cast<unsigned long long>(geometry.size()),
                static_cast<unsigned long long>(geometry_bytes),
                static_cast<unsigned long long>(GEOMETRY_MAX_ENTRIES),
                static_cast<unsigned long long>(GEOMETRY_MAX_BYTES),
                static_cast<unsigned long long>(geometry_evictions));
        }
        void add(RatioKey ratio, Py_ssize_t rank, PyObject *lc, PyObject *signature,
                 Py_ssize_t terms, Py_ssize_t bits, MonomialKey monomial) {
            rows.emplace_back(ratio, rank, lc, signature, terms, bits, monomial);
        }
        void set_order(PyObject *ordered, MonomialKey mask) {
            positions.resize(rows.size());
            guards = mask;
            for (Py_ssize_t j = 0; j < PyList_GET_SIZE(ordered); ++j) {
                const Py_ssize_t i = PyLong_AsSsize_t(PyList_GET_ITEM(ordered, j));
                positions[static_cast<size_t>(i)] = j;
            }
        }
        GvwzzNativeReducerChoice select_native_impl(
                         PyObject *ids, const std::vector<Py_ssize_t> *native_ids,
                         RatioKey target, Py_ssize_t target_rank,
                         mpz_srcptr coefficient, mpz_srcptr signature, bool mixed,
                         bool euclidean, int policy, bool signature_first) {
            const int preferred = policy == 0 ? 1 : 2; // regular-first, singular-first, first
            Py_ssize_t candidates = 0, skipped = 0, saved = -1, best = -1;
            int saved_kind = 0, best_kind = 0, best_priority = 0;
            const Py_ssize_t size = native_ids ? static_cast<Py_ssize_t>(native_ids->size())
                                               : PyTuple_GET_SIZE(ids);
            for (Py_ssize_t j = 0; j < size; ++j) {
                const Py_ssize_t i = native_ids ? (*native_ids)[static_cast<size_t>(j)]
                                               : PyLong_AsSsize_t(PyTuple_GET_ITEM(ids, j));
                if (i < 0 && PyErr_Occurred()) return {-2, NULL, 0, candidates, skipped};
                const Row &row = rows[static_cast<size_t>(i)];
                if (gvwzz_ratio_less(target, row.ratio) || (row.ratio == target && row.rank > target_rank)) continue;
                ++candidates;
                const bool exact = mpz_divisible_p(coefficient, row.lc);
                if (!exact && !(mixed && euclidean)) continue;
                if (exact) mpz_divexact(scalar, coefficient, row.lc);
                else {
                    mpz_abs(divisor, row.lc);
                    mpz_fdiv_q_2exp(half, divisor, 1);
                    mpz_add(product, coefficient, half);
                    mpz_fdiv_q(scalar, product, divisor);
                    if (mpz_sgn(row.lc) < 0) mpz_neg(scalar, scalar);
                    mpz_mul(product, scalar, row.lc);
                    mpz_sub(remainder, coefficient, product);
                    if (!mpz_sgn(scalar) || mpz_cmpabs(remainder, coefficient) >= 0) continue;
                }
                int kind;
                if (gvwzz_ratio_less(row.ratio, target) || (row.ratio == target && row.rank < target_rank)) kind = 1;
                else {
                    mpz_mul(product, scalar, row.signature);
                    kind = mpz_cmp(product, signature) == 0 ? 3 : 2;
                }
                if (!exact) {
                    if (kind == 3) { ++skipped; continue; }
                    const int priority = signature_first && kind != preferred ? 1 : 0;
                    bool better = best < 0;
                    if (!better) {
                        const Row &previous = rows[static_cast<size_t>(best)];
                        if (priority != best_priority) better = priority < best_priority;
                        else {
                            const int compare = mpz_cmpabs(remainder, best_remainder);
                            if (compare) better = compare < 0;
                            else if (row.terms != previous.terms) better = row.terms < previous.terms;
                            else if (row.bits != previous.bits) better = row.bits < previous.bits;
                            else better = i < best;
                        }
                    }
                    if (better) {
                        best = i; best_kind = kind; best_priority = priority;
                        mpz_set(best_remainder, remainder); mpz_set(best_scalar, scalar);
                    }
                    continue;
                }
                if (!mixed) {
                    if (kind == 1) return {i, scalar, kind, candidates, skipped};
                    continue;
                }
                if (kind == 3) continue;
                if (policy == 2 || kind == preferred) return {i, scalar, kind, candidates, skipped};
                if (saved < 0) {
                    saved = i; saved_kind = kind; mpz_set(saved_scalar, scalar);
                }
            }
            if (signature_first && saved >= 0 && best >= 0 && policy != 2 && best_kind == preferred)
                return {best, best >= 0 ? best_scalar : NULL, best_kind + 10, candidates, skipped};
            if (saved >= 0) return {saved, saved_scalar, saved_kind, candidates, skipped};
            return {best, best >= 0 ? best_scalar : NULL, best_kind + 10, candidates, skipped};
        }
        PyObject *select(PyObject *ids, RatioKey target, Py_ssize_t target_rank,
                         PyObject *coefficient, PyObject *signature, bool mixed,
                         bool euclidean, int policy, bool signature_first) {
            const auto choice = select_native_impl(ids, NULL, target, target_rank,
                               MPZ(coefficient), MPZ(signature), mixed, euclidean,
                               policy, signature_first);
            if (choice.id == -2 && PyErr_Occurred()) return NULL;
            return answer(choice.id, choice.scalar, choice.kind, choice.candidates, choice.skipped);
        }
        GvwzzNativeReducerChoice select_geometry_native(
                         MonomialKey monomial, RatioKey target, Py_ssize_t target_rank,
                         mpz_srcptr coefficient, mpz_srcptr signature, bool mixed,
                         bool euclidean, int policy, bool signature_first) {
            auto found = geometry.find(monomial);
            if (found == geometry.end()) {
                geometry_recent.push_back(monomial);
                try {
                    found = geometry.emplace(monomial, GeometryEntry()).first;
                } catch (...) {
                    geometry_recent.pop_back();
                    throw;
                }
                found->second.recent = --geometry_recent.end();
                geometry_bytes += GEOMETRY_ENTRY_BYTES;
            } else {
                geometry_recent.splice(geometry_recent.end(), geometry_recent,
                                       found->second.recent);
            }
            GeometryEntry &entry = found->second;
            if (entry.count != rows.size()) {
                for (size_t i = entry.count; i < rows.size(); ++i) {
                    if ((((monomial | guards) - rows[i].monomial) & guards) == guards) {
                        const size_t old_capacity = entry.ids.capacity();
                        entry.ids.push_back(static_cast<Py_ssize_t>(i));
                        geometry_bytes += (entry.ids.capacity() - old_capacity)
                            * sizeof(Py_ssize_t);
                    }
                    // If allocation fails, a later query resumes without
                    // duplicating the IDs already appended to this entry.
                    entry.count = i + 1;
                }
                std::sort(entry.ids.begin(), entry.ids.end(),
                    [this](Py_ssize_t a, Py_ssize_t b) { return positions[a] < positions[b]; });
            }
            const auto choice = select_native_impl(NULL, &entry.ids, target, target_rank,
                               coefficient, signature, mixed, euclidean, policy, signature_first);
            // The returned scalar belongs to selector scratch, not this entry.
            // Eviction can therefore include the active entry after selection.
            trim_geometry();
            return choice;
        }
        mpz_srcptr row_signature(Py_ssize_t id) const {
            return rows[static_cast<size_t>(id)].signature;
        }
        PyObject *select_geometry(MonomialKey monomial, RatioKey target, Py_ssize_t target_rank,
                         PyObject *coefficient, PyObject *signature, bool mixed,
                         bool euclidean, int policy, bool signature_first) {
            const auto choice = select_geometry_native(monomial, target, target_rank,
                                MPZ(coefficient), MPZ(signature), mixed, euclidean,
                                policy, signature_first);
            return answer(choice.id, choice.scalar, choice.kind, choice.candidates, choice.skipped);
        }
    };
    using GvwzzReducerRows = GvwzzReducerRowsT<uint64_t, int64_t, std::hash<uint64_t>>;
    using GvwzzWideReducerRows = GvwzzReducerRowsT<GvwzzWideKey, GvwzzWideKey, GvwzzWideHash>;
    #else
    class GvwzzReducerRows {
    public:
        GvwzzReducerRows() {}
        PyObject *geometry_cache_info() const { Py_RETURN_NONE; }
        void add(int64_t, Py_ssize_t, PyObject *, PyObject *, Py_ssize_t, Py_ssize_t, uint64_t) {}
        void set_order(PyObject *, uint64_t) {}
        PyObject *select_geometry(uint64_t, int64_t, Py_ssize_t, PyObject *, PyObject *, bool, bool, int, bool) { Py_RETURN_NONE; }
        PyObject *select(PyObject *, int64_t, Py_ssize_t, PyObject *, PyObject *, bool, bool, int, bool) { Py_RETURN_NONE; }
    };
    class GvwzzWideReducerRows {
    public:
        GvwzzWideReducerRows() {}
        PyObject *geometry_cache_info() const { Py_RETURN_NONE; }
        void add(GvwzzWideKey, Py_ssize_t, PyObject *, PyObject *, Py_ssize_t, Py_ssize_t, GvwzzWideKey) {}
        void set_order(PyObject *, GvwzzWideKey) {}
        PyObject *select_geometry(GvwzzWideKey, GvwzzWideKey, Py_ssize_t, PyObject *, PyObject *, bool, bool, int, bool) { Py_RETURN_NONE; }
        PyObject *select(PyObject *, GvwzzWideKey, Py_ssize_t, PyObject *, PyObject *, bool, bool, int, bool) { Py_RETURN_NONE; }
    };
    #endif
    """
    cppclass GvwzzReducerRows:
        GvwzzReducerRows() except +
        object geometry_cache_info()
        void add(long long ratio, Py_ssize_t rank, PyObject *lc, PyObject *signature,
                 Py_ssize_t terms, Py_ssize_t bits, uint64_t monomial) except +
        void set_order(PyObject *ordered, uint64_t mask) except +
        object select_geometry(uint64_t monomial, long long target, Py_ssize_t target_rank,
                      PyObject *coefficient, PyObject *signature, bint mixed,
                      bint euclidean, int policy, bint signature_first) except +
        object select(PyObject *ids, long long target, Py_ssize_t target_rank,
                      PyObject *coefficient, PyObject *signature, bint mixed,
                      bint euclidean, int policy, bint signature_first)


    cppclass GvwzzWideReducerRows:
        GvwzzWideReducerRows() except +
        object geometry_cache_info()
        void add(GvwzzWideKey ratio, Py_ssize_t rank, PyObject *lc, PyObject *signature,
                 Py_ssize_t terms, Py_ssize_t bits, GvwzzWideKey monomial) except +
        void set_order(PyObject *ordered, GvwzzWideKey mask) except +
        object select_geometry(GvwzzWideKey monomial, GvwzzWideKey target,
                      Py_ssize_t target_rank, PyObject *coefficient, PyObject *signature,
                      bint mixed, bint euclidean, int policy, bint signature_first) except +
        object select(PyObject *ids, GvwzzWideKey target, Py_ssize_t target_rank,
                      PyObject *coefficient, PyObject *signature, bint mixed,
                      bint euclidean, int policy, bint signature_first)
