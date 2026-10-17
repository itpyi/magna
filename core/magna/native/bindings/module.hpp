// Conversion helpers and every production binding, shared by the _native module
// (bindings.cpp) and the test-only oracle module built by tests/support/native_testing.py.
#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <chrono>
#include <cmath>
#include <random>
#include <exception>
#include <memory>
#include "../evaluate/global.hpp"
#include "../evaluate/boundary_trace.hpp"
#include "../evaluate/boundary_totals.hpp"
#include "../foundation/numeric.hpp"
#include "../evaluate/algebra.hpp"
#include "../compile/frontend.hpp"
#include "../compile/file_lowering.hpp"
#include "../compile/expressions.hpp"
#include "../compile/deferred.hpp"
#include "../compile/program.hpp"
#include "../compile/instrument.hpp"
#include "../runtime/branch.hpp"
#include "../runtime/proposal.hpp"
#include "../runtime/certificate_cache.hpp"
#include "../runtime/estimate.hpp"
#include "../runtime/boundary_service.hpp"
#include "../runtime/operation_boundary.hpp"
#include "../runtime/operation_transport.hpp"
#include "../runtime/full_record.hpp"
#include "../compile/task.hpp"
#include "../runtime/causal.hpp"
#include "../compile/terminal.hpp"
#include "../compile/file_program.hpp"
#include "../runtime/file_execution.hpp"
#include "../runtime/file_enumeration.hpp"
#include "../runtime/file_statistics.hpp"
#include "../runtime/file_batch.hpp"
#include "../runtime/file_parallel.hpp"
#include "../runtime/history_efficiency.hpp"
#include "../runtime/marginal.hpp"
#include "../runtime/noise.hpp"
#include "../runtime/noise_compile.hpp"
#include "../runtime/noise_model.hpp"
#include "../runtime/noise_layout.hpp"
#include "../runtime/sampling.hpp"
#include "../runtime/low_weight.hpp"
using namespace magic_native;
namespace {
struct Object {
    PyObject* p;
    explicit Object(PyObject* v) : p(v) {
        if(!p) throw std::invalid_argument("Python conversion failed");
    }
    ~Object() { Py_XDECREF(p); }
    Object(const Object&) = delete;
};
struct Seq : Object {
    explicit Seq(PyObject* v) : Object(PySequence_Fast(v, "expected sequence")) {}
    size_t size() const { return size_t(PySequence_Fast_GET_SIZE(p)); }
    PyObject* operator[](size_t i) const {
        if(i >= size()) throw std::invalid_argument("sequence shape");
        return PySequence_Fast_GET_ITEM(p, Py_ssize_t(i));
    }
};
PyObject* field(PyObject* d, const char* k) {
    auto* p = PyDict_GetItemString(d, k);
    if(!p) throw std::invalid_argument(std::string("missing plan field: ") + k);
    return p;
}
int64_t integer(PyObject* p) {
    if(!PyLong_CheckExact(p)) throw std::invalid_argument("integer required");
    auto v = PyLong_AsLongLong(p);
    if(PyErr_Occurred()) throw std::invalid_argument("integer outside index range");
    return v;
}
size_t natural(PyObject* p) {
    auto v = integer(p);
    if(v < 0) throw std::invalid_argument("negative dimension");
    return size_t(v);
}
int index_int(PyObject* p) {
    auto v = integer(p);
    if(v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
        throw std::invalid_argument("index outside native int range");
    return int(v);
}
Bits bits(PyObject* p) {
    if(!PyLong_CheckExact(p)) throw std::invalid_argument("packed integer required");
    Object s(PyNumber_ToBase(p, 16));
    const char* t = PyUnicode_AsUTF8(s.p);
    if(!t || t[0] == '-') throw std::invalid_argument("negative bit mask");
    std::string h = t;
    if(h.rfind("0x", 0) != 0) throw std::invalid_argument("packed integer encoding");
    Bits b;
    for(size_t j = h.size(); j > 2;) {
        size_t first = j - 2 > 16 ? j - 16 : 2;
        b.w.push_back(std::stoull(h.substr(first, j - first), nullptr, 16));
        j = first;
    }
    b.trim();
    return b;
}
Integer big(PyObject* p) {
    if(!PyLong_CheckExact(p)) throw std::invalid_argument("coefficient integer required");
    Object s(PyNumber_ToBase(p, 16));
    std::string h = PyUnicode_AsUTF8(s.p);
    Integer v;
    v.neg = h[0] == '-';
    if(v.neg) h.erase(0, 1);
    for(size_t j = h.size(); j > 2;) {
        size_t first = j - 2 > 8 ? j - 8 : 2;
        v.d.push_back(uint32_t(std::stoul(h.substr(first, j - first), nullptr, 16)));
        j = first;
    }
    v.trim();
    return v;
}
// An integer phase exponent of T, S or PHASE, reduced modulo eight.
int phase_exponent(PyObject* p) {
    auto n = big(p);
    int exponent = n.zero() ? 0 : int(n.d[0] & 7);
    return n.neg ? mod8(-exponent) : exponent;
}
std::vector<Bits> bitvec(PyObject* p) {
    Seq s(p);
    std::vector<Bits> v;
    v.reserve(s.size());
    for(size_t i = 0; i < s.size(); i++) v.push_back(bits(s[i]));
    return v;
}
std::vector<int> intvec(PyObject* p) {
    Seq s(p);
    std::vector<int> v;
    for(size_t i = 0; i < s.size(); i++) v.push_back(index_int(s[i]));
    return v;
}
void masks(const std::vector<Bits>& v, size_t width) { validate_masks(v, width); }
Ket ket(PyObject* p) {
    Seq s(p);
    if(s.size() != 7) throw std::invalid_argument("ket shape");
    Ket k;
    k.width = natural(s[0]);
    k.base = bits(s[1]);
    k.columns = bitvec(s[2]);
    k.linear = intvec(s[3]);
    k.upper = bitvec(s[4]);
    k.power = integer(s[5]);
    k.phase = mod8(index_int(s[6]));
    if(k.power < -1000000 || k.power > 1000000) throw std::invalid_argument("oracle power budget");
    auto n = k.columns.size();
    if(k.base.size() > k.width || k.linear.size() != n || k.upper.size() != n || k.width > 1000000 || n > 1000000)
        throw std::invalid_argument("ket dimensions");
    masks(k.columns, k.width);
    masks(k.upper, n);
    for(size_t j = 0; j < n; j++)
        if(k.linear[j] < 0 || k.linear[j] > 3 || k.upper[j] != k.upper[j].above(j))
            throw std::invalid_argument("quadratic convention");
    return k;
}
Scalar scalar(PyObject* p) {
    Seq s(p);
    if(s.size() != 2) throw std::invalid_argument("scalar shape");
    Seq c(s[0]);
    if(c.size() != 4) throw std::invalid_argument("four coefficients required");
    Scalar v;
    for(size_t j = 0; j < 4; j++) v.c[j] = big(c[j]);
    v.power = integer(s[1]);
    if(v.power < -1000000 || v.power > 1000000) throw std::invalid_argument("oracle power budget");
    v.normalize();
    return v;
}
PyObject* output_big(const Integer& v) {
    if(v.zero()) return PyLong_FromLong(0);
    static const char* d = "0123456789abcdef";
    std::string h = v.neg ? "-0x" : "0x";
    for(size_t i = v.d.size(); i-- > 0;)
        for(int j = 7; j >= 0; j--) h += d[(v.d[i] >> (4 * j)) & 15];
    return PyLong_FromString(h.data(), nullptr, 16);
}
PyObject* output_bits(const Bits& v) {
    Integer x;
    for(auto w : v.w) {
        x.d.push_back(uint32_t(w));
        x.d.push_back(uint32_t(w >> 32));
    }
    x.trim();
    return output_big(x);
}
PyObject* output_scalar(const Scalar& v) {
    Object c(PyTuple_New(4));
    for(size_t j = 0; j < 4; j++) {
        auto* p = output_big(v.c[j]);
        if(!p) throw std::bad_alloc();
        PyTuple_SET_ITEM(c.p, j, p);
    }
    return Py_BuildValue("(OL)", c.p, (long long)v.power);
}
PyObject* output_ket(const Ket& k) {
    if(k.zero) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    Object cols(PyTuple_New(k.columns.size())), upper(PyTuple_New(k.upper.size())),
            linear(PyTuple_New(k.linear.size()));
    for(size_t j = 0; j < k.columns.size(); j++) {
        PyTuple_SET_ITEM(cols.p, j, output_bits(k.columns[j]));
        PyTuple_SET_ITEM(upper.p, j, output_bits(k.upper[j]));
        PyTuple_SET_ITEM(linear.p, j, PyLong_FromLong(k.linear[j]));
    }
    Object base(output_bits(k.base));
    return Py_BuildValue(
            "(nOOOOLi)", Py_ssize_t(k.width), base.p, cols.p, linear.p, upper.p, (long long)k.power, k.phase);
}
PyObject* resource_limit_exception = nullptr;
PyObject* limit_value(double v) {
    return v == std::floor(v) && std::fabs(v) < 9007199254740992. ? PyLong_FromLongLong((long long)v)
                                                                  : PyFloat_FromDouble(v);
}
// ResourceLimit carries the setting, its value and a lower bound on the need (or None).
void raise_resource_limit(const ResourceLimitError& e) {
    auto* type = resource_limit_exception ? resource_limit_exception : PyExc_RuntimeError;
    PyErr_Clear();
    PyObject* value = PyObject_CallFunction(type, "s", e.what());
    if(!value) return;
    PyObject *option = PyUnicode_FromString(e.option.c_str()), *limit = limit_value(e.limit),
             *needed = e.needed ? limit_value(*e.needed) : Py_NewRef(Py_None);
    if(option && limit && needed && !PyObject_SetAttrString(value, "option", option) &&
            !PyObject_SetAttrString(value, "limit", limit) && !PyObject_SetAttrString(value, "needed", needed))
        PyErr_SetObject(type, value);
    Py_XDECREF(option);
    Py_XDECREF(limit);
    Py_XDECREF(needed);
    Py_DECREF(value);
}
// Configured budgets raise ResourceLimit and allocation or fixed native capacity
// failures MemoryError. Remaining native failures are internal invariant failures.
// Thrown after a Python error (such as KeyboardInterrupt) has been set; error() keeps it.
struct PendingPythonError : std::exception {};
PyObject* error() {
    try {
        throw;
    } catch(const PendingPythonError&) {
    } catch(const std::bad_alloc&) { PyErr_NoMemory(); } catch(const ResourceLimitError& e) {
        raise_resource_limit(e);
    } catch(const std::length_error& e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
    } catch(const std::invalid_argument& e) {
        if(!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, e.what());
    } catch(const std::out_of_range& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch(const std::overflow_error& e) {
        PyErr_SetString(PyExc_OverflowError, e.what());
    } catch(const std::exception& e) { PyErr_SetString(PyExc_RuntimeError, e.what()); } catch(...) {
        // No native exception may cross into CPython frames.
        PyErr_SetString(PyExc_RuntimeError, "unexpected native exception");
    }
    return nullptr;
}
#include "algebra_bindings.hpp"
#include "compiler_bindings.hpp"
#include "phase_bindings.hpp"
#include "marginal_bindings.hpp"
#include "frontend_bindings.hpp"
#include "file_lowering_bindings.hpp"
#include "expression_bindings.hpp"
#include "program_bindings.hpp"
#include "instrument_bindings.hpp"
#include "causal_bindings.hpp"
#include "terminal_bindings.hpp"

Assignment assignment(PyObject* data, const Plan& p) {
    Seq r(data);
    if(r.size() != 2 && r.size() != 4) throw std::invalid_argument("batch row shape");
    Assignment a;
    a.record = bits(r[0]);
    if(a.record.size() > p.nrecords) throw std::invalid_argument("record word outside plan");
    Seq fs(r[1]);
    for(size_t i = 0; i < fs.size(); i++) {
        Seq f(fs[i]);
        if(f.size() != 3) throw std::invalid_argument("fault shape");
        Fault fault{natural(f[0]), index_int(f[1]), index_int(f[2])};
        if(fault.location >= p.locations.size() || fault.pauli < 0 || fault.pauli > 3)
            throw std::invalid_argument("fault index/code");
        a.faults.push_back(fault);
    }
    if(r.size() == 4) {
        a.feedback = true;
        a.actual_magic = intvec(r[2]);
        a.feedback_phase = index_int(r[3]);
        if(a.actual_magic.size() != p.magic.size() || a.feedback_phase < 0 || a.feedback_phase > 7)
            throw std::invalid_argument("feedback dimensions/phase");
        for(size_t j = 0; j < p.magic.size(); j++)
            if(a.actual_magic[j] < 0 || a.actual_magic[j] > 7 || mod8(a.actual_magic[j] - p.magic[j]) % 2)
                throw std::invalid_argument("feedback changes the odd phase signature");
    }
    return a;
}

PyObject* contract_ket(PyObject*, PyObject* arg) {
    try {
        Workspace ws;
        return output_ket(contract(ket(arg), ws));
    } catch(...) { return error(); }
}
#include "global_bindings.hpp"
#include "boundary_trace_bindings.hpp"
#include "file_noise_bindings.hpp"
#include "history_efficiency_bindings.hpp"
#include "noise_model_bindings.hpp"
#include "noise_layout_bindings.hpp"
#include "low_weight_bindings.hpp"
#include "file_statistics_bindings.hpp"
#include "file_execution_bindings.hpp"
#include "file_enumeration_bindings.hpp"
#include "branch_bindings.hpp"
#include "proposal_bindings.hpp"
#include "certificate_cache_bindings.hpp"
#include "estimate_bindings.hpp"
#include "boundary_service_bindings.hpp"
#include "operation_boundary_bindings.hpp"
#include "operation_transport_bindings.hpp"
#include "full_record_bindings.hpp"
#include "task_bindings.hpp"
} // namespace
