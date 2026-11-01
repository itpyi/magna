// Test-only extension: the production bindings plus independent oracles that the
// installed package does not ship. Built and cached by tests/support/native_testing.py.
#include "bindings/module.hpp"
#include "oracles.hpp"
namespace {
PyMethodDef methods[] = {
        {"cuda_oversized_weight_plan", cuda_oversized_weight_plan, METH_NOARGS, "Construct an oversized entry plan."},
        {"file_execution_faults", file_execution_faults, METH_VARARGS,
                                 "Export native original or feedback assignments."},
        {"summarize_scientific_draws", summarize_scientific_draws, METH_VARARGS,
                "Range-safe native contribution statistics."},
        {"enumerate_noise_categories", enumerate_noise_categories, METH_VARARGS,
                "Bounded exhaustive categorical noise assignments by event weight."},
        {"make_file_noise", make_file_noise, METH_VARARGS, "Prepare immutable exact original-noise categories."},
        {"draw_file_noise", draw_file_noise, METH_VARARGS, "Batch seeded draws from prepared noise categories."},
        {"draw_file_noise_actions", draw_file_noise_actions, METH_VARARGS,
                "Sparse actions from immutable native noise table."},
        {"make_feedback_template", make_feedback_template, METH_VARARGS,
                "Compile immutable Pauli and virtual-control templates."},
        {"feedback_template_faults", feedback_template_faults, METH_VARARGS,
                "Supply reached decision offsets to a fixed template."},
        {"lower_circuit_file", lower_circuit_file, METH_VARARGS,
                "Lower physical syntax and fixed templates into native operations."},
        {"make_file_efficiency", make_file_efficiency, METH_NOARGS,
                "Create constant-storage history efficiency statistics."},
        {"add_file_efficiency", add_file_efficiency, METH_VARARGS, "Add a completed weighted history contribution."},
        {"file_efficiency_summary", file_efficiency_summary, METH_O, "Read history efficiency diagnostics."},
        {"debug_prepare_compiler", debug_prepare_compiler, METH_O, "Independent whole-instrument preparation oracle."},
        {"trace_probability_tensor", trace_probability_tensor, METH_VARARGS,
                "Symbolic single-sided probability tensor."},
        {"boundary_commuting_rows", boundary_commuting_rows, METH_VARARGS,
                "Deterministic record-parity kernel of common transported phase axes."},
        {"phase_database_validate", phase_database_test, METH_O, "Reconstruct all supplied database points exactly."},
        {"get_file_execution_boundary", get_file_execution_boundary, METH_VARARGS,
                "Retrieve a native boundary from its bounded repository."},
        {"inspect_file_boundary", inspect_file_boundary, METH_O, "Export a boundary inspection snapshot."},
        {"scalar_arithmetic", scalar_test, METH_VARARGS, "Arbitrary integer scalar oracle entry."},
        {"sample_boundary_terms", sample_boundary_terms, METH_VARARGS,
                "Exact C++ sampling of a completed global probability tensor."},
        {"sample_file_categories", sample_file_categories, METH_VARARGS,
                "Exact rational original-noise category draws."},
        {"summarize_file_draws", summarize_file_draws, METH_VARARGS,
                "C++ joint contribution means and paired ratio standard errors."},
        {"file_scalar_ratios", file_scalar_ratios, METH_VARARGS,
                "Range-safe C++ conversion of exact scalar ratios to binary scientific values."},
        {"summarize_file_weights", summarize_file_weights, METH_VARARGS,
                "Range-safe importance-weight mean and effective sample size."},
        {"phase_polynomial", phase_polynomial_test, METH_VARARGS, "Exact polynomial quotient."},
        {"overlap", overlap_test, METH_VARARGS, "Exact overlap of two stabilizer kets."},
        {"quadratic_restrict", quadratic_restrict, METH_VARARGS, "Restrict a quadratic form to an affine subspace."},
        {"gf2_solve", gf2_solve, METH_VARARGS, "Least-first-pivot GF(2) solution with kernel and reduced rows."},
        {"gf2_right_inverse", gf2_right_inverse, METH_VARARGS, "Right-inverse columns of independent GF(2) rows."},
        {"gaussian_scalar", gaussian_scalar, METH_VARARGS, "Exact Gaussian integer times a root of unity."},
        {"real_sign", real_sign_oracle, METH_O, "Exact sign of a real cyclotomic scalar."},
        {"algebra_bernoulli", algebra_bernoulli, METH_VARARGS, "Exact Bernoulli draw from a one-bit RNG stream."},
        {"model_weighted_choice", model_weighted_choice, METH_VARARGS,
                "Exact categorical draw using caller randrange."},
        {"native_boundary_coordinates", native_boundary_coordinates, METH_VARARGS,
                "Classify postselection, feedback and traced record coordinates."},
        {"boundary_service_counts", boundary_service_counts, METH_O, "Native boundary layout cache sizes."},
        {"indexed_record_row", indexed_record_row_binding, METH_VARARGS, "Map named records to a boundary parity."},
        {"is_affine_rule", affine_rule_check, METH_O, "Whether a declared rule is affine."},
        {"phase_feedback_metadata", phase_feedback_metadata, METH_O, "Axes and baseline of a feedback family."},
        {"resolve_phase_feedback", resolve_phase_feedback, METH_VARARGS,
                "Resolve actual feedback gates within a declared family."},
        {"attach_ideal_target", attach_ideal_target, METH_VARARGS, "Map the target effect to selected outputs."},
        {"attach_terminal_recovery", attach_terminal_recovery, METH_VARARGS,
                "Map recovery wires, records and declared controls."},
        {"check_file_execution", check_file_execution, METH_O, "Check native cooperative execution budgets."},
        {"draw_file_noise_random", draw_file_noise_random, METH_VARARGS,
                "Draw original noise batches using caller seeds."},
        {"file_noise_factors", file_noise_factors, METH_O, "Native category likelihood factors."},
        {"branch_schedule_stats", branch_schedule_stats, METH_O, "Native feedback schedule counters."},
        {"certificate_cache_info", certificate_cache_info, METH_O, "Certificate cache counters."},
        {"certificate_cache_keys", certificate_cache_keys, METH_O, "Certificate cache keys, all and with probability."},
        {"certificate_cache_clear", certificate_cache_clear, METH_O, "Empty a certificate cache."},
        {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_native_testing", nullptr, -1, methods, nullptr, nullptr, nullptr, nullptr};
}
PyMODINIT_FUNC PyInit__native_testing() {
    // Share ResourceLimit with the production module so tests catch one class.
    PyObject* production = PyImport_ImportModule("magna._native");
    if(!production) return nullptr;
    resource_limit_exception = PyObject_GetAttrString(production, "ResourceLimit");
    Py_DECREF(production);
    if(!resource_limit_exception) return nullptr;
    return PyModule_Create(&module);
}
