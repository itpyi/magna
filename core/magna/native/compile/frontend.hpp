// Owned circuit-file data and static classical semantics. No callback execution.
#pragma once
#include "../foundation/rational.hpp"
#include "../foundation/limits.hpp"
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
namespace magic_native {
struct SourceLocation {
    std::string filename;
    size_t line = 0, column = 1;
    std::vector<size_t> repeat;
};
struct CircuitSyntaxError : std::invalid_argument {
    SourceLocation source;
    CircuitSyntaxError(std::string message, SourceLocation where)
        : std::invalid_argument(std::move(message)), source(std::move(where)) {}
};
struct FileTarget {
    std::string kind;
    size_t index = 0;
    int64_t version = -1;
    bool inverted = false;
    std::string pauli;
    auto key() const { return std::tie(kind, index, version, inverted, pauli); }
    bool operator<(const FileTarget& b) const { return key() < b.key(); }
    bool operator==(const FileTarget& b) const { return key() == b.key(); }
    bool operator!=(const FileTarget& b) const { return !(*this == b); }
};
struct AffineExpression {
    int constant = 0;
    std::set<FileTarget> terms;
    AffineExpression& operator^=(const AffineExpression& b) {
        constant ^= b.constant;
        for(const auto& term : b.terms)
            if(!terms.erase(term)) terms.insert(term);
        return *this;
    }
    bool operator==(const AffineExpression& b) const { return constant == b.constant && terms == b.terms; }
};
struct FileInstruction {
    std::string name;
    std::vector<Rational> args;
    std::vector<FileTarget> targets;
    SourceLocation source;
    std::string op_id;
    std::vector<FileTarget> outputs;
    std::vector<size_t> records;
    std::string control_name;
    std::optional<FileTarget> guard;
    std::vector<std::vector<FileTarget>> products;
    std::string chain_id;
};
struct CircuitData {
    std::vector<FileInstruction> instructions;
    size_t num_records = 0, num_qubits = 0;
    std::map<FileTarget, AffineExpression> affine;
    std::vector<AffineExpression> detectors, required;
    std::vector<size_t> external;
    std::string source, text;
    AffineExpression expression(const FileTarget& target) const;
    std::string format_version() const;
    // Sorted indices of the qubits that some instruction acts on. Indices up to
    // num_qubits that no instruction names start in |0> and are traced out, so
    // they contribute a factor of one and need no wire.
    std::vector<size_t> qubits() const;
};
const std::set<std::string>& supported_instructions();
// Default for the max_qubits parsing budget: every index below the largest
// qubit index used becomes a boundary coordinate of the compiled networks.
constexpr size_t default_max_qubits = 65536;
CircuitData parse_circuit(const std::string& text, const std::string& source, const std::set<std::string>& controllers,
        size_t max_instructions, size_t max_qubits = default_max_qubits);
} // namespace magic_native
