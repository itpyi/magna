// Incremental GF(2) basis of observed record parities and named record rows.
// Compilation, evaluation and causal scheduling share these classical types.
#pragma once
#include "bits.hpp"
#include <limits>
#include <map>
#include <limits>
#include <optional>
#include <string>
namespace magic_native {
struct CausalRequest {
    std::vector<Bits> pending;
    std::vector<std::pair<std::optional<size_t>, int>> outputs;
};
struct CausalKnowledge {
    std::vector<Bits> rows;
    std::vector<int> values;
    // Rows keep their actual packed length. Only the pivot index is dense;
    // elimination neither pads sparse rows nor copies a row for tree storage.
    static constexpr size_t no_pivot = std::numeric_limits<size_t>::max();
    std::vector<size_t> pivots;
    std::vector<std::pair<Bits, int>> basis;
    std::pair<Bits, int> reduce(Bits mask, int bit = 0) const {
        size_t words = mask.w.size();
        while(words) {
            auto pivot = 64 * (words - 1) + 63 - __builtin_clzll(mask.w[words - 1]);
            if(pivot >= pivots.size() || pivots[pivot] == no_pivot) break;
            const auto& row = basis[pivots[pivot]];
            // Matching highest pivots guarantee equal active word lengths.
            // Track that length locally and canonicalize just once on return.
            for(size_t j = 0; j < words; j++) mask.w[j] ^= row.first.w[j];
            bit ^= row.second;
            while(words && !mask.w[words - 1]) words--;
        }
        mask.w.resize(words);
        return {std::move(mask), bit};
    }
    std::pair<bool, int> insert(const Bits& mask, int bit = 0) {
        auto [residual, value] = reduce(mask, bit);
        if(residual.empty()) return {false, value};
        auto pivot = residual.size() - 1;
        if(pivot >= pivots.size()) pivots.resize(pivot + 1, no_pivot);
        pivots[pivot] = basis.size();
        basis.push_back({std::move(residual), value});
        return {true, value};
    }
    CausalRequest request(const std::vector<std::pair<Bits, int>>& expressions) const {
        CausalRequest request;
        request.outputs.reserve(expressions.size());
        // Sorted references deduplicate equal residuals without owning a
        // second mask. Large requests use a tree to avoid quadratic insertion
        // shifts. Both retain the original first-seen pending-label order.
        struct PendingLess {
            bool operator()(const Bits* a, const Bits* b) const { return BitsLess{}(*a, *b); }
        };
        const bool small = expressions.size() <= 64;
        std::vector<size_t> indices;
        std::map<const Bits*, size_t, PendingLess> large_indices;
        BitsLess less;
        for(const auto& [mask, offset] : expressions) {
            auto [residual, known] = reduce(mask);
            if(residual.empty()) {
                request.outputs.push_back({{}, offset ^ known});
                continue;
            }
            if(request.pending.empty()) {
                // This bound keeps every pointer key stable for this request.
                request.pending.reserve(expressions.size());
                if(small) indices.reserve(expressions.size());
            }
            size_t index;
            if(small) {
                auto found = std::lower_bound(indices.begin(), indices.end(), residual,
                        [&](size_t j, const Bits& row) { return less(request.pending[j], row); });
                if(found == indices.end() || less(residual, request.pending[*found])) {
                    index = request.pending.size();
                    indices.insert(found, index);
                    request.pending.push_back(std::move(residual));
                } else index = *found;
            } else {
                auto found = large_indices.lower_bound(&residual);
                if(found == large_indices.end() || less(residual, *found->first)) {
                    index = request.pending.size();
                    request.pending.push_back(std::move(residual));
                    large_indices.emplace_hint(found, &request.pending.back(), index);
                } else index = found->second;
            }
            request.outputs.push_back({index, offset ^ known});
        }
        return request;
    }
    std::vector<int> observe(const CausalRequest& request, const Bits& word) {
        if(word.size() > request.pending.size()) throw std::invalid_argument("sampled word outside requested labels");
        for(size_t j = 0; j < request.pending.size(); j++) {
            const auto& mask = request.pending[j];
            int bit = word.bit(j);
            auto [added, value] = insert(mask, bit);
            if(added) {
                rows.push_back(mask);
                values.push_back(bit);
            } else if(value) throw std::runtime_error("sampled parities violate an existing linear relation");
        }
        std::vector<int> result;
        result.reserve(request.outputs.size());
        for(const auto& [index, bit] : request.outputs) result.push_back(bit ^ (index ? word.bit(*index) : 0));
        return result;
    }
};
using NamedRows = std::vector<std::vector<std::string>>;
} // namespace magic_native
