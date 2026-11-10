// Native collection of complete histories; no per-draw language conversion.
// Statistics accumulate as draws arrive; only requested sampled records are kept.
#pragma once
#include "file_execution.hpp"
#include "history_efficiency.hpp"
namespace magic_native {
struct FileBatch {
    ScientificAccumulator values;
    WeightAccumulator weights;
    HistoryEfficiency efficiency;
    std::vector<std::vector<int>> records;
    std::vector<std::optional<ScientificValue>> row;
    size_t count() const { return values.count; }
    void add(const HistoryContribution& contribution, const std::vector<int>& samples) {
        row.clear();
        row.reserve(contribution.values.size());
        for(const auto& value : contribution.values) row.push_back(value.scientific);
        // Every check precedes the first update, so a failed draw leaves the batch consistent.
        WeightAccumulator::validate(contribution.weight);
        std::vector<int> record;
        if(!samples.empty()) {
            record = samples;
            if(records.size() == records.capacity()) {
                if(records.size() == records.max_size()) throw std::length_error("too many sampled records");
                auto capacity = records.size() > records.max_size() / 2
                        ? records.max_size() : std::max(size_t(1), records.size() * 2);
                records.reserve(capacity);
            }
        }
        values.add(row);
        weights.add(contribution.weight);
        efficiency.add(*row[0], row[1], contribution.weight, contribution.noise_weight);
        if(!samples.empty()) records.push_back(std::move(record));
    }
    void add(const HistoryResult& result) { add(result.contribution(), result.samples); }
};
} // namespace magic_native
