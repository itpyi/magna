// The fixed causal observation schedule of a compiled experiment.
// Requested record masks, their GF(2) reductions, sampling-parent coordinates and
// the observation echelon depend only on the declared control syntax, never on a
// reported value. They are prepared once; each draw replays only bit values.
#pragma once
#include "causal.hpp"
#include "../evaluate/reference_network.hpp"
namespace magic_native {
// One causal request: a noise guard, a controller call or the sampled records.
struct ObservationRequest {
    bool guard = false;
    size_t index = 0, pending = 0;
    bool affine = false;
    // Per expression: its pending row, if any, and the inserted rows whose
    // reduced values give its already known part.
    std::vector<std::optional<size_t>> outputs;
    IndexLists known;
    // Per pending row: its inserted row, or absent when an earlier row of this
    // request determines it, and the inserted rows its reduction passes through.
    std::vector<size_t> inserted;
    IndexLists chains;
    // Affine rows report the parity of their label coordinates on the virtual
    // sector and of the X components of their physical frame bits.
    IndexLists sector_bits, frame_bits;
    // Other rows are queried on the sampling parent in these coordinates.
    ReferenceRows rows;
};
struct ObservationTemplate {
    std::vector<ObservationRequest> protocol, recovery;
    std::optional<ObservationRequest> samples;
    // Inserted record masks and their sampling-parent coordinates, in order.
    std::vector<Bits> rows;
    ReferenceRows coordinates;
    std::shared_ptr<const ReferenceHistoryLayout> layout;
    // Terminal cap rows of each parent, resolved through the complete echelon.
    std::map<std::string, std::vector<std::optional<HistoryRecipe>>> caps;
};
// Prepare the next request of a schedule from its record masks. The knowledge,
// layout and inserted rows advance exactly as the reported values will at draw time.
inline ObservationRequest prepare_request(const std::vector<Bits>& masks, CausalKnowledge& known,
        const ReferenceNetwork& parent, const ReferenceRecordMap& map, ReferenceHistoryLayout& layout,
        ObservationTemplate& schedule) {
    ObservationRequest r;
    std::vector<std::pair<Bits, int>> expressions;
    expressions.reserve(masks.size());
    for(const auto& mask : masks) expressions.emplace_back(mask, 0);
    std::vector<std::vector<size_t>> chains;
    auto causal = known.request(expressions, &chains);
    for(size_t e = 0; e < masks.size(); e++) {
        r.outputs.push_back(causal.outputs[e].first);
        r.known.push(chains[e]);
    }
    r.pending = causal.pending.size();
    if(!r.pending) return r;
    const size_t closed = parent.closed.size();
    auto rows = map(causal.pending);
    r.affine = std::all_of(
            rows.labels.begin(), rows.labels.end(), [&](const Bits& row) { return row.size() <= closed; });
    std::vector<size_t> chain, bits;
    for(size_t j = 0; j < r.pending; j++) {
        const auto& mask = causal.pending[j];
        if(r.affine) {
            r.sector_bits.push(rows.labels[j].indices());
            bits.clear();
            mask.each([&](size_t bit) { bits.push_back(map.physical.at(bit)); });
            r.frame_bits.push(bits);
        }
        chain.clear();
        bool inserted = known.insert(mask, 0, &chain).first;
        r.chains.push(chain);
        r.inserted.push_back(inserted ? schedule.rows.size() : ReferenceHistoryLayout::absent);
        if(!inserted) continue;
        Bits physical;
        const bool direct = rows.labels[j].size() <= closed;
        if(direct) mask.each([&](size_t bit) { physical.set(map.physical[bit]); });
        layout.append(rows.labels[j], direct ? &physical : nullptr);
        schedule.rows.push_back(mask);
        schedule.coordinates.labels.push_back(rows.labels[j]);
    }
    if(!r.affine) r.rows = std::move(rows);
    return r;
}
// Reported values of an affine request's pending rows under the actual virtual
// sector and frame.
inline Bits affine_word(const ObservationRequest& request, const Bits& sector, const Frame& frame) {
    Bits word;
    for(size_t j = 0; j < request.pending; j++) {
        int value = 0;
        for(auto bit = request.sector_bits.begin(j); bit != request.sector_bits.end(j); ++bit)
            value ^= sector.bit(*bit);
        for(auto bit = request.frame_bits.begin(j); bit != request.frame_bits.end(j); ++bit)
            value ^= frame.x.bit(*bit);
        if(value) word.set(j);
    }
    return word;
}
inline ObservationTemplate prepare_observation(const ReferenceNetwork& parent, const ReferenceRecordMap& map,
        const FileControlMetadata& protocol, const FileControlMetadata* recovery, size_t protocol_records,
        const std::vector<size_t>& sample_records, const std::map<std::string, std::vector<Bits>>& completed_caps) {
    ObservationTemplate out;
    CausalKnowledge known;
    auto layout = std::make_shared<ReferenceHistoryLayout>(parent);
    auto request = [&](const std::vector<Bits>& masks) {
        return prepare_request(masks, known, parent, map, *layout, out);
    };
    // Guards and calls interleave in instruction order, as the causal loop reaches them.
    auto stage = [&](const FileControlMetadata& metadata, size_t offset, std::vector<ObservationRequest>& requests) {
        size_t callback = 0, conditional = 0;
        while(callback < metadata.callbacks.size() || conditional < metadata.conditional_noise.size()) {
            if(conditional < metadata.conditional_noise.size() &&
                    (callback == metadata.callbacks.size() ||
                            metadata.conditional_noise[conditional].instruction <
                                    metadata.callbacks[callback].instruction)) {
                auto r = request({metadata.conditional_noise[conditional].guard.records << offset});
                r.guard = true;
                r.index = conditional++;
                requests.push_back(std::move(r));
            } else {
                std::vector<Bits> masks;
                for(const auto& input : metadata.callbacks[callback].inputs) masks.push_back(input.records << offset);
                auto r = request(masks);
                r.index = callback++;
                requests.push_back(std::move(r));
            }
        }
    };
    stage(protocol, 0, out.protocol);
    if(!sample_records.empty()) {
        std::vector<Bits> masks;
        for(auto record : sample_records) masks.push_back(Bits::unit(record));
        out.samples = request(masks);
    }
    if(recovery) stage(*recovery, protocol_records, out.recovery);
    for(const auto& [purpose, rows] : completed_caps) {
        auto& recipes = out.caps[purpose];
        for(const auto& row : rows) recipes.push_back(layout->resolve(row));
    }
    layout->finish();
    out.layout = std::move(layout);
    return out;
}
} // namespace magic_native
