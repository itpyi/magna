// Plans for the experimental CUDA sampler, built from a prepared CompiledExperiment.
// The device kernels in native/cuda read these plans; the CPU references here
// evaluate the same plans so that tests can check them without a GPU.
#pragma once
#include "file_execution.hpp"
#include "noise.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace magic_native {

constexpr size_t cuda_max_shots = 1000000;

// Noise laws in the block layout of the fused_noise kernel.
struct CudaNoiseTables {
    size_t sites = 0;
    std::vector<uint64_t> thresholds, conditional_thresholds;
    std::vector<uint32_t> block_offsets{0}, block_sites, choice_offsets{0}, choices;
    std::vector<uint8_t> certain, uniform, conditional_certain;
};

// Packed Pauli-frame columns of every noise action, the noise tables and the
// importance likelihood table of one compiled task.
struct CudaExperimentPlan {
    std::shared_ptr<FileBoundary> engine, target;
    size_t frame_bits = 0, words = 0;
    std::vector<uint64_t> columns, base;
    std::vector<size_t> action_sites;
    CudaNoiseTables noise;
    bool has_importance_weights = false;
    std::vector<uint8_t> weighted_sites;
    std::vector<double> likelihood_values;
};

// The plan stays valid only while the compiled task keeps the networks it was built from.
inline void cuda_check_current(const CompiledExperiment& run, const CudaExperimentPlan& plan) {
    auto found = run.plans.prepared.find(run.source.stage("recovery"));
    if(!run.complete || found == run.plans.prepared.end() || found->second != plan.engine)
        throw std::invalid_argument("compiled circuit was reconfigured; recreate the CUDA experiment");
    if(plan.target) {
        auto target = run.plans.prepared.find("target");
        if(target == run.plans.prepared.end() || target->second != plan.target)
            throw std::invalid_argument("target was reconfigured; recreate the CUDA experiment");
    }
}

inline void cuda_pack_frame(const Frame& frame, size_t words, uint64_t* out) {
    if(frame.x.w.size() > words || frame.z.w.size() > words) throw std::logic_error("CUDA frame width mismatch");
    std::fill(out, out + 2 * words, uint64_t(0));
    std::copy(frame.x.w.begin(), frame.x.w.end(), out);
    std::copy(frame.z.w.begin(), frame.z.w.end(), out + words);
}

// A single native count table is an exact sufficient statistic for these
// homogeneous, independent importance proposals. Mixed factor groups and
// channels requiring draw-time arbitrary precision remain unsupported.
inline void cuda_prepare_likelihood(const FileNoisePlan& noise, CudaExperimentPlan& plan) {
    plan.weighted_sites.assign(noise.categories.size(), 0);
    if(noise.factor_powers.empty()) {
        plan.likelihood_values = {1.};
        return;
    }
    auto unsupported = []() {
        throw std::invalid_argument("CUDA importance sampling requires 2 to 512 scaled noise locations, all with the same error probability");
    };
    if(noise.count_weights.size() != 1 || noise.count_groups.size() != noise.categories.size() ||
            noise.factor_groups.size() != noise.categories.size()) unsupported();
    size_t counted = 0;
    for(size_t site = 0; site < noise.categories.size(); site++) {
        if(noise.count_groups[site] == 0) {
            if(noise.categories[site]->occurrence.certain || !noise.categories[site]->occurrence.threshold)
                throw std::invalid_argument("CUDA importance proposal must have a nontrivial uint64 occurrence probability");
            plan.weighted_sites[site] = 1;
            counted++;
        } else if(noise.count_groups[site] != -1 || noise.factor_groups[site] != std::array<int64_t, 2>{-1, -1})
            unsupported();
    }
    if(counted < 2 || counted > 512 || noise.count_weights[0].size() != counted + 1 ||
            noise.count_scientific.size() != counted + 1) unsupported();
    for(const auto& [mantissa, exponent] : noise.count_scientific) {
        if(exponent < std::numeric_limits<int32_t>::min() || exponent > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument("CUDA likelihood exponent exceeds int32 range");
        double value = std::ldexp(mantissa, int(exponent));
        if(!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("CUDA importance likelihood table exceeds positive float64 range");
        plan.likelihood_values.push_back(value);
    }
    plan.has_importance_weights = true;
}

inline CudaNoiseTables cuda_noise_tables(const FileNoisePlan& noise) {
    if(noise.categories.size() > std::numeric_limits<uint32_t>::max())
        throw std::length_error("CUDA noise site count exceeds uint32");
    CudaNoiseTables out;
    out.sites = noise.categories.size();
    for(const auto& group : noise.occurrences) {
        for(size_t first = 0; first < group.sites.size(); first += 64) {
            out.thresholds.push_back(group.coin.threshold);
            out.certain.push_back(group.coin.certain);
            for(size_t j = first; j < std::min(first + 64, group.sites.size()); j++)
                out.block_sites.push_back(uint32_t(group.sites[j]));
            out.block_offsets.push_back(uint32_t(out.block_sites.size()));
        }
    }
    for(size_t site = 0; site < noise.categories.size(); site++) {
        const auto& category = *noise.categories[site];
        out.uniform.push_back(category.uniform);
        for(size_t j = 0; j < category.nonzero.size(); j++) {
            out.choices.push_back(uint32_t(noise.offsets[site] + category.nonzero[j]));
            if(!category.uniform && j < category.conditional.size()) {
                out.conditional_thresholds.push_back(category.conditional[j].threshold);
                out.conditional_certain.push_back(category.conditional[j].certain);
            } else {
                out.conditional_thresholds.push_back(0);
                out.conditional_certain.push_back(0);
            }
        }
        out.choice_offsets.push_back(uint32_t(out.choices.size()));
    }
    return out;
}

// Rejects unsupported tasks, prepares the task's networks and packs every noise action.
inline CudaExperimentPlan build_cuda_experiment(CompiledExperiment& run) {
    if(!run.source.guarded_ids.empty() || run.noise.guarded)
        throw std::invalid_argument("CUDA sampler does not support conditional noise");
    if(!run.task.boundaries.sample_records.empty())
        throw std::invalid_argument("CUDA sampler does not support sample_records");
    CudaExperimentPlan out;
    cuda_prepare_likelihood(run.noise, out);
    if(!run.task.boundaries.observables.empty())
        throw std::invalid_argument("CUDA sampler does not support observables");
    if(!run.complete) run.prepare();
    out.engine = run.plans.get(run.source.stage("recovery"));
    if(run.source.target) {
        out.target = run.plans.get("target");
        if(!out.target->plan->compiled.plan.continuation)
            throw std::invalid_argument("CUDA experiment requires target frame continuation");
    }
    const auto& plan = out.engine->plan->compiled.plan;
    out.frame_bits = plan.physical + plan.magic.size();
    out.words = std::max(size_t(1), (out.frame_bits + 63) / 64);
    size_t stride = 2 * out.words, actions = run.noise.actions.size();
    if(actions > std::numeric_limits<uint32_t>::max() || actions > out.columns.max_size() / stride)
        throw std::length_error("CUDA action table exceeds index capacity");
    out.columns.resize(actions * stride);
    out.base.resize(stride);
    out.action_sites.resize(actions);
    cuda_pack_frame(out.engine->feedback_constant, out.words, out.base.data());
    for(size_t site = 0; site < run.noise.categories.size(); site++)
        for(size_t choice = 0; choice < run.noise.categories[site]->size; choice++) {
            auto index = run.noise.offsets[site] + choice;
            out.action_sites.at(index) = site;
            FileNoiseDraw selected;
            selected.selected.push_back(int64_t(index));
            auto noise = noise_actions(run.noise, selected);
            auto packed = out.engine->pack(run.source.original_faults(noise), noise.flips);
            auto frame = plan.fault_frame(std::move(packed.faults), false);
            cuda_pack_frame(frame, out.words, out.columns.data() + index * stride);
        }
    out.noise = cuda_noise_tables(run.noise);
    return out;
}

// Philox4x32-10 addressed by (draw index, block counter), as in noise_sampling.cu.
struct CudaPhilox {
    uint64_t shot, seed, block = 0, saved = 0;
    bool has_saved = false;
    CudaPhilox(uint64_t s, uint64_t k) : shot(s), seed(k) {}
    uint64_t operator()() {
        if(has_saved) { has_saved = false; return saved; }
        uint32_t a = uint32_t(shot), b = uint32_t(shot >> 32);
        uint32_t c = uint32_t(block), d = uint32_t(block >> 32);
        uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
        block++;
        for(unsigned round = 0; round < 10; round++) {
            uint64_t p0 = uint64_t(0xd2511f53U) * a, p1 = uint64_t(0xcd9e8d57U) * c;
            a = uint32_t(p1 >> 32) ^ b ^ k0;
            b = uint32_t(p1);
            c = uint32_t(p0 >> 32) ^ d ^ k1;
            d = uint32_t(p0);
            k0 += 0x9e3779b9U;
            k1 += 0xbb67ae85U;
        }
        saved = uint64_t(c) | (uint64_t(d) << 32);
        has_saved = true;
        return uint64_t(a) | (uint64_t(b) << 32);
    }
};

// CPU reference of fused_noise for draw `shot`: writes its frame, appends the
// chosen action IDs to `actions` when given, and returns the weighted-site count.
inline size_t cuda_reference_draw(const FileNoisePlan& noise, const CudaExperimentPlan& plan,
        uint64_t seed, uint64_t shot, uint64_t* frame, std::vector<uint32_t>* actions) {
    CudaPhilox rng(shot, seed);
    size_t stride = 2 * plan.words, weight_count = 0;
    std::copy(plan.base.begin(), plan.base.end(), frame);
    auto select = [&](size_t site) {
        const auto& category = *noise.categories[site];
        size_t selected = category.nonzero.size() - 1;
        if(category.uniform) selected = uniform_index(category.nonzero.size(), rng);
        else for(size_t j = 0; j < category.conditional.size(); j++)
            if(category.conditional[j].draw(rng)) { selected = j; break; }
        size_t action = noise.offsets[site] + category.nonzero[selected];
        const auto* column = plan.columns.data() + action * stride;
        for(size_t word = 0; word < stride; word++) frame[word] ^= column[word];
        if(actions) actions->push_back(uint32_t(action));
        weight_count += plan.weighted_sites[site];
    };
    for(const auto& group : noise.occurrences) {
        for(size_t first = 0; first < group.sites.size(); first += 64) {
            size_t count = std::min(size_t(64), group.sites.size() - first);
            if(count < 8) {
                for(size_t j = 0; j < count; j++)
                    if(group.coin.draw(rng)) select(group.sites[first + j]);
            } else {
                uint64_t live = count == 64 ? UINT64_MAX : (uint64_t(1) << count) - 1;
                auto selected = group.coin.draw_bits(rng, live);
                while(selected) {
                    size_t bit = unsigned(__builtin_ctzll(selected));
                    selected &= selected - 1;
                    select(group.sites[first + bit]);
                }
            }
        }
    }
    return weight_count;
}

// Exact probability queries.

struct CudaQueryAffine {
    Bits row;
    int constant = 0;
};
inline CudaQueryAffine cuda_query_xor(CudaQueryAffine a, const CudaQueryAffine& b) {
    a.row ^= b.row;
    a.constant ^= b.constant;
    return a;
}
struct CudaQueryPlan {
    size_t n = 0, frame_words = 0, input_bits = 0;
    int mode = 0; // 0 zero; 1 small Gauss; 2 static Gauss; 3 histogram; 4 dynamic Gauss; 5 entry sum.
    int constant = 0, power_sqrt2 = 0, output_power = 0;
    size_t phase_rank = 0;
    std::vector<CudaQueryAffine> features;
    std::map<std::pair<Bits, int>, uint32_t, bool(*)(const std::pair<Bits,int>&, const std::pair<Bits,int>&)> feature_index;
    std::vector<uint32_t> support, guards, linear_shift;
    std::vector<int32_t> constant_linear, constant_pairs, magic;
    std::vector<uint64_t> magic_masks, upper;
    std::vector<uint8_t> linear, phase_table;
    std::vector<int32_t> steps;
    std::vector<uint32_t> step_offsets, step_indices;
    std::vector<uint8_t> step_tags;
    std::vector<CudaQueryPlan> components;
    std::vector<Scalar> weights;
    std::shared_ptr<const CudaQueryPlan> shared_base;
    size_t base_index = 0, disabled_caps = 0;
    CudaQueryPlan() : feature_index([](const auto& a, const auto& b) {
        if(a.first != b.first) return BitsLess{}(a.first, b.first);
        return a.second < b.second;
    }) {}
    uint32_t feature(CudaQueryAffine value) {
        auto key = std::make_pair(value.row, value.constant);
        auto found = feature_index.find(key);
        if(found != feature_index.end()) return found->second;
        if(features.size() >= 1000000) throw std::length_error("CUDA query feature budget exceeded");
        auto index = uint32_t(features.size());
        feature_index.emplace(std::move(key), index);
        features.push_back(std::move(value));
        return index;
    }
};

// Compose every query-coordinate parity with the exact prefix continuation.
// This removes a separate target-frame transfer/kernel without changing its map.
inline std::pair<std::vector<Bits>, std::vector<Bits>> cuda_query_frame_rows(
        const CudaExperimentPlan& experiment, const FileBoundary& engine, bool target) {
    const auto& plan = engine.plan->compiled.plan;
    size_t bits = plan.physical + plan.magic.size();
    std::vector<Bits> x(bits), z(bits);
    for(size_t axis = 0; axis < 2; axis++)
        for(size_t bit = 0; bit < experiment.frame_bits; bit++) {
            Frame image;
            (axis ? image.z : image.x).set(bit);
            if(target) image = plan.continuation->apply(image);
            const size_t input = axis * 64 * experiment.words + bit;
            image.x.each([&](size_t j) { x.at(j).set(input); });
            image.z.each([&](size_t j) { z.at(j).set(input); });
        }
    return {std::move(x), std::move(z)};
}

inline std::vector<uint8_t> cuda_query_feature_values(const CudaQueryPlan& plan, const uint64_t* frame) {
    std::vector<uint8_t> values;
    values.reserve(plan.features.size());
    for(const auto& feature : plan.features) {
        uint64_t combined = 0;
        for(size_t w = 0; w < feature.row.w.size(); w++) combined ^= feature.row.w[w] & frame[w];
        values.push_back(uint8_t(feature.constant ^ (__builtin_popcountll(combined) & 1)));
    }
    return values;
}

// Build the same actual Clifford data as CompiledNetwork::evaluate_fixed.
// Non-Clifford odd phases stay in phase_table for the exact histogram route.
inline int cuda_query_actual(const CudaQueryPlan& p, const std::vector<uint8_t>& f,
        std::vector<int>& l, std::vector<Bits>& upper, bool make_upper = true) {
    l.assign(p.linear.begin(), p.linear.end());
    size_t words = (p.n + 63) / 64;
    if(make_upper) {
        upper.resize(p.n);
        for(size_t j = 0; j < p.n; j++) {
            upper[j].w.assign(p.upper.begin() + j * words, p.upper.begin() + (j + 1) * words);
            upper[j].trim();
        }
    }
    int phase = p.constant;
    for(size_t j = 0; j < p.constant_linear.size(); j += 2)
        phase = mod8(phase + p.constant_linear[j + 1] * f[size_t(p.constant_linear[j])]);
    for(size_t j = 0; j < p.constant_pairs.size(); j += 2)
        phase = mod8(phase + 4 * f[size_t(p.constant_pairs[j])] * f[size_t(p.constant_pairs[j + 1])]);
    for(size_t j = 0; j < p.n; j++) l[j] = mod4(l[j] + 2 * f[p.linear_shift[j]]);
    for(size_t j = 0; j < p.magic.size() / 8; j++) {
        const auto* m = p.magic.data() + 8 * j;
        int actual = mod8((f[size_t(m[4])] ? -m[0] : m[0]) + 4 * f[size_t(m[5])]);
        int before[2] = {m[2], m[3]}, after[2] = {m[2] ^ f[size_t(m[6])], m[3] ^ f[size_t(m[7])]};
        for(size_t side = 0; side < 2; side++) {
            int sign = side ? 1 : -1;
            phase = mod8(phase + sign * (actual * after[side] - m[1] * before[side]));
            int delta = mod8(sign * (actual * (after[side] ? -1 : 1) - m[1] * (before[side] ? -1 : 1)));
            if(delta & 1) throw std::logic_error("CUDA query update is not Clifford");
            int c = delta / 2;
            if(!c) continue;
            Bits mask;
            auto first = p.magic_masks.begin() + (2 * j + side) * words;
            mask.w.assign(first, first + words);
            mask.trim();
            mask.each([&](size_t k) {
                l[k] = mod4(l[k] + c);
                if(make_upper && (c & 1)) upper[k] ^= mask.above(k);
            });
        }
    }
    return mod8(phase);
}

// A binary Fourier multiplier enforces an affine cap without changing the
// certified phase resource: delta(Lx=b) = sum_y (-1)^(y(Lx+b))/2.
inline CudaQueryPlan cuda_query_cap(const CudaQueryPlan& mother, const std::vector<Bits>& rows,
        const std::vector<CudaQueryAffine>& rhs) {
    CudaQueryPlan out = mother;
    size_t before = mother.n, old_words = (before + 63) / 64;
    out.n += rows.size();
    if(out.n > 4096) throw std::invalid_argument("CUDA history Fourier cap dimension exceeds 4096");
    out.mode = out.n <= 64 ? 1 : 4;
    out.phase_rank = 0;
    out.guards.clear(); out.steps.clear(); out.step_offsets.clear(); out.step_indices.clear(); out.step_tags.clear();
    size_t words = (out.n + 63) / 64;
    out.upper.assign(out.n * words, 0);
    for(size_t j = 0; j < before; j++)
        std::copy(mother.upper.begin() + j * old_words, mother.upper.begin() + (j + 1) * old_words, out.upper.begin() + j * words);
    out.magic_masks.assign(mother.magic.size() / 4 * words, 0);
    for(size_t j = 0; j < mother.magic.size() / 4; j++)
        std::copy(mother.magic_masks.begin() + j * old_words, mother.magic_masks.begin() + (j + 1) * old_words,
                out.magic_masks.begin() + j * words);
    for(size_t j = 0, index = before; j < rows.size(); j++) {
        rows[j].each([&](size_t bit) { out.upper[bit * words + index / 64] ^= uint64_t(1) << (index % 64); });
        out.linear.push_back(0);
        out.linear_shift.push_back(out.feature(rhs[j]));
        out.power_sqrt2 -= 2;
        index++;
    }
    return out;
}

inline CudaQueryPlan cuda_query_entries(const CudaQueryPlan& mother, const PhaseResource& resource) {
    CudaQueryPlan out;
    out.mode = 5; out.n = mother.n; out.frame_words = mother.frame_words; out.input_bits = mother.input_bits;
    for(const auto& entry : resource_entries(resource)) {
        if(entry.coefficient.zero()) continue;
        auto component = mother;
        auto linear = resource.quotient.linear;
        auto upper = resource.quotient.upper;
        PhaseTerms parities;
        for(const auto& [mask, coefficient] : entry.parities)
            parities.emplace_back(xors(resource.quotient.axes, mask), 2 * mod4(coefficient));
        phase_add_even(linear, upper, parities);
        component.linear.assign(linear.begin(), linear.end());
        size_t words = (component.n + 63) / 64;
        component.upper.assign(component.n * words, 0);
        for(size_t j = 0; j < component.n; j++)
            std::copy(upper[j].w.begin(), upper[j].w.end(), component.upper.begin() + j * words);
        std::vector<Bits> equations;
        std::vector<CudaQueryAffine> rhs;
        for(const auto& [mask, value] : entry.constraints) {
            equations.push_back(xors(resource.quotient.axes, mask));
            rhs.push_back({{}, value});
        }
        out.components.push_back(cuda_query_cap(component, equations, rhs));
        out.weights.push_back(entry.coefficient);
    }
    return out;
}
inline CudaQueryPlan cuda_query_scale(CudaQueryPlan query, const Scalar& scale) {
    if(!query.mode || (scale.c == Scalar(1).c && scale.power == 0)) return query;
    if(query.mode == 5) {
        for(auto& weight : query.weights) weight = weight * scale;
        return query;
    }
    CudaQueryPlan result;
    result.mode = 5; result.n = query.n; result.frame_words = query.frame_words; result.input_bits = query.input_bits;
    result.components.push_back(std::move(query)); result.weights.push_back(scale);
    return result;
}

inline CudaQueryPlan build_cuda_query(const CompiledExperiment& run, const CudaExperimentPlan& experiment,
        const FileBoundary& engine, const char* purpose, bool target,
        const CompiledNetwork* supplied = nullptr, const std::vector<CudaQueryAffine>* outcomes = nullptr,
        size_t auxiliary_words = 0, const std::vector<Bits>* extra_changes = nullptr,
        const PhaseContraction* pullback = nullptr, bool matrix_only = false) {
    CudaQueryPlan out;
    out.frame_words = 2 * experiment.words + auxiliary_words;
    out.input_bits = experiment.frame_bits;
    const auto& parent = *engine.reference_networks.at(purpose);
    const auto& network = supplied ? *supplied : *parent.reference;
    if((!matrix_only && !pullback && !network.boundary.keep.empty()) ||
            (!outcomes && (!parent.closed.empty() || parent.capped_history)))
        throw std::invalid_argument("CUDA exact queries require scalar fixed-history boundaries");
    if(network.traced.clifford.zero) return out;
    const auto* certificate = network.fixed_certificate();
    if(!certificate) throw std::invalid_argument("CUDA exact query requires a fixed phase certificate");
    const auto& resource = certificate->reference;
    out.n = network.restricted.clifford.n;
    out.phase_rank = pullback ? 0 : resource.quotient.axes.size();
    if(out.n > (outcomes ? 4096 : 1024))
        throw std::invalid_argument(outcomes ? "CUDA history query dimension exceeds 4096" : "CUDA exact query dimension exceeds 1024");
    if(!matrix_only && outcomes && out.phase_rank && out.n > 12) {
        auto mother = build_cuda_query(run, experiment, engine, purpose, target, supplied, outcomes,
                auxiliary_words, extra_changes, pullback, true);
        return cuda_query_entries(mother, resource);
    }
    if(!matrix_only && out.phase_rank && out.n > 12)
        throw std::invalid_argument("CUDA non-Clifford histogram dimension exceeds 12; no CPU fallback");
    if(resource.quotient.linear.size() != out.n)
        throw std::logic_error("CUDA fixed certificate does not use the scalar query domain");
    if(!matrix_only && !out.phase_rank && !pullback) {
        const auto& entries = resource_entries(resource);
        if(entries.size() != 1 || !entries[0].constraints.empty() || !entries[0].parities.empty() ||
                entries[0].coefficient.c != Scalar(1).c || entries[0].coefficient.power != 0)
            throw std::invalid_argument("CUDA Gauss query requires the certified unit entry");
    }
    out.mode = pullback ? 2 : (matrix_only ? 4 : (out.phase_rank ? 3 : (out.n <= 64 ? 1 : 2)));
    const auto& base = pullback ? pullback->pulled_reference.linear :
            (out.phase_rank && !matrix_only ? network.restricted.clifford.linear : resource.quotient.linear);
    const auto& base_upper = pullback ? pullback->pulled_reference.upper :
            (out.phase_rank && !matrix_only ? network.restricted.clifford.upper : resource.quotient.upper);
    out.linear.assign(base.begin(), base.end());
    size_t words = (out.n + 63) / 64;
    out.upper.assign(out.n * words, 0);
    for(size_t j = 0; j < out.n; j++)
        for(size_t w = 0; w < base_upper[j].w.size(); w++) out.upper[j * words + w] = base_upper[j].w[w];
    out.constant = mod8(network.restricted.clifford.constant + network.phase_constant);
    auto power = network.restricted.clifford.power;
    if(power < -1000000 || power > 1000000) throw std::invalid_argument("CUDA query scale exceeds int32 budget");
    out.power_sqrt2 = int(power);
    auto target_power = target ? run.source.target->traced.size() : 0;
    if(target_power > 1000000) throw std::invalid_argument("CUDA target trace scale exceeds int32 budget");
    out.output_power = int(target_power);
    const auto& plan = engine.plan->compiled.plan;
    auto [xrows, zrows] = cuda_query_frame_rows(experiment, engine, target);
    std::vector<CudaQueryAffine> rhs;
    for(size_t j = 0; j < network.boundary.events.size(); j++) {
        auto value = outcomes ? outcomes->at(j) : CudaQueryAffine{{}, parent.selection[j]};
        value.row ^= xors(xrows, network.boundary.events[j]);
        value.constant ^= network.traced.clifford.boundary[network.boundary.keep.size() + j].second;
        rhs.push_back(std::move(value));
    }
    auto combine = [](const std::vector<CudaQueryAffine>& values, const Bits& mask) {
        CudaQueryAffine result;
        mask.each([&](size_t j) { result = cuda_query_xor(std::move(result), values.at(j)); });
        return result;
    };
    const auto& factor = network.event_support->factor;
    for(const auto& row : factor.zeros) out.support.push_back(out.feature(combine(rhs, row)));
    std::vector<CudaQueryAffine> shift(network.traced.clifford.n);
    for(const auto& [j, pair] : factor.pivots)
        shift[j] = cuda_query_xor(combine(shift, pair.first), combine(rhs, pair.second));
    const auto& traced = network.traced.clifford;
    for(size_t j = 0; j < shift.size(); j++) {
        auto a = out.feature(shift[j]);
        if(traced.linear[j]) out.constant_linear.insert(out.constant_linear.end(), {int32_t(a), 2 * traced.linear[j]});
        traced.upper[j].each([&](size_t k) {
            out.constant_pairs.insert(out.constant_pairs.end(), {int32_t(a), int32_t(out.feature(shift[k]))});
        });
    }
    auto observation = CudaQueryAffine{xors(zrows, network.boundary.x) ^ xors(xrows, network.boundary.z), 0};
    out.constant_linear.insert(out.constant_linear.end(), {int32_t(out.feature(observation)), 4});
    auto event_polar = network.maps->event_polar;
    if(pullback) {
        event_polar.clear();
        for(const auto& axis : pullback->change) event_polar.push_back(xors(network.maps->event_polar, axis));
    }
    for(const auto& row : event_polar) out.linear_shift.push_back(out.feature(combine(shift, row)));
    auto dual = pullback ? transpose(pullback->change, pullback->dimension) : std::vector<Bits>{};
    std::map<std::pair<size_t, size_t>, CudaQueryAffine> structural;
    for(size_t j = 0; j < network.magic.size(); j++) {
        CudaQueryAffine xm{xrows.at(plan.physical + j), 0}, zm{zrows.at(plan.physical + j), 0};
        auto bm = combine(shift, network.maps->bra_magic[j]), km = combine(shift, network.maps->ket_magic[j]);
        out.magic.insert(out.magic.end(), {mod8(plan.magic[j]), mod8(network.reference_exponents[j]),
                network.bra_offsets[j], network.ket_offsets[j], int32_t(out.feature(xm)), int32_t(out.feature(zm)),
                int32_t(out.feature(bm)), int32_t(out.feature(km))});
        Bits mapped[2] = {network.maps->bra_reduced[j], network.maps->ket_reduced[j]};
        if(pullback) for(auto& row : mapped) row = xors(dual, row);
        const Bits* masks[2] = {&mapped[0], &mapped[1]};
        CudaQueryAffine changes[2] = {cuda_query_xor(xm, bm), cuda_query_xor(xm, km)};
        for(size_t side = 0; side < 2; side++) {
            auto start = out.magic_masks.size();
            out.magic_masks.resize(start + words);
            std::copy(masks[side]->w.begin(), masks[side]->w.end(), out.magic_masks.begin() + start);
            if(out.mode == 2 && (plan.magic[j] & 1)) {
                auto indices = masks[side]->indices();
                for(size_t a = 0; a < indices.size(); a++)
                    for(size_t b = a; b < indices.size(); b++) {
                        auto key = std::make_pair(indices[a], indices[b]);
                        structural[key] = cuda_query_xor(std::move(structural[key]), changes[side]);
                    }
            }
        }
    }
    if(out.mode == 3) {
        out.phase_table.resize(size_t(1) << out.n);
        for(size_t value = 0; value < out.phase_table.size(); value++) {
            int phase = 0;
            for(const auto& [mask, coefficient] : network.fixed_phase)
                phase = mod8(phase + coefficient * mask.parity_and(Bits(value)));
            out.phase_table[value] = uint8_t(mod8(phase));
        }
    }
    if(out.mode == 2) {
        // A large query is eliminated once, on the base frame, so every noise
        // column must leave its quadratic geometry unchanged.
        std::vector<Bits> structural_rows;
        for(const auto& [position, value] : structural)
            if(!value.row.empty()) structural_rows.push_back(value.row);
        auto independent = phase_image_basis(structural_rows);
        Bits allowed_base;
        allowed_base.w = experiment.base;
        allowed_base.trim();
        bool dynamic = false;
        for(const auto& row : independent) {
            for(size_t a = 0; a < experiment.action_sites.size(); a++) {
                uint64_t combined = 0;
                for(size_t w = 0; w < std::min(row.w.size(), 2 * experiment.words); w++)
                    combined ^= row.w[w] & experiment.columns[a * 2 * experiment.words + w];
                if(__builtin_popcountll(combined) & 1) {
                    if(!outcomes) throw std::invalid_argument("CUDA large query has varying quadratic geometry on the noise span");
                    dynamic = true;
                }
            }
            if(extra_changes)
                for(const auto& column : *extra_changes)
                    if(row.parity_and(column)) dynamic = true;
            out.guards.push_back(out.feature({row, row.parity_and(allowed_base)}));
        }
        if(dynamic) {
            out.mode = 4;
            out.guards.clear();
            return out;
        }
        auto base_frame = experiment.base;
        base_frame.resize(out.frame_words);
        auto features = cuda_query_feature_values(out, base_frame.data());
        std::vector<int> l;
        std::vector<Bits> upper;
        cuda_query_actual(out, features, l, upper);
        CliffordContraction contraction(pullback ? pullback->columns : std::vector<Bits>(out.n), Quadratic{0, l, upper});
        if(!pullback && !contraction.keep.empty()) throw std::logic_error("CUDA scalar contraction retained outputs");
        out.step_offsets.push_back(0);
        for(const auto& step : contraction.steps) {
            out.steps.insert(out.steps.end(), {int32_t(step.h), int32_t(step.p), int32_t(step.odd)});
            const auto& entries = step.odd ? step.neighbors : step.support | step.neighbors;
            entries.each([&](size_t k) {
                out.step_indices.push_back(uint32_t(k));
                out.step_tags.push_back(uint8_t(step.support.bit(k) | (2 * step.neighbors.bit(k))));
            });
            out.step_offsets.push_back(uint32_t(out.step_indices.size()));
        }
    }
    return out;
}

// Query 0 is acceptance; query 1, when the task has an ideal state, is target success.
inline std::vector<CudaQueryPlan> build_cuda_queries(const CompiledExperiment& run, const CudaExperimentPlan& experiment) {
    std::vector<CudaQueryPlan> out;
    out.push_back(build_cuda_query(run, experiment, *experiment.engine, "acceptance", false));
    if(experiment.target) out.push_back(build_cuda_query(run, experiment, *experiment.target, "fidelity", true));
    return out;
}

// CPU reference of exact_query in probability_queries.cu.
inline Scalar cuda_query_cpu_value(const CudaQueryPlan& p, const uint64_t* frame, size_t disabled_caps = 0) {
    if(!p.mode) return {};
    if(p.mode == 6) return cuda_query_cpu_value(*p.shared_base, frame, p.disabled_caps);
    if(p.mode == 5) {
        Scalar result;
        for(size_t j = 0; j < p.components.size(); j++) result = result + cuda_query_cpu_value(p.components[j], frame, disabled_caps) * p.weights[j];
        return result;
    }
    auto f = cuda_query_feature_values(p, frame);
    for(auto guard : p.guards)
        if(f[guard]) throw std::invalid_argument("frame is outside the certified large-query quadratic geometry");
    for(auto event : p.support)
        if(f[event]) return {};
    std::vector<int> l;
    std::vector<Bits> upper;
    int phase = cuda_query_actual(p, f, l, upper, p.mode != 2);
    size_t active = p.n - disabled_caps;
    int adjusted_power = p.power_sqrt2 + 2 * int(disabled_caps);
    if(disabled_caps) {
        l.resize(active); upper.resize(active);
        for(auto& row : upper) row = row.low(active);
    }
    Scalar value;
    if(p.mode == 3) {
        std::array<int64_t, 8> counts{};
        for(size_t u = 0; u < p.phase_table.size(); u++)
            counts[mod8(phase + 2 * qvalue(Bits(u), l, upper) + p.phase_table[u])]++;
        for(size_t k = 0; k < 8; k++) value = value + Scalar(counts[k]) * omega(int(k), p.power_sqrt2);
    } else if(p.mode == 1 || p.mode == 4) {
        CliffordContraction contraction(std::vector<Bits>(active), Quadratic{0, l, upper});
        auto ket = contraction.evaluate(0, {}, std::move(l), adjusted_power, phase);
        if(!ket.zero) value = omega(ket.phase, ket.power);
    } else {
        int power = p.power_sqrt2;
        for(size_t s = 0; s < p.steps.size() / 3; s++) {
            auto h = p.steps[3 * s], pivot = p.steps[3 * s + 1], odd = p.steps[3 * s + 2];
            int lh = l[size_t(h)];
            if(odd) {
                int sign = lh == 1 ? -1 : 1;
                for(auto k = p.step_offsets[s]; k < p.step_offsets[s + 1]; k++)
                    l[p.step_indices[k]] = mod4(l[p.step_indices[k]] + sign);
                power++;
                phase = mod8(phase + (lh == 1 ? 1 : -1));
            } else {
                power += 2;
                int rhs = lh / 2;
                if(size_t(pivot) == p.n) { if(rhs) return {}; continue; }
                int lp = l[size_t(pivot)];
                phase = mod8(phase + 2 * rhs * lp);
                for(auto k = p.step_offsets[s]; k < p.step_offsets[s + 1]; k++) {
                    int sj = p.step_tags[k] & 1, nj = (p.step_tags[k] >> 1) & 1;
                    auto j = p.step_indices[k];
                    l[j] = mod4(l[j] + lp * sj + 2 * sj * nj + 2 * rhs * (nj ^ ((lp & 1) * sj)));
                }
            }
        }
        value = omega(phase, power);
    }
    return value * Scalar(1, p.output_power);
}

} // namespace magic_native
