// Python-independent native implementation.
#pragma once
#include "../foundation/rational_random.hpp"
#include "../foundation/numeric.hpp"
#include <map>
#include <optional>
#include <memory>
#include <random>
#include <string>

namespace magic_native {
struct FileFraction {
    Integer n, d;
    FileFraction(Integer numerator = Integer(0), Integer denominator = Integer(1))
        : n(std::move(numerator)), d(std::move(denominator)) {}
};
inline FileFraction file_fraction_subtract(const FileFraction& a, const FileFraction& b) {
    return {a.n * b.d - b.n * a.d, a.d * b.d};
}

struct FileNoiseCategory {
    std::vector<FileFraction> probabilities;
    size_t size;
    RationalCoin occurrence{Integer(0), Integer(1)};
    std::vector<size_t> nonzero;
    std::vector<RationalCoin> conditional;
    bool uniform = true;
    explicit FileNoiseCategory(std::vector<FileFraction> input)
        : probabilities(std::move(input)), size(probabilities.size()) {
        FileFraction remaining(Integer(1));
        for(size_t k = 0; k < size; k++) {
            const auto& p = probabilities[k];
            if(p.n.neg || p.d.neg || p.d.zero()) throw std::invalid_argument("invalid rational probability");
            remaining =
                    remaining.d == p.d ? FileFraction(remaining.n - p.n, p.d) : file_fraction_subtract(remaining, p);
            if(remaining.n.neg) throw std::invalid_argument("category probability sum exceeds one");
            if(!p.n.zero()) {
                if(!nonzero.empty()) {
                    const auto& first = probabilities[nonzero.front()];
                    uniform = uniform && p.n * first.d == first.n * p.d;
                }
                nonzero.push_back(k);
            }
        }
        FileFraction total(remaining.d - remaining.n, remaining.d);
        occurrence = RationalCoin(total.n, total.d);
        if(uniform) return;
        remaining = std::move(total);
        for(size_t i = 0; i + 1 < nonzero.size(); i++) {
            const auto& p = probabilities[nonzero[i]];
            conditional.emplace_back(p.n * remaining.d, p.d * remaining.n);
            remaining =
                    remaining.d == p.d ? FileFraction(remaining.n - p.n, p.d) : file_fraction_subtract(remaining, p);
        }
    }
    int64_t draw(std::mt19937_64& rng) const {
        if(!occurrence.draw(rng)) return -1;
        return draw_present(rng);
    }
    int64_t draw_present(std::mt19937_64& rng) const {
        if(uniform) return int64_t(nonzero[uniform_index(nonzero.size(), rng)]);
        for(size_t i = 0; i < conditional.size(); i++)
            if(conditional[i].draw(rng)) return int64_t(nonzero[i]);
        return int64_t(nonzero.back());
    }
};
struct FileNoiseAction {
    std::string anchor, report;
    std::vector<std::pair<char, size_t>> gates;
};
// Fixed channels retain one site, not one object per virtual alternative.
// Explicit user actions retain their arbitrary ordered gate sequences.
class FileNoiseActions {
public:
    enum class Template { X, Y, Z, One, Two, Report };

private:
    struct Site {
        size_t offset;
        std::string anchor, report;
        std::array<size_t, 2> targets;
        Template channel;
    };
    std::vector<Site> sites_;
    std::vector<std::pair<size_t, FileNoiseAction>> explicit_;
    size_t size_ = 0;

public:
    size_t size() const { return size_; }
    void push_back(FileNoiseAction action) { explicit_.emplace_back(size_++, std::move(action)); }
    void add(Template channel, std::string anchor, size_t first = 0, size_t second = 0, std::string report = {}) {
        sites_.push_back({size_, std::move(anchor), std::move(report), {first, second}, channel});
        size_ += channel == Template::Two ? 15 : channel == Template::One ? 3 : 1;
    }
    FileNoiseAction operator[](size_t index) const {
        auto custom = std::lower_bound(explicit_.begin(), explicit_.end(), index,
                [](const auto& entry, size_t value) { return entry.first < value; });
        if(custom != explicit_.end() && custom->first == index) return custom->second;
        auto site = std::upper_bound(sites_.begin(), sites_.end(), index,
                [](size_t value, const Site& entry) { return value < entry.offset; });
        --site;
        FileNoiseAction out{site->anchor, site->report, {}};
        const size_t choice = index - site->offset;
        auto gate = [&](char pauli, size_t target) {
            if(pauli != 'I') out.gates.emplace_back(pauli, target);
        };
        switch(site->channel) {
        case Template::X: gate('X', site->targets[0]); break;
        case Template::Y: gate('Y', site->targets[0]); break;
        case Template::Z: gate('Z', site->targets[0]); break;
        case Template::One: gate("XYZ"[choice], site -> targets[0]); break;
        case Template::Two:
            gate("IXYZ"[(choice + 1) / 4], site -> targets[0]);
            gate("IXYZ"[(choice + 1) % 4], site -> targets[1]);
            break;
        case Template::Report: break;
        }
        return out;
    }
    FileNoiseAction at(size_t index) const {
        if(index >= size_) throw std::out_of_range("noise action index");
        return (*this)[index];
    }
    std::array<size_t, 3> storage_counts() const {
        size_t gates = 0;
        for(const auto& entry : explicit_) gates += entry.second.gates.capacity() * sizeof(std::pair<char, size_t>);
        return {sites_.capacity() * sizeof(Site) + explicit_.capacity() * sizeof(explicit_[0]), gates, sites_.size()};
    }
};
struct FileNoisePlan {
    struct Occurrences {
        RationalCoin coin{Integer(0), Integer(1)};
        std::vector<size_t> sites;
        std::vector<uint64_t> unconditional;
    };
    std::vector<Occurrences> occurrences;
    // Counts of absent likelihood factors, with/without guarded sites.
    std::array<std::vector<size_t>, 2> absent_counts;
    std::vector<std::shared_ptr<const FileNoiseCategory>> categories;
    std::vector<std::array<FileFraction, 2>> factors;
    // Equal likelihood factors share prepared powers; draws count uses only.
    std::vector<std::array<int64_t, 2>> factor_groups;
    std::vector<std::vector<FileFraction>> factor_powers;
    // Unconditional sites with the same two factors have a binomial count
    // sufficient for their likelihood. This table contains no sampled results.
    std::vector<int64_t> count_groups;
    std::vector<std::vector<FileFraction>> count_weights;
    std::vector<std::pair<double, int64_t>> count_scientific;
    void prepare_weights() {
        occurrences.clear();
        std::map<std::pair<uint64_t, bool>, size_t> occurrence_groups;
        for(size_t j = 0; j < categories.size(); j++) {
            const auto& coin = categories[j]->occurrence;
            if(!coin.certain && !coin.threshold) continue;
            bool guarded = !conditional.empty() && conditional[j];
            auto key = std::make_pair(coin.threshold, coin.certain);
            auto [it, inserted] = occurrence_groups.emplace(key, occurrences.size());
            if(inserted) occurrences.push_back({coin, {}, {}});
            auto& group = occurrences[it->second];
            auto bit = group.sites.size();
            if(bit % 64 == 0) group.unconditional.push_back(0);
            if(!guarded) group.unconditional.back() |= uint64_t(1) << (bit % 64);
            group.sites.push_back(j);
        }
        factor_groups.clear();
        factor_powers.clear();
        count_groups.clear();
        count_weights.clear();
        count_scientific.clear();
        using Key = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;
        std::map<Key, size_t> groups;
        std::vector<size_t> counts;
        for(const auto& pair : factors) {
            std::array<int64_t, 2> indices{-1, -1};
            for(size_t k = 0; k < 2; k++)
                if(!(pair[k].n == pair[k].d)) {
                    auto key = Key{pair[k].n.d, pair[k].d.d};
                    auto it = groups.find(key);
                    if(it == groups.end()) {
                        size_t id = factor_powers.size();
                        it = groups.emplace(std::move(key), id).first;
                        factor_powers.push_back({pair[k]});
                        counts.push_back(0);
                    }
                    indices[k] = int64_t(it->second);
                    counts[it->second]++;
                }
            factor_groups.push_back(indices);
        }
        for(size_t j = 0; j < counts.size(); j++)
            for(size_t n = counts[j]; n > 1; n >>= 1) {
                auto& powers = factor_powers[j];
                const auto& f = powers.back();
                powers.push_back({f.n * f.n, f.d * f.d});
            }
        std::map<std::array<int64_t, 2>, std::vector<size_t>> pairs;
        count_groups.assign(factors.size(), -1);
        for(size_t j = 0; j < factors.size(); j++)
            if((conditional.empty() || !conditional[j]) && factor_groups[j] != std::array<int64_t, 2>{-1, -1})
                pairs[factor_groups[j]].push_back(j);
        size_t available = 1024 * 1024;
        for(const auto& [pair, sites] : pairs) {
            size_t count = sites.size();
            if(count < 2 || count > 512) continue;
            size_t words = 1;
            for(auto id : pair)
                if(id >= 0) {
                    const auto& f = factor_powers[size_t(id)][0];
                    words = std::max(words, std::max(f.n.d.size(), f.d.d.size()));
                }
            // Conservative limb bound, including multiplication's extra limb.
            // Large groups retain the existing logarithmic power product.
            if(words > available / (8 * (count + 1) * (count + 1))) continue;
            size_t bytes = (count + 1) * (sizeof(FileFraction) + 8 * (count * words + 1));
            if(bytes > available) continue;
            available -= bytes;
            std::vector<FileFraction> table;
            table.reserve(count + 1);
            for(size_t present = 0; present <= count; present++) {
                FileFraction weight(Integer(1));
                for(size_t k = 0; k < 2; k++)
                    if(pair[k] >= 0)
                        for(size_t n = k ? present : count - present, bit = 0; n; n >>= 1, bit++)
                            if(n & 1) {
                                const auto& f = factor_powers[size_t(pair[k])][bit];
                                weight.n = weight.n * f.n;
                                weight.d = weight.d * f.d;
                            }
                table.push_back(std::move(weight));
            }
            for(auto site : sites) count_groups[site] = int64_t(count_weights.size());
            count_weights.push_back(std::move(table));
        }
        // Reuse the rounded ratio as well when a single table supplies the
        // complete weight. Mixed groups still round their exact final product.
        bool complete = count_weights.size() == 1;
        for(size_t j = 0; j < factor_groups.size(); j++)
            if(count_groups[j] < 0 && factor_groups[j] != std::array<int64_t, 2>{-1, -1}) complete = false;
        if(complete)
            for(const auto& f : count_weights.front())
                count_scientific.push_back(
                        f.n.zero() ? std::pair<double, int64_t>{0, 0} : numeric_detail::integer_ratio(f.n, f.d));
        for(size_t skip = 0; skip < 2; skip++) {
            absent_counts[skip].assign(factor_powers.size(), 0);
            for(size_t j = 0; j < factor_groups.size(); j++) {
                if(skip && !conditional.empty() && conditional[j]) continue;
                auto id = factor_groups[j][0];
                if(count_groups[j] < 0 && id >= 0) absent_counts[skip][size_t(id)]++;
            }
        }
    }
    FileNoiseActions actions;
    std::vector<size_t> offsets;
    // Guarded category indices by physical instruction; ordinary draws skip these.
    std::map<size_t, std::vector<size_t>> conditional_sites;
    std::vector<bool> conditional;
    bool has_actions = false;
    size_t tilted = 0, guarded = 0, deterministic = 0;
};

struct FileNoiseDraw {
    std::vector<int64_t> selected;
    FileFraction weight{Integer(1)};
    std::optional<std::pair<double, int64_t>> scientific_weight;
};
inline FileNoiseDraw draw_noise(const FileNoisePlan& plan, uint64_t seed, bool sparse, bool skip_conditional = false) {
    if(sparse && !plan.has_actions) throw std::invalid_argument("noise action table required");
    std::mt19937_64 rng(seed);
    FileNoiseDraw out;
    if(!sparse) out.selected.assign(plan.categories.size(), -1);
    auto counts = plan.absent_counts[skip_conditional];
    std::vector<size_t> present(plan.count_weights.size());
    auto select = [&](size_t j) {
        auto choice = plan.categories[j]->draw_present(rng);
        if(sparse) out.selected.push_back(int64_t(plan.offsets[j] + size_t(choice)));
        else out.selected[j] = choice;
        if(!plan.factor_groups.empty()) {
            auto table = plan.count_groups[j];
            if(table >= 0) present[size_t(table)]++;
            else {
                auto pair = plan.factor_groups[j];
                if(pair[0] >= 0) counts[size_t(pair[0])]--;
                if(pair[1] >= 0) counts[size_t(pair[1])]++;
            }
        }
    };
    for(const auto& group : plan.occurrences) {
        for(size_t start = 0; start < group.sites.size(); start += 64) {
            auto count = std::min(size_t(64), group.sites.size() - start);
            auto live = count == 64 ? ~uint64_t(0) : (uint64_t(1) << count) - 1;
            if(skip_conditional) live &= group.unconditional[start / 64];
            if(count < 8) {
                for(size_t k = 0; k < count; k++)
                    if((live & (uint64_t(1) << k)) && group.coin.draw(rng)) select(group.sites[start + k]);
                continue;
            }
            auto selected = group.coin.draw_bits(rng, live);
            while(selected) {
                auto bit = unsigned(__builtin_ctzll(selected));
                selected &= selected - 1;
                select(group.sites[start + bit]);
            }
        }
    }
    // Groups may interleave in the source. Preserve original action order,
    // including arbitrary ordered Pauli products and report flips.
    if(sparse) std::sort(out.selected.begin(), out.selected.end());
    for(size_t j = 0; j < present.size(); j++) {
        const auto& f = plan.count_weights[j][present[j]];
        if(j == 0) out.weight = f;
        else {
            out.weight.n = out.weight.n * f.n;
            out.weight.d = out.weight.d * f.d;
        }
    }
    for(size_t j = 0; j < counts.size(); j++)
        for(size_t n = counts[j], bit = 0; n; n >>= 1, bit++)
            if(n & 1) {
                const auto& f = plan.factor_powers[j][bit];
                out.weight.n = out.weight.n * f.n;
                out.weight.d = out.weight.d * f.d;
            }
    if(plan.factor_powers.empty()) out.scientific_weight = std::pair<double, int64_t>{0.5, 1};
    else if(!plan.count_scientific.empty()) out.scientific_weight = plan.count_scientific[present[0]];
    return out;
}

} // namespace magic_native
