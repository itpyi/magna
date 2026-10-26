// Bounded complete-certificate snapshots; arbitrary Python serialization is an adapter.
#pragma once
#include "../foundation/scalar.hpp"
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
namespace magic_native {
struct CertificateCache {
    struct Entry {
        std::string key, blob;
        std::optional<Scalar> probability;
        size_t scalar_bytes;
        double cost;
    };
    size_t budget, max_entries, bytes = 0, hits = 0, misses = 0, evictions = 0, oversize = 0;
    double saved_seconds = 0;
    std::list<Entry> entries;
    std::map<std::string, std::list<Entry>::iterator> index;
    mutable std::mutex mutex;
    CertificateCache(size_t budget, size_t max_entries) : budget(budget), max_entries(max_entries) {}
    static size_t size(const Entry& e) { return e.key.size() + e.blob.size() + e.scalar_bytes; }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        entries.clear();
        index.clear();
        bytes = hits = misses = evictions = oversize = 0;
        saved_seconds = 0;
    }
    std::optional<Entry> get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = index.find(key);
        if(found == index.end()) {
            misses++;
            return {};
        }
        hits++;
        entries.splice(entries.end(), entries, found->second);
        saved_seconds += found->second->cost;
        return *found->second;
    }
    void put(Entry value) {
        std::lock_guard<std::mutex> lock(mutex);
        if(!budget || !max_entries) return;
        auto added = size(value);
        if(added > budget) {
            oversize++;
            return;
        }
        if(auto found = index.find(value.key); found != index.end()) {
            bytes -= size(*found->second);
            entries.erase(found->second);
            index.erase(found);
        }
        while(!entries.empty() && (added > budget - bytes || entries.size() >= max_entries)) {
            bytes -= size(entries.front());
            index.erase(entries.front().key);
            entries.pop_front();
            evictions++;
        }
        entries.push_back(std::move(value));
        index[entries.back().key] = std::prev(entries.end());
        bytes += added;
    }
};
} // namespace magic_native
