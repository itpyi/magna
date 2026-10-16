// Parallel execution behind the ordinary batch interface. Every runtime owns its
// mutable plans; contributions are consumed in draw order by the existing statistics.
#pragma once
#include "file_batch.hpp"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <random>
#include <thread>

namespace magic_native {
inline void advance_parallel_file_batch(CompiledExperiment& main,
        std::vector<std::unique_ptr<CompiledExperiment>>& cached, FileBatch& batch, size_t count,
        size_t requested_workers, const ExecutionHooks& hooks, const std::function<void()>& checkpoint) {
    // The bound applies even to direct native callers, independently of progress reporting.
    constexpr size_t window_size = 4096;
    while(count) {
        size_t size = std::min(count, window_size), workers = std::min(size, requested_workers);
        while(cached.size() + 1 < workers)
            cached.push_back(std::make_unique<CompiledExperiment>(
                    main.source, main.task, main.noise_owner, main.limits, main.database));
        std::vector<CompiledExperiment*> runs{&main};
        for(size_t j = 1; j < workers; j++) runs.push_back(cached[j - 1].get());
        auto budget = std::make_shared<RunContext::SharedBudget>();
        budget->queries = main.context.queries - main.context.query_base;
        std::vector<RunContext> before;
        before.reserve(workers);
        for(auto* run : runs) before.push_back(run->context);
        struct ContextSync {
            CompiledExperiment& main;
            const std::vector<CompiledExperiment*>& runs;
            ~ContextSync() {
                for(auto* run : runs) run->context.shared_budget.reset();
                main.parallel_active = false;
            }
        } sync{main, runs};
        main.parallel_active = true;
        for(auto* run : runs) {
            run->context.started = main.context.started;
            run->context.shared_budget = budget;
        }
        std::vector<uint64_t> seeds;
        seeds.reserve(size);
        for(size_t j = 0; j < size; j++) seeds.push_back(hooks.seed());
        struct Draw {
            HistoryContribution contribution;
            std::vector<int> samples;
        };
        std::vector<std::optional<Draw>> draws(size);
        std::vector<std::exception_ptr> errors(size);
        std::atomic<size_t> next{0}, failed{size};
        std::atomic<bool> cancelled{false};
        std::mutex mutex;
        std::condition_variable ready;
        size_t running = 0;
        std::vector<std::thread> threads;
        threads.reserve(workers);
        std::exception_ptr interrupted;
        try {
            for(auto* run : runs) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    running++;
                }
                try {
                    threads.emplace_back([&, run]() {
                        for(;;) {
                            auto j = next.fetch_add(1, std::memory_order_relaxed);
                            if(j >= size || j >= failed.load(std::memory_order_relaxed) || cancelled.load()) break;
                            try {
                                std::mt19937_64 rng(seeds[j]);
                                auto local = hooks;
                                local.seed = [&]() { return rng(); };
                                auto result = run->sample(local);
                                draws[j] = Draw{result.contribution(), std::move(result.samples)};
                            } catch(...) {
                                errors[j] = std::current_exception();
                                auto first = failed.load(std::memory_order_relaxed);
                                while(j < first && !failed.compare_exchange_weak(first, j, std::memory_order_relaxed)) {}
                                break;
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            running--;
                        }
                        ready.notify_one();
                    });
                } catch(...) {
                    std::lock_guard<std::mutex> lock(mutex);
                    running--;
                    throw;
                }
            }
            std::unique_lock<std::mutex> lock(mutex);
            while(running) {
                ready.wait_for(lock, std::chrono::milliseconds(100));
                lock.unlock();
                checkpoint();
                lock.lock();
            }
        } catch(...) {
            interrupted = std::current_exception();
            cancelled = true;
        }
        // Always join before inspecting results, restoring Python errors, or releasing runtimes.
        for(auto& thread : threads) thread.join();
        // Allocation while merging diagnostic names may fail; keep it outside a destructor.
        for(size_t j = 1; j < runs.size(); j++) main.context.include(runs[j]->context, before[j]);
        auto first = failed.load();
        for(size_t j = 0; j < first && draws[j]; j++) batch.add(draws[j]->contribution, draws[j]->samples);
        if(interrupted) std::rethrow_exception(interrupted);
        if(first < size) std::rethrow_exception(errors[first]);
        count -= size;
    }
}
} // namespace magic_native
