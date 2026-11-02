// Parallel execution behind the ordinary batch interface. Every runtime owns its
// mutable draw state and shares prepared plans; contributions are consumed in draw order.
#pragma once
#include "file_batch.hpp"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <random>
#include <thread>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace magic_native {
// A compiled circuit keeps its threads alongside its worker-owned runtimes. A
// window borrows its job only until every participating thread has completed, so
// no callback or Python object is retained by an idle worker.
class ParallelFileThreads {
    struct Worker {
        std::condition_variable ready;
        std::thread thread;
        bool assigned = false;
    };
    std::vector<std::unique_ptr<Worker>> workers;
    std::mutex mutex;
    std::condition_variable completed;
    bool stopping = false;
    size_t running = 0;
    const std::function<void(size_t)>* job = nullptr;
    std::atomic<bool>* cancelled = nullptr;
    std::exception_ptr worker_error;

    void serve(Worker& worker, size_t index) {
        std::unique_lock<std::mutex> lock(mutex);
        for(;;) {
            worker.ready.wait(lock, [&]() { return stopping || worker.assigned; });
            if(stopping) return;
            worker.assigned = false;
            lock.unlock();
            std::exception_ptr error;
            try {
                (*job)(index);
            } catch(...) {
                error = std::current_exception();
                cancelled->store(true, std::memory_order_relaxed);
            }
            lock.lock();
            if(error && !worker_error) worker_error = error;
            if(--running == 0) completed.notify_one();
        }
    }
    void ensure(size_t count) {
        workers.reserve(count);
        while(workers.size() < count) {
            auto index = workers.size();
            workers.push_back(std::make_unique<Worker>());
            auto* worker = workers.back().get();
            try {
                worker->thread = std::thread([this, worker, index]() { serve(*worker, index); });
            } catch(...) {
                workers.pop_back();
                throw;
            }
        }
    }

public:
    ~ParallelFileThreads() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        for(auto& worker : workers) worker->ready.notify_one();
        for(auto& worker : workers) worker->thread.join();
    }
    std::exception_ptr execute(size_t count, const std::function<void(size_t)>& work,
            const std::function<void()>& checkpoint, std::atomic<bool>& cancel) {
        ensure(count);
        std::unique_lock<std::mutex> lock(mutex);
        job = &work;
        cancelled = &cancel;
        worker_error = nullptr;
        running = count;
        for(size_t j = 0; j < count; j++) {
            workers[j]->assigned = true;
            workers[j]->ready.notify_one();
        }
        std::exception_ptr interrupted;
        while(running) {
            completed.wait_for(lock, std::chrono::milliseconds(100));
            lock.unlock();
            try {
                checkpoint();
            } catch(...) {
                interrupted = std::current_exception();
                cancel.store(true, std::memory_order_relaxed);
            }
            lock.lock();
            if(interrupted) completed.wait(lock, [&]() { return running == 0; });
        }
        job = nullptr;
        cancelled = nullptr;
        auto error = interrupted ? interrupted : worker_error;
        worker_error = nullptr;
        return error;
    }
};

class ParallelFileExecutor {
    std::unique_ptr<ParallelFileThreads> threads;
#if defined(__unix__) || defined(__APPLE__)
    pid_t process = getpid();
#endif
    void discard_inherited() {
#if defined(__unix__) || defined(__APPLE__)
        auto current = getpid();
        if(current != process) {
            // After an idle circuit is inherited through fork, its worker
            // threads no longer exist. Neither joining their handles nor
            // destroying their inherited synchronization objects is safe.
            // Abandon this small copied pool until child exit; the independent
            // runtime caches remain usable and the child starts fresh threads.
            threads.release();
            process = current;
        }
#endif
    }
public:
    ~ParallelFileExecutor() { discard_inherited(); }
    std::exception_ptr execute(size_t count, const std::function<void(size_t)>& work,
            const std::function<void()>& checkpoint, std::atomic<bool>& cancel) {
        discard_inherited();
        if(!threads) threads = std::make_unique<ParallelFileThreads>();
        return threads->execute(count, work, checkpoint, cancel);
    }
};

inline void advance_parallel_file_batch(CompiledExperiment& main,
        std::vector<std::unique_ptr<CompiledExperiment>>& cached, ParallelFileExecutor& executor,
        FileBatch& batch, size_t count, size_t requested_workers, const ExecutionHooks& hooks,
        const std::function<void()>& checkpoint,
        const std::function<std::vector<uint64_t>(size_t)>& seed_batch) {
    // The bound applies even to direct native callers, independently of progress reporting.
    constexpr size_t window_size = 4096;
    while(count) {
        size_t size = std::min(count, window_size), workers = std::min(size, requested_workers);
        if(!main.complete) main.prepare();
        while(cached.size() + 1 < workers)
            cached.push_back(std::make_unique<CompiledExperiment>(main));
        std::vector<CompiledExperiment*> runs{&main};
        for(size_t j = 1; j < workers; j++) {
            // Configuration changes invalidate every runtime before the main
            // plan is rebuilt. Reuse that new owner, including for idle workers
            // which return after a later increase in the requested count.
            cached[j - 1]->share_prepared(main);
            runs.push_back(cached[j - 1].get());
        }
        for(auto* run : runs) run->prepare_parallel_resources();
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
        auto seeds = seed_batch(size);
        struct Draw {
            HistoryContribution contribution;
            std::vector<int> samples;
        };
        std::vector<std::optional<Draw>> draws(size);
        std::vector<std::exception_ptr> errors(size);
        std::atomic<size_t> next{0}, failed{size};
        std::atomic<bool> cancelled{false};
        auto interrupted = executor.execute(workers, [&](size_t worker) {
            auto* run = runs[worker];
            // Contiguous claims reduce contention and adjacent result writes;
            // short chunks still balance histories with different costs.
            const size_t chunk_size = std::min(size_t(16), std::max(size_t(1), size / (workers * 4)));
            for(;;) {
                auto begin = next.fetch_add(chunk_size, std::memory_order_relaxed);
                if(begin >= size) break;
                auto end = std::min(begin + chunk_size, size);
                for(size_t j = begin; j < end; j++) {
                    if(j >= failed.load(std::memory_order_relaxed) || cancelled.load(std::memory_order_relaxed)) return;
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
                        return;
                    }
                }
            }
        }, checkpoint, cancelled);
        // Every worker is idle before inspecting results or restoring Python errors.
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
