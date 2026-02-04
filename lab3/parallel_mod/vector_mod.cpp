#include "vector_mod.h"
#include "num_threads.h"
#include <thread>
#include <condition_variable>
#include <mutex>
#include "mod_ops.h"
#include <vector>

IntegerWord pow_mod(IntegerWord base, IntegerWord power, IntegerWord mod) {
    IntegerWord result = 1;
    while (power > 0) {
        if (power % 2 != 0) {
            result = mul_mod(result, base, mod);
        }
        power >>= 1;
        base = mul_mod(base, base, mod);
    }
    return result;
}

IntegerWord word_pow_mod(size_t power, IntegerWord mod) {
    return pow_mod((-mod) % mod, power, mod);
}


class barrier {
    const unsigned m_max_threads;
    unsigned m_curruent_threads = 0;
    std::condition_variable m_cv;
    std::mutex m_mtx;
    bool m_curr_gen = true;

    public:
        barrier (unsigned T): m_max_threads(T) {};
        void arrive_and_wait(){
            std::unique_lock l{m_mtx};
            if (++m_curruent_threads < m_max_threads){
                unsigned curr_gen = m_curr_gen;
                while (m_curr_gen == curr_gen){
                    m_cv.wait(l);
                }
            }
            else {
                m_curr_gen = !m_curr_gen;
                m_curruent_threads = 0;
                m_cv.notify_all();
            }
        };
};


struct partials_t{
    alignas(std::hardware_destructive_interference_size) IntegerWord v;
};

struct thread_range {
	unsigned long start;
    size_t end;
    thread_range(unsigned long B, size_t e){
        this->start = B;
        this->end = e;
    }
};

thread_range vector_thread_range(size_t n, unsigned T, unsigned t) {
	auto B = n % T;
	auto S = n / T;
	if (t < B) B = ++S * t;
	else B += S * t;
	size_t e = B + S;
	return thread_range(B, e);
}

IntegerWord vector_mod(const IntegerWord* V, std::size_t N, IntegerWord mod){
    unsigned T = get_num_threads();
    barrier bar{T};
    auto partials = std::make_unique<partials_t[]>(T);
    std::vector<std::thread> workers(T-1);
    
    auto worker_proc = [T, &partials, &bar, V, N, mod](unsigned t){
        IntegerWord r = 0;
        thread_range range = vector_thread_range(N, T, t);

        for (auto i = range.start; i < range.end; ++i){
            r = add_mod(times_word(r, mod), V[range.end - (i + 1 - range.start)], mod);
        }

        partials[t].v = mul_mod(r, pow_mod(-mod, range.start, mod), mod);

        for (size_t step = 1, round = 2; step < T; step = round, round += round){
            bar.arrive_and_wait();
            if (!(t % round) && t + step < T){
                partials[t].v = add_mod(partials[t].v, partials[t+step].v, mod);
            }
        }
    };

    
    for (unsigned t=0; t < T-1; t++) workers[t] = std::thread(worker_proc, t + 1);
    worker_proc(0);

    for (auto & worker: workers) worker.join();
    return partials[0].v;

};