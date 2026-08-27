/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file benchmark_common.h
 * @brief RL benchmark 的参数、单调时钟、周期调度与统计公共实现
 */
#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

#include <sched.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef RL_BENCHMARK_BUILD_TYPE
#define RL_BENCHMARK_BUILD_TYPE "unknown"
#endif

#ifndef RL_BENCHMARK_CXX_FLAGS
#define RL_BENCHMARK_CXX_FLAGS "unknown"
#endif

namespace rl_benchmark {

using Clock = std::chrono::steady_clock;

enum class Mode {
    THROUGHPUT,
    PERIODIC,
};

enum class OverrunPolicy {
    BACKLOG,
    DROP,
};

struct Options {
    int warmup = 100;
    int rounds = 1000;
    double hz = 50.0;
    bool hz_overridden = false;
    Mode mode = Mode::THROUGHPUT;
    OverrunPolicy overrun = OverrunPolicy::DROP;
    std::string provider = "auto";
    int threads = 1;
    std::string affinity;
    std::string ep_affinity;
    bool ep_dump_subgraphs = false;
    std::string ep_profile_prefix;
    bool ort_spinning = true;
    int measure_start_delay_ms = 0;
    std::string csv_path;
    bool verbose_after_timing = false;
};

struct TimingSample {
    std::uint64_t release_index = 0;
    int dropped_before = 0;
    int backlog_releases = 0;
    double release_jitter_ms = 0.0;
    double service_ms = 0.0;
    double response_ms = 0.0;
    double deadline_lateness_ms = 0.0;
};

struct TimingResult {
    std::vector<TimingSample> samples;
    std::uint64_t dropped_releases = 0;
    int max_backlog_releases = 0;
    int deadline_misses = 0;
};

struct Stats {
    double min = 0.0;
    double avg = 0.0;
    double std_dev = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double p9999 = 0.0;
    double max = 0.0;
    bool has_p999 = false;
    bool has_p9999 = false;
};

inline const char *ModeName(Mode mode) {
    return mode == Mode::PERIODIC ? "periodic" : "throughput";
}

inline const char *OverrunPolicyName(OverrunPolicy policy) {
    return policy == OverrunPolicy::BACKLOG ? "backlog" : "drop";
}

inline bool ShouldConfigureSpaceMITProvider(const std::string &provider) {
    if (provider == "spacemit")
        return true;
    if (provider != "auto")
        return false;
#if defined(cpu_rv64) || defined(__riscv)
    return true;
#else
    return false;
#endif
}

inline int ParseInt(const char *text, const std::string &name) {
    if (!text || *text == '\0') {
        throw std::invalid_argument(name + " requires an integer");
    }
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(name + " has an invalid integer: " + text);
    }
    return static_cast<int>(value);
}

inline double ParseDouble(const char *text, const std::string &name) {
    if (!text || *text == '\0') {
        throw std::invalid_argument(name + " requires a number");
    }
    errno = 0;
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (errno != 0 || !end || *end != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument(name + " has an invalid number: " + text);
    }
    return value;
}

inline Options ParseOptions(int argc, char *argv[], int first_option, double default_hz) {
    Options options;
    options.hz = default_hz;

    const auto require_value = [&](int index, const std::string &name) {
        if (index + 1 >= argc) {
            throw std::invalid_argument(name + " requires a value");
        }
    };

    for (int i = first_option; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--warmup") {
            require_value(i, arg);
            options.warmup = ParseInt(argv[++i], arg);
        } else if (arg == "--rounds") {
            require_value(i, arg);
            options.rounds = ParseInt(argv[++i], arg);
        } else if (arg == "--hz") {
            require_value(i, arg);
            options.hz = ParseDouble(argv[++i], arg);
            options.hz_overridden = true;
        } else if (arg == "--mode") {
            require_value(i, arg);
            const std::string value = argv[++i];
            if (value == "throughput") {
                options.mode = Mode::THROUGHPUT;
            } else if (value == "periodic") {
                options.mode = Mode::PERIODIC;
            } else {
                throw std::invalid_argument("--mode must be throughput or periodic");
            }
        } else if (arg == "--overrun") {
            require_value(i, arg);
            const std::string value = argv[++i];
            if (value == "backlog") {
                options.overrun = OverrunPolicy::BACKLOG;
            } else if (value == "drop") {
                options.overrun = OverrunPolicy::DROP;
            } else {
                throw std::invalid_argument("--overrun must be backlog or drop");
            }
        } else if (arg == "--provider") {
            require_value(i, arg);
            options.provider = argv[++i];
        } else if (arg == "--threads") {
            require_value(i, arg);
            options.threads = ParseInt(argv[++i], arg);
        } else if (arg == "--affinity") {
            require_value(i, arg);
            options.affinity = argv[++i];
        } else if (arg == "--ep-affinity") {
            require_value(i, arg);
            options.ep_affinity = argv[++i];
        } else if (arg == "--ep-dump-subgraphs") {
            options.ep_dump_subgraphs = true;
        } else if (arg == "--ep-profile") {
            require_value(i, arg);
            options.ep_profile_prefix = argv[++i];
            if (options.ep_profile_prefix.empty()) {
                throw std::invalid_argument("--ep-profile requires a non-empty prefix");
            }
        } else if (arg == "--ort-spinning") {
            require_value(i, arg);
            const std::string value = argv[++i];
            if (value == "on") {
                options.ort_spinning = true;
            } else if (value == "off") {
                options.ort_spinning = false;
            } else {
                throw std::invalid_argument("--ort-spinning must be on or off");
            }
        } else if (arg == "--measure-start-delay-ms") {
            require_value(i, arg);
            options.measure_start_delay_ms = ParseInt(argv[++i], arg);
        } else if (arg == "--csv") {
            require_value(i, arg);
            options.csv_path = argv[++i];
        } else if (arg == "--verbose") {
            options.verbose_after_timing = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (options.warmup < 0) {
        throw std::invalid_argument("--warmup must be zero or greater");
    }
    if (options.rounds <= 0) {
        throw std::invalid_argument("--rounds must be greater than zero");
    }
    if (!std::isfinite(options.hz) || options.hz <= 0.0) {
        throw std::invalid_argument("--hz/default rl_dt must produce a positive frequency");
    }
    if (options.threads <= 0) {
        throw std::invalid_argument("--threads must be greater than zero");
    }
    if (options.measure_start_delay_ms < 0) {
        throw std::invalid_argument(
            "--measure-start-delay-ms must be zero or greater");
    }
    if (options.provider != "auto" && options.provider != "cpu" && options.provider != "spacemit") {
        throw std::invalid_argument("--provider must be auto, cpu, or spacemit");
    }
    if (options.provider == "cpu"
        && (options.ep_dump_subgraphs || !options.ep_profile_prefix.empty())) {
        throw std::invalid_argument("EP diagnostics require --provider auto or spacemit");
    }
    if (options.provider == "cpu" && !options.ep_affinity.empty()) {
        throw std::invalid_argument("--ep-affinity requires --provider auto or spacemit");
    }
    if (options.rounds >= 100000 && options.csv_path.empty()) {
        throw std::invalid_argument(
            "--rounds >= 100000 requires --csv so P99.99 samples are retained");
    }
    return options;
}

inline void PrintCommonUsage(std::ostream &out) {
    out << "  --mode throughput|periodic  背靠背吞吐或绝对时钟周期释放\n"
        << "  --warmup N                  预热次数，默认 100\n"
        << "  --rounds N                  测量次数，默认 1000\n"
        << "  --hz HZ                     周期频率；默认读取 YAML rl_policy.rl_dt\n"
        << "  --overrun drop|backlog      periodic 过载策略，默认 drop\n"
        << "  --provider auto|cpu|spacemit\n"
        << "  --threads N                 CPU intra-op 或 SpaceMIT EP 线程数\n"
        << "  --ort-spinning on|off      ORT intra-op worker busy-spin，默认 on\n"
        << "  --affinity CPU_LIST         Linux 进程 affinity，例如 0 或 0,1\n"
        << "  --ep-affinity CPU_LIST      SpaceMIT EP affinity；默认沿用 --affinity\n"
        << "  --ep-dump-subgraphs         导出 SpaceMIT EP 实际编译出的子图\n"
        << "  --ep-profile PREFIX         导出 SpaceMIT EP 执行 profile JSON\n"
        << "  --measure-start-delay-ms N  START marker 后、计时前等待监控器，默认 0\n"
        << "  --csv PATH                  保存逐轮原始数据\n"
        << "  --verbose                   计时结束后打印逐轮数据\n";
}

inline std::vector<int> ParseCpuList(const std::string &text) {
    if (text.empty())
        return {};
    std::vector<int> cpus;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::string token
            = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (token.empty()) {
            throw std::invalid_argument("invalid empty CPU in --affinity: " + text);
        }
        const std::size_t dash = token.find('-');
        const int first = ParseInt(token.substr(0, dash).c_str(), "--affinity");
        const int last = dash == std::string::npos
            ? first
            : ParseInt(token.substr(dash + 1).c_str(), "--affinity");
        if (first < 0 || last < first || last >= CPU_SETSIZE) {
            throw std::invalid_argument("invalid CPU range in --affinity: " + token);
        }
        for (int cpu = first; cpu <= last; ++cpu)
            cpus.push_back(cpu);
        if (comma == std::string::npos)
            break;
        begin = comma + 1;
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

inline std::string EpAffinityFromCpuList(
    const std::string &requested, int threads, bool configure_ep) {
    if (!configure_ep || requested.empty())
        return {};
    const auto cpus = ParseCpuList(requested);
    if (static_cast<int>(cpus.size()) != threads) {
        throw std::invalid_argument("SpaceMIT EP affinity CPU count must equal --threads: cpus="
            + std::to_string(cpus.size()) + ", threads=" + std::to_string(threads));
    }
    std::string result;
    for (const int cpu : cpus) {
        if (!result.empty())
            result.push_back(';');
        result += std::to_string(cpu);
    }
    return result;
}

inline std::string FormatCpuSet(const cpu_set_t &set) {
    std::string result;
    int range_begin = -1;
    int previous = -1;
    const auto append_range = [&](int first, int last, std::string *target) {
        if (!target->empty())
            target->append(",");
        target->append(std::to_string(first));
        if (last != first)
            target->append("-").append(std::to_string(last));
    };
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set))
            continue;
        if (range_begin < 0) {
            range_begin = cpu;
        } else if (cpu != previous + 1) {
            append_range(range_begin, previous, &result);
            range_begin = cpu;
        }
        previous = cpu;
    }
    if (range_begin >= 0)
        append_range(range_begin, previous, &result);
    return result.empty() ? "none" : result;
}

inline std::string ApplyAndGetAffinity(const std::string &requested) {
    if (!requested.empty()) {
        cpu_set_t requested_set;
        CPU_ZERO(&requested_set);
        for (const int cpu : ParseCpuList(requested))
            CPU_SET(cpu, &requested_set);
        if (sched_setaffinity(0, sizeof(requested_set), &requested_set) != 0) {
            throw std::runtime_error(
                "sched_setaffinity failed for " + requested + ": " + std::strerror(errno));
        }
    }

    cpu_set_t effective_set;
    CPU_ZERO(&effective_set);
    if (sched_getaffinity(0, sizeof(effective_set), &effective_set) != 0) {
        throw std::runtime_error(std::string("sched_getaffinity failed: ") + std::strerror(errno));
    }
    return FormatCpuSet(effective_set);
}

template <typename Duration> inline double ToMilliseconds(Duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

template <typename Work> TimingResult MeasureRounds(const Options &options, Work work) {
    TimingResult result;
    result.samples.reserve(options.rounds);

    if (options.mode == Mode::THROUGHPUT) {
        for (int i = 0; i < options.rounds; ++i) {
            const auto start = Clock::now();
            work(static_cast<std::uint64_t>(i));
            const auto finish = Clock::now();
            TimingSample sample;
            sample.release_index = static_cast<std::uint64_t>(i);
            sample.service_ms = ToMilliseconds(finish - start);
            sample.response_ms = sample.service_ms;
            result.samples.push_back(sample);
        }
        return result;
    }

    const auto period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0 / options.hz));
    if (period <= Clock::duration::zero()) {
        throw std::runtime_error("period is below steady_clock resolution");
    }

    auto scheduled_release = Clock::now() + period;
    std::uint64_t release_index = 0;
    for (int i = 0; i < options.rounds; ++i) {
        TimingSample sample;
        auto before_sleep = Clock::now();
        if (before_sleep > scheduled_release) {
            const auto late = before_sleep - scheduled_release;
            const auto backlog = static_cast<int>(late / period);
            sample.backlog_releases = backlog;
            result.max_backlog_releases = std::max(result.max_backlog_releases, backlog);
            if (options.overrun == OverrunPolicy::DROP && backlog > 0) {
                sample.dropped_before = backlog;
                result.dropped_releases += static_cast<std::uint64_t>(backlog);
                release_index += static_cast<std::uint64_t>(backlog);
                scheduled_release += period * backlog;
            }
        }

        std::this_thread::sleep_until(scheduled_release);
        const auto start = Clock::now();
        sample.release_index = release_index;
        sample.release_jitter_ms = ToMilliseconds(start - scheduled_release);
        work(release_index);
        const auto finish = Clock::now();

        sample.service_ms = ToMilliseconds(finish - start);
        sample.response_ms = ToMilliseconds(finish - scheduled_release);
        sample.deadline_lateness_ms = std::max(0.0, sample.response_ms - 1000.0 / options.hz);
        if (sample.deadline_lateness_ms > 0.0)
            ++result.deadline_misses;
        result.samples.push_back(sample);

        ++release_index;
        scheduled_release += period;
    }
    return result;
}

inline double Percentile(const std::vector<double> &sorted, double fraction) {
    const auto rank
        = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted.size())));
    const std::size_t index
        = std::min(sorted.size() - 1, rank > 0 ? rank - 1 : static_cast<std::size_t>(0));
    return sorted[index];
}

inline Stats ComputeStats(std::vector<double> values) {
    if (values.empty())
        throw std::invalid_argument("cannot compute empty statistics");
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double avg = sum / static_cast<double>(values.size());
    double square_sum = 0.0;
    for (const double value : values) {
        square_sum += (value - avg) * (value - avg);
    }

    Stats stats;
    stats.min = values.front();
    stats.avg = avg;
    stats.std_dev = std::sqrt(square_sum / static_cast<double>(values.size()));
    stats.p50 = Percentile(values, 0.50);
    stats.p95 = Percentile(values, 0.95);
    stats.p99 = Percentile(values, 0.99);
    stats.max = values.back();
    stats.has_p999 = values.size() >= 10000;
    stats.has_p9999 = values.size() >= 100000;
    if (stats.has_p999)
        stats.p999 = Percentile(values, 0.999);
    if (stats.has_p9999)
        stats.p9999 = Percentile(values, 0.9999);
    return stats;
}

inline std::vector<double> ExtractMetric(
    const std::vector<TimingSample> &samples, double TimingSample::*member) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto &sample : samples)
        values.push_back(sample.*member);
    return values;
}

inline std::string CsvEscape(const std::string &value) {
    if (value.find_first_of(",\"\n") == std::string::npos)
        return value;
    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '\"')
            escaped.push_back('\"');
        escaped.push_back(ch);
    }
    escaped.push_back('\"');
    return escaped;
}

inline std::ofstream OpenCsv(const std::string &path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open CSV output: " + path);
    }
    output << std::fixed << std::setprecision(6);
    return output;
}

inline void BeginMeasuredRegion(const Options &options) {
    std::cout << "Measure...\nBENCHMARK_MEASURE_READY\n" << std::flush;
    if (options.measure_start_delay_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(options.measure_start_delay_ms));
    }
    std::cout << "BENCHMARK_MEASURE_START\n" << std::flush;
}

inline void EndMeasuredRegion() {
    std::cout << "BENCHMARK_MEASURE_END\n" << std::flush;
}

template <typename T>
inline void WriteVectorEvidence(
    std::ostream &output, const std::string &name, const std::vector<T> &values) {
    std::size_t nonfinite = 0;
    double sum = 0.0;
    double square_sum = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const T raw_value : values) {
        const double value = static_cast<double>(raw_value);
        if (!std::isfinite(value)) {
            ++nonfinite;
            continue;
        }
        sum += value;
        square_sum += value * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    output << "# " << name << "_count=" << values.size() << "\n"
        << "# " << name << "_nonfinite=" << nonfinite << "\n"
        << std::fixed << std::setprecision(9)
        << "# " << name << "_sum=" << sum << "\n"
        << "# " << name << "_l2=" << std::sqrt(square_sum) << "\n";
    if (values.size() > nonfinite) {
        output << "# " << name << "_min=" << minimum << "\n"
            << "# " << name << "_max=" << maximum << "\n";
    }
    output << "# " << name << "_values=";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0)
            output << ';';
        output << static_cast<double>(values[index]);
    }
    output << "\n" << std::setprecision(6);
}

inline void PrintBuildMetadata() {
    std::cout << "Build type: " << RL_BENCHMARK_BUILD_TYPE << "\n"
            << "CXX flags:  " << RL_BENCHMARK_CXX_FLAGS << "\n";
}

}  // namespace rl_benchmark

#endif  // BENCHMARK_COMMON_H
