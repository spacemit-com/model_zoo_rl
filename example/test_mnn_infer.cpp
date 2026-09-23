/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file test_mnn_infer.cpp
 * @brief MNN FLOAT32 model smoke test, independent of PolicyExecutor.
 */

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mnn_infer.h"

namespace fs = std::filesystem;

static bool TestModel(const std::string &path, bool random_input, int iterations) {
    try {
        std::cout << "[model] " << path << std::endl;
        mnn_runtime::MnnRuntimeClass runtime;
        if (!runtime.Init(path)) {
            throw std::runtime_error(runtime.GetLastError());
        }
        runtime.PrintModelInfo();
        if (runtime.GetInputCount() == 0 || runtime.GetOutputCount() == 0) {
            throw std::runtime_error("model must have inputs and outputs");
        }

        // Fixed seed makes smoke runs reproducible. Initialize every input.
        std::mt19937 generator(0);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (int i = 0; i < runtime.GetInputCount(); ++i) {
            const auto &info = runtime.GetInputInfo(i);
            if (!runtime.CanSetInputFromFloat(i) || info.total_size <= 0) {
                throw std::runtime_error("requires a nonempty FLOAT32 input: " + info.name);
            }
            std::vector<float> input(static_cast<std::size_t>(info.total_size), 0.0f);
            if (random_input) {
                for (float &value : input) {
                    value = distribution(generator);
                }
            }
            runtime.SetInputFromFloat(i, input.data(), input.size());
        }
        for (int i = 0; i < runtime.GetOutputCount(); ++i) {
            const auto &info = runtime.GetOutputInfo(i);
            if (!runtime.CanGetOutputAsFloat(i) || info.total_size <= 0) {
                throw std::runtime_error("requires a nonempty FLOAT32 output: " + info.name);
            }
        }

        double total_ms = 0.0;
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = 0.0;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            if (!runtime.Run()) {
                throw std::runtime_error(runtime.GetLastError());
            }
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            total_ms += elapsed_ms;
            min_ms = std::min(min_ms, elapsed_ms);
            max_ms = std::max(max_ms, elapsed_ms);

            for (int i = 0; i < runtime.GetOutputCount(); ++i) {
                const auto &info = runtime.GetOutputInfo(i);
                const auto &output = runtime.GetOutput(i);
                if (output.size() != static_cast<std::size_t>(info.total_size)) {
                    throw std::runtime_error("output size mismatch: " + info.name);
                }
                for (float value : output) {
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("non-finite output: " + info.name);
                    }
                }
            }
        }
        std::cout << "[PASS] " << path << "\n  " << iterations
            << " iterations, avg/min/max: " << std::fixed << std::setprecision(3)
            << total_ms / iterations << " / " << min_ms << " / " << max_ms << " ms\n";
        return true;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << path << ": " << error.what() << '\n';
        return false;
    }
}

static int ScanModels(const std::string &directory, bool random_input, int iterations) {
    std::vector<std::string> models;
    for (const auto &entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mnn") {
            models.push_back(entry.path().string());
        }
    }
    std::sort(models.begin(), models.end());
    if (models.empty()) {
        throw std::runtime_error("no .mnn models found in " + directory);
    }

    std::size_t passed = 0;
    for (const auto &path : models) {
        // Isolate crashes and bound each model so the remaining scan can finish.
        std::cout.flush();
        std::cerr.flush();
        const pid_t child = fork();
        if (child < 0) {
            throw std::runtime_error("failed to fork model test");
        }
        if (child == 0) {
            alarm(120);
            const bool ok = TestModel(path, random_input, iterations);
            std::cout.flush();
            std::cerr.flush();
            _exit(ok ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        int status = 0;
        pid_t result;
        do {
            result = waitpid(child, &status, 0);
        } while (result < 0 && errno == EINTR);
        if (result < 0) {
            throw std::runtime_error("failed to wait for model test");
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS) {
            ++passed;
        } else if (WIFSIGNALED(status)) {
            std::cerr << "[FAIL] " << path << ": signal " << WTERMSIG(status) << '\n';
        }
    }
    std::cout << "MNN scan: " << passed << " passed, " << models.size() - passed << " failed\n";
    return passed == models.size() ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char *argv[]) {
    try {
        if (argc < 2) {
            throw std::invalid_argument("missing model path or --scan directory");
        }
        const bool scan = std::string(argv[1]) == "--scan";
        if (scan && argc < 3) {
            throw std::invalid_argument("--scan requires a directory");
        }
        const std::string path = argv[scan ? 2 : 1];
        bool random_input = false;
        int iterations = 50;
        for (int i = scan ? 3 : 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--random") {
                random_input = true;
            } else if (option == "--iterations" && i + 1 < argc) {
                const std::string value = argv[++i];
                std::size_t consumed = 0;
                iterations = std::stoi(value, &consumed);
                if (consumed != value.size() || iterations <= 0) {
                    throw std::invalid_argument("--iterations requires a positive integer");
                }
            } else {
                throw std::invalid_argument("unknown or incomplete option: " + option);
            }
        }
        return scan ? ScanModels(path, random_input, iterations)
            : (TestModel(path, random_input, iterations) ? EXIT_SUCCESS : EXIT_FAILURE);
    } catch (const std::exception &error) {
        std::cerr << "[ERROR] " << error.what() << "\nUsage:\n  " << argv[0]
            << " <model.mnn> [--random] [--iterations N]\n  " << argv[0]
            << " --scan <directory> [--random] [--iterations N]\n";
        return EXIT_FAILURE;
    }
}
