/**
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file test_onnx_infer.cpp
 * @brief OnnxRuntimeClass 推理测试（临时测试，不依赖 policy_executor）
 *
 * 用法：
 *   单模型测试:  ./test_onnx_infer <model.onnx> [--random]
 *   批量扫描:    ./test_onnx_infer --scan <robot_dir>
 *               ./test_onnx_infer --scan   (默认扫描 ../../../application/robot)
 */

#include <execinfo.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>  // NOLINT(build/c++17)
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "onnx_infer.h"

namespace fs = std::filesystem;
using onnx_runtime::OnnxRuntimeClass;
using onnx_runtime::TensorInfo;

// ---- 信号处理器：崩溃时将 backtrace 写入 stdout（管道）----
static void crash_handler(int) {
    const char *marker = "BACKTRACE:\n";
    write(STDOUT_FILENO, marker, strlen(marker));
    void *trace[32];
    int n = backtrace(trace, 32);
    backtrace_symbols_fd(trace, n, STDOUT_FILENO);
    fsync(STDOUT_FILENO);
    _exit(1);
}

// ---- 迭代次数 ----
static const int NUM_ITERATIONS = 50;

// ---- 单模型测试，返回是否成功 ----
struct ModelResult {
    std::string path;
    bool load_ok = false;
    bool infer_ok = false;
    int input_count = 0;
    int output_count = 0;
    std::vector<std::string> input_shapes;
    std::vector<std::string> output_shapes;
    double avg_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;
    std::string error;
    std::vector<std::string> backtrace;
};

static std::string shape_str(const TensorInfo &info) {
    std::string s = "[";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        s += std::to_string(info.shape[i]);
        if (i + 1 < info.shape.size())
            s += ",";
    }
    return s + "]";
}

static ModelResult test_model(const std::string &model_path, bool use_random) {
    ModelResult r;
    r.path = model_path;

    OnnxRuntimeClass ort;
    if (!ort.Init(model_path)) {
        r.error = "模型加载失败";
        return r;
    }
    r.load_ok = true;
    r.input_count = ort.GetInputCount();
    r.output_count = ort.GetOutputCount();

    for (int i = 0; i < r.input_count; ++i) {
        const auto &info = ort.GetInputInfo(i);
        r.input_shapes.push_back(info.name + shape_str(info));
        auto &input = ort.GetInput(i);
        if (use_random) {
            for (int j = 0; j < input.size(); ++j)
                input[j] = static_cast<float>(std::rand()) / RAND_MAX;
        } else {
            input.setZero();
        }
    }
    for (int i = 0; i < r.output_count; ++i) {
        const auto &info = ort.GetOutputInfo(i);
        r.output_shapes.push_back(info.name + shape_str(info));
    }

    double total_ms = 0;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        ort.Run();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        r.min_ms = std::min(r.min_ms, ms);
        r.max_ms = std::max(r.max_ms, ms);
    }
    r.avg_ms = total_ms / NUM_ITERATIONS;
    r.infer_ok = true;
    return r;
}

// ---- 递归扫描目录下所有 .onnx 文件 ----
static std::vector<std::string> find_onnx(const std::string &root) {
    std::vector<std::string> files;
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ---- ANSI 颜色 ----
#define CLR_GREEN "\033[32m"
#define CLR_RED "\033[31m"
#define CLR_YELLOW "\033[33m"
#define CLR_BOLD "\033[1m"
#define CLR_RESET "\033[0m"

// ---- 打印分析报告 ----
static void print_report(const std::vector<ModelResult> &results) {
    int pass = 0, fail = 0;

    std::cout << "\n";
    std::cout << CLR_BOLD << std::string(90, '=') << CLR_RESET << "\n";
    std::cout << CLR_BOLD << "  ONNX 推理批量测试报告" << CLR_RESET << "\n";
    std::cout << CLR_BOLD << std::string(90, '=') << CLR_RESET << "\n";

    for (const auto &r : results) {
        std::cout << "\n";
        std::cout << std::string(90, '-') << "\n";

        // 取相对路径显示
        std::string name = r.path;
        auto pos = name.find("application/robot/");
        if (pos != std::string::npos)
            name = name.substr(pos + 18);

        if (r.infer_ok) {
            std::cout << "  " << CLR_GREEN << CLR_BOLD << "[ PASS ]" << CLR_RESET << "  "
                    << CLR_BOLD << name << CLR_RESET << "\n";
            ++pass;
        } else {
            std::cout << "  " << CLR_RED << CLR_BOLD << "[ FAIL ]" << CLR_RESET << "  " << CLR_BOLD
                    << name << CLR_RESET << "\n";
            ++fail;
        }

        if (r.load_ok) {
            std::cout << "  │\n";
            std::cout << "  ├─ 输入(" << r.input_count << "): ";
            for (const auto &s : r.input_shapes)
                std::cout << s << "  ";
            std::cout << "\n";
            std::cout << "  ├─ 输出(" << r.output_count << "): ";
            for (const auto &s : r.output_shapes)
                std::cout << s << "  ";
            std::cout << "\n";
        }
        if (r.infer_ok) {
            std::cout << "  └─ 推理耗时 (Avg/Min/Max): " << std::fixed << std::setprecision(2)
                    << r.avg_ms << " / " << r.min_ms << " / " << r.max_ms << " ms ("
                    << NUM_ITERATIONS << " 次迭代)\n";
        }
        if (!r.error.empty()) {
            std::cout << "  └─ " << CLR_RED << "错误: " << r.error << CLR_RESET << "\n";
            for (size_t i = 0; i < r.backtrace.size(); ++i)
                std::cout << "     " << CLR_YELLOW << r.backtrace[i] << CLR_RESET << "\n";
        }
    }

    std::cout << "\n" << CLR_BOLD << std::string(90, '=') << CLR_RESET << "\n";
    std::cout << "  共 " << results.size() << " 个模型    " << CLR_GREEN << CLR_BOLD
            << "通过: " << pass << CLR_RESET << "    " << CLR_RED << CLR_BOLD << "失败: " << fail
            << CLR_RESET << "\n";
    std::cout << CLR_BOLD << std::string(90, '=') << CLR_RESET << "\n";
}

int main(int argc, char *argv[]) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // ---- 子进程单模型模式（供批量扫描调用）----
    if (argc >= 3 && std::string(argv[1]) == "--single") {
        const std::string path = argv[2];

        // 安装崩溃信号处理器，将 backtrace 写入管道
        signal(SIGSEGV, crash_handler);
        signal(SIGABRT, crash_handler);

        OnnxRuntimeClass ort;

        // 阶段1：加载模型
        if (!ort.Init(path)) {
            std::cout << "STAGE:load_fail\n";
            std::cout.flush();
            return 1;
        }

        // 阶段2：立即输出 I/O 结构（Run 崩溃前已写入管道）
        int in_cnt = ort.GetInputCount();
        int out_cnt = ort.GetOutputCount();
        std::string io_line =
            "IOINFO:" + std::to_string(in_cnt) + ":" + std::to_string(out_cnt) + ":";
        for (int i = 0; i < in_cnt; ++i)
            io_line += ort.GetInputInfo(i).name + shape_str(ort.GetInputInfo(i)) +
                        (i + 1 < in_cnt ? "|" : "");
        io_line += ":";
        for (int i = 0; i < out_cnt; ++i)
            io_line += ort.GetOutputInfo(i).name + shape_str(ort.GetOutputInfo(i)) +
                        (i + 1 < out_cnt ? "|" : "");
        std::cout << io_line << "\n";
        std::cout.flush();  // 确保崩溃前已写入管道

        // 阶段3：填充输入
        for (int i = 0; i < in_cnt; ++i)
            ort.GetInput(i).setZero();

        // 阶段4：推理（可能崩溃）
        double total_ms = 0;
        double min_ms = 1e9, max_ms = 0;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            ort.Run();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }
        double avg_ms = total_ms / NUM_ITERATIONS;

        std::cout << "RESULT:ok:avg:" << std::fixed << std::setprecision(2) << avg_ms
                << ":min:" << min_ms << ":max:" << max_ms << "\n";
        std::cout.flush();
        return 0;
    }

    // ---- 批量扫描模式（fork 子进程，防止 EP 崩溃影响主进程）----
    if (argc >= 2 && std::string(argv[1]) == "--scan") {
        std::string robot_dir = (argc >= 3) ? argv[2] : "";
        if (robot_dir.empty()) {
            std::cerr << "[scan] 请提供 robot 目录路径\n";
            return 1;
        }
        robot_dir = fs::canonical(robot_dir).string();
        std::cout << "[scan] 扫描目录: " << robot_dir << "\n";

        auto files = find_onnx(robot_dir);
        if (files.empty()) {
            std::cerr << "[scan] 未找到任何 .onnx 文件\n";
            return 1;
        }
        std::cout << "[scan] 发现 " << files.size() << " 个模型，开始测试...\n\n";

        std::vector<ModelResult> results;
        std::string self = argv[0];

        for (const auto &f : files) {
            std::cout << "测试: " << f << " ... " << std::flush;

            // 创建管道读取子进程输出
            int pipefd[2];
            pipe(pipefd);

            pid_t pid = fork();
            if (pid == 0) {
                // 子进程：重定向 stdout 到管道，stderr 丢弃
                close(pipefd[0]);
                dup2(pipefd[1], STDOUT_FILENO);
                close(pipefd[1]);
                int devnull = open("/dev/null", O_WRONLY);
                dup2(devnull, STDERR_FILENO);
                execl(self.c_str(), self.c_str(), "--single", f.c_str(), nullptr);
                _exit(1);
            }

            // 父进程：读取管道输出
            close(pipefd[1]);
            std::string output;
            char buf[256];
            ssize_t n;
            while ((n = read(pipefd[0], buf, sizeof(buf) - 1)) > 0) {
                buf[n] = '\0';
                output += buf;
            }
            close(pipefd[0]);

            int status = 0;
            waitpid(pid, &status, 0);
            bool crashed = WIFSIGNALED(status) ||
                            (WIFEXITED(status) && output.find("BACKTRACE:") != std::string::npos);

            ModelResult r;
            r.path = f;

            // 解析 IOINFO 行（无论是否崩溃，只要 Init 成功就有）
            auto io_pos = output.find("IOINFO:");
            if (io_pos != std::string::npos) {
                r.load_ok = true;
                // 只取这一行，不跨行
                auto eol = output.find('\n', io_pos);
                std::string line = output.substr(
                    io_pos + 7, eol == std::string::npos ? std::string::npos : eol - io_pos - 7);
                auto tok = [&](std::string &s, char sep = ':') {
                    auto p = s.find(sep);
                    std::string t = s.substr(0, p);
                    s = (p == std::string::npos) ? "" : s.substr(p + 1);
                    return t;
                };
                auto split = [](const std::string &s) {
                    std::vector<std::string> v;
                    std::string cur;
                    for (char c : s) {
                        if (c == '|') {
                            v.push_back(cur);
                            cur.clear();
                        } else {
                        cur += c;
                    }
                    }
                    if (!cur.empty())
                        v.push_back(cur);
                    return v;
                };
                r.input_count = std::stoi(tok(line));
                r.output_count = std::stoi(tok(line));
                r.input_shapes = split(tok(line));
                r.output_shapes = split(tok(line));
            }

            // 解析 RESULT 行
            auto res_pos = output.find("RESULT:ok:");
            if (res_pos != std::string::npos) {
                r.infer_ok = true;
                std::string line = output.substr(res_pos + 10);
                auto tok = [&](std::string &s, char sep = ':') {
                    auto p = s.find(sep);
                    std::string t = s.substr(0, p);
                    s = (p == std::string::npos) ? "" : s.substr(p + 1);
                    return t;
                };
                try {
                    while (!line.empty()) {
                        std::string key = tok(line);
                        std::string val = tok(line);
                        if (key == "avg")
                            r.avg_ms = std::stod(val);
                        else if (key == "min")
                            r.min_ms = std::stod(val);
                        else if (key == "max")
                            r.max_ms = std::stod(val);
                    }
                } catch (...) {
                }
            } else if (crashed) {
                r.error = r.load_ok ? "推理时进程崩溃 (SIGSEGV/SIGABRT)"
                                    : "加载时进程崩溃 (SIGSEGV/SIGABRT)";
            } else if (output.find("STAGE:load_fail") != std::string::npos) {
                r.error = "模型加载失败";
            } else if (!r.infer_ok) {
                r.error = "推理失败（未知原因）";
            }

            // 解析 BACKTRACE 行（信号处理器写入）
            auto bt_pos = output.find("BACKTRACE:\n");
            if (bt_pos != std::string::npos) {
                std::string bt_section = output.substr(bt_pos + 11);
                std::istringstream ss(bt_section);
                std::string line;
                while (std::getline(ss, line)) {
                    if (!line.empty())
                        r.backtrace.push_back(line);
                }
            }

            std::cout << (r.infer_ok ? "OK" : (crashed ? "CRASH" : "FAIL")) << "\n";
            results.push_back(std::move(r));
        }

        print_report(results);
        return 0;
    }

    // ---- 单模型模式 ----
    if (argc < 2) {
        std::cerr << "用法:\n"
                << "  单模型: " << argv[0] << " <model.onnx> [--random]\n"
                << "  批量:   " << argv[0] << " --scan [robot_dir]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    bool use_random = (argc >= 3 && std::string(argv[2]) == "--random");

    OnnxRuntimeClass ort;
    if (!ort.Init(model_path)) {
        std::cerr << "[错误] 模型加载失败\n";
        return 1;
    }
    ort.PrintModelInfo();

    for (int i = 0; i < ort.GetInputCount(); ++i) {
        auto &input = ort.GetInput(i);
        if (use_random) {
            for (int j = 0; j < input.size(); ++j)
                input[j] = static_cast<float>(std::rand()) / RAND_MAX;
            std::cout << "[test] 输入[" << i << "] 随机值，维度=" << input.size() << "\n";
        } else {
            input.setZero();
            std::cout << "[test] 输入[" << i << "] 全零，维度=" << input.size() << "\n";
        }
    }

    std::cout << "\n[test] 执行推理 (" << NUM_ITERATIONS << " 次迭代)...\n";
    double total_ms = 0;
    double min_ms = 1e9, max_ms = 0;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        ort.Run();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }
    std::cout << "[test] 推理完成. 平均耗时: " << (total_ms / NUM_ITERATIONS)
            << " ms, Min: " << min_ms << " ms, Max: " << max_ms << " ms\n\n";

    for (int i = 0; i < ort.GetOutputCount(); ++i) {
        const auto &output = ort.GetOutput(i);
        const auto &info = ort.GetOutputInfo(i);
        std::cout << "输出[" << i << "] " << info.name << "  维度=" << output.size() << "\n  值: [";
        int n = std::min(static_cast<int>(output.size()), 20);
        for (int j = 0; j < n; ++j) {
            std::cout << output[j];
            if (j < n - 1)
                std::cout << ", ";
        }
        if (output.size() > 20)
            std::cout << ", ...";
        std::cout << "]\n";
    }

    std::cout << "\n[test] 测试成功\n";
    return 0;
}
