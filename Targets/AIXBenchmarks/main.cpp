//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "common.hpp"
// External includes
#include <docopt/docopt.h>
#include <yaml-cpp/yaml.h>
// System includes
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>


void registerAllBenchmarks()
{
    REGISTER_BENCHMARK(BenchmarkTensorAddF3210M);
    REGISTER_BENCHMARK(BenchmarkTensorSubF3210M);
    REGISTER_BENCHMARK(BenchmarkTensorMulF3210M);
    REGISTER_BENCHMARK(BenchmarkTensorDivF3210M);
    REGISTER_BENCHMARK(BenchmarkTensorMatMulF32800);
    REGISTER_BENCHMARK(BenchmarkDeviceAddF3210M);
    REGISTER_BENCHMARK(BenchmarkDeviceSubF3210M);
    REGISTER_BENCHMARK(BenchmarkDeviceMulF3210M);
    REGISTER_BENCHMARK(BenchmarkDeviceDivF3210M);
    REGISTER_BENCHMARK(BenchmarkDeviceMatMulF32800);
}


AIXBenchmarkResult runBenchmark(const std::shared_ptr<BenchmarkBase>& benchmark, const AIXBenchmarkConfigs& configs)
{
    AIXBenchmarkResult result;
    double avgDurationSum = 0;
    for (size_t i=0; i<configs.samplingCount + configs.warmupCount; ++i)
    {
        // Setup
        benchmark->setup(configs);

        // Run
        auto timeStart = std::chrono::high_resolution_clock::now();
        benchmark->run(configs);
        auto timeEnd = std::chrono::high_resolution_clock::now();

        // CleanUp
        benchmark->cleanUp();

        if (i < configs.warmupCount) continue;   // Skip timing for warm-up.

        auto duration = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
        auto avgDuration = duration / static_cast<double>(configs.iterationCount);
        result.min = std::min(result.min, avgDuration);
        result.max = std::max(result.max, avgDuration);
        avgDurationSum += avgDuration;
    }

    result.avg = avgDurationSum / static_cast<double>(configs.samplingCount);

    return result;
}


void saveBenchmarks(const std::string& filename, const AIXBenchmarkConfigs& configs)
{
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "config" << YAML::BeginMap;
    out << YAML::Key << "deviceType" << YAML::Value << static_cast<size_t>(configs.deviceType);
    out << YAML::Key << "warmupCount" << YAML::Value << configs.warmupCount;
    out << YAML::Key << "samplingCount" << YAML::Value << configs.samplingCount;
    out << YAML::Key << "iterationCount" << YAML::Value << configs.iterationCount;
    out << YAML::EndMap;

    out << YAML::Key << "benchmarks" << YAML::BeginMap;
    for (auto& benchmark : registeredBenchmarksList)
    {
        if (!configs.testList.empty() && configs.testList.find(benchmark->name()) == configs.testList.end()) continue;
        auto result = runBenchmark(benchmark, configs);

        out << YAML::Key << benchmark->name() << YAML::BeginMap;
        out << YAML::Key << "min" << YAML::Value << result.min;
        out << YAML::Key << "max" << YAML::Value << result.max;
        out << YAML::Key << "avg" << YAML::Value << result.avg;
        out << YAML::EndMap;

        std::cout << "[" << benchmark->name().c_str() << "] : done!" << std::endl;
    }
    out << YAML::EndMap;
    out << YAML::EndMap;

    std::ofstream file(filename);
    file << out.c_str();
    file.close();
}


void compareBenchmarks(const std::string& filename, const AIXBenchmarkConfigs& configs)
{
    auto config = YAML::LoadFile(filename);
    auto benchmarks = config["benchmarks"];

    for (auto& benchmark : registeredBenchmarksList)
    {
        if (!configs.testList.empty() && configs.testList.find(benchmark->name()) == configs.testList.end()) continue;
        auto test = benchmarks[benchmark->name()];
        if (test)
        {
            auto results = runBenchmark(benchmark, configs);
            auto avgChange = 100 * (test["avg"].as<double>() - results.avg) / test["avg"].as<double>();
            std::cout << "[" << benchmark->name().c_str() << "]"
                      << " avg base:" << test["avg"].as<double>() << "ms"
                      << " new:" << results.avg << "ms"
                      << " change:" << avgChange  << "%"
                      << std::endl;
        }
    }
}


void listBenchmarkNames()
{
    for (const auto& benchmark : registeredBenchmarksList)
    {
        std::cout << benchmark->name() << std::endl;
    }
}


int main(int argc, const char* argv[])
{
    static const char USAGE[] =
    R"(
    AIXBenchmarks - Copyright (c) 2024-Present, Arkin Terli. All rights reserved.

    Usage:
        AIXBenchmarks (save|compare) --file=<name> --device=<name> [--testName=<name>...] [--wc=<number>]
                                                                   [--sc=<number>] [--ic=<number>]
        AIXBenchmarks list

    Options:
        save                Save benchmark results to a file. No comparison.
        compare             Run benchmarks and compare results within the specified file.
        list                List test names.

        --file=<name>       YAML file that stores benchmark results.
        --device=<name>     Device Type: CPU|MCS.  [default: CPU]
        --wc=<number>       Warm-up count.         [default: 1]
        --sc=<number>       Sampling count.        [default: 1]
        --ic=<number>       Iteration count.       [default: 1000]
)";

    std::map <std::string, docopt::value>  args;

    try
    {
        // Check cmd-line parameters.
        std::vector<std::string>  baseArgs{ argv + 1, argv + argc };
        if (baseArgs.empty() || (baseArgs[0] != "save" && baseArgs[0] != "compare" && baseArgs[0] != "list"))
        {
            std::cout << USAGE << std::endl;
            return -1;
        }

        // Parse cmd-line parameters.
        args = docopt::docopt(USAGE, {argv + 1, argv + argc}, false, "AIXBenchmarks 0.0.0");

        auto file = args["--file"] ? args["--file"].asString() : "";
        if (args["compare"].asBool() && !std::filesystem::exists(file))
        {
            std::cerr << "Could not find file: " << file << std::endl;
            return -1;
        }

        auto deviceTypeStr = args["--device"] ? args["--device"].asString() : "CPU";
        if (deviceTypeStr != "CPU" && deviceTypeStr != "MCS")
        {
            std::cerr << "Unknown device type: " << deviceTypeStr << std::endl;
            return -1;
        }
        aix::DeviceType deviceType = aix::DeviceType::kCPU;
        if (deviceTypeStr == "MCS")
            deviceType = aix::DeviceType::kGPU_METAL;

        // Check if this device is available in this platform.
        auto device = aix::createDevice(deviceType);
        if (!device)
        {
            std::cerr << "Device is not supported in this platform" << std::endl;
            return -1;
        }
        device.release();

        auto samplingCount = args["--sc"] ? args["--sc"].asLong() : 1;
        if (samplingCount < 1)
        {
            std::cerr << "Sampling count must be bigger than one." << std::endl;
            return -1;
        }

        auto iterationCount = args["--ic"] ? args["--ic"].asLong() : 1000;
        if (iterationCount < 1)
        {
            std::cerr << "Iteration count must be bigger than one." << std::endl;
            return -1;
        }

        auto warmupCount = args["--wc"] ? args["--wc"].asLong() : 1;
        if (warmupCount < 1)
        {
            std::cerr << "Warm-up count must be bigger than one." << std::endl;
            return -1;
        }

        auto testsToRun = args["--testName"] ? args["--testName"].asStringList() : std::vector<std::string>();

        AIXBenchmarkConfigs benchConfigs
        {
            .deviceType     = deviceType,
            .warmupCount    = static_cast<size_t>(warmupCount),
            .samplingCount  = static_cast<size_t>(samplingCount),
            .iterationCount = static_cast<size_t>(iterationCount),
            .testList       = std::unordered_set<std::string>(testsToRun.begin(), testsToRun.end())
        };

        registerAllBenchmarks();

        if (args["save"].asBool())
        {
            saveBenchmarks(file, benchConfigs);
        }
        else if (args["compare"].asBool())
        {
            compareBenchmarks(file, benchConfigs);
        }
        else if (args["list"].asBool())
        {
            listBenchmarkNames();
        }
        else
        {
            std::cerr << "Unknown command usage." << std::endl;
            return -1;
        }
    }
    catch (...)
    {
        std::cerr << "Invalid commandline parameter usage." << std::endl;
        return -1;
    }

    return 0;
}
