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
// System includes


// --------------------------------------------------------------------------------
// ADD
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceAdd : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->add(m_t1.value().data(), m_t2.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceAddF3210M = BenchmarkDeviceAdd<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceAddF3210M, "device_add_f32_10m");

// --------------------------------------------------------------------------------
// SUB
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSub : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->sub(m_t1.value().data(), m_t2.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSubF3210M = BenchmarkDeviceSub<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSubF3210M, "device_sub_f32_10m");

// --------------------------------------------------------------------------------
// MUL
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceMul : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->mul(m_t1.value().data(), m_t2.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceMulF3210M = BenchmarkDeviceMul<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceMulF3210M, "device_mul_f32_10m");

// --------------------------------------------------------------------------------
// DIV
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceDiv : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->div(m_t1.value().data(), m_t2.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceDivF3210M = BenchmarkDeviceDiv<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceDivF3210M, "device_div_f32_10m");

// --------------------------------------------------------------------------------
// MATMUL
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t M>
class BenchmarkDeviceMatMul : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({M, M}, opt);
        m_t2 = aix::randn({M, M}, opt);
        m_result = aix::Tensor(aix::Shape{M, M}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->matmul(m_t1.value().data(), {M, M}, m_t2.value().data(), {M, M},
                             m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceMatMulF32800 = BenchmarkDeviceMatMul<aix::DataType::kFloat32,800>;

BENCHMARK(BenchmarkDeviceMatMulF32800, "device_matmul_f32_800");
