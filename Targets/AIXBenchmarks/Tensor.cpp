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
class BenchmarkTensorAdd : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);;
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 + m_t2;
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorAddF3210M = BenchmarkTensorAdd<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorAddF3210M, "tensor_add_f32_10m");

// --------------------------------------------------------------------------------
// SUB
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorSub : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);;
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 - m_t2;
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorSubF3210M = BenchmarkTensorSub<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorSubF3210M, "tensor_sub_f32_10m");

// --------------------------------------------------------------------------------
// MUL
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorMul : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);;
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 * m_t2;
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorMulF3210M = BenchmarkTensorMul<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorMulF3210M, "tensor_mul_f32_10m");

// --------------------------------------------------------------------------------
// DIV
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorDiv : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);;
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 / m_t2;
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorDivF3210M = BenchmarkTensorDiv<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorDivF3210M, "tensor_div_f32_10m");

// --------------------------------------------------------------------------------
// MATMUL
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t M>
class BenchmarkTensorMatMul : public BenchmarkBase
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
            auto t = matmul(m_t1, m_t2);
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

using BenchmarkTensorMatMulF32800 = BenchmarkTensorMatMul<aix::DataType::kFloat32,800>;

BENCHMARK(BenchmarkTensorMatMulF32800, "tensor_matmul_f32_800");
