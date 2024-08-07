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
// UNARY
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceUnary : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->unary(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceUnaryF3210M = BenchmarkDeviceUnary<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceUnaryF3210M, "device_unary_f32_10m");

// --------------------------------------------------------------------------------
// FILL
// --------------------------------------------------------------------------------

template<aix::DataType srcDataType, aix::DataType dstDataType, size_t elementCount>
class BenchmarkDeviceFill : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dstDataType, .device=m_device.get() };
        m_result = aix::Tensor(aix::Shape{elementCount}, opt);  // No fill
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        double scalar = 5;  // Type and accuracy do not matter here.
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->fill(&scalar, srcDataType, elementCount, m_result.value().data(), m_result.dataType());
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceFillF32F3210M = BenchmarkDeviceFill<aix::DataType::kFloat32, aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceFillF32F3210M, "device_fill_f32_f32_10m");

// --------------------------------------------------------------------------------
// SUM
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSum : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->sum(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSumF3210M = BenchmarkDeviceSum<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSumF3210M, "device_sum_f32_10m");

// --------------------------------------------------------------------------------
// MAX
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceMax : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->max(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceMaxF3210M = BenchmarkDeviceMax<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceMaxF3210M, "device_max_f32_10m");

// --------------------------------------------------------------------------------
// SQRT
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSqrt : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->sqrt(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSqrtF3210M = BenchmarkDeviceSqrt<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSqrtF3210M, "device_sqrt_f32_10m");

// --------------------------------------------------------------------------------
// SIN
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSin : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->sin(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSinF3210M = BenchmarkDeviceSin<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSinF3210M, "device_sin_f32_10m");

// --------------------------------------------------------------------------------
// COS
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceCos : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->cos(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceCosF3210M = BenchmarkDeviceCos<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceCosF3210M, "device_cos_f32_10m");

// --------------------------------------------------------------------------------
// TANH
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceTanh : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->tanh(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceTanhF3210M = BenchmarkDeviceTanh<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceTanhF3210M, "device_tanh_f32_10m");

// --------------------------------------------------------------------------------
// LOG
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceLog : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->log(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceLogF3210M = BenchmarkDeviceLog<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceLogF3210M, "device_log_f32_10m");

// --------------------------------------------------------------------------------
// EXP
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceExp : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->exp(m_t.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceExpF3210M = BenchmarkDeviceExp<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceExpF3210M, "device_exp_f32_10m");

// --------------------------------------------------------------------------------
// POW
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDevicePow : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_exp = 2 + aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->pow(m_t.value().data(), m_exp.value().data(), elementCount, m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_exp, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDevicePowF3210M = BenchmarkDevicePow<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDevicePowF3210M, "device_pow_f32_10m");

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

// --------------------------------------------------------------------------------
// TRANSPOSE
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t M>
class BenchmarkDeviceTranspose : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .dtype=dataType, .device=m_device.get() };
        m_t = aix::randn({M, M}, opt);
        m_result = aix::Tensor(aix::Shape{M, M}, opt);
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->transpose(0, 1, m_t.value().data(), {M, M}, m_t.value().strides(), m_result.value().strides(),
                                m_result.value().size(), m_result.value().data(), dataType);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceTransposeF325K = BenchmarkDeviceTranspose<aix::DataType::kFloat32,5000>;

BENCHMARK(BenchmarkDeviceTransposeF325K, "device_transpose_f32_5k");

// --------------------------------------------------------------------------------
// COPY
// --------------------------------------------------------------------------------

template<aix::DataType srcDataType, aix::DataType dstDataType, size_t elementCount>
class BenchmarkDeviceCopy : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        m_src = aix::Tensor(aix::Shape{elementCount}, { .dtype=srcDataType, .device=m_device.get() });  // No fill
        m_dst = aix::Tensor(aix::Shape{elementCount}, { .dtype=dstDataType, .device=m_device.get() });  // No fill
        m_device->commitAndWait();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            m_device->copy(m_src.value().data(), srcDataType, m_dst.value().data(), dstDataType, elementCount);
            m_device->commitAndWait();
        }
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_src, m_dst;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceCopyF32F3210M = BenchmarkDeviceCopy<aix::DataType::kFloat32, aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceCopyF32F3210M, "device_copy_f32_f32_10m");
