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
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 + m_t2;
            m_device->commitAndWait();
        }
        m_device->synchronize();
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
BENCHMARK(BenchmarkTensorAddF3210M, "tensor_add_f32_10m")

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
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 - m_t2;
            m_device->commitAndWait();
        }
        m_device->synchronize();
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
BENCHMARK(BenchmarkTensorSubF3210M, "tensor_sub_f32_10m")

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
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 * m_t2;
            m_device->commitAndWait();
        }
        m_device->synchronize();
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
BENCHMARK(BenchmarkTensorMulF3210M, "tensor_mul_f32_10m")

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
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t1 / m_t2;
            m_device->commitAndWait();
        }
        m_device->synchronize();
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
BENCHMARK(BenchmarkTensorDivF3210M, "tensor_div_f32_10m")

// --------------------------------------------------------------------------------
// UNARY
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorUnary : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = -m_t;
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorUnaryF3210M = BenchmarkTensorUnary<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorUnaryF3210M, "tensor_unary_f32_10m")

// --------------------------------------------------------------------------------
// SUM
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorSum : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.sum();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorSumF3210M = BenchmarkTensorSum<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorSumF3210M, "tensor_sum_f32_10m")

// --------------------------------------------------------------------------------
// SUM WITH DIMENSION
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorSumWithDim : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({elementCount,elementCount,elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.sum(1, false);
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorSumWithDimF32300 = BenchmarkTensorSumWithDim<aix::DataType::kFloat32, 300>;
BENCHMARK(BenchmarkTensorSumWithDimF32300, "tensor_sum_wd_f32_300")

// --------------------------------------------------------------------------------
// MAX
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorMax : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.max();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorMaxF3210M = BenchmarkTensorMax<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorMaxF3210M, "tensor_max_f32_10m")

// --------------------------------------------------------------------------------
// MAX WITH DIMENSION
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorMaxWithDim : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({elementCount, elementCount, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.max(1, false);
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorMaxWithDimF32300 = BenchmarkTensorMaxWithDim<aix::DataType::kFloat32, 300>;
BENCHMARK(BenchmarkTensorMaxWithDimF32300, "tensor_max_wd_f32_300")

// --------------------------------------------------------------------------------
// SQRT
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorSqrt : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.sqrt();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorSqrtF3210M = BenchmarkTensorSqrt<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorSqrtF3210M, "tensor_sqrt_f32_10m")

// --------------------------------------------------------------------------------
// SIN
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorSin : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.sin();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorSinF3210M = BenchmarkTensorSin<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorSinF3210M, "tensor_sin_f32_10m")

// --------------------------------------------------------------------------------
// COS
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorCos : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.cos();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorCosF3210M = BenchmarkTensorCos<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorCosF3210M, "tensor_cos_f32_10m")

// --------------------------------------------------------------------------------
// TANH
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorTanh : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.tanh();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorTanhF3210M = BenchmarkTensorTanh<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorTanhF3210M, "tensor_tanh_f32_10m")

// --------------------------------------------------------------------------------
// LOG
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorLog : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.log();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorLogF3210M = BenchmarkTensorLog<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorLogF3210M, "tensor_log_f32_10m")

// --------------------------------------------------------------------------------
// EXP
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorExp : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.exp();
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorExpF3210M = BenchmarkTensorExp<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorExpF3210M, "tensor_exp_f32_10m")

// --------------------------------------------------------------------------------
// POW
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorPow : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_exp = 2 + aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.pow(m_exp);
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_exp;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorPowF3210M = BenchmarkTensorPow<aix::DataType::kFloat32, 10000000>;
BENCHMARK(BenchmarkTensorPowF3210M, "tensor_pow_f32_10m")

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
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({M, M}, opt);
        m_t2 = aix::randn({M, M}, opt);
        m_result = aix::Tensor(aix::Shape{M, M}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = matmul(m_t1, m_t2);
            m_device->commitAndWait();
        }
        m_device->synchronize();
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

BENCHMARK(BenchmarkTensorMatMulF32800, "tensor_matmul_f32_800")

// --------------------------------------------------------------------------------
// TRANSPOSE
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t M>
class BenchmarkTensorTranspose : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({M, M}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.transpose(0, 1);
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorTransposeF325K = BenchmarkTensorTranspose<aix::DataType::kFloat32,5000>;

BENCHMARK(BenchmarkTensorTransposeF325K, "tensor_transpose_f32_5k")

// --------------------------------------------------------------------------------
// SLICE
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorSlice : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({elementCount, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = m_t.slice(0, 0, elementCount, 1);
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorSliceF323K = BenchmarkTensorSlice<aix::DataType::kFloat32, 3000>;
BENCHMARK(BenchmarkTensorSliceF323K, "tensor_slice_f32_3k")

// --------------------------------------------------------------------------------
// CAT
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount, ssize_t dim>
class BenchmarkTensorCat : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        for (size_t i=0; i<10; ++i)
        {
            m_tensors.emplace_back(aix::randn({elementCount, elementCount}, opt));
        }
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto t = aix::cat(m_tensors, dim);
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    std::vector<aix::Tensor>  m_tensors;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorCatF32V = BenchmarkTensorCat<aix::DataType::kFloat32, 500, 0>;
BENCHMARK(BenchmarkTensorCatF32V, "tensor_cat_v_f32")

using BenchmarkTensorCatF32H = BenchmarkTensorCat<aix::DataType::kFloat32, 500, 1>;
BENCHMARK(BenchmarkTensorCatF32H, "tensor_cat_h_f32")

// --------------------------------------------------------------------------------
// INDEX SELECT
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount, ssize_t dim>
class BenchmarkTensorIndexSelect : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        assert(dim < 3);
        auto shape = aix::Shape{elementCount, elementCount, elementCount};
        m_indices = aix::arange(0, shape[dim], { .m_dtype=aix::DataType::kInt32, .m_device=m_device.get()});
        m_tensor  = aix::randn(shape, { .m_dtype=dataType, .m_device=m_device.get() }).to(dataType);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs& configs) final
    {
        for (size_t i=0; i<configs.iterationCount; ++i)
        {
            auto selected = m_tensor.indexSelect(dim, m_indices);
            m_device->commitAndWait();
        }
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_tensor;
    aix::Tensor  m_indices;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkTensorIndexSelectF322001 = BenchmarkTensorIndexSelect<aix::DataType::kFloat32, 200, 1>;
BENCHMARK(BenchmarkTensorIndexSelectF322001, "tensor_isel_f32_200_1")
