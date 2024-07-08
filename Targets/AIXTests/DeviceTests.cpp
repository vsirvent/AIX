//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "Utils.hpp"
#include <aix.hpp>
#include <aixDevices.hpp>
// External includes
#include <doctest/doctest.h>
// System includes

using namespace aix;

#define EPSILON             1e-5
#define EPSILON_F16         1e-2
//#define DEBUG_LOG

std::uniform_real_distribution<float>  distr(-1, 1);

std::vector<size_t>  testSizes = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                                   127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049 };

std::vector<DeviceType>  testDeviceTypes = { aix::DeviceType::kGPU_METAL };


bool verifyResults(const aix::TensorValue & tv1, const aix::TensorValue & tv2, float epsilon = EPSILON)
{
    if (tv1.dataType() != tv2.dataType())
    {
        throw std::invalid_argument("Tensor data types do no match for test result comparison.");
    }

    if (static_cast<size_t>(tv1.dataType()) >= aix::DataTypeCount)
    {
        throw std::invalid_argument("CheckVectorApproxValues does not support the new data type.");
    }

    if (tv1.size() != tv2.size())
    {
        std::cout << "Matrix element sizes does not match!" << std::endl;
    }

    if (tv1.dataType() == aix::DataType::kFloat64)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<double>()[i] - tv2.data<double>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kFloat32)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<float>()[i] - tv2.data<float>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kFloat16)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<float16_t>()[i] - tv2.data<float16_t>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kBFloat16)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<bfloat16_t>()[i] - tv2.data<bfloat16_t>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt64)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int64_t>()[i] != tv2.data<int64_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt32)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int32_t>()[i] != tv2.data<int32_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt16)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int16_t>()[i] != tv2.data<int16_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt8)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int8_t>()[i] != tv2.data<int8_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kUInt8)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<uint8_t>()[i] != tv2.data<uint8_t>()[i])
            {
                return false;
            }
        }
    }

    return true;
}


aix::Shape createRandomShape(ssize_t min, ssize_t max)
{
    std::uniform_int_distribution<size_t> distr_int(min, max);

    Shape shape;
    auto n = distr_int(randGen);
    for (size_t i=0; i<n; i++)
    {
        shape.emplace_back(distr_int(randGen));
    }

    return shape;
}


bool testAllocate(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto cpuBuf    = refDevice.allocate(n, dtype);
        auto deviceBuf = testDevice->allocate(n, dtype);

        // Device should be able to allocate memory, and it should be accessible by CPU to read and write.
        for (size_t i=0; i < n * Device::dataTypeSize(dtype); ++i)
        {
            static_cast<uint8_t*>(cpuBuf)[i]    = 5;
            static_cast<uint8_t*>(deviceBuf)[i] = 5;
        }

        refDevice.deallocate(cpuBuf);
        testDevice->deallocate(deviceBuf);
    }

    return true;
}


bool testAdd(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype);
        auto array2 = aix::randn({1, n}).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.add(array1.value().data(), array2.value().data(), n, cpuResult.data(), dtype);
        testDevice->add(array1.value().data(), array2.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testSub(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && (dtype == DataType::kFloat64)) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype);
        auto array2 = aix::randn({1, n}).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.add(array1.value().data(), array2.value().data(), n, cpuResult.data(), dtype);
        testDevice->add(array1.value().data(), array2.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testUnary(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.unary(array1.value().data(), n, cpuResult.data(), dtype);
        testDevice->unary(array1.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testSqrt(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1       = (50 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.sqrt(array1.value().data(), n, cpuResult.data(), dtype);
        testDevice->sqrt(array1.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testSin(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1       = (50 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.sin(array1.value().data(), n, cpuResult.data(), dtype);
        testDevice->sin(array1.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testCos(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1       = (50 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.cos(array1.value().data(), n, cpuResult.data(), dtype);
        testDevice->cos(array1.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testTanh(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array        = (50 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.tanh().value();
        auto deviceResult = array.to(*testDevice).tanh().value();
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testLog(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array        = (11 + 10 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.log().value();
        auto deviceResult = array.to(*testDevice).log().value();
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testExp(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array        = (1 + aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.exp().value();
        auto deviceResult = array.to(*testDevice).exp().value();
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testPow(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1       = 2 * aix::randn({1, n}).to(dtype);
        auto exp          = 3 + aix::randn({1, n}).to(dtype) * 2;       // Random numbers in [1,5]
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.pow(array1.value().data(), exp.value().data(), n, cpuResult.data(), dtype);
        testDevice->pow(array1.value().data(), exp.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult,
            !(dtype == DataType::kFloat16 || dtype == DataType::kBFloat16) ? EPSILON : EPSILON_F16 * 10))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Exponents" << std::endl << exp.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testMul(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype);
        auto array2 = aix::randn({1, n}).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.mul(array1.value().data(), array2.value().data(), n, cpuResult.data(), dtype);
        testDevice->mul(array1.value().data(), array2.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testDiv(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto array1 = (21 + 20 * aix::randn({1, n})).to(dtype);
        auto array2 = (21 + 20 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.div(array1.value().data(), array2.value().data(), n, cpuResult.data(), dtype);
        testDevice->div(array1.value().data(), array2.value().data(), n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16 * 10))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testMatMul(Device* testDevice, size_t n, size_t inner, size_t m)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL &&
            (dtype == DataType::kFloat64 || dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        auto matA = (11 + 10 * aix::randn({n, inner})).to(dtype);
        auto matB = (11 + 10 * aix::randn({inner, m})).to(dtype);
        auto cpuResult    = aix::TensorValue({n, m}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({n, m}, testDevice).to(dtype);

        refDevice.matmul(matA.value().data(), {n, inner},
                         matB.value().data(), {inner, m},
                         cpuResult.data(), dtype);

        testDevice->matmul(matA.value().data(), {n, inner},
                           matB.value().data(), {inner, m},
                           deviceResult.data(), dtype);

        testDevice->commitAndWait();

        // Compare true/cpu result with gpu result
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "MatA" << std::endl << matA.value() << std::endl;
            std::cout << "MatB" << std::endl << matB.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testTranspose2D(Device* testDevice, size_t n, size_t m)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.

        size_t dim0 = 0;
        size_t dim1 = 1;
        auto tensor = aix::randn({n, m}).to(dtype);

        Shape newShape = tensor.shape();
        std::swap(newShape[dim0], newShape[dim1]);
        auto cpuResult    = aix::TensorValue(newShape, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue(newShape, testDevice).to(dtype);

        refDevice.transpose(dim0, dim1, tensor.value().data(), tensor.shape(), tensor.value().strides(), cpuResult.strides(),
                            cpuResult.size(), cpuResult.data(), dtype);

        testDevice->transpose(dim0, dim1, tensor.value().data(), tensor.shape(), tensor.value().strides(), deviceResult.strides(),
                              deviceResult.size(), deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare true/cpu result with gpu result
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Dim (" << dim0 << "," << dim1 << ")" << std::endl;
            std::cout << "Tensor" << std::endl << tensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testTranspose(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.
        ssize_t maxDim = 5;
        auto tensor = aix::randn(createRandomShape(1, maxDim)).to(dtype);

        std::uniform_int_distribution<size_t> distr_int(0, 1000);
        size_t dim0 = distr_int(randGen) % tensor.shape().size();
        size_t dim1 = distr_int(randGen) % tensor.shape().size();

        Shape newShape = tensor.shape();
        std::swap(newShape[dim0], newShape[dim1]);

        auto cpuResult    = aix::TensorValue(newShape, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue(newShape, testDevice).to(dtype);

        refDevice.transpose(dim0, dim1, tensor.value().data(), tensor.shape(), tensor.value().strides(), cpuResult.strides(),
                            cpuResult.size(), cpuResult.data(), dtype);

        testDevice->transpose(dim0, dim1, tensor.value().data(), tensor.shape(), tensor.value().strides(), deviceResult.strides(),
                              deviceResult.size(), deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare true/cpu result with gpu result
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Dim (" << dim0 << "," << dim1 << ")" << std::endl;
            std::cout << "Tensor" << std::endl << tensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testCopy(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        for (size_t j=0; j<aix::DataTypeCount; ++j)
        {
            auto srcDType = static_cast<DataType>(i);
            auto dstDType = static_cast<DataType>(j);
            auto hasFloat64 = srcDType == DataType::kFloat64 || dstDType == DataType::kFloat64;
            auto hasFloat16 = srcDType == DataType::kFloat16 || dstDType == DataType::kFloat16;

            // Apple Metal Framework does not support kFloat64 data type.
            if (testDevice->type() == DeviceType::kGPU_METAL && hasFloat64) continue;

            aix::Device  refDevice;     // Reference/CPU device.
            auto src = aix::randn({1, n}).to(srcDType);
            auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dstDType);
            auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dstDType);

            refDevice.copy(src.value().data(), srcDType, cpuResult.data(), dstDType, n);
            testDevice->copy(src.value().data(), srcDType, deviceResult.data(), dstDType, n);
            testDevice->commitAndWait();

            // Compare results with the true/reference results
            if (!verifyResults(cpuResult, deviceResult, hasFloat16 ? EPSILON_F16 : EPSILON))
            {
                #ifdef DEBUG_LOG
                std::cout << "----------------------" << std::endl;
                std::cout << "Source" << std::endl << src.value() << std::endl;
                std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
                std::cout << "Device Result" << std::endl << deviceResult << std::endl;
                #endif
                return false;
            }
        }
    }
    return true;
}


bool testFill(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::Device  refDevice;     // Reference/CPU device.
        auto scalar       = distr(randGen);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.fill(&scalar, n, cpuResult.data(), dtype);
        testDevice->fill(&scalar, n, deviceResult.data(), dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Scalar: " << scalar << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testBroadcastTo(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        auto shape    = createRandomShape(1, 5);
        auto newShape = createRandomShape(1, 5);
        // Skip this test if the two random shapes cannot be broadcasted.
        if (!TensorValue::checkBroadcastShapes(shape, newShape)) return true;

        aix::Device  refDevice;     // Reference/CPU device.
        auto srcTensor    = aix::randn(shape).to(dtype);
        auto cpuResult    = aix::TensorValue(newShape, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue(newShape, testDevice).to(dtype);

        refDevice.broadcastTo(srcTensor.value().data(), cpuResult.data(), cpuResult.size(), shape, newShape, dtype);
        testDevice->broadcastTo(srcTensor.value().data(), deviceResult.data(), deviceResult.size(), shape, newShape, dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Source" << std::endl << srcTensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testReduceTo(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        auto shape    = createRandomShape(1, 5);
        auto newShape = createRandomShape(1, 5);
        // If we can broadcast a tensor from shape to newShape, then we can reduce from newShape to shape.
        if (!TensorValue::checkBroadcastTo(shape, newShape)) return true;

        aix::Device  refDevice;     // Reference/CPU device.

        auto srcTensor    = aix::randn(newShape).to(dtype);
        // Must initialize result tensor values since reduceTo has sum operation.
        auto cpuResult    = aix::TensorValue(0, shape, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue(0, shape, testDevice).to(dtype);

        refDevice.reduceTo(srcTensor.value().data(),   cpuResult.data(),    cpuResult.size(),    shape, newShape, dtype);
        testDevice->reduceTo(srcTensor.value().data(), deviceResult.data(), deviceResult.size(), shape, newShape, dtype);
        testDevice->commitAndWait();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Source" << std::endl << srcTensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "Device Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


TEST_CASE("Device Tests - createDevice")
{
    std::vector<aix::DeviceType> deviceTypes
    {
        aix::DeviceType::kCPU,
        aix::DeviceType::kGPU_METAL
    };

    for (const auto type : deviceTypes)
    {
        auto device = aix::createDevice(type);
        CHECK(device->type() == type);
    }
}


TEST_CASE("Device Tests - Allocate")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testAllocate(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Add")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testAdd(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Sub")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSub(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Unary")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testUnary(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Sqrt")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSqrt(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Sin")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSin(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Cos")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testCos(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Tanh")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testTanh(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Log")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testLog(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Exp")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testExp(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Pow")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testPow(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Mul")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testMul(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Div")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testDiv(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - MatMul")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 1; n < 8; n+=2)
        {
            for (size_t i = 1; i < 8; i+=2)
            {
                for (size_t m = 1; m < 8; m+=2)
                {
                    auto device2 = aix::createDevice(deviceType);
                    CHECK(testMatMul(&*device2, n, i, m));
                }
            }
        }

        CHECK(testMatMul(&*device, 257, 129, 513));
        CHECK(testMatMul(&*device, 258, 130, 514));
        CHECK(testMatMul(&*device, 256, 128, 512));
        CHECK(testMatMul(&*device, 255, 127, 511));
        CHECK(testMatMul(&*device, 254, 126, 510));
    }
}


TEST_CASE("Device Tests - Transpose2D")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 1; n < 8; n+=2)
        {
            for (size_t m = 1; m < 8; m+=2)
            {
                auto device2 = aix::createDevice(deviceType);
                CHECK(testTranspose2D(&*device2, n, m));
            }
        }

        CHECK(testTranspose2D(&*device, 129, 513));
        CHECK(testTranspose2D(&*device, 130, 514));
        CHECK(testTranspose2D(&*device, 128, 512));
        CHECK(testTranspose2D(&*device, 127, 511));
        CHECK(testTranspose2D(&*device, 126, 510));
    }
}


TEST_CASE("Device Tests - Transpose")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 0; n < 100; ++n)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testTranspose(&*device2));
        }
    }
}


TEST_CASE("Device Tests - Copy")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testCopy(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - Fill")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testFill(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - broadcastTo")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<100; ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testBroadcastTo(&*device2));
        }
    }
}


TEST_CASE("Device Tests - reduceTo")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<100; ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testReduceTo(&*device2));
        }
    }
}


TEST_CASE("Device Tests - batch compute")
{
    // If a device uses an advanced command queuing method, subsequent commands should be executed properly once the
    // commitAndWait method is called.

    Shape shape{2,3};
    std::initializer_list<float> data1{1.0, 2.0, 3.0,  4.0,  5.0,  6.0};
    std::initializer_list<float> data2{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    size_t queueSize = 200;

    SUBCASE("Add")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x + y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x + y;
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({1608.0,2010.0,2412.0,2814.0,3216.0,3618.0},  shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
        }
    }

    SUBCASE("Sub")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x - y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z - x - y;
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({-1606.0,-2006.0,-2406.0,-2806.0,-3206.0,-3606.0},  shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({-199.0,-199.0,-199.0,-199.0,-199.0,-199.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({-201.0,-201.0,-201.0,-201.0,-201.0,-201.0}, shape).value());
        }
    }

    SUBCASE("Mul")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x * y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x * y;
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({1407.0,3216.0,5427.0,8040.0,11055.0,14472.0}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({1407.0,1608.0,1809.0,2010.0,2211.0,2412.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201.0,402.0,603.0,804.0,1005.0,1206.0}, shape).value());
        }
    }

    SUBCASE("Div")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x / y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x / y;
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({28.7143,50.25,66.9999,80.4002,91.3635,100.5}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({28.7143,25.125,22.3333,20.1,18.2727,16.75}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({-4.1020,-6.2812,-7.4444,-8.0400,-8.3058,-8.3750}, shape).value());
        }
    }

    SUBCASE("Sum")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.sum() + y.sum();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.sum() + y.sum();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor(15678));
            CheckVectorApproxValues(x.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
        }
    }

    SUBCASE("Mean")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.mean() + y.mean();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.mean() + y.mean();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor(2613));
            CheckVectorApproxValues(x.grad(), aix::tensor({33.5,33.5,33.5,33.5,33.5,33.5}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({33.5,33.5,33.5,33.5,33.5,33.5}, shape).value());
        }
    }

    SUBCASE("sqrt")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.sqrt() + y.sqrt();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.sqrt() + y.sqrt();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({732.7961,852.7692,951.1430,1037.6198,1116.0948,1188.6305}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({100.5000,71.0643,58.0236,50.2500,44.9449,41.0290}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({37.9855,35.5321,33.5000,31.7808,30.3018,29.0118}, shape).value());
        }
    }

    SUBCASE("sin")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.sin() + y.sin();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.sin() + y.sin();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({301.1897,381.6301,111.2009,-261.4660,-393.7420,-164.0143}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({108.6004,-83.6453,-198.9882,-131.3821,57.0160,192.9943}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({151.5342,-29.2455,-183.1375,-168.6533,0.889566,169.6149}, shape).value());
        }
    }

    SUBCASE("cos")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.cos() + y.cos();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.cos() + y.cos();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({260.1352,-112.8908,-382.1257,-300.0356,57.9055,362.6089}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({-169.1358,-182.7688,-28.3652,152.1176,192.7436,56.1625}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({-132.0546,-198.8614,-82.8357,109.3483,200.9977,107.8513}, shape).value());
        }
    }

    SUBCASE("log")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.log() + y.log();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.log() + y.log();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({391.1286,557.2905,662.4633,741.4656,805.4725,859.6092}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({201.0000,100.5000,66.9999,50.2500,40.2001,33.5000}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({28.7143,25.1250,22.3333,20.1000,18.2727,16.7500}, shape).value());
        }
    }

    SUBCASE("exp")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.exp() + y.exp();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.exp() + y.exp();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({220970.0,600657.0,1.63276e+06,4.43829e+06,1.20645e+07,3.27948e+07}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({546.375,1485.2,4037.19,10974.3,29831.1,81089.3}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({220424.0,599173.0,1.62872e+06,4.42732e+06,1.20347e+07,3.27137e+07}, shape).value());
        }
    }

    SUBCASE("pow")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);
            auto exp = aix::tensor(2.0);

            auto z = x.pow(exp) + y.pow(exp);
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.pow(exp) + y.pow(exp);
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({10050.0,13668.0,18090.0,23316.0,29346.0,36180.0}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({402.0,804.0,1206.0,1608.0,2010.0,2412.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({2814.0,3216.0,3618.0,4020.0,4422.0,4824.0}, shape).value());
        }
    }

    SUBCASE("tanh")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x.tanh() + y.tanh();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.tanh() + y.tanh();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({354.0808,394.7695,401.0063,401.8651,401.9816,401.9982}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({84.4149,14.2008,1.98307,0.269514,0.0365,0.00493598}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({0.00067091,9.58443e-05,2.39611e-05,0.0,0.0,0.0}, shape).value());
        }
    }

    SUBCASE("matmul")
    {
        Shape matShape{2, 2};
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor({1.0, 2.0, 3.0, 4.0}, matShape, true);
            auto y = aix::tensor({5.0, 6.0, 7.0, 8.0}, matShape, true);

            auto z = x.matmul(y) + y.matmul(x);
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.matmul(y) + y.matmul(x);
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({8442.0,11256.0,14874.0,19296.0}, matShape));
            CheckVectorApproxValues(x.grad(), aix::tensor({4623.0,5427.0,5025.0,5829.0}, matShape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({1407.0,2211.0,1809.0,2613.0}, matShape).value());
        }
    }

    SUBCASE("complex equation")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, true);
            auto y = aix::tensor(data2, shape, true);

            auto z = x + y - (x * y).log() + y/x.exp() + (x-y) * x * x.sin() / y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x + y*y / x.sum() - (x * y).sin()- y / y.exp() + (x-y) * x * x.sin() / y.tanh() + (y * y) / (x*x).mean();
            }
            z.backward();
            device->commitAndWait();

            CheckVectorApproxValues(z, aix::tensor({178.2879,-264.8438,1748.9033,6566.8809,9716.0850,6445.9224}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({-2764.4619,1425.6165,3467.7439,4074.8318,-2422.4048,-5619.4829}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({1.1793,384.1357,500.5189,1594.3270,1437.5804,2042.0659}, shape).value());
        }
    }
}


TEST_CASE("Tensor - broadcast")
{
    SUBCASE("([],[1],[1,1],[1,3],[2,3]) op [2x3]")
    {
        std::vector<Shape> shapes{{}, {1}, {1,1}, {1,3}, {2,3}};
        for (const auto& shape : shapes)
        {
            Shape newShape{2,3};
            size_t newSize = 6;
            auto x = Tensor(2.0, shape);
            auto y = tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape);

            auto a1 = x + y;
            CHECK(a1.value().size() == newSize);
            CHECK(a1.shape() == newShape);
            CheckVectorApproxValues(a1, tensor({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape));

            // Try reverse order
            auto a2 = y + x;
            CHECK(a2.value().size() == newSize);
            CHECK(a2.shape() == newShape);
            CheckVectorApproxValues(a2, tensor({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape));

            auto s1 = x - y;
            CHECK(s1.value().size() == newSize);
            CHECK(s1.shape() == newShape);
            CheckVectorApproxValues(s1, tensor({1.0, 0.0, -1.0, -2.0, -3.0, -4.0}, newShape));

            // Try reverse order
            auto s2 = y - x;
            CHECK(s2.value().size() == newSize);
            CHECK(s2.shape() == newShape);
            CheckVectorApproxValues(s2, tensor({-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, newShape));

            auto m1 = x * y;
            CHECK(m1.value().size() == newSize);
            CHECK(m1.shape() == newShape);
            CheckVectorApproxValues(m1, tensor({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape));

            // Try reverse order
            auto m2 = y * x;
            CHECK(m2.value().size() == newSize);
            CHECK(m2.shape() == newShape);
            CheckVectorApproxValues(m2, tensor({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape));

            auto d1 = x / y;
            CHECK(d1.value().size() == newSize);
            CHECK(d1.shape() == newShape);
            CheckVectorApproxValues(d1, tensor({2.0, 1.0, 0.666667, 0.5, 0.4, 0.333334}, newShape));

            // Try reverse order
            auto d2 = y / x;
            CHECK(d2.value().size() == newSize);
            CHECK(d2.shape() == newShape);
            CheckVectorApproxValues(d2, tensor({0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, newShape));
        }
    }

    SUBCASE("[2x3] [3x2]")
    {
        std::initializer_list<float> data{1.0, 2.0, 3.0,4.0, 5.0, 6.0};
        Shape shape1{2,3};
        Shape shape2{3,2};
        auto tensor1 = tensor(data, shape1);
        auto tensor2 = tensor(data, shape2);

        // Add
        CHECK_THROWS_AS(tensor1 + tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 + tensor1, std::invalid_argument);

        // Sub
        CHECK_THROWS_AS(tensor1 - tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 - tensor1, std::invalid_argument);

        // Mul
        CHECK_THROWS_AS(tensor1 * tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 * tensor1, std::invalid_argument);

        // Div
        CHECK_THROWS_AS(tensor1 / tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 / tensor1, std::invalid_argument);
    }
}
