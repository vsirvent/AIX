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

#define EPSILON             1e-6
#define EPSILON_PERCENT     1
//#define DEBUG_LOG

std::uniform_real_distribution<aix::DataType>  distr(-1, 1);

std::vector<size_t>  testSizes = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                                   127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
                                   4095, 4096, 4097, 8191, 8192, 8193, 100'000 };

std::vector<DeviceType>  testDeviceTypes = { aix::DeviceType::kCPU , aix::DeviceType::kGPU_METAL };


bool verifyResults(const aix::TensorValue & tv1, const aix::TensorValue & tv2, float epsilon = EPSILON)
{
    if (tv1.size() != tv2.size())
    {
        std::cout << "Matrix element sizes does not match!" << std::endl;
    }
    for (size_t i=0; i < tv1.size(); ++i)
    {
        if (std::abs(tv1.data()[i] - tv2.data()[i]) > epsilon)
        {
            return false;
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
};


bool testAdd(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto array2 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.add(array1.value().data(), array2.value().data(), n, cpuResult.data());
    testDevice->add(array1.value().data(), array2.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
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

    return true;
}


bool testSub(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto array2 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.add(array1.value().data(), array2.value().data(), n, cpuResult.data());
    testDevice->add(array1.value().data(), array2.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
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

    return true;
}


bool testAdd_A_S(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto scalar = distr(randGen);
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.add(array1.value().data(), scalar, n, cpuResult.data());
    testDevice->add(array1.value().data(), scalar, n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Scalar " << scalar << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testAdd_S_A(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto scalar = distr(randGen);
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.sub(scalar, array1.value().data(), n, cpuResult.data());
    testDevice->sub(scalar, array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Scalar " << scalar << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testMul_A_S(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto scalar = distr(randGen);
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.mul(array1.value().data(), scalar, n, cpuResult.data());
    testDevice->mul(array1.value().data(), scalar, n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Scalar " << scalar << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testDiv_A_S(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto scalar = distr(randGen);
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.div(array1.value().data(), scalar, n, cpuResult.data());
    testDevice->div(array1.value().data(), scalar, n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Scalar " << scalar << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testDiv_S_A(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto scalar = distr(randGen);
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.div(scalar, array1.value().data(), n, cpuResult.data());
    testDevice->div(scalar, array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Scalar " << scalar << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testUnary(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.unary(array1.value().data(), n, cpuResult.data());
    testDevice->unary(array1.value().data(), n, deviceResult.data());
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

    return true;
}


bool testSqrt(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.sqrt(array1.value().data(), n, cpuResult.data());
    testDevice->sqrt(array1.value().data(), n, deviceResult.data());
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

    return true;
}


bool testSin(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.sin(array1.value().data(), n, cpuResult.data());
    testDevice->sin(array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult, EPSILON))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testCos(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.cos(array1.value().data(), n, cpuResult.data());
    testDevice->cos(array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult, EPSILON))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testTanh(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.tanh(array1.value().data(), n, cpuResult.data());
    testDevice->tanh(array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult, EPSILON))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testLog(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.log(array1.value().data(), n, cpuResult.data());
    testDevice->log(array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult, EPSILON))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testExp(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.exp(array1.value().data(), n, cpuResult.data());
    testDevice->exp(array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult, EPSILON))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Array1" << std::endl << array1.value() << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testMean(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    TensorValue cpuResult(0, {}, &refDevice);
    TensorValue deviceResult(0, {}, testDevice);

    refDevice.mean(array1.value().data(), n, cpuResult.data());
    testDevice->mean(array1.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    auto errorPercent = std::abs((cpuResult.item() - deviceResult.item()) / cpuResult.item());
    if ( errorPercent > EPSILON_PERCENT)
    {
        #ifdef DEBUG_LOG
        std::cout << "---------------------- " << std::endl;
        std::cout << "Error Percent   : " << errorPercent << std::endl;
        std::cout << "Array Size      : " << n << std::endl;
        std::cout << "Expected Result : " << cpuResult << std::endl;
        std::cout << "Device Result   : " << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testMul(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto array2 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.mul(array1.value().data(), array2.value().data(), n, cpuResult.data());
    testDevice->mul(array1.value().data(), array2.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
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

    return true;
}


bool testDiv(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto array1 = aix::randn({1, n});
    auto array2 = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.div(array1.value().data(), array2.value().data(), n, cpuResult.data());
    testDevice->div(array1.value().data(), array2.value().data(), n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
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

    return true;
}


bool testMatMul(Device* testDevice, size_t n, size_t inner, size_t m)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto matA = aix::randn({n, inner});
    auto matB = aix::randn({inner, m});
    auto cpuResult    = aix::TensorValue({n, m}, &refDevice);
    auto deviceResult = aix::TensorValue({n, m}, testDevice);

    refDevice.matmul(matA.value().data(), {n, inner},
                     matB.value().data(), {inner, m},
                     cpuResult.data());

    testDevice->matmul(matA.value().data(), {n, inner},
                       matB.value().data(), {inner, m},
                       deviceResult.data());

    testDevice->commitAndWait();

    // Compare true/cpu result with gpu result
    if (!verifyResults(cpuResult, deviceResult, EPSILON))
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

    return true;
}


bool testTranspose(Device* testDevice, size_t n, size_t m)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto mat = aix::randn({n, m});
    auto cpuResult    = aix::TensorValue({n, m}, &refDevice);
    auto deviceResult = aix::TensorValue({n, m}, testDevice);

    refDevice.transpose(mat.value().data(), {n, m}, cpuResult.data());
    testDevice->transpose(mat.value().data(), {n, m}, deviceResult.data());
    testDevice->commitAndWait();

    // Compare true/cpu result with gpu result
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Mat" << std::endl << mat.value() << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testCopy(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto src = aix::randn({1, n});
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.copy(src.value().data(), cpuResult.data(), n);
    testDevice->copy(src.value().data(), deviceResult.data(), n);
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Source" << std::endl << src.value() << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testFill(Device* testDevice, size_t n)
{
    aix::Device  refDevice;     // Reference/CPU device.

    auto scalar       = distr(randGen);
    auto cpuResult    = aix::TensorValue({1, n}, &refDevice);
    auto deviceResult = aix::TensorValue({1, n}, testDevice);

    refDevice.fill(scalar, n, cpuResult.data());
    testDevice->fill(scalar, n, deviceResult.data());
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    if (!verifyResults(cpuResult, deviceResult))
    {
        #ifdef DEBUG_LOG
        std::cout << "----------------------" << std::endl;
        std::cout << "Scalar: " << scalar << std::endl;
        std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
        std::cout << "Device Result" << std::endl << deviceResult << std::endl;
        #endif
        return false;
    }

    return true;
}


bool testBroadcastTo(Device* testDevice)
{
    auto shape    = createRandomShape(1, 5);
    auto newShape = createRandomShape(1, 5);
    // Skip this test if the two random shapes cannot be broadcasted.
    if (!TensorValue::checkBroadcastShapes(shape, newShape)) return true;

    aix::Device  refDevice;     // Reference/CPU device.

    auto srcTensor    = aix::randn(shape);
    auto cpuResult    = aix::TensorValue(newShape, &refDevice);
    auto deviceResult = aix::TensorValue(newShape, testDevice);

    refDevice.broadcastTo(srcTensor.value().data(), cpuResult.data(), cpuResult.size(), shape, newShape);
    testDevice->broadcastTo(srcTensor.value().data(), deviceResult.data(), deviceResult.size(), shape, newShape);
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

    return true;
}


bool testReduceTo(Device* testDevice)
{
    auto shape    = createRandomShape(1, 5);
    auto newShape = createRandomShape(1, 5);
    // If we can broadcast a tensor from shape to newShape, then we can reduce from newShape to shape.
    if (!TensorValue::checkBroadcastTo(shape, newShape)) return true;

    aix::Device  refDevice;     // Reference/CPU device.

    auto srcTensor    = aix::randn(newShape);
    // Must initialize result tensor values since reduceTo has sum operation.
    auto cpuResult    = aix::TensorValue(0, shape, &refDevice);
    auto deviceResult = aix::TensorValue(0, shape, testDevice);

    refDevice.reduceTo(srcTensor.value().data(),   cpuResult.data(),    cpuResult.size(),    shape, newShape);
    testDevice->reduceTo(srcTensor.value().data(), deviceResult.data(), deviceResult.size(), shape, newShape);
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

    return true;
}


TEST_CASE("Device Tests - Add")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testAdd(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testAdd(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Sub")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testSub(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testSub(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Add_Array_Scalar")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testAdd_A_S(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testAdd_A_S(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Add_Scalar_Array")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testAdd_S_A(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testAdd_S_A(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Mul_Array_Scalar")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testMul_A_S(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testMul_A_S(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Div_Array_Scalar")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testDiv_A_S(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testDiv_A_S(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Div_Scalar_Array")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testDiv_S_A(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testDiv_S_A(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Unary")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testUnary(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testUnary(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Sqrt")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testSqrt(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testSqrt(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Sin")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testSin(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testSin(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Cos")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testCos(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testCos(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Tanh")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testTanh(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testTanh(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Log")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testLog(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testLog(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Exp")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testExp(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testExp(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Mean")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testMean(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testMean(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Mul")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testMul(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testMul(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Div")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testDiv(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testDiv(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - MatMul")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 1; n < 8; n+=2)
        {
            for (size_t i = 1; i < 8; i+=2)
            {
                for (size_t m = 1; m < 8; m+=2)
                {
                    CHECK(testMatMul(device, n, i, m));
                }
            }
        }

        CHECK(testMatMul(device, 257, 129, 513));
        CHECK(testMatMul(device, 258, 130, 514));
        CHECK(testMatMul(device, 256, 128, 512));
        CHECK(testMatMul(device, 255, 127, 511));
        CHECK(testMatMul(device, 254, 126, 510));

        delete device;
    }
}


TEST_CASE("Device Tests - Transpose")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 1; n < 8; n+=2)
        {
            for (size_t m = 1; m < 8; m+=2)
            {
                CHECK(testTranspose(device, n, m));
            }
        }

        CHECK(testTranspose(device, 129, 513));
        CHECK(testTranspose(device, 130, 514));
        CHECK(testTranspose(device, 128, 512));
        CHECK(testTranspose(device, 127, 511));
        CHECK(testTranspose(device, 126, 510));

        delete device;
    }
}


TEST_CASE("Device Tests - Copy")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testCopy(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testCopy(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - Fill")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (auto size: testSizes)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testFill(device, size));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (auto size: testSizes)
        {
            CHECK(testFill(device, size));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - broadcastTo")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (size_t i=0; i<100; ++i)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testBroadcastTo(device));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (size_t i=0; i<100; ++i)
        {
            CHECK(testBroadcastTo(device));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - reduceTo")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aixDeviceFactory::CreateDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.
        delete device;

        // Create a new device per test
        for (size_t i=0; i<100; ++i)
        {
            device = aixDeviceFactory::CreateDevice(deviceType);
            CHECK(testReduceTo(device));
            delete device;
        }

        // Use the same device per test
        device = aixDeviceFactory::CreateDevice(deviceType);
        for (size_t i=0; i<100; ++i)
        {
            CHECK(testReduceTo(device));
        }
        delete device;
    }
}


TEST_CASE("Device Tests - batch compute")
{
    // If a device uses an advanced command queuing method, subsequent commands should be executed properly once the
    // commitAndWait method is called.

    Shape shape{2,3};
    std::vector<DataType> data1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<DataType> data2{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    size_t queueSize = 200;

    SUBCASE("Add")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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

            CheckVectorApproxValues(z, aix::tensor({1608,2010,2412,2814,3216,3618},  shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({201,201,201,201,201,201}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201,201,201,201,201,201}, shape).value());
        }
    }

    SUBCASE("Sub")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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

            CheckVectorApproxValues(z, aix::tensor({-1606,-2006,-2406,-2806,-3206,-3606},  shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({-199,-199,-199,-199,-199,-199}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({-201,-201,-201,-201,-201,-201}, shape).value());
        }
    }

    SUBCASE("Mul")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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

            CheckVectorApproxValues(z, aix::tensor({1407,3216,5427,8040,11055,14472}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({1407,1608,1809,2010,2211,2412}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201,402,603,804,1005,1206}, shape).value());
        }
    }

    SUBCASE("Div")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            CheckVectorApproxValues(x.grad(), aix::tensor({201,201,201,201,201,201}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201,201,201,201,201,201}, shape).value());
        }
    }

    SUBCASE("Mean")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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

            CheckVectorApproxValues(z, aix::tensor({220970,600657,1.63276e+06,4.43829e+06,1.20645e+07,3.27948e+07}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({546.375,1485.2,4037.19,10974.3,29831.1,81089.3}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({220424,599173,1.62872e+06,4.42732e+06,1.20347e+07,3.27137e+07}, shape).value());
        }
    }

    SUBCASE("tanh")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
            CheckVectorApproxValues(y.grad(), aix::tensor({0.00067091,9.58443e-05,2.39611e-05,0,0,0}, shape).value());
        }
    }

    SUBCASE("matmul")
    {
        Shape matShape{2, 2};
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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

            CheckVectorApproxValues(z, aix::tensor({8442,11256,14874,19296}, matShape));
            CheckVectorApproxValues(x.grad(), aix::tensor({4623,5427,5025,5829}, matShape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({1407,2211,1809,2613}, matShape).value());
        }
    }

    SUBCASE("complex equation")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = std::unique_ptr<aix::Device>(aixDeviceFactory::CreateDevice(deviceType));
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
        std::vector<DataType> data{1.0, 2.0, 3.0,4.0, 5.0, 6.0};
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
