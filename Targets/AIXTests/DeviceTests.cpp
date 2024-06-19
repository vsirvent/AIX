//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
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

std::vector<DeviceType>  testDeviceTypes = { aix::DeviceType::kGPU_METAL };


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
    float cpuResult    = 0;
    float deviceResult = 0;

    refDevice.mean(array1.value().data(), n, cpuResult);
    testDevice->mean(array1.value().data(), n, deviceResult);
    testDevice->commitAndWait();

    // Compare results with the true/reference results
    auto errorPercent = std::abs((cpuResult - deviceResult) / cpuResult);
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
