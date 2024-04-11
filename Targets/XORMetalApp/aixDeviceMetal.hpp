//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Project includes
#include <aix.hpp>
#include "aixDeviceMetalShaders.hpp"
// External includes
#include <Metal/Metal.hpp>
// System includes


namespace aix
{

class DeviceMetal : public aix::Device
{
public:
    // Constructor
    DeviceMetal()
    {
        m_pool = NS::AutoreleasePool::alloc()->init();      // Create autorelease pool.
        m_mtlDevice = reinterpret_cast<MTL::Device*>(MTL::CopyAllDevices()->object(0));   // Get first available device.
        auto defaultLibrary = createLibrary(aix::shaders::aixDeviceMetalShaders);
        // Compile time evaluation.
        if constexpr (std::is_same_v<DataType, float>)
        {
            m_compFuncPSOAdd          = createComputeFuncPSO(defaultLibrary, "add_float");
            m_compFuncPSOSub          = createComputeFuncPSO(defaultLibrary, "sub_float");
            m_compFuncPSOMul          = createComputeFuncPSO(defaultLibrary, "mul_float");
            m_compFuncPSODiv          = createComputeFuncPSO(defaultLibrary, "div_float");
            m_compFuncPSOAdd_A_S      = createComputeFuncPSO(defaultLibrary, "add_a_s_float");
            m_compFuncPSOSub_S_A      = createComputeFuncPSO(defaultLibrary, "sub_s_a_float");
            m_compFuncPSOMul_A_S      = createComputeFuncPSO(defaultLibrary, "mul_a_s_float");
            m_compFuncPSODiv_A_S      = createComputeFuncPSO(defaultLibrary, "div_a_s_float");
            m_compFuncPSODiv_S_A      = createComputeFuncPSO(defaultLibrary, "div_s_a_float");
            m_compFuncPSOSqrt         = createComputeFuncPSO(defaultLibrary, "sqrt_a_float");
            m_compFuncPSOSin          = createComputeFuncPSO(defaultLibrary, "sin_a_float");
            m_compFuncPSOCos          = createComputeFuncPSO(defaultLibrary, "cos_a_float");
            m_compFuncPSOTanh         = createComputeFuncPSO(defaultLibrary, "tanh_a_float");
            m_compFuncPSOLog          = createComputeFuncPSO(defaultLibrary, "log_a_float");
            m_compFuncPSOExp          = createComputeFuncPSO(defaultLibrary, "exp_a_float");
            m_compFuncPSOMatMul       = createComputeFuncPSO(defaultLibrary, "matrix_mul_float");
            m_compFuncPSOMatTranspose = createComputeFuncPSO(defaultLibrary, "matrix_transpose_float");
            m_compFuncPSOCopy_A_A     = createComputeFuncPSO(defaultLibrary, "copy_a_a_float");
            m_compFuncPSOCopy_S_A     = createComputeFuncPSO(defaultLibrary, "copy_s_a_float");
        }
        else
            throw std::invalid_argument("Metal device supports only float data type for now.");

        m_cmdQueue = createCommandQueue();
    }

    // Destructor
    virtual ~DeviceMetal()
    {
        m_cmdQueue->release();
        m_compFuncPSOAdd->release();
        m_compFuncPSOSub->release();
        m_compFuncPSOMul->release();
        m_compFuncPSODiv->release();
        m_compFuncPSOAdd_A_S->release();
        m_compFuncPSOSub_S_A->release();
        m_compFuncPSOMul_A_S->release();
        m_compFuncPSODiv_A_S->release();
        m_compFuncPSODiv_S_A->release();
        m_compFuncPSOSqrt->release();
        m_compFuncPSOSin->release();
        m_compFuncPSOCos->release();
        m_compFuncPSOTanh->release();
        m_compFuncPSOLog->release();
        m_compFuncPSOExp->release();
        m_compFuncPSOMatMul->release();
        m_compFuncPSOMatTranspose->release();
        m_compFuncPSOCopy_A_A->release();
        m_compFuncPSOCopy_S_A->release();
        m_mtlDevice->release();
        // No need to release MTL Buffer objects in m_allocMap.
        m_pool->release();
    }

    DeviceType type() const override { return DeviceType::kGPU_METAL; }

    // Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
    void * allocate(size_t size) override
    {
        // Allocate GPU memory and save the mtl buffer to be used later.
        auto mtlBuf = m_mtlDevice->newBuffer(size, MTL::ResourceStorageModeShared);
        auto contentPtr = mtlBuf->contents();
        m_allocMap[contentPtr] = mtlBuf;
        return contentPtr;
    }

    // Deallocate GPU memory if it's allocated by current device.
    void deallocate(void * memory) override
    {
        if (m_allocMap.find(memory) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::deallocate() - Found different type of memory to free.");
        auto mtlBuf = m_allocMap[memory];
        m_allocMap.erase(memory);
        mtlBuf->release();
    }

    void add(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::add(a1, a2, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::add() result must have GPU memory.");

        m_buf1 = getReadOnlyMTLBuffer(a1, size);      // Memory could be a GPU allocated memory or system memory.
        m_buf2 = getReadOnlyMTLBuffer(a2, size);      // Memory could be a GPU allocated memory or system memory.
        m_bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOAdd->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = Final matrix size
        sendComputeCommandDoubleBuffer(m_compFuncPSOAdd, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a1);
        freeTemporaryBuffer(m_buf2, a2);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_buf2 = nullptr;
        m_bufResult = nullptr;
    }

    void sub(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::sub(a1, a2, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::sub() result must have GPU memory.");

        m_buf1 = getReadOnlyMTLBuffer(a1, size);      // Memory could be a GPU allocated memory or system memory.
        m_buf2 = getReadOnlyMTLBuffer(a2, size);      // Memory could be a GPU allocated memory or system memory.
        m_bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOSub->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = Final matrix size
        sendComputeCommandDoubleBuffer(m_compFuncPSOSub, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a1);
        freeTemporaryBuffer(m_buf2, a2);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_buf2 = nullptr;
        m_bufResult = nullptr;
    }

    void mul(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::mul(a1, a2, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::mul() result must have GPU memory.");

        m_buf1 = getReadOnlyMTLBuffer(a1, size);      // Memory could be a GPU allocated memory or system memory.
        m_buf2 = getReadOnlyMTLBuffer(a2, size);      // Memory could be a GPU allocated memory or system memory.
        m_bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOMul->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = Final matrix size
        sendComputeCommandDoubleBuffer(m_compFuncPSOMul, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a1);
        freeTemporaryBuffer(m_buf2, a2);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_buf2 = nullptr;
        m_bufResult = nullptr;
    }

    void div(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::div(a1, a2, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::div() result must have GPU memory.");

        m_buf1 = getReadOnlyMTLBuffer(a1, size);      // Memory could be a GPU allocated memory or system memory.
        m_buf2 = getReadOnlyMTLBuffer(a2, size);      // Memory could be a GPU allocated memory or system memory.
        m_bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSODiv->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = Final matrix size
        sendComputeCommandDoubleBuffer(m_compFuncPSODiv, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a1);
        freeTemporaryBuffer(m_buf2, a2);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_buf2 = nullptr;
        m_bufResult = nullptr;
    }

    void add(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::add(a, scalar, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::add() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = scalar;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOAdd_A_S->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOAdd_A_S, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    // TODO: This should be handled in TensorValue.
    void sub(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        add(a, -scalar, size, result);
    }

    void sub(DataType scalar, const DataType * a, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::sub(scalar, a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::sub() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = scalar;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOSub_S_A->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOSub_S_A, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void mul(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::mul(a, scalar, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::mul() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = scalar;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOMul_A_S->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOMul_A_S, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void div(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::div(a, scalar, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::div() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = scalar;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSODiv_A_S->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSODiv_A_S, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void div(DataType scalar, const DataType * a, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::mul(scalar, a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::div() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = scalar;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSODiv_S_A->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSODiv_S_A, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void unary(const DataType * a, const size_t size, DataType * result) override
    {
        mul(a, DataType(-1), size, result);
    }

    void fill(DataType scalar, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::fill(scalar, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::div() result must have GPU memory.");

        m_buf1   = nullptr;
        m_scalar = scalar;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOCopy_S_A->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOCopy_S_A, gridSize, threadsPerThreadGroup);

        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

/*
    // TODO: Add GPU support for the following device methods.
    // Unimplemented GPU implementations will use CPU by default and be called from base Device.
    void sum(const DataType * a, const size_t size, DataType & result) override {}
    void mean(const DataType * a, const size_t size, DataType & result) override {}
*/

    void sqrt(const DataType * a, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::sqrt(a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::sqrt() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = 0;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOSqrt->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOSqrt, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void sin(const DataType * a, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::sin(a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::sin() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = 0;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOSin->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOSin, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void cos(const DataType * a, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::cos(a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::cos() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = 0;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOCos->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOCos, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void tanh(const DataType * a, const size_t size, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::tanh(a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::tanh() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = 0;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOTanh->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOTanh, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void log(const DataType* a, const size_t size, DataType* result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::log(a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::log() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = 0;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOLog->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOLog, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void exp(const DataType* a, const size_t size, DataType* result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::exp(a, size, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::exp() result must have GPU memory.");

        m_buf1   = getReadOnlyMTLBuffer(a, size);
        m_scalar = 0;
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOExp->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandArrayScalar(m_compFuncPSOExp, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void matmul(const DataType * a1, const Shape & s1, const DataType * a2, const Shape & s2, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::matmul(a1,s1,a2,s2,result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

        m_buf1 = getReadOnlyMTLBuffer(a1, s1[0] * s1[1]);  // Memory could be a GPU allocated memory or system memory.
        m_buf2 = getReadOnlyMTLBuffer(a2, s2[0] * s2[1]);  // Memory could be a GPU allocated memory or system memory.
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = s1[0];
        m_buf1Size.cols = s1[1];
        m_buf2Size.rows = s2[0];
        m_buf2Size.cols = s2[1];

        // Calculate maximum thread group dimensions
        NS::UInteger w = m_compFuncPSOMatMul->threadExecutionWidth();
        NS::UInteger h = m_compFuncPSOMatMul->maxTotalThreadsPerThreadgroup() / w;
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, h, 1);
        MTL::Size gridSize = MTL::Size(m_buf2Size.cols, m_buf1Size.rows, 1);    // gridSize = Final matrix size
        sendComputeCommandDoubleBuffer(m_compFuncPSOMatMul, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, a1);
        freeTemporaryBuffer(m_buf2, a2);
        // Never release buffers since they will be in use by Arrays.
        m_buf1 = nullptr;
        m_buf2 = nullptr;
        m_bufResult = nullptr;
    }

    void transpose(const DataType * mat, const Shape & shape, DataType * result) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::transpose(a, shape, result); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

        // Memory could be a GPU allocated memory or system memory.
        m_buf1 = getReadOnlyMTLBuffer(mat, shape[0] * shape[1]);
        m_bufResult = m_allocMap[result];

        m_buf1Size.rows = shape[0];
        m_buf1Size.cols = shape[1];

        // Calculate maximum thread group dimensions
        NS::UInteger w = m_compFuncPSOMatTranspose->threadExecutionWidth();
        NS::UInteger h = m_compFuncPSOMatTranspose->maxTotalThreadsPerThreadgroup() / w;
        MTL::Size threadsPerThreadGroup = MTL::Size(w, h, 1);
        MTL::Size gridSize = MTL::Size(m_buf1Size.rows, m_buf1Size.cols, 1);
        sendComputeCommandSingleBuffer(m_compFuncPSOMatTranspose, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, mat);
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

    void copy(const DataType * src, DataType * dst, size_t size) override
    {
        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        //Device::copy(src, dst, size); return;

        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(dst) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::copy() result must have GPU memory.");

        // Memory could be a GPU allocated memory or system memory.
        m_buf1 = getReadOnlyMTLBuffer(src, size);
        m_bufResult = m_allocMap[dst];

        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOCopy_A_A->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        MTL::Size threadsPerThreadGroup = MTL::Size(w, 1, 1);
        MTL::Size gridSize = MTL::Size(size, 1, 1);    // gridSize = array size
        sendComputeCommandSingleBuffer(m_compFuncPSOCopy_A_A, gridSize, threadsPerThreadGroup);

        freeTemporaryBuffer(m_buf1, src);
        m_buf1 = nullptr;
        m_bufResult = nullptr;
    }

protected:
    inline MTL::Buffer* getReadOnlyMTLBuffer(const DataType * address, size_t size)
    {
        // Memory could be from other devices. Create a temporary buffer for read only case.
        if (m_allocMap.find(address) == m_allocMap.end())
        {
            return m_mtlDevice->newBuffer(address, size * sizeof(DataType), MTL::ResourceStorageModeShared);
        }

        return m_allocMap[address];    // Return MTL Buffer if the memory is from the current device.
    }

    inline void freeTemporaryBuffer(MTL::Buffer * buffer, const DataType * address)
    {
        // Release only temporary buffer.
        if (m_allocMap.find(address) == m_allocMap.end())
        {
            buffer->release();
        }
    }

    MTL::Library* createLibrary(const char* shaders)
    {
        // Create a compile options.
        MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc()->init();
        compileOptions->setFastMathEnabled(false);

        NS::Error* error = nullptr;
        MTL::Library* defaultLibrary = m_mtlDevice->newLibrary(NS::String::string(shaders, NS::UTF8StringEncoding),
                                                               compileOptions, &error);
        compileOptions->release();

        if (!defaultLibrary)
        {
            std::cerr << "Failed to load default library. Details: " << error->localizedDescription()->utf8String() << "\n";
            exit(-1);
        }

        return defaultLibrary;
    }

    MTL::CommandQueue* createCommandQueue()
    {
        auto cmdQueue = m_mtlDevice->newCommandQueue();
        if (!cmdQueue)
        {
            std::cerr << "Failed to create command queue.\n";
            exit(-1);
        }

        return cmdQueue;
    }

    MTL::ComputePipelineState* createComputeFuncPSO(MTL::Library* library, const std::string & kernelName)
    {
        auto funcName = NS::String::string(kernelName.c_str(), NS::ASCIIStringEncoding);
        auto compFunc = library->newFunction(funcName);
        if (!compFunc)
        {
            std::cerr << "Failed to find the compute function.\n";
            // No need to halt the application here.
        }

        NS::Error* error = nullptr;
        auto compFuncPSO = m_mtlDevice->newComputePipelineState(compFunc, &error);
        if (!compFuncPSO)
        {
            std::cerr << "Failed to create the pipeline state object.\n";
            exit(-1);
        }

        return compFuncPSO;
    }

    void encodeComputeCommandSingleBuffer(MTL::ComputeCommandEncoder * computeEncoder,
                                          MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize,
                                          MTL::Size & threadsPerTG) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(m_buf1, 0, 0);
        computeEncoder->setBuffer(m_bufResult, 0, 1);
        computeEncoder->setBytes(&m_buf1Size, sizeof(MatrixSize), 2);
        computeEncoder->dispatchThreads(gridSize, threadsPerTG);
    }

    void encodeComputeCommandDoubleBuffer(MTL::ComputeCommandEncoder * computeEncoder,
                                          MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize,
                                          MTL::Size & threadsPerTG) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(m_buf1, 0, 0);
        computeEncoder->setBuffer(m_buf2, 0, 1);
        computeEncoder->setBuffer(m_bufResult, 0, 2);
        computeEncoder->setBytes(&m_buf1Size, sizeof(MatrixSize), 3);
        computeEncoder->setBytes(&m_buf2Size, sizeof(MatrixSize), 4);
        computeEncoder->dispatchThreads(gridSize, threadsPerTG);
    }

    void encodeComputeCommandArrayScalar(MTL::ComputeCommandEncoder*  computeEncoder,
                                         MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize,
                                         MTL::Size & threadsPerTG) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(m_buf1, 0, 0);
        computeEncoder->setBytes(&m_scalar, sizeof(DeviceType), 1);
        computeEncoder->setBytes(&m_buf1Size, sizeof(MatrixSize), 2);
        computeEncoder->setBuffer(m_bufResult, 0, 3);
        computeEncoder->dispatchThreads(gridSize, threadsPerTG);
    }

    void sendComputeCommandSingleBuffer(MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize,
                                        MTL::Size & threadsPerTG)
    {
        MTL::CommandBuffer* cmdBuffer = m_cmdQueue->commandBuffer();                    // Create a command buffer
        MTL::ComputeCommandEncoder* compEncoder = cmdBuffer->computeCommandEncoder();   // Start a compute pass
        // Serialize resource and states to be called by GPU
        encodeComputeCommandSingleBuffer(compEncoder, compFuncPSO, gridSize, threadsPerTG);
        compEncoder->endEncoding();         // End the compute pass
        cmdBuffer->commit();                // Execute the command
        cmdBuffer->waitUntilCompleted();    // Wait until the work is done
    }

    void sendComputeCommandDoubleBuffer(MTL::ComputePipelineState* compFuncPSO, MTL::Size & gridSize,
                                        MTL::Size & threadsPerTG)
    {
        MTL::CommandBuffer* cmdBuffer = m_cmdQueue->commandBuffer();                    // Create a command buffer
        MTL::ComputeCommandEncoder* compEncoder = cmdBuffer->computeCommandEncoder();   // Start a compute pass
        // Serialize resource and states to be called by GPU
        encodeComputeCommandDoubleBuffer(compEncoder, compFuncPSO, gridSize, threadsPerTG);
        compEncoder->endEncoding();         // End the compute pass
        cmdBuffer->commit();                // Execute the command
        cmdBuffer->waitUntilCompleted();    // Wait until the work is done
    }

    void sendComputeCommandArrayScalar(MTL::ComputePipelineState* compFuncPSO, MTL::Size & gridSize,
                                       MTL::Size & threadsPerTG)
    {
        MTL::CommandBuffer* cmdBuffer = m_cmdQueue->commandBuffer();                    // Create a command buffer
        MTL::ComputeCommandEncoder* compEncoder = cmdBuffer->computeCommandEncoder();   // Start a compute pass
        // Serialize resource and states to be called by GPU
        encodeComputeCommandArrayScalar(compEncoder, compFuncPSO, gridSize, threadsPerTG);
        compEncoder->endEncoding();         // End the compute pass
        cmdBuffer->commit();                // Execute the command
        cmdBuffer->waitUntilCompleted();    // Wait until the work is done
    }

    struct MatrixSize
    {
        uint rows;
        uint cols;
    };

    NS::AutoreleasePool*   m_pool{nullptr};
    MTL::Device*           m_mtlDevice{nullptr};
    MTL::CommandQueue*     m_cmdQueue{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOAdd{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSub{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMul{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODiv{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOAdd_A_S{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSub_S_A{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMul_A_S{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODiv_A_S{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODiv_S_A{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSqrt{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSin{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCos{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTanh{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOLog{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOExp{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMul{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatTranspose{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCopy_A_A{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCopy_S_A{nullptr};
    MTL::Buffer*   m_buf1{nullptr};
    MTL::Buffer*   m_buf2{nullptr};
    MTL::Buffer*   m_bufResult{nullptr};
    MatrixSize     m_buf1Size{0, 0};
    MatrixSize     m_buf2Size{0, 0};
    DataType       m_scalar{0};
    std::unordered_map<const void*, MTL::Buffer*>  m_allocMap;
};

}   // namespace
