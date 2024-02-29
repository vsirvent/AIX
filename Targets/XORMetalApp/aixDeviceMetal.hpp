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
        auto defaultLibrary = loadLibrary("default.metallib");
        // Compile time evaluation.
        if constexpr (std::is_same_v<DataType, float>)
        {
            m_compFuncPSOMatMul       = createComputeFuncPSO(defaultLibrary, "matrix_mul_float");
            m_compFuncPSOMatTranspose = createComputeFuncPSO(defaultLibrary, "matrix_transpose_float");
        }
        else
            throw std::invalid_argument("Metal device supports only float data type for now.");

        m_cmdQueue = createCommandQueue();
    }

    // Destructor
    virtual ~DeviceMetal()
    {
        m_cmdQueue->release();
        m_compFuncPSOMatMul->release();
        m_compFuncPSOMatTranspose->release();
        m_mtlDevice->release();
        m_pool->release();
    }

    DeviceType type() const override { return DeviceType::kGPU_METAL; }

/*
    // TODO: Add GPU support for the following device methods.
    // Unimplemented GPU implementations will use CPU by default and be called from base Device.

    void add(const Array & a1, const Array & a2, const size_t size, Array & result) override {}
    void sub(const Array & a1, const Array & a2, const size_t size, Array & result) override {}
    void mul(const Array & a1, const Array & a2, const size_t size, Array & result) override {}
    void div(const Array & a1, const Array & a2, const size_t size, Array & result) override {}
    void add(const Array & a, DataType scalar, const size_t size, Array & result) override {}
    void sub(const Array & a, DataType scalar, const size_t size, Array & result) override {}
    void sub(DataType scalar, const Array & a, const size_t size, Array & result) override {}
    void mul(const Array & a, DataType scalar, const size_t size, Array & result) override {}
    void div(const Array & a, DataType scalar, const size_t size, Array & result) override {}
    void div(DataType scalar, const Array & a, const size_t size, Array & result) override {}
    void unary(const Array & a, const size_t size, Array & result) override {}
    void fill(DataType value, const size_t size, Array & result) override {}
    void mean(const Array & a, const size_t size, DataType & result) override {}
    void sqrt(const Array & a, const size_t size, Array & result) override {}
    void sin(const Array & a, const size_t size, Array & result) override {}
    void cos(const Array & a, const size_t size, Array & result) override {}
    void tanh(const Array & a, const size_t size, Array & result) override {}
*/

    void matmul(const Array & a1, const Shape & s1, const Array & a2, const Shape & s2, Array & result) override
    {
        size_t mat1Size = s1[0] * s1[1];
        size_t mat2Size = s2[0] * s2[1];
        size_t matResultSize = s1[0] * s2[1];

        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::matmul(...);

        m_buf1 = m_mtlDevice->newBuffer(static_cast<const void*>(a1.data()), mat1Size * sizeof(DataType),
                                        MTL::ResourceStorageModeShared);
        m_buf2 = m_mtlDevice->newBuffer(static_cast<const void*>(a2.data()), mat2Size * sizeof(DataType),
                                        MTL::ResourceStorageModeShared);

        m_bufResult = m_mtlDevice->newBuffer(static_cast<void*>(result.data()), matResultSize * sizeof(DataType),
                                             MTL::ResourceStorageModeShared);

        m_buf1Size.rows = s1[0];
        m_buf1Size.cols = s1[1];
        m_buf2Size.rows = s2[0];
        m_buf2Size.cols = s2[1];

        auto gridSize = MTL::Size(m_buf1Size.cols, m_buf2Size.rows, 1);    // Final buffer size
        sendComputeCommandDoubleBuffer(m_compFuncPSOMatMul, gridSize);

        m_buf1->release();
        m_buf2->release();
        m_bufResult->release();
    }

    void transpose(const Array & a, const Shape & shape, Array & result) override
    {
        size_t matSize = shape[0] * shape[1];

        // TODO: If the tensor size is small, we can call base CPU implementation to reduce GPU call overhead.
        // Device::transpose(...);

        m_buf1 = m_mtlDevice->newBuffer(static_cast<const void*>(a.data()), matSize * sizeof(DataType),
                                        MTL::ResourceStorageModeShared);

        m_bufResult = m_mtlDevice->newBuffer(static_cast<void*>(result.data()), matSize * sizeof(DataType),
                                             MTL::ResourceStorageModeShared);

        m_buf1Size.rows = shape[0];
        m_buf1Size.cols = shape[1];

        auto gridSize = MTL::Size(m_buf1Size.rows, m_buf1Size.cols, 1);    // Final buffer size
        sendComputeCommandSingleBuffer(m_compFuncPSOMatTranspose, gridSize);

        m_buf1->release();
        m_bufResult->release();
    }

protected:
    MTL::Library* loadLibrary(const std::string & libName)
    {
        NS::Error* error = nullptr;

        auto library = NS::String::string(libName.c_str(), NS::UTF8StringEncoding);
        auto defaultLibrary = m_mtlDevice->newLibrary(library, &error);
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
                                          MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(m_buf1, 0, 0);
        computeEncoder->setBuffer(m_bufResult, 0, 1);
        computeEncoder->setBytes(&m_buf1Size, sizeof(MatrixSize), 3);

        setupGrid(computeEncoder, compFuncPSO, gridSize);
    }

    void encodeComputeCommandDoubleBuffer(MTL::ComputeCommandEncoder * computeEncoder,
                                          MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(m_buf1, 0, 0);
        computeEncoder->setBuffer(m_buf2, 0, 1);
        computeEncoder->setBuffer(m_bufResult, 0, 2);
        computeEncoder->setBytes(&m_buf1Size, sizeof(MatrixSize), 3);
        computeEncoder->setBytes(&m_buf2Size, sizeof(MatrixSize), 4);

        setupGrid(computeEncoder, compFuncPSO, gridSize);
    }

    void sendComputeCommandSingleBuffer(MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize)
    {
        MTL::CommandBuffer* cmdBuffer = m_cmdQueue->commandBuffer();                    // Create a command buffer
        MTL::ComputeCommandEncoder* compEncoder = cmdBuffer->computeCommandEncoder();   // Start a compute pass
        // Serialize resource and states to be called by GPU
        encodeComputeCommandSingleBuffer(compEncoder, compFuncPSO, gridSize);
        compEncoder->endEncoding();         // End the compute pass
        cmdBuffer->commit();                // Execute the command
        cmdBuffer->waitUntilCompleted();    // Wait until the work is done
    }

    void sendComputeCommandDoubleBuffer(MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize)
    {
        MTL::CommandBuffer* cmdBuffer = m_cmdQueue->commandBuffer();                    // Create a command buffer
        MTL::ComputeCommandEncoder* compEncoder = cmdBuffer->computeCommandEncoder();   // Start a compute pass
        // Serialize resource and states to be called by GPU
        encodeComputeCommandDoubleBuffer(compEncoder, compFuncPSO, gridSize);
        compEncoder->endEncoding();         // End the compute pass
        cmdBuffer->commit();                // Execute the command
        cmdBuffer->waitUntilCompleted();    // Wait until the work is done
    }

    void setupGrid(MTL::ComputeCommandEncoder * computeEncoder, MTL::ComputePipelineState*  compFuncPSO,
                   MTL::Size & gridSize) const
    {
        // Calculate maximum thread group dimensions
        NS::UInteger w = compFuncPSO->threadExecutionWidth();
        NS::UInteger h = compFuncPSO->maxTotalThreadsPerThreadgroup() / w;

        // Encode the compute command. IMPORTANT: Assuming the device supports non-uniform grid size.
        computeEncoder->dispatchThreads(gridSize, MTL::Size(w, h, 1));
    }

    struct MatrixSize
    {
        uint rows;
        uint cols;
    };

    NS::AutoreleasePool*   m_pool{nullptr};
    MTL::Device*           m_mtlDevice{nullptr};
    MTL::CommandQueue*     m_cmdQueue{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMul{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatTranspose{nullptr};
    MTL::Buffer*   m_buf1{nullptr};
    MTL::Buffer*   m_buf2{nullptr};
    MTL::Buffer*   m_bufResult{nullptr};
    MatrixSize     m_buf1Size{0, 0};
    MatrixSize     m_buf2Size{0, 0};
};

}   // namespace
