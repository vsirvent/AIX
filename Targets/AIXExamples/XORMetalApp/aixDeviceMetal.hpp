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

#define MAX_CMD_BATCH_SIZE 1

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
        m_cmdBuffer = m_cmdQueue->commandBuffer();
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
        executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOAdd, "add");
    }

    void sub(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOSub, "sub");
    }

    void mul(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOMul, "mul");
    }

    void div(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override
    {
        executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSODiv, "div");
    }

    void add(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSOAdd_A_S, "add");
    }

    void sub(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, -scalar, size, result, m_compFuncPSOAdd_A_S, "sub");
    }

    void sub(DataType scalar, const DataType * a, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSOSub_S_A, "sub");
    }

    void mul(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSOMul_A_S, "mul");
    }

    void div(const DataType * a, DataType scalar, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSODiv_A_S, "div");
    }

    void div(DataType scalar, const DataType * a, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSODiv_S_A, "div");
    }

    void unary(const DataType * a, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, DataType(-1), size, result, m_compFuncPSOMul_A_S, "unary");
    }

    void fill(DataType scalar, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(nullptr, scalar, size, result, m_compFuncPSOCopy_S_A, "copy");
    }

    void sum(const DataType * a, const size_t size, DataType & result) override
    {
        commitAndWait();
        // TODO: Add GPU support for the following device methods.
        Device::sum(a, size, result);
    }

    void mean(const DataType * a, const size_t size, DataType & result) override
    {
        commitAndWait();
        // TODO: Add GPU support for the following device methods.
        Device::mean(a, size, result);
    }

    void sqrt(const DataType * a, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSqrt, "sqrt");
    }

    void sin(const DataType * a, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSin, "sin");
    }

    void cos(const DataType * a, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOCos, "cos");
    }

    void tanh(const DataType * a, const size_t size, DataType * result) override
    {
        executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOTanh, "tanh");
    }

    void log(const DataType* a, const size_t size, DataType* result) override
    {
        executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOLog, "log");
    }

    void exp(const DataType* a, const size_t size, DataType* result) override
    {
        executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOExp, "exp");
    }

    void matmul(const DataType * a1, const Shape & s1, const DataType * a2, const Shape & s2, DataType * result) override
    {
        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

        auto buf1 = getReadOnlyMTLBuffer(a1, s1[0] * s1[1]);  // Memory could be a GPU allocated memory or system memory.
        auto buf2 = getReadOnlyMTLBuffer(a2, s2[0] * s2[1]);  // Memory could be a GPU allocated memory or system memory.
        auto bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = m_compFuncPSOMatMul->threadExecutionWidth();
        NS::UInteger h = m_compFuncPSOMatMul->maxTotalThreadsPerThreadgroup() / w;
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        sendComputeCommandDoubleBuffer(buf1, {s1[0], s1[1]}, buf2, {s2[0], s2[1]}, bufResult,
                                       m_compFuncPSOMatMul, {s2[1], s1[0], 1}, {w, h, 1});

        freeTemporaryBuffer(buf1);
        freeTemporaryBuffer(buf2);
    }

    void transpose(const DataType * mat, const Shape & shape, DataType * result) override
    {
        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

        // Memory could be a GPU allocated memory or system memory.
        auto buf1 = getReadOnlyMTLBuffer(mat, shape[0] * shape[1]);
        auto bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = m_compFuncPSOMatTranspose->threadExecutionWidth();
        NS::UInteger h = m_compFuncPSOMatTranspose->maxTotalThreadsPerThreadgroup() / w;

        sendComputeCommandSingleBuffer(buf1, {shape[0], shape[1]}, bufResult,
                                       m_compFuncPSOMatTranspose, {shape[0], shape[1], 1}, {w, h, 1});

        freeTemporaryBuffer(buf1);
    }

    void copy(const DataType * src, DataType * dst, size_t size) override
    {
        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(dst) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::copy() result must have GPU memory.");

        // Memory could be a GPU allocated memory or system memory.
        auto buf1 = getReadOnlyMTLBuffer(src, size);
        auto bufResult = m_allocMap[dst];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, m_compFuncPSOCopy_A_A->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        sendComputeCommandSingleBuffer(buf1, {1, size}, bufResult, m_compFuncPSOCopy_A_A, {size, 1, 1}, {w, 1, 1});

        freeTemporaryBuffer(buf1);
    }

    void copy_immediate(const DataType* src, DataType* dst, size_t size) override
    {
        copy(src, dst, size);
        commitAndWait();
    }

    void commitAndWait() override
    {
        // Execute only if there is at least one command encoded.
        if (!m_compEncoder) return;

        m_compEncoder->endEncoding();
        m_cmdBuffer->commit();                // Execute the command
        m_cmdBuffer->waitUntilCompleted();    // Wait until the work is done
        m_cmdBuffer = m_cmdQueue->commandBuffer();
        m_compEncoder = nullptr;

        // Update batch size metrics.
        m_maxBatchSize = std::max(m_currentBatchSize, m_maxBatchSize);
        m_currentBatchSize = 0;
    }

protected:
    struct MatrixSize
    {
        size_t rows;
        size_t cols;
    };

    inline MTL::Buffer* getReadOnlyMTLBuffer(const DataType * address, size_t size)
    {
        // Memory could be from other devices. Create a temporary buffer for read only case.
        if (m_allocMap.find(address) == m_allocMap.end())
        {
            return m_mtlDevice->newBuffer(address, size * sizeof(DataType), MTL::ResourceStorageModeShared);
        }

        return m_allocMap[address];    // Return MTL Buffer if the memory is from the current device.
    }

    inline void freeTemporaryBuffer(MTL::Buffer * buffer)
    {
        // Release only temporary buffer.
        if (m_allocMap.find(buffer->contents()) == m_allocMap.end())
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

    void encodeComputeCommandSingleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size, MTL::Buffer* bufResult,
                                          MTL::ComputeCommandEncoder* computeEncoder,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(buf1, 0, 0);
        computeEncoder->setBuffer(bufResult, 0, 1);
        computeEncoder->setBytes(&buf1Size, sizeof(MatrixSize), 2);
        computeEncoder->dispatchThreads(gridSize, threadsPerTG);
    }

    void encodeComputeCommandDoubleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
                                          const MTL::Buffer* buf2, const MatrixSize& buf2Size, MTL::Buffer* bufResult,
                                          MTL::ComputeCommandEncoder * computeEncoder,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(buf1, 0, 0);
        computeEncoder->setBuffer(buf2, 0, 1);
        computeEncoder->setBuffer(bufResult, 0, 2);
        computeEncoder->setBytes(&buf1Size, sizeof(MatrixSize), 3);
        computeEncoder->setBytes(&buf2Size, sizeof(MatrixSize), 4);
        computeEncoder->dispatchThreads(gridSize, threadsPerTG);
    }

    void encodeComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
                                         const DataType scalar, MTL::Buffer* bufResult,
                                         MTL::ComputeCommandEncoder* computeEncoder,
                                         const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                         const MTL::Size & threadsPerTG) const
    {
        // Encode the pipeline state object and its parameters.
        computeEncoder->setComputePipelineState(compFuncPSO);
        computeEncoder->setBuffer(buf1, 0, 0);
        computeEncoder->setBytes(&scalar, sizeof(DataType), 1);
        computeEncoder->setBytes(&buf1Size, sizeof(MatrixSize), 2);
        computeEncoder->setBuffer(bufResult, 0, 3);
        computeEncoder->dispatchThreads(gridSize, threadsPerTG);
    }

    void sendComputeCommandSingleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size, MTL::Buffer* bufResult,
                                        const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                        const MTL::Size & threadsPerTG)
    {
        if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
        // Serialize resource and states to be called by GPU
        encodeComputeCommandSingleBuffer(buf1, buf1Size, bufResult,
                                         m_compEncoder, compFuncPSO, gridSize, threadsPerTG);
        m_currentBatchSize++;
        if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();
    }

    void sendComputeCommandDoubleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
                                        const MTL::Buffer* buf2, const MatrixSize& buf2Size, MTL::Buffer* bufResult,
                                        const MTL::ComputePipelineState* compFuncPSO, const MTL::Size & gridSize,
                                        const MTL::Size & threadsPerTG)
    {
        if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
        // Serialize resource and states to be called by GPU
        encodeComputeCommandDoubleBuffer(buf1, buf1Size, buf2, buf2Size, bufResult,
                                         m_compEncoder, compFuncPSO, gridSize, threadsPerTG);
        m_currentBatchSize++;
        if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();
    }

    void sendComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size, const DataType scalar,
                                       MTL::Buffer* bufResult, const MTL::ComputePipelineState* compFuncPSO,
                                       const MTL::Size & gridSize, const MTL::Size & threadsPerTG)
    {
        if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
        // Serialize resource and states to be called by GPU
        encodeComputeCommandArrayScalar(buf1, buf1Size, scalar, bufResult,
                                        m_compEncoder, compFuncPSO, gridSize, threadsPerTG);
        m_currentBatchSize++;
        if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();
    }

    void executeArrayScalarCmd(const DataType * a,
                               DataType scalar,
                               size_t size,
                               DataType * result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName)
    {
        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

        // Set constants
        auto buf1      = a ? getReadOnlyMTLBuffer(a, size) : nullptr;
        auto bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, compFuncPSO->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        sendComputeCommandArrayScalar(buf1, {1, size}, scalar, bufResult, compFuncPSO, {size, 1, 1}, {w, 1, 1});

        if (a)
        {
            // Buf1 could be temporary buffer. It should be deleted if temporary.
            freeTemporaryBuffer(buf1);
            // Note: We never release result buffer since it will be used.
        }
    }

    void executeDoubleArrayCmd(const DataType * a1,
                               const DataType * a2,
                               size_t size,
                               DataType * result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName)
    {
        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

        auto buf1 = getReadOnlyMTLBuffer(a1, size);   // Memory could be a GPU allocated memory or system memory.
        auto buf2 = getReadOnlyMTLBuffer(a2, size);   // Memory could be a GPU allocated memory or system memory.
        auto bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, compFuncPSO->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        sendComputeCommandDoubleBuffer(buf1, {0,0}, buf2, {0,0}, bufResult, compFuncPSO, {size, 1, 1}, {w, 1, 1});

        // Buf 1 and 2 could be temporary buffer. It should be deleted if temporary.
        freeTemporaryBuffer(buf1);
        freeTemporaryBuffer(buf2);
        // Note: We never release result buffer since it will be used.
    }

    NS::AutoreleasePool*   m_pool{nullptr};
    MTL::Device*           m_mtlDevice{nullptr};
    MTL::CommandQueue*     m_cmdQueue{nullptr};
    MTL::CommandBuffer*    m_cmdBuffer{nullptr};
    MTL::ComputeCommandEncoder*  m_compEncoder{nullptr};
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
    std::unordered_map<const void*, MTL::Buffer*>  m_allocMap;
    size_t   m_currentBatchSize{0};
    size_t   m_maxBatchSize{0};
};

}   // namespace
