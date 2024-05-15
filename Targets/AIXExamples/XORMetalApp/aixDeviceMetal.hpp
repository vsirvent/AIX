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
        computeEncoder->setBytes(&m_scalar, sizeof(DataType), 1);
        computeEncoder->setBytes(&m_buf1Size, sizeof(MatrixSize), 2);
        computeEncoder->setBuffer(m_bufResult, 0, 3);
        computeEncoder->dispatchThreads(gridSize, threadsPerTG);
    }

    void sendComputeCommandSingleBuffer(MTL::ComputePipelineState*  compFuncPSO, MTL::Size & gridSize,
                                        MTL::Size & threadsPerTG)
    {
        if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
        // Serialize resource and states to be called by GPU
        encodeComputeCommandSingleBuffer(m_compEncoder, compFuncPSO, gridSize, threadsPerTG);
        m_currentBatchSize++;
        if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();
    }

    void sendComputeCommandDoubleBuffer(MTL::ComputePipelineState* compFuncPSO, MTL::Size & gridSize,
                                        MTL::Size & threadsPerTG)
    {
        if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
        // Serialize resource and states to be called by GPU
        encodeComputeCommandDoubleBuffer(m_compEncoder, compFuncPSO, gridSize, threadsPerTG);
        m_currentBatchSize++;
        if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();
    }

    void sendComputeCommandArrayScalar(MTL::ComputePipelineState* compFuncPSO, MTL::Size & gridSize,
                                       MTL::Size & threadsPerTG)
    {
        if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
        // Serialize resource and states to be called by GPU
        encodeComputeCommandArrayScalar(m_compEncoder, compFuncPSO, gridSize, threadsPerTG);
        m_currentBatchSize++;
        if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();
    }

    void executeArrayScalarCmd(const DataType * a,
                               DataType scalar,
                               size_t size,
                               DataType * result,
                               MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName)
    {
        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

        // Set constants
        m_buf1          = a ? getReadOnlyMTLBuffer(a, size) : nullptr;
        m_buf2          = nullptr;
        m_bufResult     = m_allocMap[result];
        m_scalar        = scalar;
        m_buf1Size.rows = 1;
        m_buf1Size.cols = size;

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, compFuncPSO->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        auto threadsPerThreadGroup = MTL::Size{w, 1, 1};
        auto gridSize = MTL::Size{size, 1, 1};            // gridSize = array size
        sendComputeCommandArrayScalar(compFuncPSO, gridSize, threadsPerThreadGroup);

        if (a)
        {
            // Buf1 could be temporary buffer. It should be deleted if temporary.
            freeTemporaryBuffer(m_buf1, a);
            // Note: We never release result buffer since it will be used.
        }
    }

    void executeDoubleArrayCmd(const DataType * a1,
                               const DataType * a2,
                               size_t size,
                               DataType * result,
                               MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName)
    {
        // Result buffer has to be allocated in advance and has to be a GPU memory.
        if (m_allocMap.find(result) == m_allocMap.end())
            throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

        m_buf1 = getReadOnlyMTLBuffer(a1, size);   // Memory could be a GPU allocated memory or system memory.
        m_buf2 = getReadOnlyMTLBuffer(a2, size);   // Memory could be a GPU allocated memory or system memory.
        m_bufResult = m_allocMap[result];

        // Calculate maximum thread group dimensions
        NS::UInteger w = std::min(size, compFuncPSO->maxTotalThreadsPerThreadgroup());
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        auto threadsPerThreadGroup = MTL::Size{w, 1, 1};
        auto gridSize = MTL::Size{size, 1, 1};    // gridSize = Final matrix size
        sendComputeCommandDoubleBuffer(compFuncPSO, gridSize, threadsPerThreadGroup);

        // Buf 1 and 2 could be temporary buffer. It should be deleted if temporary.
        freeTemporaryBuffer(m_buf1, a1);
        freeTemporaryBuffer(m_buf2, a2);
        // Note: We never release result buffer since it will be used.
    }

    struct MatrixSize
    {
        uint rows;
        uint cols;
    };

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
    MTL::Buffer*   m_buf1{nullptr};
    MTL::Buffer*   m_buf2{nullptr};
    MTL::Buffer*   m_bufResult{nullptr};
    MatrixSize     m_buf1Size{0, 0};
    MatrixSize     m_buf2Size{0, 0};
    DataType       m_scalar{0};
    size_t         m_currentBatchSize{0};
    size_t         m_maxBatchSize{0};
    std::unordered_map<const void*, MTL::Buffer*>  m_allocMap;
};

}   // namespace
