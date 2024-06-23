//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once


// Project includes
#include "aix.hpp"
// External includes
// System includes


// Forward declarations
namespace NS
{
    class AutoreleasePool;
}

// Forward declarations
namespace MTL
{
    class Buffer;
    class ComputePipelineState;
    class CommandQueue;
    class CommandBuffer;
    class ComputeCommandEncoder;
    class Device;
    class Library;
    struct Size;
}


namespace aix
{

#define MAX_CMD_BATCH_SIZE                  10
#define MAX_THREADS_PER_THREADGROUP         1024
#define MIN_BUFFER_SIZE_TO_CPU_FALLBACK     1024
#define ALIGNMENT_SIZE                      64

class DeviceMetal : public aix::Device
{
public:
    // Constructor
    DeviceMetal();

    // Destructor
    virtual ~DeviceMetal();

    DeviceType type() const override { return DeviceType::kGPU_METAL; }

    size_t align(size_t size, size_t alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);       // Padding for alignment.
    }

    // Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
    void * allocate(size_t size) override;

    // Deallocate GPU memory if it's allocated by current device.
    void deallocate(void * memory) override;

    void add(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override;

    void sub(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override;

    void mul(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override;

    void div(const DataType * a1, const DataType * a2, const size_t size, DataType * result) override;

    void add(const DataType * a, DataType scalar, const size_t size, DataType * result) override;

    void sub(const DataType * a, DataType scalar, const size_t size, DataType * result) override;

    void sub(DataType scalar, const DataType * a, const size_t size, DataType * result) override;

    void mul(const DataType * a, DataType scalar, const size_t size, DataType * result) override;

    void div(const DataType * a, DataType scalar, const size_t size, DataType * result) override;

    void div(DataType scalar, const DataType * a, const size_t size, DataType * result) override;

    void unary(const DataType * a, const size_t size, DataType * result) override;

    void fill(DataType scalar, const size_t size, DataType * result) override;

    void sum(const DataType * a, const size_t size, DataType* result) override;

    void mean(const DataType * a, const size_t size, DataType* result) override;

    void sqrt(const DataType * a, const size_t size, DataType * result) override;

    void sin(const DataType * a, const size_t size, DataType * result) override;

    void cos(const DataType * a, const size_t size, DataType * result) override;

    void tanh(const DataType * a, const size_t size, DataType * result) override;

    void log(const DataType* a, const size_t size, DataType* result) override;

    void exp(const DataType* a, const size_t size, DataType* result) override;

    void pow(const DataType* a, const DataType* exp, const size_t size, DataType* result) override;

    void matmul(const DataType * a1, const Shape & s1, const DataType * a2, const Shape & s2, DataType * result) override;

    virtual void transpose(size_t dim0, size_t dim1, const DataType* data, [[maybe_unused]] const Shape& shape,
                           const Stride& strides, const Stride& newStrides, const size_t size, DataType* result) override;

    void copy(const DataType * src, DataType * dst, size_t size) override;

    void copyImmediate(const DataType* src, DataType* dst, size_t size) override;

    void broadcastTo(const DataType* src, DataType* dst, size_t size, const Shape& shape, const Shape& newShape) override;

    void reduceTo(const DataType* src, DataType* dst, size_t size, const Shape& shape, const Shape& newShape) override;

    void commitAndWait() override;

protected:
    struct MatrixSize
    {
        size_t rows;
        size_t cols;
    };

    MTL::Buffer* newBuffer(size_t size);

    MTL::Buffer* newBufferWithAddress(const void* address, size_t size);

    MTL::Buffer* getReadOnlyMTLBuffer(const void * address, size_t size, size_t sizeofType);

    void freeTemporaryBuffer(MTL::Buffer * buffer);

    MTL::Library* createLibrary(const char* shaders);

    MTL::CommandQueue* createCommandQueue();

    MTL::ComputePipelineState* createComputeFuncPSO(MTL::Library* library, const std::string & kernelName);

    void encodeComputeCommandSingleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size, MTL::Buffer* bufResult,
                                          MTL::ComputeCommandEncoder* computeEncoder,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const;

    void encodeComputeCommandDoubleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
                                          const MTL::Buffer* buf2, const MatrixSize& buf2Size, MTL::Buffer* bufResult,
                                          MTL::ComputeCommandEncoder * computeEncoder,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const;

    void encodeComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
                                         const DataType scalar, MTL::Buffer* bufResult,
                                         MTL::ComputeCommandEncoder* computeEncoder,
                                         const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                         const MTL::Size & threadsPerTG) const;

    void sendComputeCommandSingleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size, MTL::Buffer* bufResult,
                                        const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                        const MTL::Size & threadsPerTG);

    void sendComputeCommandDoubleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
                                        const MTL::Buffer* buf2, const MatrixSize& buf2Size, MTL::Buffer* bufResult,
                                        const MTL::ComputePipelineState* compFuncPSO, const MTL::Size & gridSize,
                                        const MTL::Size & threadsPerTG);

    void sendComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size, const DataType scalar,
                                       MTL::Buffer* bufResult, const MTL::ComputePipelineState* compFuncPSO,
                                       const MTL::Size & gridSize, const MTL::Size & threadsPerTG);

    void executeArrayScalarCmd(const DataType * a,
                               DataType scalar,
                               size_t size,
                               DataType * result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName);

    void executeDoubleArrayCmd(const DataType * a1,
                               const DataType * a2,
                               size_t size,
                               DataType * result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName);

    // Common method for broadcastTo and reduceTo methods.
    void translation(const DataType* src, DataType* dst, size_t size, const Shape& shape, const Shape& newShape,
                     const MTL::ComputePipelineState *computePSO, const std::string & name);

    void transpose2D(const DataType * mat, const Shape& shape, DataType * result);

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
    MTL::ComputePipelineState*   m_compFuncPSOPow{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSum{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMul{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatTranspose2D{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatTranspose{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCopy_A_A{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCopy_S_A{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOBroadcastTo{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOReduceTo{nullptr};
    std::vector<MTL::Buffer*>    m_tempBuffers;
    std::unordered_map<const void*, MTL::Buffer*>  m_allocMap;
    size_t   m_currentBatchSize{0};
    size_t   m_maxBatchSize{0};
};

}   // namespace
