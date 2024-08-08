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

namespace aix::metal
{

#define MAX_CMD_BATCH_SIZE                  1000
#define MAX_THREADS_PER_THREADGROUP         1024
#define ALLOCATION_BYTE_ALIGNMENT_SIZE      32      // Should be power of two and min 32 bytes.
#define VECTOR_TYPE_COMPONENT_COUNT         4       // i.e. float4 has 4 components.
#define BATCH_PROCESS_SIZE_PER_THREAD       1       // i.e. each GPU thread will access/process 16 of float4 per dispatch.
#define TOTAL_COMPONENT_COUNT               (BATCH_PROCESS_SIZE_PER_THREAD * VECTOR_TYPE_COMPONENT_COUNT)

class MTLBufferCache;

class DeviceMetal : public aix::Device
{
public:
    // Constructor
    explicit DeviceMetal(size_t deviceIndex = 0);

    // Destructor
    ~DeviceMetal() override;

    DeviceType type() const override { return DeviceType::kGPU_METAL; }
    std::string name() const override { return "MCS"; }             // MCS = Metal Custom/Compute Shaders

    // Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
    void* allocate(size_t size) override;

    void* allocate(size_t size, DataType dtype) override;

    // Deallocate GPU memory if it's allocated by current device.
    void deallocate(void * memory) override;

    void add(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void sub(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void mul(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void div(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void unary(const void* a, size_t size, void* result, DataType dtype) override;

    void fill(const void* scalar, DataType srcDType, size_t size, void* result, DataType dstDType) override;

    void fillMin(DataType dtype, size_t size, void* result) override;

    void sum(const void* a, size_t size, void* result, DataType dtype) override;

    void sqrt(const void* a, size_t size, void* result, DataType dtype) override;

    void sin(const void* a, size_t size, void* result, DataType dtype) override;

    void cos(const void* a, size_t size, void* result, DataType dtype) override;

    void tanh(const void* a, size_t size, void* result, DataType dtype) override;

    void log(const void* a, size_t size, void* result, DataType dtype) override;

    void exp(const void* a, size_t size, void* result, DataType dtype) override;

    void pow(const void* a, const void* exp, size_t size, void* result, DataType dtype) override;

    void max(const void* a, size_t size, void* result, DataType dtype) override;

    void argmax(const void* a, size_t size, void* result, DataType dtype, DataType resultDtype) override;

    void argmaxIndices(const void* a, size_t size, void* result, DataType dtype, DataType resultDtype) override;

    void matmul(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result, DataType dtype) override;

    void transpose(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                   const Stride& strides, const Stride& newStrides, size_t size, void* result, DataType dtype) override;

    void copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) override;

    void copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) override;

    void broadcastTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype) override;

    void reduceTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype) override;

    void maxTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype) override;

    void argmaxTo(const void* src, void* dst, size_t srcSize, size_t dstSize, const Shape& shape, const Shape& newShape,
                  const Shape& strides, size_t dim, DataType dtype, DataType resultDtype) override;

    void argmaxIndicesTo(const void* src, void* dst, size_t srcSize, size_t dstSize,
                         const Shape& shape, const Shape& newShape, DataType dtype, DataType resultDtype) override;

    void commitAndWait() override;

protected:
    void commit();
    void commitBatchQueue();

    inline static void validateDataType(DataType dtype);

    struct MatrixSize
    {
        size_t rows;
        size_t cols;
    };

    static size_t align(size_t size, size_t alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);       // Padding for alignment.
    }

    inline bool isDeviceBuffer(const void* bufPtr)
    {
        return m_allocMap.find(bufPtr) != m_allocMap.end();
    }

    MTL::Buffer* newBuffer(size_t size);

    MTL::Buffer* getReadOnlyMTLBuffer(const void * address, size_t size, size_t sizeofType,
                                      size_t alignSize = TOTAL_COMPONENT_COUNT);

    void freeTemporaryBuffer(MTL::Buffer * buffer);

    MTL::Device* createMTLDevice(size_t deviceIndex) const;

    MTL::Library* createLibrary(const char* shaders);

    MTL::CommandQueue* createCommandQueue();

    MTL::ComputePipelineState* createComputeFuncPSO(MTL::Library* library, const std::string & kernelName);

    void encodeComputeCommandDoubleBuffer(const MTL::Buffer* buf, MTL::Buffer* bufResult,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const;

    void encodeComputeCommandTripleBuffer(const MTL::Buffer* buf1, const MTL::Buffer* buf2, MTL::Buffer* bufResult,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const;

    void executeDoubleArrayCmd(const void* a1, size_t size, void* result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               DataType dtype, const std::string & cmdName);

    void executeTripleArrayCmd(const void* a1, const void* a2, size_t size, void* result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               DataType dtype, const std::string & cmdName);

    // Common method for broadcastTo and reduceTo methods.
    void translation(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                     const MTL::ComputePipelineState *computePSO, DataType dtype, const std::string & name);

    void transpose2D(const void* mat, const Shape& shape, void* result, DataType dtype);

    static const std::string& toString(size_t dtype);
    inline static const std::string& toString(DataType dtype);

    static void CheckCommandBufferStatus(const MTL::CommandBuffer* commandBuffer);

    NS::AutoreleasePool*   m_pool{nullptr};
    MTL::Device*           m_mtlDevice{nullptr};
    MTL::CommandQueue*     m_cmdQueue{nullptr};
    MTL::CommandBuffer*    m_cmdBuffer{nullptr};
    MTL::CommandBuffer*    m_committedCmdBuffer{nullptr};
    MTL::ComputeCommandEncoder*  m_compEncoder{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOAdd[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOSub[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOMul[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSODiv[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOUnary[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOSqrt[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOSin[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOCos[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOTanh[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOLog[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOExp[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOPow[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOSum[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOMax[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOMatMul[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOTranspose2D[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOTranspose[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOCopyAA[aix::DataTypeCount][aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOFill[aix::DataTypeCount][aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOFillMin[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOBroadcastTo[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOReduceTo[aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOMaxTo[aix::DataTypeCount];
    std::vector<std::pair<MTL::Buffer*, void*>>    m_tempBuffers;
    std::unordered_map<const void*, MTL::Buffer*>  m_allocMap;
    std::unique_ptr<MTLBufferCache>  m_bufferCache;
    size_t   m_currentBatchSize{0};
    size_t   m_maxBatchSize{0};
    size_t   m_maxWorkingSetSize{0};
    size_t   m_currentWorkingSetSize{0};
};

}   // namespace
