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

#define MAX_CMD_BATCH_SIZE                  1000
#define MAX_THREADS_PER_THREADGROUP         1024
#define ALIGNMENT_SIZE                      64

class DeviceMetal : public aix::Device
{
public:
    // Constructor
    DeviceMetal();

    // Destructor
    ~DeviceMetal() override;

    DeviceType type() const override { return DeviceType::kGPU_METAL; }

    // Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
    void* allocate(size_t size) override;

    void* allocate(size_t size, DataType dtype) override;

    // Deallocate GPU memory if it's allocated by current device.
    void deallocate(void * memory) override;

    void add(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void sub(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void mul(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void div(const void* a1, const void* a2, size_t size, void* result, DataType dtype) override;

    void addAS(const void* a, const void* scalar, size_t size, void* result, DataType dtype) override;

    void subAS(const void* a, const void* scalar, size_t size, void* result, DataType dtype) override;

    void subSA(const void* scalar, const void* a, size_t size, void* result, DataType dtype) override;

    void mulAS(const void* a, const void* scalar, size_t size, void* result, DataType dtype) override;

    void divAS(const void* a, const void* scalar, size_t size, void* result, DataType dtype) override;

    void divSA(const void* scalar, const void* a, size_t size, void* result, DataType dtype) override;

    void unary(const void* a, size_t size, void* result, DataType dtype) override;

    void fill(const void* scalar, size_t size, void* result, DataType dtype) override;

    void sum(const void* a, size_t size, void* result, DataType dtype) override;

    void mean(const void* a, size_t size, void* result, DataType dtype) override;

    void sqrt(const void* a, size_t size, void* result, DataType dtype) override;

    void sin(const void* a, size_t size, void* result, DataType dtype) override;

    void cos(const void* a, size_t size, void* result, DataType dtype) override;

    void tanh(const void* a, size_t size, void* result, DataType dtype) override;

    void log(const void* a, size_t size, void* result, DataType dtype) override;

    void exp(const void* a, size_t size, void* result, DataType dtype) override;

    void pow(const void* a, const void* exp, size_t size, void* result, DataType dtype) override;

    void matmul(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result, DataType dtype) override;

    void transpose(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                   const Stride& strides, const Stride& newStrides, size_t size, void* result, DataType dtype) override;

    void copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) override;

    void copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) override;

    void broadcastTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype) override;

    void reduceTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype) override;

    void commitAndWait() override;

protected:
    void addF64(const void* a1, const void* a2, size_t size, void* result);
    void addF32(const void* a1, const void* a2, size_t size, void* result);

    void subF64(const void* a1, const void* a2, size_t size, void* result);
    void subF32(const void* a1, const void* a2, size_t size, void* result);

    void mulF64(const void* a1, const void* a2, size_t size, void* result);
    void mulF32(const void* a1, const void* a2, size_t size, void* result);

    void divF64(const void* a1, const void* a2, size_t size, void* result);
    void divF32(const void* a1, const void* a2, size_t size, void* result);

    void addASF64(const void* a1, const void* scalar, size_t size, void* result);
    void addASF32(const void* a1, const void* scalar, size_t size, void* result);

    void subASF64(const void* a1, const void* scalar, size_t size, void* result);
    void subASF32(const void* a1, const void* scalar, size_t size, void* result);

    void subSAF64(const void* scalar, const void* a, size_t size, void* result);
    void subSAF32(const void* scalar, const void* a, size_t size, void* result);

    void mulASF64(const void* a, const void* scalar, size_t size, void* result);
    void mulASF32(const void* a, const void* scalar, size_t size, void* result);

    void divASF64(const void* a, const void* scalar, size_t size, void* result);
    void divASF32(const void* a, const void* scalar, size_t size, void* result);

    void divSAF64(const void* scalar, const void* a, size_t size, void* result);
    void divSAF32(const void* scalar, const void* a, size_t size, void* result);

    void unaryF64(const void* a, size_t size, void* result);
    void unaryF32(const void* a, size_t size, void* result);

    void fillF64(const void* scalar, size_t size, void* result);
    void fillF32(const void* scalar, size_t size, void* result);

    void sumF64(const void* a, size_t size, void* result);
    void sumF32(const void* a, size_t size, void* result);

    void meanF64(const void* a, size_t size, void* result);
    void meanF32(const void* a, size_t size, void* result);

    void sqrtF64(const void* a, size_t size, void* result);
    void sqrtF32(const void* a, size_t size, void* result);

    void sinF64(const void* a, size_t size, void* result);
    void sinF32(const void* a, size_t size, void* result);

    void cosF64(const void* a, size_t size, void* result);
    void cosF32(const void* a, size_t size, void* result);

    void tanhF64(const void* a, size_t size, void* result);
    void tanhF32(const void* a, size_t size, void* result);

    void logF64(const void* a, size_t size, void* result);
    void logF32(const void* a, size_t size, void* result);

    void expF64(const void* a, size_t size, void* result);
    void expF32(const void* a, size_t size, void* result);

    void powF64(const void* a, const void* exp, size_t size, void* result);
    void powF32(const void* a, const void* exp, size_t size, void* result);

    void matmulF64(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result);
    void matmulF32(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result);

    void transposeF64(size_t dim0, size_t dim1, const void* data, const Shape& shape,
                      const Stride& strides, const Stride& newStrides, size_t size, void* result);
    void transposeF32(size_t dim0, size_t dim1, const void* data, const Shape& shape,
                      const Stride& strides, const Stride& newStrides, size_t size, void* result);

    void copyF64(const void* src, void* dst, size_t size);
    void copyF32(const void* src, void* dst, size_t size);

    void copyImmediateF64(const void* src, void* dst, size_t size);
    void copyImmediateF32(const void* src, void* dst, size_t size);

    void broadcastToF64(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape);
    void broadcastToF32(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape);

    void reduceToF64(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape);
    void reduceToF32(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape);

    void notImplementedF64() const;

    struct MatrixSize
    {
        size_t rows;
        size_t cols;
    };

    static size_t align(size_t size, size_t alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);       // Padding for alignment.
    }

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
                                         float scalar, MTL::Buffer* bufResult,
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

    void sendComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size, float scalar,
                                       MTL::Buffer* bufResult, const MTL::ComputePipelineState* compFuncPSO,
                                       const MTL::Size & gridSize, const MTL::Size & threadsPerTG);

    void executeArrayScalarCmd(const void* a,
                               float scalar,
                               size_t size,
                               void* result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName);

    void executeDoubleArrayCmd(const void* a1,
                               const void* a2,
                               size_t size,
                               void* result,
                               const MTL::ComputePipelineState* compFuncPSO,
                               const std::string & cmdName);

    // Common method for broadcastTo and reduceTo methods.
    void translationF32(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                        const MTL::ComputePipelineState *computePSO, const std::string & name);

    void transpose2DF32(const void* mat, const Shape& shape, void* result);

    NS::AutoreleasePool*   m_pool{nullptr};
    MTL::Device*           m_mtlDevice{nullptr};
    MTL::CommandQueue*     m_cmdQueue{nullptr};
    MTL::CommandBuffer*    m_cmdBuffer{nullptr};
    MTL::ComputeCommandEncoder*  m_compEncoder{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOAddF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSubF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMulF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODivF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOAddASF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSubSAF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMulASF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODivASF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODivSAF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSqrtF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSinF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCosF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTanhF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOLogF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOExpF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOPowF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSumF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMulF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTranspose2DF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTransposeF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCopyAAF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCopySAF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOBroadcastToF32{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOReduceToF32{nullptr};
    std::vector<MTL::Buffer*>    m_tempBuffers;
    std::unordered_map<const void*, MTL::Buffer*>  m_allocMap;
    size_t   m_currentBatchSize{0};
    size_t   m_maxBatchSize{0};
};

}   // namespace
