//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Project includes
#include "aixDeviceMetal.hpp"
#include "aixDeviceMetalShaders.hpp"
// External includes
#include <Metal/Metal.hpp>
// System includes


namespace aix
{

DeviceMetal::DeviceMetal()
{
    m_pool = NS::AutoreleasePool::alloc()->init();      // Create autorelease pool.
    m_mtlDevice = reinterpret_cast<MTL::Device*>(MTL::CopyAllDevices()->object(0));   // Get first available device.
    auto defaultLibrary = createLibrary(aix::shaders::aixDeviceMetalShaders);
    // Compile time evaluation.
    m_compFuncPSOAddF32         = createComputeFuncPSO(defaultLibrary, "add_float");
    m_compFuncPSOSubF32         = createComputeFuncPSO(defaultLibrary, "sub_float");
    m_compFuncPSOMulF32         = createComputeFuncPSO(defaultLibrary, "mul_float");
    m_compFuncPSODivF32         = createComputeFuncPSO(defaultLibrary, "div_float");
    m_compFuncPSOAddASF32       = createComputeFuncPSO(defaultLibrary, "add_a_s_float");
    m_compFuncPSOSubSAF32       = createComputeFuncPSO(defaultLibrary, "sub_s_a_float");
    m_compFuncPSOMulASF32       = createComputeFuncPSO(defaultLibrary, "mul_a_s_float");
    m_compFuncPSODivASF32       = createComputeFuncPSO(defaultLibrary, "div_a_s_float");
    m_compFuncPSODivSAF32       = createComputeFuncPSO(defaultLibrary, "div_s_a_float");
    m_compFuncPSOSqrtF32        = createComputeFuncPSO(defaultLibrary, "sqrt_a_float");
    m_compFuncPSOSinF32         = createComputeFuncPSO(defaultLibrary, "sin_a_float");
    m_compFuncPSOCosF32         = createComputeFuncPSO(defaultLibrary, "cos_a_float");
    m_compFuncPSOTanhF32        = createComputeFuncPSO(defaultLibrary, "tanh_a_float");
    m_compFuncPSOLogF32         = createComputeFuncPSO(defaultLibrary, "log_a_float");
    m_compFuncPSOExpF32         = createComputeFuncPSO(defaultLibrary, "exp_a_float");
    m_compFuncPSOPowF32         = createComputeFuncPSO(defaultLibrary, "pow_float");
    m_compFuncPSOSumF32         = createComputeFuncPSO(defaultLibrary, "sum_a_float");
    m_compFuncPSOMatMulF32      = createComputeFuncPSO(defaultLibrary, "matrix_mul_float");
    m_compFuncPSOTranspose2DF32 = createComputeFuncPSO(defaultLibrary, "transpose2D_float");
    m_compFuncPSOTransposeF32   = createComputeFuncPSO(defaultLibrary, "transpose_float");
    m_compFuncPSOCopyAAF32      = createComputeFuncPSO(defaultLibrary, "copy_a_a_float");
    m_compFuncPSOCopySAF32      = createComputeFuncPSO(defaultLibrary, "copy_s_a_float");
    m_compFuncPSOBroadcastToF32 = createComputeFuncPSO(defaultLibrary, "broadcastTo_float");
    m_compFuncPSOReduceToF32    = createComputeFuncPSO(defaultLibrary, "reduceTo_float");

    m_cmdQueue = createCommandQueue();
    m_cmdBuffer = m_cmdQueue->commandBuffer();
}

// Destructor
DeviceMetal::~DeviceMetal()
{
    m_cmdQueue->release();
    m_compFuncPSOAddF32->release();
    m_compFuncPSOSubF32->release();
    m_compFuncPSOMulF32->release();
    m_compFuncPSODivF32->release();
    m_compFuncPSOAddASF32->release();
    m_compFuncPSOSubSAF32->release();
    m_compFuncPSOMulASF32->release();
    m_compFuncPSODivASF32->release();
    m_compFuncPSODivSAF32->release();
    m_compFuncPSOSqrtF32->release();
    m_compFuncPSOSinF32->release();
    m_compFuncPSOCosF32->release();
    m_compFuncPSOTanhF32->release();
    m_compFuncPSOLogF32->release();
    m_compFuncPSOExpF32->release();
    m_compFuncPSOPowF32->release();
    m_compFuncPSOSumF32->release();
    m_compFuncPSOMatMulF32->release();
    m_compFuncPSOTranspose2DF32->release();
    m_compFuncPSOTransposeF32->release();
    m_compFuncPSOCopyAAF32->release();
    m_compFuncPSOCopySAF32->release();
    m_compFuncPSOBroadcastToF32->release();
    m_compFuncPSOReduceToF32->release();
    m_mtlDevice->release();
    // No need to release MTL Buffer objects in m_allocMap.
    m_pool->release();
}

// Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
void* DeviceMetal::allocate(size_t size)
{
    auto mtlBuf = newBuffer(size);
    auto contentPtr = mtlBuf->contents();
    m_allocMap[contentPtr] = mtlBuf;
    return contentPtr;
}

void* DeviceMetal::allocate(size_t size, DataType dtype)
{
    return allocate(size * dataTypeSize(dtype));
}

// Deallocate GPU memory if it's allocated by current device.
void DeviceMetal::deallocate(void * memory)
{
    if (m_allocMap.find(memory) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::deallocate() - Found different type of memory to free.");
    auto mtlBuf = m_allocMap[memory];
    m_allocMap.erase(memory);
    mtlBuf->release();
}

void DeviceMetal::add(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::addF32,
    };
    // Call the appropriate function from the table.
    (this->*funcTable[static_cast<size_t>(dtype)])(a1, a2, size, result);
}

void DeviceMetal::sub(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::subF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a1, a2, size, result);
}

void DeviceMetal::mul(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::mulF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a1, a2, size, result);
}

void DeviceMetal::div(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::divF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a1, a2, size, result);
}

void DeviceMetal::addAS(const void* a1, const void* scalar, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::addASF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a1, scalar, size, result);
}

void DeviceMetal::subAS(const void* a1, const void* scalar, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::subASF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a1, scalar, size, result);
}

void DeviceMetal::subSA(const void* scalar, const void* a1, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::subSAF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(scalar, a1, size, result);
}

void DeviceMetal::mulAS(const void* a, const void* scalar, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::mulASF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, scalar, size, result);
}

void DeviceMetal::divAS(const void* a, const void* scalar, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::divASF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, scalar, size, result);
}

void DeviceMetal::divSA(const void* scalar, const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::divSAF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(scalar, a, size, result);
}

void DeviceMetal::unary(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::unaryF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::fill(const void* scalar, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::fillF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(scalar, size, result);
}

void DeviceMetal::sum(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::sumF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::mean(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::meanF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::sqrt(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::sqrtF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::sin(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::sinF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::cos(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::cosF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::tanh(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::tanhF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::log(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::logF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::exp(const void* a, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::expF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, size, result);
}

void DeviceMetal::pow(const void* a, const void* exp, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::powF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a, exp, size, result);
}

void DeviceMetal::matmul(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::matmulF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(a1, s1, a2, s2, result);
}

void DeviceMetal::transpose(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                            const Stride& strides, const Stride& newStrides, size_t size, void* result, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::transposeF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(dim0, dim1, data, shape, strides, newStrides, size, result);
}

void DeviceMetal::copy(const void* src, void* dst, size_t size, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::copyF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(src, dst, size);
}

void DeviceMetal::copyImmediate(const void* src, void* dst, size_t size, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::copyImmediateF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(src, dst, size);
}

void DeviceMetal::broadcastTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::broadcastToF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(src, dst, size, shape, newShape);
}

void DeviceMetal::reduceTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype)
{
    static const auto funcTable = std::array
    {
        &DeviceMetal::reduceToF32,
    };
    // Call the appropriate function from the table.
    return (this->*funcTable[static_cast<size_t>(dtype)])(src, dst, size, shape, newShape);
}

void DeviceMetal::addF32(const void* a1, const void* a2, const size_t size, void* result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOAddF32, "addFloat");
}

void DeviceMetal::subF32(const void* a1, const void* a2, size_t size, void* result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOSubF32, "subFloat");
}

void DeviceMetal::mulF32(const void* a1, const void* a2, size_t size, void* result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOMulF32, "mulFloat");
}

void DeviceMetal::divF32(const void* a1, const void* a2, size_t size, void* result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSODivF32, "divFloat");
}

void DeviceMetal::addASF32(const void* a, const void* scalar, size_t size, void* result)
{
    executeArrayScalarCmd(a, *(float*)scalar, size, result, m_compFuncPSOAddASF32, "addASFloat");
}

void DeviceMetal::subASF32(const void* a1, const void* scalar, size_t size, void* result)
{
    executeArrayScalarCmd(a1, -(*(float*)scalar), size, result, m_compFuncPSOAddASF32, "subASFloat");
}

void DeviceMetal::subSAF32(const void* scalar, const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, *(float*)scalar, size, result, m_compFuncPSOSubSAF32, "subSAFloat");
}

void DeviceMetal::mulASF32(const void* a, const void* scalar, size_t size, void* result)
{
    executeArrayScalarCmd(a, *(float*)scalar, size, result, m_compFuncPSOMulASF32, "mulASFloat");
}

void DeviceMetal::divASF32(const void* a, const void* scalar, size_t size, void* result)
{
    executeArrayScalarCmd(a, *(float*)scalar, size, result, m_compFuncPSODivASF32, "divASFloat");
}

void DeviceMetal::divSAF32(const void* scalar, const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, *(float*)scalar, size, result, m_compFuncPSODivSAF32, "divSAFloat");
}

void DeviceMetal::unaryF32(const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, -1.0f, size, result, m_compFuncPSOMulASF32, "unaryFloat");
}

void DeviceMetal::fillF32(const void* scalar, size_t size, void* result)
{
    executeArrayScalarCmd(nullptr, *(float*)scalar, size, result, m_compFuncPSOCopySAF32, "copyFloat");
}

void DeviceMetal::sumF32(const void* a, size_t size, void* result)
{
    size_t maxThreadsPerTG = std::min<size_t>(MAX_THREADS_PER_THREADGROUP,
                                              m_compFuncPSOSumF32->maxTotalThreadsPerThreadgroup());

    auto buf1      = getReadOnlyMTLBuffer(a, size, sizeof(float));
    auto bufResult = newBuffer((1 + size / maxThreadsPerTG) * sizeof(float));
    auto bufRec = buf1;     // Recursive data buffer pointer.

    // Apply Parallel Reduction Sum.
    size_t length = size - 1;
    while (length > 0)
    {
        // Calculate maximum thread group dimensions.
        NS::UInteger w = std::min<size_t>(length+1, maxThreadsPerTG);
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        sendComputeCommandArrayScalar(bufRec, {1, length+1}, 0, bufResult,
                                      m_compFuncPSOSumF32, {length + 1, 1, 1}, {w, 1, 1});
        length = (length-1) / maxThreadsPerTG;
        bufRec = bufResult;
    }

    // Buf1 could be temporary buffer. It should be deleted if temporary.
    freeTemporaryBuffer(buf1);
    commitAndWait();

    *(float*)result = static_cast<float*>(bufResult->contents())[0];  // Read the final result.
    bufResult->release();       // Release temporary buffer.
}

void DeviceMetal::meanF32(const void* a, size_t size, void* result)
{
    sumF32(a, size, result);
    *(float*)result /= size;
}

void DeviceMetal::sqrtF32(const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSqrtF32, "sqrtFloat");
}

void DeviceMetal::sinF32(const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSinF32, "sinFloat");
}

void DeviceMetal::cosF32(const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOCosF32, "cosFloat");
}

void DeviceMetal::tanhF32(const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOTanhF32, "tanhFloat");
}

void DeviceMetal::logF32(const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOLogF32, "logFloat");
}

void DeviceMetal::expF32(const void* a, size_t size, void* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOExpF32, "expFloat");
}

void DeviceMetal::powF32(const void* a, const void* exp, size_t size, void* result)
{
    executeDoubleArrayCmd(a, exp, size, result, m_compFuncPSOPowF32, "powFloat");
}

void DeviceMetal::matmulF32(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::matmul() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a1, s1[0] * s1[1], sizeof(float));
    auto buf2 = getReadOnlyMTLBuffer(a2, s2[0] * s2[1], sizeof(float));
    auto bufResult = m_allocMap[result];

    // Calculate maximum thread group dimensions
    NS::UInteger w = m_compFuncPSOMatMulF32->threadExecutionWidth();
    NS::UInteger h = m_compFuncPSOMatMulF32->maxTotalThreadsPerThreadgroup() / w;
    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandDoubleBuffer(buf1, {s1[0], s1[1]}, buf2, {s2[0], s2[1]}, bufResult,
                                   m_compFuncPSOMatMulF32, {s2[1], s1[0], 1}, {w, h, 1});

    freeTemporaryBuffer(buf1);
    freeTemporaryBuffer(buf2);
}

void DeviceMetal::transposeF32(size_t dim0, size_t dim1, const void* data, const Shape& shape,
                                 const Stride& strides, const Stride& newStrides, size_t size, void* result)
{
    // Use fast and simplified version of the general transpose for matrix transpose operations.
    if (shape.size() == 2 && dim0 == 0 && dim1 == 1)
    {
        transpose2DF32(data, shape, result);
        return;
    }

    if (strides.size() > 16)
        throw std::invalid_argument("Metal device does not support tensors with more than 16 dimensions for acceleration.");

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto bufData       = getReadOnlyMTLBuffer(data, size, sizeof(float));
    auto bufResult     = m_allocMap[result];
    auto bufStrides    = getReadOnlyMTLBuffer(strides.data(), strides.size(), sizeof(size_t));
    size_t stridesSize = strides.size();
    auto bufNewStrides = getReadOnlyMTLBuffer(newStrides.data(), newStrides.size(), sizeof(size_t));
    size_t newStridesSize = newStrides.size();

    if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(m_compFuncPSOTransposeF32);
    m_compEncoder->setBuffer(bufData,        0,                       0);
    m_compEncoder->setBuffer(bufResult,      0,                       1);
    m_compEncoder->setBytes(&dim0,           sizeof(dim0),            2);
    m_compEncoder->setBytes(&dim1,           sizeof(dim1),            3);
    m_compEncoder->setBuffer(bufStrides,     0,                       4);
    m_compEncoder->setBytes(&stridesSize,    sizeof(stridesSize),     5);
    m_compEncoder->setBuffer(bufNewStrides,  0,                       6);
    m_compEncoder->setBytes(&newStridesSize, sizeof(newStridesSize),  7);
    m_compEncoder->setBytes(&size,           sizeof(size),            8);

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(size, m_compFuncPSOTransposeF32->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({size, 1, 1}, {w, 1, 1});

    m_currentBatchSize++;
    if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();

    freeTemporaryBuffer(bufData);
    freeTemporaryBuffer(bufResult);
    freeTemporaryBuffer(bufStrides);
    freeTemporaryBuffer(bufNewStrides);
}

void DeviceMetal::copyF32(const void* src, void* dst, size_t size)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(dst) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::copy() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(src, size, sizeof(float));
    auto bufResult = m_allocMap[dst];

    // Calculate maximum thread group dimensions
    auto asize = align(size, ALIGNMENT_SIZE);
    NS::UInteger w = std::min(asize, m_compFuncPSOCopyAAF32->maxTotalThreadsPerThreadgroup()) / ALIGNMENT_SIZE;
    asize /= ALIGNMENT_SIZE;

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandSingleBuffer(buf1, {1, size}, bufResult, m_compFuncPSOCopyAAF32, {asize, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(buf1);
}

void DeviceMetal::copyImmediateF32(const void* src, void* dst, size_t size)
{
    copyF32(src, dst, size);
    commitAndWait();
}

void DeviceMetal::broadcastToF32(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape)
{
    translationF32(src, dst, size, shape, newShape, m_compFuncPSOBroadcastToF32, "broadcastToFloat");
}

void DeviceMetal::reduceToF32(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape)
{
    translationF32(src, dst, size, shape, newShape, m_compFuncPSOReduceToF32, "reduceToFloat");
    // NOTE: The ReduceTo function performs a sum operation. The order of these operations by GPU threads is not
    // guaranteed, which might result in minor differences in the final results due to floating-point precision limits.
}

void DeviceMetal::commitAndWait()
{
    // Execute only if there is at least one command encoded.
    if (!m_compEncoder) return;

    m_compEncoder->endEncoding();
    m_cmdBuffer->commit();                // Execute the command
    m_cmdBuffer->waitUntilCompleted();    // Wait until the work is done

    // Release all temporary buffers.
    for (auto buf : m_tempBuffers)
        buf->release();

    m_tempBuffers.clear();
    m_tempBuffers.reserve(MAX_CMD_BATCH_SIZE * 2);

    m_cmdBuffer = m_cmdQueue->commandBuffer();
    m_compEncoder = nullptr;

    // Update batch size metrics.
    m_maxBatchSize = std::max(m_currentBatchSize, m_maxBatchSize);
    m_currentBatchSize = 0;
}


MTL::Buffer* DeviceMetal::newBuffer(size_t size)
{
    assert(size > 0);
    auto buffer = m_mtlDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    assert(buffer);
    return buffer;
}


MTL::Buffer* DeviceMetal::newBufferWithAddress(const void* address, size_t size)
{
    assert(size > 0);
    auto buffer = m_mtlDevice->newBuffer(address, size, MTL::ResourceStorageModeShared);
    assert(buffer);
    return buffer;
}


MTL::Buffer* DeviceMetal::getReadOnlyMTLBuffer(const void * address, size_t size, size_t sizeofType)
{
    // Memory could be from other devices. Create a temporary buffer for read only case.
    if (m_allocMap.find(address) == m_allocMap.end())
    {
        size = align(size, ALIGNMENT_SIZE);
        return newBufferWithAddress(address, size * sizeofType);
    }

    return m_allocMap[address];    // Return MTL Buffer if the memory is from the current device.
}


void DeviceMetal::freeTemporaryBuffer(MTL::Buffer * buffer)
{
    // Release only temporary buffer.
    if (m_allocMap.find(buffer->contents()) == m_allocMap.end())
    {
        // Add the buffer to the list to be released when commitAndWait() is executed.
        // Until then, the buffer could be in use, especially when a batch command is used.
        m_tempBuffers.emplace_back(buffer);
    }
}

MTL::Library* DeviceMetal::createLibrary(const char* shaders)
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

MTL::CommandQueue* DeviceMetal::createCommandQueue()
{
    auto cmdQueue = m_mtlDevice->newCommandQueue();
    if (!cmdQueue)
    {
        std::cerr << "Failed to create command queue.\n";
        exit(-1);
    }

    return cmdQueue;
}

MTL::ComputePipelineState* DeviceMetal::createComputeFuncPSO(MTL::Library* library, const std::string & kernelName)
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

void DeviceMetal::encodeComputeCommandSingleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size, MTL::Buffer* bufResult,
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

void DeviceMetal::encodeComputeCommandDoubleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
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

void DeviceMetal::encodeComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
                                                  float scalar, MTL::Buffer* bufResult,
                                                  MTL::ComputeCommandEncoder* computeEncoder,
                                                  const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                                  const MTL::Size & threadsPerTG) const
{
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(compFuncPSO);
    computeEncoder->setBuffer(buf1, 0, 0);
    computeEncoder->setBytes(&scalar, sizeof(float), 1);
    computeEncoder->setBytes(&buf1Size, sizeof(MatrixSize), 2);
    computeEncoder->setBuffer(bufResult, 0, 3);
    computeEncoder->dispatchThreads(gridSize, threadsPerTG);
}

void DeviceMetal::sendComputeCommandSingleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size, MTL::Buffer* bufResult,
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

void DeviceMetal::sendComputeCommandDoubleBuffer(const MTL::Buffer* buf1, const MatrixSize& buf1Size,
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

void DeviceMetal::sendComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size, float scalar,
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

void DeviceMetal::executeArrayScalarCmd(const void* a,
                                        float scalar,
                                        size_t size,
                                        void* result,
                                        const MTL::ComputePipelineState* compFuncPSO,
                                        const std::string & cmdName)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

    // Set constants
    auto buf1      = a ? getReadOnlyMTLBuffer(a, size, sizeof(float)) : nullptr;
    auto bufResult = m_allocMap[result];

    // Calculate maximum thread group dimensions
    auto asize = align(size, ALIGNMENT_SIZE);
    NS::UInteger w = std::min(asize, compFuncPSO->maxTotalThreadsPerThreadgroup()) / ALIGNMENT_SIZE;
    asize /= ALIGNMENT_SIZE;
    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandArrayScalar(buf1, {1, size}, scalar, bufResult, compFuncPSO, {asize, 1, 1}, {w, 1, 1});

    if (a)
    {
        // Buf1 could be temporary buffer. It should be deleted if temporary.
        freeTemporaryBuffer(buf1);
        // Note: We never release result buffer since it will be used.
    }
}

void DeviceMetal::executeDoubleArrayCmd(const void* a1,
                                        const void* a2,
                                        size_t size,
                                        void* result,
                                        const MTL::ComputePipelineState* compFuncPSO,
                                        const std::string & cmdName)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a1, size, sizeof(float));
    auto buf2 = getReadOnlyMTLBuffer(a2, size, sizeof(float));
    auto bufResult = m_allocMap[result];

    // Calculate maximum thread group dimensions
    auto asize = align(size, ALIGNMENT_SIZE);
    NS::UInteger w = std::min(asize, compFuncPSO->maxTotalThreadsPerThreadgroup()) / ALIGNMENT_SIZE;
    asize /= ALIGNMENT_SIZE;
    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandDoubleBuffer(buf1, {0, 0}, buf2, {0, 0}, bufResult, compFuncPSO, {asize, 1, 1}, {w, 1, 1});

    // Buf 1 and 2 could be temporary buffer. It should be deleted if temporary.
    freeTemporaryBuffer(buf1);
    freeTemporaryBuffer(buf2);
    // Note: We never release result buffer since it will be used.
}

void DeviceMetal::translationF32(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                                 const MTL::ComputePipelineState* computePSO, const std::string & name)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(dst) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::" + name + "() result must have GPU memory.");

    size_t shapeSize    = shape.size();
    size_t newShapeSize = newShape.size();
    size_t srcBufSize   = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    assert(srcBufSize > 0);

    // Memory could be a GPU allocated memory or system memory.
    auto bufSrc    = getReadOnlyMTLBuffer(src,             srcBufSize,   sizeof(float));
    auto bufShape1 = getReadOnlyMTLBuffer(shape.data(),    shapeSize,    sizeof(size_t));
    auto bufShape2 = getReadOnlyMTLBuffer(newShape.data(), newShapeSize, sizeof(size_t));
    auto bufDst    = m_allocMap[dst];

    if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(computePSO);
    m_compEncoder->setBuffer(bufSrc,    0, 0);
    m_compEncoder->setBuffer(bufDst,    0, 1);
    m_compEncoder->setBuffer(bufShape1, 0, 2);
    m_compEncoder->setBuffer(bufShape2, 0, 3);
    m_compEncoder->setBytes(&shapeSize,    sizeof(shapeSize),    4);
    m_compEncoder->setBytes(&newShapeSize, sizeof(newShapeSize), 5);

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(size, computePSO->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({size, 1, 1}, {w, 1, 1});

    m_currentBatchSize++;
    if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();

    freeTemporaryBuffer(bufSrc);
    freeTemporaryBuffer(bufShape1);
    freeTemporaryBuffer(bufShape2);
}

void DeviceMetal::transpose2DF32(const void* mat, const Shape& shape, void* result)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::transpose2DF32() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(mat, shape[0] * shape[1], sizeof(float));
    auto bufResult = m_allocMap[result];

    // Calculate maximum thread group dimensions
    NS::UInteger w = m_compFuncPSOTranspose2DF32->threadExecutionWidth();
    NS::UInteger h = m_compFuncPSOTranspose2DF32->maxTotalThreadsPerThreadgroup() / w;

    sendComputeCommandSingleBuffer(buf1, {shape[0], shape[1]}, bufResult,
                                   m_compFuncPSOTranspose2DF32, {shape[0], shape[1], 1}, {w, h, 1});

    freeTemporaryBuffer(buf1);
}

}   // namespace
