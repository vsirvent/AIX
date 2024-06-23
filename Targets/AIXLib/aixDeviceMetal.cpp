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
        m_compFuncPSOPow          = createComputeFuncPSO(defaultLibrary, "pow_float");
        m_compFuncPSOSum          = createComputeFuncPSO(defaultLibrary, "sum_a_float");
        m_compFuncPSOMatMul       = createComputeFuncPSO(defaultLibrary, "matrix_mul_float");
        m_compFuncPSOMatTranspose2D = createComputeFuncPSO(defaultLibrary, "transpose2D_float");
        m_compFuncPSOMatTranspose = createComputeFuncPSO(defaultLibrary, "transpose_float");
        m_compFuncPSOCopy_A_A     = createComputeFuncPSO(defaultLibrary, "copy_a_a_float");
        m_compFuncPSOCopy_S_A     = createComputeFuncPSO(defaultLibrary, "copy_s_a_float");
        m_compFuncPSOBroadcastTo  = createComputeFuncPSO(defaultLibrary, "broadcastTo_float");
        m_compFuncPSOReduceTo     = createComputeFuncPSO(defaultLibrary, "reduceTo_float");
    }
    else
        throw std::invalid_argument("Metal device supports only float data type for now.");

    m_cmdQueue = createCommandQueue();
    m_cmdBuffer = m_cmdQueue->commandBuffer();
}

// Destructor
DeviceMetal::~DeviceMetal()
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
    m_compFuncPSOPow->release();
    m_compFuncPSOSum->release();
    m_compFuncPSOMatMul->release();
    m_compFuncPSOMatTranspose2D->release();
    m_compFuncPSOMatTranspose->release();
    m_compFuncPSOCopy_A_A->release();
    m_compFuncPSOCopy_S_A->release();
    m_compFuncPSOBroadcastTo->release();
    m_compFuncPSOReduceTo->release();
    m_mtlDevice->release();
    // No need to release MTL Buffer objects in m_allocMap.
    m_pool->release();
}

// Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
void * DeviceMetal::allocate(size_t size)
{
    // Size in bytes.
    size = align(size, ALIGNMENT_SIZE * sizeof(DataType));
    // Allocate GPU memory and save the mtl buffer to be used later.
    auto mtlBuf = newBuffer(size);
    auto contentPtr = mtlBuf->contents();
    m_allocMap[contentPtr] = mtlBuf;
    return contentPtr;
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

void DeviceMetal::add(const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOAdd, "add");
}

void DeviceMetal::sub(const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOSub, "sub");
}

void DeviceMetal::mul(const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOMul, "mul");
}

void DeviceMetal::div(const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSODiv, "div");
}

void DeviceMetal::add(const DataType * a, DataType scalar, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSOAdd_A_S, "add");
}

void DeviceMetal::sub(const DataType * a, DataType scalar, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, -scalar, size, result, m_compFuncPSOAdd_A_S, "sub");
}

void DeviceMetal::sub(DataType scalar, const DataType * a, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSOSub_S_A, "sub");
}

void DeviceMetal::mul(const DataType * a, DataType scalar, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSOMul_A_S, "mul");
}

void DeviceMetal::div(const DataType * a, DataType scalar, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSODiv_A_S, "div");
}

void DeviceMetal::div(DataType scalar, const DataType * a, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, scalar, size, result, m_compFuncPSODiv_S_A, "div");
}

void DeviceMetal::unary(const DataType * a, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, DataType(-1), size, result, m_compFuncPSOMul_A_S, "unary");
}

void DeviceMetal::fill(DataType scalar, size_t size, DataType * result)
{
    executeArrayScalarCmd(nullptr, scalar, size, result, m_compFuncPSOCopy_S_A, "copy");
}

void DeviceMetal::sum(const DataType * a, size_t size, DataType* result)
{
    size_t maxThreadsPerTG = std::min<size_t>(MAX_THREADS_PER_THREADGROUP,
                                              m_compFuncPSOSum->maxTotalThreadsPerThreadgroup());

    auto buf1      = getReadOnlyMTLBuffer(a, size, sizeof(DataType));
    auto bufResult = newBuffer((1 + size / maxThreadsPerTG) * sizeof(DataType));
    auto bufRec = buf1;     // Recursive data buffer pointer.

    // Apply Parallel Reduction Sum.
    size_t length = size - 1;
    while (length > 0)
    {
        // Calculate maximum thread group dimensions.
        NS::UInteger w = std::min<size_t>(length+1, maxThreadsPerTG);
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        sendComputeCommandArrayScalar(bufRec, {1, length+1}, 0, bufResult,
                                      m_compFuncPSOSum, {length+1, 1, 1}, {w, 1, 1});
        length = (length-1) / maxThreadsPerTG;
        bufRec = bufResult;
    }

    // Buf1 could be temporary buffer. It should be deleted if temporary.
    freeTemporaryBuffer(buf1);
    commitAndWait();

    *result = static_cast<DataType*>(bufResult->contents())[0];  // Read the final result.
    bufResult->release();       // Release temporary buffer.
}

void DeviceMetal::mean(const DataType * a, size_t size, DataType* result)
{
    sum(a, size, result);
    *result /= DataType(size);
}

void DeviceMetal::sqrt(const DataType * a, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSqrt, "sqrt");
}

void DeviceMetal::sin(const DataType * a, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSin, "sin");
}

void DeviceMetal::cos(const DataType * a, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOCos, "cos");
}

void DeviceMetal::tanh(const DataType * a, size_t size, DataType * result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOTanh, "tanh");
}

void DeviceMetal::log(const DataType* a, size_t size, DataType* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOLog, "log");
}

void DeviceMetal::exp(const DataType* a, size_t size, DataType* result)
{
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOExp, "exp");
}

void DeviceMetal::pow(const DataType* a, const DataType* exp, size_t size, DataType* result)
{
    executeDoubleArrayCmd(a, exp, size, result, m_compFuncPSOPow, "pow");
}

void DeviceMetal::matmul(const DataType * a1, const Shape & s1, const DataType * a2, const Shape & s2, DataType * result)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::matmul() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a1, s1[0] * s1[1], sizeof(DataType));
    auto buf2 = getReadOnlyMTLBuffer(a2, s2[0] * s2[1], sizeof(DataType));
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

void DeviceMetal::transpose(size_t dim0, size_t dim1, const DataType* data, const Shape& shape,
                            const Stride& strides, const Stride& newStrides, size_t size, DataType* result)
{
    // Use fast and simplified version of the general transpose for matrix transpose operations.
    if (shape.size() == 2 && dim0 == 0 && dim1 == 1)
    {
        transpose2D(data, shape, result);
        return;
    }

    if (strides.size() > 16)
        throw std::invalid_argument("Metal device does not support tensors with more than 16 dimensions for acceleration.");

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto bufData       = getReadOnlyMTLBuffer(data, size, sizeof(DataType));
    auto bufResult     = m_allocMap[result];
    auto bufStrides    = getReadOnlyMTLBuffer(strides.data(), strides.size(), sizeof(size_t));
    size_t stridesSize = strides.size();
    auto bufNewStrides = getReadOnlyMTLBuffer(newStrides.data(), newStrides.size(), sizeof(size_t));
    size_t newStridesSize = newStrides.size();

    if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(m_compFuncPSOMatTranspose);
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
    NS::UInteger w = std::min(size, m_compFuncPSOMatTranspose->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({size, 1, 1}, {w, 1, 1});

    m_currentBatchSize++;
    if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();

    freeTemporaryBuffer(bufData);
    freeTemporaryBuffer(bufResult);
    freeTemporaryBuffer(bufStrides);
    freeTemporaryBuffer(bufNewStrides);
}

void DeviceMetal::copy(const DataType * src, DataType * dst, size_t size)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(dst) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::copy() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(src, size, sizeof(DataType));
    auto bufResult = m_allocMap[dst];

    // Calculate maximum thread group dimensions
    auto asize = align(size, ALIGNMENT_SIZE);
    NS::UInteger w = std::min(asize, m_compFuncPSOCopy_A_A->maxTotalThreadsPerThreadgroup()) / ALIGNMENT_SIZE;
    asize /= ALIGNMENT_SIZE;

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandSingleBuffer(buf1, {1, size}, bufResult, m_compFuncPSOCopy_A_A, {asize, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(buf1);
}

void DeviceMetal::copyImmediate(const DataType* src, DataType* dst, size_t size)
{
    copy(src, dst, size);
    commitAndWait();
}

void DeviceMetal::broadcastTo(const DataType* src, DataType* dst, size_t size, const Shape& shape, const Shape& newShape)
{
    translation(src, dst, size, shape, newShape, m_compFuncPSOBroadcastTo, "broadcastTo");
}

void DeviceMetal::reduceTo(const DataType* src, DataType* dst, size_t size, const Shape& shape, const Shape& newShape)
{
    translation(src, dst, size, shape, newShape, m_compFuncPSOReduceTo, "reduceTo");
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
                                                  DataType scalar, MTL::Buffer* bufResult,
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

void DeviceMetal::sendComputeCommandArrayScalar(const MTL::Buffer* buf1, const MatrixSize& buf1Size, DataType scalar,
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

void DeviceMetal::executeArrayScalarCmd(const DataType * a,
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
    auto buf1      = a ? getReadOnlyMTLBuffer(a, size, sizeof(DataType)) : nullptr;
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

void DeviceMetal::executeDoubleArrayCmd(const DataType * a1,
                                        const DataType * a2,
                                        size_t size,
                                        DataType * result,
                                        const MTL::ComputePipelineState* compFuncPSO,
                                        const std::string & cmdName)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a1, size, sizeof(DataType));
    auto buf2 = getReadOnlyMTLBuffer(a2, size, sizeof(DataType));
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

void DeviceMetal::translation(const DataType* src, DataType* dst, size_t size, const Shape& shape, const Shape& newShape,
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
    auto bufSrc    = getReadOnlyMTLBuffer(src,             srcBufSize,   sizeof(DataType));
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

void DeviceMetal::transpose2D(const DataType * mat, const Shape& shape, DataType * result)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (m_allocMap.find(result) == m_allocMap.end())
        throw std::invalid_argument("DeviceMetal::transpose2D() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(mat, shape[0] * shape[1], sizeof(DataType));
    auto bufResult = m_allocMap[result];

    // Calculate maximum thread group dimensions
    NS::UInteger w = m_compFuncPSOMatTranspose2D->threadExecutionWidth();
    NS::UInteger h = m_compFuncPSOMatTranspose2D->maxTotalThreadsPerThreadgroup() / w;

    sendComputeCommandSingleBuffer(buf1, {shape[0], shape[1]}, bufResult,
                                   m_compFuncPSOMatTranspose2D, {shape[0], shape[1], 1}, {w, h, 1});

    freeTemporaryBuffer(buf1);
}

}   // namespace
