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
#include <array>


namespace aix
{

DeviceMetal::DeviceMetal(size_t deviceIndex)
{
    // Create autorelease pool.
    m_pool = NS::AutoreleasePool::alloc()->init();
    m_mtlDevice = createMTLDevice(deviceIndex);
    m_maxWorkingSetSize = static_cast<size_t>(static_cast<double>(m_mtlDevice->recommendedMaxWorkingSetSize()) * 0.8);
    auto defaultLibrary = createLibrary(aix::shaders::aixDeviceMetalShaders);
    auto nullKernelName = "nullKernel";

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto iDType = static_cast<DataType>(i);
        for (size_t j=0; j<aix::DataTypeCount; ++j)
        {
            auto jDType = static_cast<DataType>(j);
            // Metal Framework does not support kFloat64 format.
            bool isNull = iDType == DataType::kFloat64 || jDType == DataType::kFloat64;
            std::string kernelName = "copy_aa_" + toString(i) + "_" + toString(j);
            m_compFuncPSOCopyAA[i][j] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : kernelName);
            kernelName = "fill_aa_" + toString(i) + "_" + toString(j);
            m_compFuncPSOFill[i][j]   = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : kernelName);
        }

        // Metal Framework does not support kFloat64 format.
        bool isNull = iDType == DataType::kFloat64;
        std::string dtypeStr = toString(i);
        m_compFuncPSOAdd[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "add_aa_" + dtypeStr);
        m_compFuncPSOSub[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sub_aa_" + dtypeStr);
        m_compFuncPSOMul[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "mul_aa_" + dtypeStr);
        m_compFuncPSODiv[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "div_aa_" + dtypeStr);
        m_compFuncPSOUnary[i]       = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "unary_a_" + dtypeStr);
        m_compFuncPSOSqrt[i]        = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sqrt_a_" + dtypeStr);
        m_compFuncPSOSin[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sin_a_" + dtypeStr);
        m_compFuncPSOCos[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "cos_a_" + dtypeStr);
        m_compFuncPSOTanh[i]        = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "tanh_a_" + dtypeStr);
        m_compFuncPSOLog[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "log_a_" + dtypeStr);
        m_compFuncPSOExp[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "exp_a_" + dtypeStr);
        m_compFuncPSOPow[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "pow_aa_" + dtypeStr);
        m_compFuncPSOSum[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sum_a_" + dtypeStr);
        m_compFuncPSOMatMul[i]      = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "matrixMul_aa_" + dtypeStr);
        m_compFuncPSOTranspose2D[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "transpose2D_a_" + dtypeStr);
        m_compFuncPSOTranspose[i]   = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "transpose_a_" + dtypeStr);
        m_compFuncPSOBroadcastTo[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "broadcastTo_a_" + dtypeStr);
        m_compFuncPSOReduceTo[i]    = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "reduceTo_a_" + dtypeStr);
    }

    m_cmdQueue = createCommandQueue();
    m_cmdBuffer = m_cmdQueue->commandBuffer();
}

// Destructor
DeviceMetal::~DeviceMetal()
{
    // Note: No need to release MTL Buffer objects in m_allocMap.

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        for (size_t j=0; j<aix::DataTypeCount; ++j)
        {
            m_compFuncPSOCopyAA[i][j]->release();
            m_compFuncPSOFill[i][j]->release();
        }
        m_compFuncPSOAdd[i]->release();
        m_compFuncPSOSub[i]->release();
        m_compFuncPSOMul[i]->release();
        m_compFuncPSODiv[i]->release();
        m_compFuncPSOUnary[i]->release();
        m_compFuncPSOSqrt[i]->release();
        m_compFuncPSOSin[i]->release();
        m_compFuncPSOCos[i]->release();
        m_compFuncPSOTanh[i]->release();
        m_compFuncPSOLog[i]->release();
        m_compFuncPSOExp[i]->release();
        m_compFuncPSOPow[i]->release();
        m_compFuncPSOSum[i]->release();
        m_compFuncPSOMatMul[i]->release();
        m_compFuncPSOTranspose2D[i]->release();
        m_compFuncPSOTranspose[i]->release();
        m_compFuncPSOBroadcastTo[i]->release();
        m_compFuncPSOReduceTo[i]->release();
    }

    m_cmdQueue->release();
    m_mtlDevice->release();
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
    if (!isDeviceBuffer(memory))
        throw std::invalid_argument("DeviceMetal::deallocate() - Found different type of memory to free.");
    auto mtlBuf = m_allocMap[memory];
    m_allocMap.erase(memory);
    // IMPORTANT: Delay all deallocations of device buffers until all commands in the batch queue are executed.
    m_tempBuffers.emplace_back(mtlBuf);
}

void DeviceMetal::add(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOAdd[iDType], dtype, "add_" + toString(dtype));
}

void DeviceMetal::sub(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOSub[iDType], dtype, "sub_" + toString(dtype));
}

void DeviceMetal::mul(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSOMul[iDType], dtype, "mul_" + toString(dtype));
}

void DeviceMetal::div(const void* a1, const void* a2, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeDoubleArrayCmd(a1, a2, size, result, m_compFuncPSODiv[iDType], dtype, "div_" + toString(dtype));
}

void DeviceMetal::unary(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOUnary[iDType], dtype, "unary_" + toString(dtype));
}

void DeviceMetal::fill(const void* scalar, DataType srcDType, size_t size, void* result, DataType dstDType)
{
    validateDataType(srcDType);
    validateDataType(dstDType);
    auto iSrcDType = static_cast<size_t>(srcDType);
    auto iDstDType = static_cast<size_t>(dstDType);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result))
        throw std::invalid_argument("DeviceMetal::fill() result must have GPU memory.");

    if (isDeviceBuffer(scalar))
        throw std::invalid_argument("DeviceMetal::fill() scalar address cannot be a device-allocated address.");

    // bufScalar is a temporary size aligned buffer to be used as vector of 4.
    auto bufScalar = newBuffer(dataTypeSize(srcDType) * 4);
    auto bufResult = m_allocMap[result];

    // Convert scalar value to a vector of 4 to be use in SIMD operation. i.e. float -> float4
    static const auto scalarToVector4FuncTable = std::array
    {
        scalarToVector4<double>,
        scalarToVector4<float>,
        scalarToVector4<float16_t>,
        scalarToVector4<bfloat16_t>,
        scalarToVector4<int64_t>,
        scalarToVector4<int32_t>,
        scalarToVector4<int16_t>,
        scalarToVector4<int8_t>,
        scalarToVector4<uint8_t>,
    };
    scalarToVector4FuncTable[static_cast<size_t>(srcDType)](scalar, bufScalar->contents());

    // Calculate maximum thread group dimensions
    auto asize = align(size, ALIGNMENT_SIZE);
    auto compFuncPSO = m_compFuncPSOFill[iSrcDType][iDstDType];
    NS::UInteger w = std::min(asize, compFuncPSO->maxTotalThreadsPerThreadgroup()) / ALIGNMENT_SIZE;
    asize /= ALIGNMENT_SIZE;

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandSingleBuffer(bufScalar, {0,0}, bufResult, compFuncPSO, {asize, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(bufScalar);
}

void DeviceMetal::sum(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    auto compFuncPSO = m_compFuncPSOSum[iDType];

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result))
        throw std::invalid_argument("DeviceMetal::sum() result must have GPU memory.");

    size_t maxThreadsPerTG = std::min<size_t>(MAX_THREADS_PER_THREADGROUP, compFuncPSO->maxTotalThreadsPerThreadgroup());

    auto buf1    = getReadOnlyMTLBuffer(a, size, dataTypeSize(dtype));
    auto bufTemp = m_allocMap[allocate(buf1->allocatedSize())];

    // TODO: Avoid the following copy if possible when changing the algorithm.
    copy(buf1->contents(), dtype, bufTemp->contents(), dtype, size);

    // Apply Parallel Reduction Sum.
    size_t length = size - 1;
    while (length > 0)
    {
        // Calculate maximum thread group dimensions.
        NS::UInteger w = std::min<size_t>(length+1, maxThreadsPerTG);
        // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
        sendComputeCommandArrayScalar(bufTemp, {1, length+1}, 0, bufTemp, compFuncPSO, {length + 1, 1, 1}, {w, 1, 1});
        length = (length - 1) / maxThreadsPerTG;
    }

    // Copy result from temp buf to result buffer.
    copy(bufTemp->contents(), dtype, result, dtype, 1);

    // Buf1 could be temporary buffer. It should be deleted if temporary.
    freeTemporaryBuffer(buf1);
    deallocate(bufTemp->contents());
}

void DeviceMetal::sqrt(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSqrt[iDType], dtype, "sqrt_" + toString(dtype));
}

void DeviceMetal::sin(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOSin[iDType], dtype, "sin_" + toString(dtype));
}

void DeviceMetal::cos(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOCos[iDType], dtype, "cos_" + toString(dtype));
}

void DeviceMetal::tanh(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOTanh[iDType], dtype, "tanh_" + toString(dtype));
}

void DeviceMetal::log(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOLog[iDType], dtype, "log_" + toString(dtype));
}

void DeviceMetal::exp(const void* a, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeArrayScalarCmd(a, 0, size, result, m_compFuncPSOExp[iDType], dtype, "exp_" + toString(dtype));
}

void DeviceMetal::pow(const void* a, const void* exp, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    executeDoubleArrayCmd(a, exp, size, result, m_compFuncPSOPow[iDType], dtype, "pow_" + toString(dtype));
}

void DeviceMetal::matmul(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result, DataType dtype)
{
    validateDataType(dtype);
    auto iDType = static_cast<size_t>(dtype);
    auto compFuncPSO = m_compFuncPSOMatMul[iDType];

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result))
        throw std::invalid_argument("DeviceMetal::matmul() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a1, s1[0] * s1[1], dataTypeSize(dtype));
    auto buf2 = getReadOnlyMTLBuffer(a2, s2[0] * s2[1], dataTypeSize(dtype));
    auto bufResult = m_allocMap[result];

    // Calculate maximum thread group dimensions
    NS::UInteger w = compFuncPSO->threadExecutionWidth();
    NS::UInteger h = compFuncPSO->maxTotalThreadsPerThreadgroup() / w;
    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandDoubleBuffer(buf1, {s1[0], s1[1]}, buf2, {s2[0], s2[1]}, bufResult,
                                   compFuncPSO, {s2[1], s1[0], 1}, {w, h, 1});

    freeTemporaryBuffer(buf1);
    freeTemporaryBuffer(buf2);
}

void DeviceMetal::transpose(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                            const Stride& strides, const Stride& newStrides, size_t size, void* result, DataType dtype)
{
    auto iDType = static_cast<size_t>(dtype);
    // Use fast and simplified version of the general transpose for matrix transpose operations.
    if (shape.size() == 2 && dim0 == 0 && dim1 == 1)
    {
        transpose2D(data, shape, result, dtype);
        return;
    }

    if (strides.size() > 16)
        throw std::invalid_argument("Metal device does not support tensors with more than 16 dimensions for acceleration.");

    validateDataType(dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result))
        throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto bufData       = getReadOnlyMTLBuffer(data, size, dataTypeSize(dtype));
    auto bufResult     = m_allocMap[result];
    auto bufStrides    = getReadOnlyMTLBuffer(strides.data(), strides.size(), sizeof(size_t));
    size_t stridesSize = strides.size();
    auto bufNewStrides = getReadOnlyMTLBuffer(newStrides.data(), newStrides.size(), sizeof(size_t));
    size_t newStridesSize = newStrides.size();

    if (!m_compEncoder) m_compEncoder = m_cmdBuffer->computeCommandEncoder();
    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(m_compFuncPSOTranspose[iDType]);
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
    NS::UInteger w = std::min(size, m_compFuncPSOTranspose[iDType]->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({size, 1, 1}, {w, 1, 1});

    m_currentBatchSize++;
    if (m_currentBatchSize >= MAX_CMD_BATCH_SIZE) commitAndWait();

    freeTemporaryBuffer(bufData);
    freeTemporaryBuffer(bufResult);
    freeTemporaryBuffer(bufStrides);
    freeTemporaryBuffer(bufNewStrides);
}

void DeviceMetal::copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
{
    validateDataType(srcDType);
    validateDataType(dstDType);
    auto iSrcDType = static_cast<size_t>(srcDType);
    auto iDstDType = static_cast<size_t>(dstDType);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst))
        throw std::invalid_argument("DeviceMetal::copy() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(src, size, dataTypeSize(srcDType));
    auto bufResult = m_allocMap[dst];

    // Calculate maximum thread group dimensions
    auto asize = align(size, ALIGNMENT_SIZE);
    auto compFuncPSO = m_compFuncPSOCopyAA[iSrcDType][iDstDType];
    NS::UInteger w = std::min(asize, compFuncPSO->maxTotalThreadsPerThreadgroup()) / ALIGNMENT_SIZE;
    asize /= ALIGNMENT_SIZE;

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    sendComputeCommandSingleBuffer(buf1, {1, size}, bufResult, compFuncPSO, {asize, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(buf1);
}

void DeviceMetal::copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
{
    copy(src, srcDType, dst, dstDType, size);
    commitAndWait();
}

void DeviceMetal::broadcastTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype)
{
    validateDataType(dtype);
    auto iDType = static_cast<size_t>(dtype);
    translation(src, dst, size, shape, newShape, m_compFuncPSOBroadcastTo[iDType], dtype, "broadcastTo_" + toString(dtype));
}

void DeviceMetal::reduceTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape, DataType dtype)
{
    validateDataType(dtype);
    // NOTE: Metal Framework supports add and sub operations for only atomic_float, atomic_uint and atomic_int.
    //       Since reduceTo uses atomic<T>, we can only allow certain formats for acceleration for now.
    if (!(dtype == DataType::kFloat32 || dtype == DataType::kInt32))
    {
        commitAndWait();
        Device::reduceTo(src, dst, size, shape, newShape, dtype);
        return;
    }

    auto iDType = static_cast<size_t>(dtype);
    translation(src, dst, size, shape, newShape, m_compFuncPSOReduceTo[iDType], dtype, "reduceTo_" + toString(dtype));
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
    m_currentWorkingSetSize = 0;
}


MTL::Buffer* DeviceMetal::newBuffer(size_t size)
{
    assert(size > 0);

    if (m_currentWorkingSetSize + size >= m_maxWorkingSetSize && m_currentBatchSize > 0)
    {
        commitAndWait();
    }

    auto buffer = m_mtlDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    m_currentWorkingSetSize += size;

    if (!buffer)
    {
        commitAndWait();
        buffer = m_mtlDevice->newBuffer(size, MTL::ResourceStorageModeShared);
        if (!buffer)
            throw std::runtime_error("GPU memory allocation has failed for size: " + std::to_string(size) + " bytes.");
    }
    assert(buffer);
    return buffer;
}


MTL::Buffer* DeviceMetal::newBufferWithAddress(const void* address, size_t size)
{
    assert(size > 0);

    if (m_currentWorkingSetSize + size >= m_maxWorkingSetSize && m_currentBatchSize > 0)
    {
        commitAndWait();
    }

    auto buffer = m_mtlDevice->newBuffer(address, size, MTL::ResourceStorageModeShared);
    m_currentWorkingSetSize += size;

    if (!buffer)
    {
        commitAndWait();
        buffer = m_mtlDevice->newBuffer(address, size, MTL::ResourceStorageModeShared);
        if (!buffer)
            throw std::runtime_error("GPU memory allocation has failed for size: " + std::to_string(size) + " bytes.");
    }
    assert(buffer);
    return buffer;
}


MTL::Buffer* DeviceMetal::getReadOnlyMTLBuffer(const void * address, size_t size, size_t sizeofType)
{
    // Memory could be from other devices. Create a temporary buffer for read only case.
    if (!isDeviceBuffer(address))
    {
        commitAndWait();
        size = align(size, ALIGNMENT_SIZE);
        return newBufferWithAddress(address, size * sizeofType);
    }

    return m_allocMap[address];    // Return MTL Buffer if the memory is from the current device.
}


void DeviceMetal::freeTemporaryBuffer(MTL::Buffer * buffer)
{
    // Release only temporary buffer.
    if (!isDeviceBuffer(buffer->contents()))
    {
        // Add the buffer to the list to be released when commitAndWait() is executed.
        // Until then, the buffer could be in use, especially when a batch command is used.
        m_tempBuffers.emplace_back(buffer);
    }
}


MTL::Device* DeviceMetal::createMTLDevice(size_t deviceIndex) const
{
    try
    {
        return reinterpret_cast<MTL::Device*>(MTL::CopyAllDevices()->object(deviceIndex));
    }
    catch (...)
    {
        throw std::invalid_argument("Device index is not supported.");
    }

    return nullptr;
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
                                        DataType dtype,
                                        const std::string & cmdName)
{
    validateDataType(dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result))
        throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

    // Set constants
    auto buf1      = a ? getReadOnlyMTLBuffer(a, size, dataTypeSize(dtype)) : nullptr;
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
                                        DataType dtype,
                                        const std::string & cmdName)
{
    validateDataType(dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result))
        throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a1, size, dataTypeSize(dtype));
    auto buf2 = getReadOnlyMTLBuffer(a2, size, dataTypeSize(dtype));
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

void DeviceMetal::translation(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                              const MTL::ComputePipelineState* computePSO, DataType dtype, const std::string & name)
{
    validateDataType(dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst))
        throw std::invalid_argument("DeviceMetal::" + name + "() result must have GPU memory.");

    // For a special case, two scalar tensors, use just a copy operation.
    if (shape.empty() && newShape.empty())
    {
        copy(src, dtype, dst, dtype, size);
        return;
    }

    size_t shapeSize    = shape.size();
    size_t newShapeSize = newShape.size();
    size_t srcBufSize   = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    assert(srcBufSize > 0);

    // NOTE: For a scalar tensor shape size could be zero.
    auto bufSrc    = getReadOnlyMTLBuffer(src, srcBufSize, dataTypeSize(dtype));
    auto bufShape1 = shapeSize    != 0 ? getReadOnlyMTLBuffer(shape.data(),    shapeSize,    sizeof(size_t)) : nullptr;
    auto bufShape2 = newShapeSize != 0 ? getReadOnlyMTLBuffer(newShape.data(), newShapeSize, sizeof(size_t)) : nullptr;
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

void DeviceMetal::transpose2D(const void* mat, const Shape& shape, void* result, DataType dtype)
{
    validateDataType(dtype);
    auto iDType = static_cast<size_t>(dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result))
        throw std::invalid_argument("DeviceMetal::transpose2D() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(mat, shape[0] * shape[1], dataTypeSize(dtype));
    auto bufResult = m_allocMap[result];

    // Calculate maximum thread group dimensions
    NS::UInteger w = m_compFuncPSOTranspose2D[iDType]->threadExecutionWidth();
    NS::UInteger h = m_compFuncPSOTranspose2D[iDType]->maxTotalThreadsPerThreadgroup() / w;

    sendComputeCommandSingleBuffer(buf1, {shape[0], shape[1]}, bufResult,
                                   m_compFuncPSOTranspose2D[iDType], {shape[0], shape[1], 1}, {w, h, 1});

    freeTemporaryBuffer(buf1);
}

const std::string& DeviceMetal::toString(size_t dtypeIndex)
{
    assert(dtypeIndex < aix::DataTypeCount);
    static std::string formatStrTable[aix::DataTypeCount] =
    {
        "f64",
        "f32",
        "f16",
        "bf16",
        "i64",
        "i32",
        "i16",
        "i8",
        "ui8",
    };
    return formatStrTable[dtypeIndex];
}

const std::string& DeviceMetal::toString(DataType dtype)
{
    return toString(static_cast<size_t>(dtype));
}

void DeviceMetal::validateDataType(DataType dtype)
{
    if (dtype == aix::DataType::kFloat64)
    {
        throw std::invalid_argument("Apple Metal Framework does not support Float64 data type.");
    }
}

}   // namespace
