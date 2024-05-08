//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#import "aixDeviceMetalShaders.hpp"
// External includes
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
// System includes
#import <string>
#import <unordered_map>

typedef float DataType;
MPSDataType mpsDataType = MPSDataTypeFloat32;

#ifdef DEBUG
    #define CHECK(value) if (!value) \
    {                                \
        NSLog(@"Failed to create %s at line %d in %s.", #value, __LINE__, __FUNCTION__ ); \
        exit(-1);                    \
    }
#else
    #define CHECK(value)
#endif

struct MatrixSize
{
    uint rows;
    uint cols;
};

struct MPSDevice
{
    NSAutoreleasePool*  pool{nullptr};
    id<MTLDevice>       device{nullptr};
    id<MTLCommandQueue> commandQueue{nullptr};
    id<MTLComputePipelineState>   compFuncPSOAdd{nullptr};
    id<MTLComputePipelineState>   compFuncPSOSub{nullptr};
    id<MTLComputePipelineState>   compFuncPSOMul{nullptr};
    id<MTLComputePipelineState>   compFuncPSODiv{nullptr};
    id<MTLComputePipelineState>   compFuncPSOAdd_A_S{nullptr};
    id<MTLComputePipelineState>   compFuncPSOSub_S_A{nullptr};
    id<MTLComputePipelineState>   compFuncPSOMul_A_S{nullptr};
    id<MTLComputePipelineState>   compFuncPSODiv_A_S{nullptr};
    id<MTLComputePipelineState>   compFuncPSODiv_S_A{nullptr};
    id<MTLComputePipelineState>   compFuncPSOCopy_S_A{nullptr};
    id<MTLComputePipelineState>   compFuncPSOSqrt{nullptr};
    id<MTLComputePipelineState>   compFuncPSOSin{nullptr};
    id<MTLComputePipelineState>   compFuncPSOCos{nullptr};
    id<MTLComputePipelineState>   compFuncPSOTanh{nullptr};
    id<MTLComputePipelineState>   compFuncPSOLog{nullptr};
    id<MTLComputePipelineState>   compFuncPSOExp{nullptr};

    id<MTLBuffer>   buf1{nullptr};
    id<MTLBuffer>   buf2{nullptr};
    id<MTLBuffer>   bufResult{nullptr};
    DataType        scalar{0};
    MatrixSize      buf1Size{0, 0};
    MatrixSize      buf2Size{0, 0};
    std::unordered_map<const void*, id<MTLBuffer>>  allocMap;

};

id<MTLBuffer> getReadOnlyMTLBuffer(MPSDevice* mpsDevice, const DataType * address, size_t size)
{
    // Memory could be from other devices. Create a temporary buffer for read only case.
    if (mpsDevice->allocMap.find(address) == mpsDevice->allocMap.end())
    {
        return [mpsDevice->device newBufferWithBytes:address length:(size * sizeof(DataType))
                                             options:MTLResourceStorageModeShared];
    }

    return mpsDevice->allocMap[address];    // Return MTL Buffer if the memory is from the current device.
}

void freeTemporaryBuffer(MPSDevice* mpsDevice, id<MTLBuffer> buffer, const DataType * address)
{
    // Release only temporary buffer.
    if (mpsDevice->allocMap.find(address) == mpsDevice->allocMap.end())
    {
        [buffer release];
    }
}


id<MTLLibrary> createLibrary(MPSDevice* mpsDevice, const char* shaders)
{
    // Create compile options
    auto compileOptions = [[MTLCompileOptions alloc] init];
    compileOptions.fastMathEnabled = NO;

    NSError* error = nullptr;
    auto shaderSource = [NSString stringWithUTF8String:shaders];
    auto defaultLibrary = [mpsDevice->device newLibraryWithSource:shaderSource
                                                          options:compileOptions
                                                            error:&error];
    [compileOptions release];

    if (!defaultLibrary)
    {
        NSLog(@"Failed to load default library. Details: %@", error.localizedDescription);
        exit(-1);
    }

    return defaultLibrary;
}

id<MTLComputePipelineState> createComputeFuncPSO(MPSDevice* mpsDevice, id<MTLLibrary> library, const char* kernelName)
{
    auto funcName = [NSString stringWithUTF8String:kernelName];
    auto compFunc = [library newFunctionWithName:funcName];
    CHECK(compFunc);
    if (!compFunc)
    {
        NSLog(@"Failed to find the compute function.");
        // No need to halt the application here.
    }

    NSError *error{nullptr};
    auto compFuncPSO = [mpsDevice->device newComputePipelineStateWithFunction:compFunc error:&error];
    if (!compFuncPSO)
    {
        NSLog(@"Failed to create the pipeline state object. Error: %@", error);
        exit(-1);
    }

    return compFuncPSO;
}

void encodeComputeCommandDoubleBuffer(MPSDevice* mpsDevice,
                                      id<MTLComputeCommandEncoder> computeEncoder,
                                      id<MTLComputePipelineState> compFuncPSO, MTLSize & gridSize,
                                      MTLSize & threadsPerTG)
{
    // Encode the pipeline state object and its parameters
    [computeEncoder setComputePipelineState:compFuncPSO];
    [computeEncoder setBuffer:mpsDevice->buf1 offset:0 atIndex:0];
    [computeEncoder setBuffer:mpsDevice->buf2 offset:0 atIndex:1];
    [computeEncoder setBuffer:mpsDevice->bufResult offset:0 atIndex:2];
    [computeEncoder setBytes:&mpsDevice->buf1Size length:sizeof(MatrixSize) atIndex:3];
    [computeEncoder setBytes:&mpsDevice->buf2Size length:sizeof(MatrixSize) atIndex:4];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerTG];
}


void encodeComputeCommandArrayScalar(MPSDevice* mpsDevice,
                                     id<MTLComputeCommandEncoder> computeEncoder,
                                     id<MTLComputePipelineState> compFuncPSO, MTLSize & gridSize,
                                     MTLSize & threadsPerTG)
{
    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:compFuncPSO];
    [computeEncoder setBuffer:mpsDevice->buf1 offset:0 atIndex:0];
    [computeEncoder setBytes:&mpsDevice->scalar length:sizeof(DataType) atIndex:1];
    [computeEncoder setBytes:&mpsDevice->buf1Size length:sizeof(MatrixSize) atIndex:2];
    [computeEncoder setBuffer:mpsDevice->bufResult offset:0 atIndex:3];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerTG];
}


void sendComputeCommandDoubleBuffer(MPSDevice* mpsDevice, id<MTLComputePipelineState> compFuncPSO, MTLSize & gridSize,
                                    MTLSize & threadsPerTG)
{
    auto cmdBuffer   = [mpsDevice->commandQueue commandBuffer];   // Create a command buffer
    auto compEncoder = [cmdBuffer computeCommandEncoder];         // Start a compute pass
    // Serialize resource and states to be called by GPU
    encodeComputeCommandDoubleBuffer(mpsDevice, compEncoder, compFuncPSO, gridSize, threadsPerTG);
    [compEncoder endEncoding];        // End the compute pass
    [cmdBuffer commit];               // Execute the command
    [cmdBuffer waitUntilCompleted];   // Wait until the work is done
}


void sendComputeCommandArrayScalar(MPSDevice* mpsDevice, id<MTLComputePipelineState> compFuncPSO, MTLSize & gridSize,
                                   MTLSize & threadsPerTG)
{
    auto cmdBuffer   = [mpsDevice->commandQueue commandBuffer];   // Create a command buffer
    auto compEncoder = [cmdBuffer computeCommandEncoder];         // Start a compute pass
    // Serialize resource and states to be called by GPU
    encodeComputeCommandArrayScalar(mpsDevice, compEncoder, compFuncPSO, gridSize, threadsPerTG);
    [compEncoder endEncoding];         // End the compute pass
    [cmdBuffer commit];                // Execute the command
    [cmdBuffer waitUntilCompleted];    // Wait until the work is done
}

void executeArrayScalarCmd(MPSDevice* mpsDevice,
                           const DataType * a,
                           DataType scalar,
                           size_t size,
                           DataType * result,
                           id<MTLComputePipelineState> compFuncPSO,
                           const std::string & cmdName)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (mpsDevice->allocMap.find(result) == mpsDevice->allocMap.end())
        throw std::invalid_argument("DeviceMPS::" + cmdName + "() result must have GPU memory.");

    // Set constants
    mpsDevice->buf1          = a ? getReadOnlyMTLBuffer(mpsDevice, a, size) : nullptr;
    mpsDevice->buf2          = nullptr;
    mpsDevice->bufResult     = mpsDevice->allocMap[result];
    mpsDevice->scalar        = scalar;
    mpsDevice->buf1Size.rows = 1;
    mpsDevice->buf1Size.cols = size;

    // Calculate maximum thread group dimensions
    NSUInteger w = std::min(size, [compFuncPSO maxTotalThreadsPerThreadgroup]);
    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    auto threadsPerThreadGroup = MTLSize{w, 1, 1};
    auto gridSize = MTLSize{size, 1, 1};            // gridSize = array size
    sendComputeCommandArrayScalar(mpsDevice, compFuncPSO, gridSize, threadsPerThreadGroup);

    if (a)
    {
        // Buf1 could be temporary buffer. It should be deleted if temporary.
        freeTemporaryBuffer(mpsDevice, mpsDevice->buf1, a);
        // Note: We never release result buffer since it will be used.
    }
}

void executeDoubleArrayCmd(MPSDevice* mpsDevice,
                           const DataType * a1,
                           const DataType * a2,
                           size_t size,
                           DataType * result,
                           id<MTLComputePipelineState> compFuncPSO,
                           const std::string & cmdName)
{
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (mpsDevice->allocMap.find(result) == mpsDevice->allocMap.end())
        throw std::invalid_argument("DeviceMPS::" + cmdName + "() result must have GPU memory.");

    mpsDevice->buf1 = getReadOnlyMTLBuffer(mpsDevice, a1, size);   // Memory could be a GPU allocated memory or system memory.
    mpsDevice->buf2 = getReadOnlyMTLBuffer(mpsDevice, a2, size);   // Memory could be a GPU allocated memory or system memory.
    mpsDevice->bufResult = mpsDevice->allocMap[result];

    // Calculate maximum thread group dimensions
    NSUInteger w = std::min(size, [compFuncPSO maxTotalThreadsPerThreadgroup]);
    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    auto threadsPerThreadGroup = MTLSize{w, 1, 1};
    auto gridSize = MTLSize{size, 1, 1};    // gridSize = Final matrix size
    sendComputeCommandDoubleBuffer(mpsDevice, compFuncPSO, gridSize, threadsPerThreadGroup);

    // Buf 1 and 2 could be temporary buffer. It should be deleted if temporary.
    freeTemporaryBuffer(mpsDevice, mpsDevice->buf1, a1);
    freeTemporaryBuffer(mpsDevice, mpsDevice->buf2, a2);
    // Note: We never release result buffer since it will be used.
}


extern "C"
{

void* createMPSDevice(int deviceIndex = 0)
{
    auto mpsDevice = new MPSDevice;
    mpsDevice->pool = [[NSAutoreleasePool alloc] init];
    mpsDevice->device = MTLCopyAllDevices()[deviceIndex];
    mpsDevice->commandQueue = [mpsDevice->device newCommandQueue];
    CHECK(mpsDevice->device);
    CHECK(mpsDevice->commandQueue);
    auto defaultLibrary = createLibrary(mpsDevice, aix::shaders::aixDeviceMetalShaders);
    CHECK(defaultLibrary);

    mpsDevice->compFuncPSOAdd      = createComputeFuncPSO(mpsDevice, defaultLibrary, "add_float");
    mpsDevice->compFuncPSOSub      = createComputeFuncPSO(mpsDevice, defaultLibrary, "sub_float");
    mpsDevice->compFuncPSOMul      = createComputeFuncPSO(mpsDevice, defaultLibrary, "mul_float");
    mpsDevice->compFuncPSODiv      = createComputeFuncPSO(mpsDevice, defaultLibrary, "div_float");
    mpsDevice->compFuncPSOAdd_A_S  = createComputeFuncPSO(mpsDevice, defaultLibrary, "add_a_s_float");
    mpsDevice->compFuncPSOSub_S_A  = createComputeFuncPSO(mpsDevice, defaultLibrary, "sub_s_a_float");
    mpsDevice->compFuncPSOMul_A_S  = createComputeFuncPSO(mpsDevice, defaultLibrary, "mul_a_s_float");
    mpsDevice->compFuncPSODiv_A_S  = createComputeFuncPSO(mpsDevice, defaultLibrary, "div_a_s_float");
    mpsDevice->compFuncPSODiv_S_A  = createComputeFuncPSO(mpsDevice, defaultLibrary, "div_s_a_float");
    mpsDevice->compFuncPSOCopy_S_A = createComputeFuncPSO(mpsDevice, defaultLibrary, "copy_s_a_float");
    mpsDevice->compFuncPSOSqrt     = createComputeFuncPSO(mpsDevice, defaultLibrary, "sqrt_a_float");
    mpsDevice->compFuncPSOSin      = createComputeFuncPSO(mpsDevice, defaultLibrary, "sin_a_float");
    mpsDevice->compFuncPSOCos      = createComputeFuncPSO(mpsDevice, defaultLibrary, "cos_a_float");
    mpsDevice->compFuncPSOTanh     = createComputeFuncPSO(mpsDevice, defaultLibrary, "tanh_a_float");
    mpsDevice->compFuncPSOLog      = createComputeFuncPSO(mpsDevice, defaultLibrary, "log_a_float");
    mpsDevice->compFuncPSOExp      = createComputeFuncPSO(mpsDevice, defaultLibrary, "exp_a_float");

    return mpsDevice;
}

void releaseMPSDevice(void* device)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    for (auto [ptr, buf]: mpsDevice->allocMap)
    {
        [buf release];
        NSLog(@"Leaking memory.");
    }

    [mpsDevice->compFuncPSOAdd release];
    [mpsDevice->compFuncPSOSub release];
    [mpsDevice->compFuncPSOMul release];
    [mpsDevice->compFuncPSODiv release];
    [mpsDevice->compFuncPSOAdd_A_S release];
    [mpsDevice->compFuncPSOSub_S_A release];
    [mpsDevice->compFuncPSOMul_A_S release];
    [mpsDevice->compFuncPSODiv_A_S release];
    [mpsDevice->compFuncPSODiv_S_A release];
    [mpsDevice->compFuncPSOCopy_S_A release];
    [mpsDevice->compFuncPSOSqrt release];
    [mpsDevice->compFuncPSOSin  release];
    [mpsDevice->compFuncPSOCos  release];
    [mpsDevice->compFuncPSOTanh release];
    [mpsDevice->compFuncPSOLog  release];
    [mpsDevice->compFuncPSOExp  release];

    [mpsDevice->commandQueue release];
    [mpsDevice->device release];
    [mpsDevice->pool release];

    delete mpsDevice;
}

// Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
void * allocate(void* device, size_t size)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);

    // Allocate GPU memory and save the mtl buffer to be used later.
    auto mtlBuf = [mpsDevice->device newBufferWithLength:size options:MTLResourceStorageModeShared];
    mpsDevice->allocMap[mtlBuf.contents] = mtlBuf;
    return mtlBuf.contents;
}

// Deallocate GPU memory if it's allocated by current device.
void deallocate(void* device, void* memory)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);

    if (mpsDevice->allocMap.find(memory) == mpsDevice->allocMap.end())
        throw std::invalid_argument("DeviceMPS::deallocate() - Found different type of memory to free.");
    auto mtlBuf = mpsDevice->allocMap[memory];
    mpsDevice->allocMap.erase(memory);
    [mtlBuf release];
}

void add_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeDoubleArrayCmd(mpsDevice, a1, a2, size, result, mpsDevice->compFuncPSOAdd, "add");
}

void sub_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeDoubleArrayCmd(mpsDevice, a1, a2, size, result, mpsDevice->compFuncPSOSub, "sub");
}

void mul_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeDoubleArrayCmd(mpsDevice, a1, a2, size, result, mpsDevice->compFuncPSOMul, "mul");
}

void div_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeDoubleArrayCmd(mpsDevice, a1, a2, size, result, mpsDevice->compFuncPSODiv, "div");
}

void  add_a_s(void* device, const DataType * a, DataType scalar, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, scalar, size, result, mpsDevice->compFuncPSOAdd_A_S, "add_a_s");
}

void sub_s_a(void* device, DataType scalar, const DataType * a, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, scalar, size, result, mpsDevice->compFuncPSOSub_S_A, "sub_s_a");
}

void mul_a_s(void* device, const DataType * a, DataType scalar, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, scalar, size, result, mpsDevice->compFuncPSOMul_A_S, "mul_a_s");
}

void div_a_s(void* device, const DataType * a, DataType scalar, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, scalar, size, result, mpsDevice->compFuncPSODiv_A_S, "div_a_s");
}

void div_s_a(void* device, DataType scalar, const DataType * a, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, scalar, size, result, mpsDevice->compFuncPSODiv_S_A, "div_s_a");
}

void sqrt_a(void* device, const DataType * a, const size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, 0, size, result, mpsDevice->compFuncPSOSqrt, "sqrt_a");
}

void sin_a(void* device, const DataType * a, const size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, 0, size, result, mpsDevice->compFuncPSOSin, "sin_a");
}

void cos_a(void* device, const DataType * a, const size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, 0, size, result, mpsDevice->compFuncPSOCos, "cos_a");
}

void tanh_a(void* device, const DataType * a, const size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, 0, size, result, mpsDevice->compFuncPSOTanh, "tanh_a");
}

void log_a(void* device, const DataType * a, const size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, 0, size, result, mpsDevice->compFuncPSOLog, "log_a");
}

void exp_a(void* device, const DataType * a, const size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, a, 0, size, result, mpsDevice->compFuncPSOExp, "exp_a");
}

void copy_s_a(void* device, DataType scalar, size_t size, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    executeArrayScalarCmd(mpsDevice, nullptr, scalar, size, result, mpsDevice->compFuncPSOCopy_S_A, "copy_s_a");
}

void copy_a_a(void* device, const DataType* src, DataType* dst, size_t size)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    CHECK(mpsDevice);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (mpsDevice->allocMap.find(dst) == mpsDevice->allocMap.end())
        throw std::invalid_argument("DeviceMPS::copy() - Result buffer must have GPU memory.");

    auto srcBuf = getReadOnlyMTLBuffer(mpsDevice, src, size);
    auto dstBuf = mpsDevice->allocMap[dst];
    CHECK(srcBuf);
    CHECK(dstBuf);

    size_t M = 1;
    size_t N = size;
    size_t rowInBytes = N * sizeof(DataType);

    // Create MPS Matrix Descriptors
    auto srcMatDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:rowInBytes dataType:mpsDataType];
    auto dstMatDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:rowInBytes dataType:mpsDataType];
    CHECK(srcMatDesc);
    CHECK(dstMatDesc);

    // Create MPS Matrices
    auto srcMat = [[MPSMatrix alloc] initWithBuffer:srcBuf descriptor:srcMatDesc];
    auto dstMat = [[MPSMatrix alloc] initWithBuffer:dstBuf descriptor:dstMatDesc];
    CHECK(srcMat);
    CHECK(dstMat);

    // Create a MPS Matrix Copy Descriptor
    auto matCopyDesc = [MPSMatrixCopyDescriptor descriptorWithSourceMatrix:srcMat
                                                         destinationMatrix:dstMat
                                                                   offsets:{0,0,0,0}];
    CHECK(matCopyDesc);

    // Create Matrix copy kernel
    MPSMatrixCopy* kernel = [[MPSMatrixCopy alloc] initWithDevice:mpsDevice->device
                                                         copyRows:M
                                                      copyColumns:N
                                             sourcesAreTransposed:false
                                        destinationsAreTransposed:false];
    CHECK(kernel);

    // Prepare and execute the command
    auto cmdBuffer = [mpsDevice->commandQueue commandBuffer];
    CHECK(cmdBuffer);

    [kernel encodeToCommandBuffer:cmdBuffer copyDescriptor:matCopyDesc];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    freeTemporaryBuffer(mpsDevice, srcBuf, src);
    // Release only if object is allocated with alloc() method.
    [kernel release];
    [srcMat release];
    [dstMat release];
}

void  matmul(void* device, const DataType * a1, size_t rows1, size_t cols1,
                           const DataType * a2, size_t rows2, size_t cols2,
                           DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    CHECK(mpsDevice);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (mpsDevice->allocMap.find(result) == mpsDevice->allocMap.end())
        throw std::invalid_argument("DeviceMPS::matmul() - Result buffer must have GPU memory.");

    auto matBuf1 = getReadOnlyMTLBuffer(mpsDevice, a1, rows1 * cols1);  // Memory could be a GPU allocated memory or system memory.
    auto matBuf2 = getReadOnlyMTLBuffer(mpsDevice, a2, rows2 * cols2);  // Memory could be a GPU allocated memory or system memory.
    auto matRes  = mpsDevice->allocMap[result];
    CHECK(matBuf1);
    CHECK(matBuf2);
    CHECK(matRes);

    size_t rowInBytes1 = cols1 * sizeof(DataType);
    size_t rowInBytes2 = cols2 * sizeof(DataType);

    // Create MPS Matrix Descriptors
    auto mat1Desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows1 columns:cols1 rowBytes:rowInBytes1 dataType:mpsDataType];
    auto mat2Desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows2 columns:cols2 rowBytes:rowInBytes2 dataType:mpsDataType];
    auto resDesc  = [MPSMatrixDescriptor matrixDescriptorWithRows:rows1 columns:cols2 rowBytes:rowInBytes2 dataType:mpsDataType];
    CHECK(mat1Desc);
    CHECK(mat2Desc);
    CHECK(resDesc);

    // Create MPS Matrices
    auto mpsMat1 = [[MPSMatrix alloc] initWithBuffer:matBuf1 descriptor:mat1Desc];
    auto mpsMat2 = [[MPSMatrix alloc] initWithBuffer:matBuf2 descriptor:mat2Desc];
    auto mpsRes  = [[MPSMatrix alloc] initWithBuffer:matRes  descriptor:resDesc];
    CHECK(mpsMat1);
    CHECK(mpsMat2);
    CHECK(mpsRes);

    // Create Matrix multiplication kernel
    auto kernel = [[MPSMatrixMultiplication alloc] initWithDevice:mpsDevice->device
                                                    transposeLeft:false
                                                   transposeRight:false
                                                       resultRows:rows1
                                                    resultColumns:cols2
                                                  interiorColumns:rows2
                                                            alpha:1
                                                             beta:0];
    CHECK(kernel);

    // Prepare and execute the command
    auto cmdBuffer = [mpsDevice->commandQueue commandBuffer];
    CHECK(cmdBuffer);

    [kernel encodeToCommandBuffer:cmdBuffer leftMatrix:mpsMat1 rightMatrix:mpsMat2 resultMatrix:mpsRes];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    freeTemporaryBuffer(mpsDevice, matBuf1, a1);
    freeTemporaryBuffer(mpsDevice, matBuf2, a2);
    // Release only if object is allocated with alloc() method.
    [kernel release];
    [mpsMat1 release];
    [mpsMat2 release];
    [mpsRes release];
}

void transpose(void* device, const DataType * mat, size_t rows, size_t cols, DataType * result)
{
    auto mpsDevice = static_cast<MPSDevice*>(device);
    CHECK(mpsDevice);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (mpsDevice->allocMap.find(result) == mpsDevice->allocMap.end())
        throw std::invalid_argument("DeviceMPS::transpose() - Result buffer must have GPU memory.");

    auto srcBuf = getReadOnlyMTLBuffer(mpsDevice, mat, rows * cols);
    auto dstBuf = mpsDevice->allocMap[result];
    CHECK(srcBuf);
    CHECK(dstBuf);

    size_t M = rows;
    size_t N = cols;
    size_t rowInBytes  = N * sizeof(DataType);
    size_t rowInBytesT = M * sizeof(DataType);

    // Create MPS Matrix Descriptors
    auto srcMatDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:rowInBytes dataType:mpsDataType];
    // Note: Matrix is set up in transposed dimensions before the copy.
    auto dstMatDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:M rowBytes:rowInBytesT dataType:mpsDataType];
    CHECK(srcMatDesc);
    CHECK(dstMatDesc);

    // Create MPS Matrices
    auto srcMat = [[MPSMatrix alloc] initWithBuffer:srcBuf descriptor:srcMatDesc];
    auto dstMat = [[MPSMatrix alloc] initWithBuffer:dstBuf descriptor:dstMatDesc];
    CHECK(srcMat);
    CHECK(dstMat);

    // Create a MPS Matrix Copy Descriptor
    auto matCopyDesc = [MPSMatrixCopyDescriptor descriptorWithSourceMatrix:srcMat
                                                         destinationMatrix:dstMat
                                                                   offsets:{0,0,0,0}];
    CHECK(matCopyDesc);

    // Create Matrix copy kernel
    auto kernel = [[MPSMatrixCopy alloc] initWithDevice:mpsDevice->device
                                               copyRows:M
                                            copyColumns:N
                                   sourcesAreTransposed:false
                              destinationsAreTransposed:true];
    CHECK(kernel);

    // Prepare and execute the command
    auto cmdBuffer = [mpsDevice->commandQueue commandBuffer];
    CHECK(cmdBuffer);

    [kernel encodeToCommandBuffer:cmdBuffer copyDescriptor:matCopyDesc];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    freeTemporaryBuffer(mpsDevice, srcBuf, mat);
    // Release only if object is allocated with alloc() method.
    [kernel release];
    [srcMat release];
    [dstMat release];
}

}   // extern "C"
