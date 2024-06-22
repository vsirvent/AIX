//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

namespace aix::shaders
{

// Requires Metal Language Version 2_2 and higher.
const char* aixDeviceMetalShaders = R"(

#include <metal_stdlib>
using namespace metal;

#define ALIGNMENT_SIZE   64
#define ITERATION_SIZE   (ALIGNMENT_SIZE/4)

struct MatrixSize
{
    size_t rows;
    size_t cols;
};


// -----------------------------------------------------------------
// Add - Two Arrays
// -----------------------------------------------------------------
template<typename T>
kernel void add(device const T* inA,
                device const T* inB,
                device T* result,
                constant MatrixSize& aSize,
                constant MatrixSize& bSize,
                uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] + inB[index + i];
}

// -----------------------------------------------------------------
// Sub - Two Arrays
// -----------------------------------------------------------------
template<typename T>
kernel void sub(device const T* inA,
                device const T* inB,
                device T* result,
                constant MatrixSize& aSize,
                constant MatrixSize& bSize,
                uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] - inB[index + i];
}

// -----------------------------------------------------------------
// Mul - Two Arrays
// -----------------------------------------------------------------
template<typename T>
kernel void mul(device const T* inA,
                device const T* inB,
                device T* result,
                constant MatrixSize& aSize,
                constant MatrixSize& bSize,
                uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] * inB[index + i];
}

// -----------------------------------------------------------------
// Div - Two Arrays
// -----------------------------------------------------------------
template<typename T>
kernel void div(device const T* inA,
                device const T* inB,
                device T* result,
                constant MatrixSize& aSize,
                constant MatrixSize& bSize,
                uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] / inB[index + i];
}

// -----------------------------------------------------------------
// Add - Array + Scalar
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void add_a_s(device const T* inA,
                    constant const T2& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] + scalar;
}

// -----------------------------------------------------------------
// Sub - Scalar - Array
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void sub_s_a(device const T* inA,
                    constant const T2& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = scalar - inA[index + i];
}

// -----------------------------------------------------------------
// Mul - Array * Scalar
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void mul_a_s(device const T* inA,
                    constant const T2& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] * scalar;
}

// -----------------------------------------------------------------
// Div - Array / Scalar
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void div_a_s(device const T* inA,
                    constant const T2& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] / scalar;
}

// -----------------------------------------------------------------
// Div - Scalar / Array
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void div_s_a(device const T* inA,
                    constant const T2& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = scalar / inA[index + i];
}

// -----------------------------------------------------------------
// Sqrt - sqrt(Array)
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void sqrt_a(device const T* inA,
                   constant const T2& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = sqrt(inA[index + i]);
}

// -----------------------------------------------------------------
// Sin - sin(Array)
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void sin_a(device const T* inA,
                  constant const T2& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = sin(inA[index + i]);
}

// -----------------------------------------------------------------
// Cos - cos(Array)
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void cos_a(device const T* inA,
                   constant const T2& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = cos(inA[index + i]);
}

// -----------------------------------------------------------------
// Tanh - tanh(Array)
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void tanh_a(device const T* inA,
                   constant const T2& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = tanh(inA[index + i]);
}

// -----------------------------------------------------------------
// Log - log(Array)
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void log_a(device const T* inA,
                  constant const T2& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = log(inA[index + i]);
}

// -----------------------------------------------------------------
// Exp - exp(Array)
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void exp_a(device const T* inA,
                  constant const T2& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = exp(inA[index + i]);
}


// -----------------------------------------------------------------
// Pow
// -----------------------------------------------------------------
template<typename T>
kernel void pow(device const T* inA,
                device const T* expA,
                device T* result,
                constant MatrixSize& aSize,
                constant MatrixSize& bSize,
                uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = pow(inA[index + i], expA[index + i]);
}


// -----------------------------------------------------------------
// Matrix Multiply - Naive implementation
// -----------------------------------------------------------------
template<typename T>
kernel void matrix_mul(device const T* inA,
                       device const T* inB,
                       device T* result,
                       constant MatrixSize& matASize,
                       constant MatrixSize& matBSize,
                       uint2 gid [[thread_position_in_grid]])
{
    T sum = 0.0;
    for (uint k = 0; k < matASize.cols; k++)
    {
        uint aIndex = gid.y * matASize.cols + k;
        uint bIndex = k * matBSize.cols + gid.x;
        sum += inA[aIndex] * inB[bIndex];
    }

    result[gid.y * matBSize.cols + gid.x] = sum;
}

// -----------------------------------------------------------------
// Transpose2D - Naive implementation
// -----------------------------------------------------------------
template<typename T>
kernel void transpose2D(device const T* mat,
                        device T* result,
                        constant MatrixSize& matSize,
                        uint2 gid [[thread_position_in_grid]],
                        uint2 tid [[thread_position_in_threadgroup]])
{
    uint ofs1 = gid.y * matSize.rows + gid.x;
    uint ofs2 = gid.x * matSize.cols + gid.y;
    result[ofs1] = mat[ofs2];
}


// -----------------------------------------------------------------
// Transpose - Naive implementation
// -----------------------------------------------------------------

size_t flattenIndex(thread size_t* indices, size_t indicesSize, device const size_t* strides)
{
    size_t index = 0;
    for (size_t i = 0; i < indicesSize; ++i)
    {
        index += indices[i] * strides[i];
    }
    return index;
}

void unflattenIndex(size_t index, device const size_t* strides, size_t stridesSize, thread size_t* outIndices)
{
    for (size_t i = 0; i < stridesSize; ++i)
    {
        outIndices[i] = index / strides[i];
        index %= strides[i];
    }
}

void swap(thread size_t& a, thread size_t& b)
{
    size_t temp = a;
    a = b;
    b = temp;
}

template<typename T>
kernel void transpose(device const T* data,
                      device T* result,
                      constant size_t& dim0,
                      constant size_t& dim1,
                      device const size_t* strides,
                      constant size_t& stridesSize,
                      device const size_t* newStrides,
                      constant size_t& newStridesSize,
                      constant size_t& size,
                      uint index [[thread_position_in_grid]])
{
    thread size_t oldIndices[16];
    unflattenIndex(index, strides, stridesSize, oldIndices);
    swap(oldIndices[dim0], oldIndices[dim1]);
    size_t newIndex = flattenIndex(oldIndices, stridesSize, newStrides);
    result[newIndex] = data[index];
}


// -----------------------------------------------------------------
// Copy - Src To Dst
// -----------------------------------------------------------------
template<typename T>
kernel void copy_a_a(device const T* src,
                     device T* dst,
                     constant MatrixSize& size,
                     uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        dst[index + i] = src[index + i];
}

// -----------------------------------------------------------------
// Copy - Scalar To Array
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void copy_s_a(device const T* inA,
                     constant const T2& scalar,
                     constant MatrixSize& aSize,
                     device T* result,
                     uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = scalar;
}


// -----------------------------------------------------------------
// Sum - Naive Parallel Reduction Sum (Array)
// -----------------------------------------------------------------
template<typename T>
kernel void sum_a(device const T* inA,
                  constant const T& scalar,
                  constant MatrixSize& aSize,
                  device float* result,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]])
{
    const size_t MAX_THREADS = 1024;
    threadgroup T sharedData[MAX_THREADS];
    sharedData[li] = inA[tgi * MAX_THREADS + li];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform parallel reduction in shared memory
    size_t size = threadsPerThreadgroup;
    for (uint stride = size / 2; stride > 0; stride >>= 1)
    {
        if (size % 2 == 1 && li == 0)
            sharedData[0] += sharedData[size-1];
        size >>= 1;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (li < stride)
            sharedData[li] += sharedData[li + stride];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (li == 0)
    {
        result[tgi] = sharedData[0];
    }
}


// -----------------------------------------------------------------
// TranslationIndex
// -----------------------------------------------------------------
size_t translationIndex(size_t index, device const size_t* shape, device const size_t* newShape,
                        size_t shapeSize, size_t newShapeSize)
{
    size_t originalIndex  = 0;
    size_t targetStride   = 1;
    size_t originalStride = 1;

    for (int64_t i = newShapeSize - 1, j = shapeSize - 1; i >= 0; --i)
    {
        size_t dimIndex = (index / targetStride) % newShape[i];
        if (j >= 0 && shape[j] == newShape[i])
        {
            originalIndex += dimIndex * originalStride;
            originalStride *= shape[--j + 1];
        }
        else if (j >= 0 && shape[j] == 1)
        {
            originalStride *= shape[--j + 1];
        }
        targetStride *= newShape[i];
    }

    return originalIndex;
}


// -----------------------------------------------------------------
// BroadcastTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void broadcastTo(device const T* src,
                        device       T* dst,
                        device const T2* shape,
                        device const T2* newShape,
                        constant T2& shapeSize,
                        constant T2& newShapeSize,
                        uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translationIndex(index, shape, newShape, shapeSize, newShapeSize);
    dst[index] = src[originalIndex];
}


// -----------------------------------------------------------------
// ReduceTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void reduceTo(device const T* src,
                     device       T* dst,
                     device const T2* shape,
                     device const T2* newShape,
                     constant T2& shapeSize,
                     constant T2& newShapeSize,
                     uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translationIndex(index, shape, newShape, shapeSize, newShapeSize);
    atomic_fetch_add_explicit((device atomic_float*)&(dst[originalIndex]), src[index], memory_order_relaxed);
}


// Templates


template [[ host_name("add_float") ]]
kernel void add(device const float4*,
                device const float4*,
                device float4*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("sub_float") ]]
kernel void sub(device const float4*,
                device const float4*,
                device float4*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("mul_float") ]]
kernel void mul(device const float4*,
                device const float4*,
                device float4*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("div_float") ]]
kernel void div(device const float4*,
                device const float4*,
                device float4*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("add_a_s_float") ]]
kernel void add_a_s(device const float4*,
                    constant const float&,
                    constant MatrixSize&,
                    device float4*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("sub_s_a_float") ]]
kernel void sub_s_a(device const float4*,
                    constant const float&,
                    constant MatrixSize&,
                    device float4*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("mul_a_s_float") ]]
kernel void mul_a_s(device const float4*,
                    constant const float&,
                    constant MatrixSize&,
                    device float4*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("div_a_s_float") ]]
kernel void div_a_s(device const float4*,
                    constant const float&,
                    constant MatrixSize&,
                    device float4*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("div_s_a_float") ]]
kernel void div_s_a(device const float4*,
                    constant const float&,
                    constant MatrixSize&,
                    device float4*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("sqrt_a_float") ]]
kernel void sqrt_a(device const float4*,
                   constant const float&,
                   constant MatrixSize&,
                   device float4*,
                   uint index [[thread_position_in_grid]]);


template [[ host_name("sin_a_float") ]]
kernel void sin_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);


template [[ host_name("cos_a_float") ]]
kernel void cos_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);


template [[ host_name("tanh_a_float") ]]
kernel void tanh_a(device const float4*,
                   constant const float&,
                   constant MatrixSize&,
                   device float4*,
                   uint index [[thread_position_in_grid]]);


template [[ host_name("log_a_float") ]]
kernel void log_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);


template [[ host_name("exp_a_float") ]]
kernel void exp_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);


template [[ host_name("pow_float") ]]
kernel void pow(device const float4*,
                device const float4*,
                device float4*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("sum_a_float") ]]
kernel void sum_a(device const float*,
                  constant const float&,
                  constant MatrixSize&,
                  device float*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);


template [[ host_name("matrix_mul_float") ]]
kernel void matrix_mul(device const float*,
                       device const float*,
                       device float*,
                       constant MatrixSize&,
                       constant MatrixSize&,
                       uint2 gid [[thread_position_in_grid]]);


template [[ host_name("transpose2D_float") ]]
kernel void transpose2D(device const float*,
                        device float*,
                        constant MatrixSize&,
                        uint2 gid,
                        uint2 tid);


template [[ host_name("transpose_float") ]]
kernel void transpose(device const float* data,
                      device float* result,
                      constant size_t& dim0,
                      constant size_t& dim1,
                      device const size_t* strides,
                      constant size_t& stridesSize,
                      device const size_t* newStrides,
                      constant size_t& newStridesSize,
                      constant size_t& size,
                      uint index [[thread_position_in_grid]]);


template [[ host_name("copy_a_a_float") ]]
kernel void copy_a_a(device const float4*,
                     device float4*,
                     constant MatrixSize&,
                     uint index);


template [[ host_name("copy_s_a_float") ]]
kernel void copy_s_a(device const float4*,
                     constant const float&,
                     constant MatrixSize&,
                     device float4*,
                     uint index [[thread_position_in_grid]]);


template [[ host_name("broadcastTo_float") ]]
kernel void broadcastTo(device const float* src,
                       device        float* dst,
                       device const size_t* shape,
                       device const size_t* newShape,
                       constant size_t& shapeSize,
                       constant size_t& newShapeSize,
                       uint index [[thread_position_in_grid]]);


template [[ host_name("reduceTo_float") ]]
kernel void reduceTo(device const float* src,
                     device       float* dst,
                     device const size_t* shape,
                     device const size_t* newShape,
                     constant size_t& shapeSize,
                     constant size_t& newShapeSize,
                     uint index [[thread_position_in_grid]]);

)";


}   // namespace