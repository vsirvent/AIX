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
// TEMPLATES
// -----------------------------------------------------------------


// Add - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void add_aa(device const T* inA,
                   device const T* inB,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] + inB[index + i];
}


// Sub - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void sub_aa(device const T* inA,
                   device const T* inB,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] - inB[index + i];
}


// Mul - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void mul_aa(device const T* inA,
                  device const T* inB,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] * inB[index + i];
}


// Div - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void div_aa(device const T* inA,
                   device const T* inB,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = inA[index + i] / inB[index + i];
}


// Sqrt - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void sqrt_a(device const T* inA,
                   constant const T2& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = static_cast<T>(sqrt(static_cast<float4>(inA[index + i])));
}


// Sin - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void sin_a(device const T* inA,
                  constant const T2& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = static_cast<T>(sin(static_cast<float4>(inA[index + i])));
}


// Cos - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void cos_a(device const T* inA,
                  constant const T2& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = static_cast<T>(cos(static_cast<float4>(inA[index + i])));
}


// Tanh - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void tanh_a(device const T* inA,
                   constant const T2& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = static_cast<T>(tanh(static_cast<float4>(inA[index + i])));
}


// Log - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void log_a(device const T* inA,
                  constant const T2& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = static_cast<T>(log(static_cast<float4>(inA[index + i])));
}


// Exp - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void exp_a(device const T* inA,
                  constant const T2& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = static_cast<T>(exp(static_cast<float4>(inA[index + i])));
}


// Pow - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void pow_aa(device const T* inA,
                   device const T* expA,
                   device T* result,
                   constant MatrixSize& aSize,
                   constant MatrixSize& bSize,
                   uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = static_cast<T>(pow(static_cast<float4>(inA[index + i]), static_cast<float4>(expA[index + i])));
}


// Matrix_Mul - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void matrixMul_aa(device const T* inA,
                         device const T* inB,
                         device T* result,
                         constant MatrixSize& matASize,
                         constant MatrixSize& matBSize,
                         uint2 gid [[thread_position_in_grid]])
{
    T sum = 0;
    for (uint k = 0; k < matASize.cols; k++)
    {
        uint aIndex = gid.y * matASize.cols + k;
        uint bIndex = k * matBSize.cols + gid.x;
        sum += inA[aIndex] * inB[bIndex];
    }

    result[gid.y * matBSize.cols + gid.x] = sum;
}


// Transpose2D - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void transpose2D_a(device const T* mat,
                          device T* result,
                          constant MatrixSize& matSize,
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]])
{
    uint ofs1 = gid.y * matSize.rows + gid.x;
    uint ofs2 = gid.x * matSize.cols + gid.y;
    result[ofs1] = mat[ofs2];
}


// Transpose - Naive Implementation
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
kernel void transpose_a(device const T* data,
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


// Copy - Naive Implementation
// -----------------------------------------------------------------
template<typename ST, typename DT>
kernel void copy_aa(device const ST* src,
                    device DT* dst,
                    constant MatrixSize& size,
                    uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        dst[index + i] = static_cast<DT>(src[index + i]);
}


// Unary - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void unary_a(device const T* inA,
                    constant const T2& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = -inA[index + i];
}


// Fill - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void fill_aa(device const T* scalar,
                    device T2* result,
                    uint index [[thread_position_in_grid]])
{
    T2 scalarVector = static_cast<T2>(scalar[0].xxxx);

    index *= ITERATION_SIZE;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<ITERATION_SIZE; ++i)
        result[index + i] = scalarVector;
}


// Sum - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
kernel void sum_a(device const T* inA,
                  constant const float&,
                  constant MatrixSize& aSize,
                  device T* result,
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


// TranslationIndex - Naive Implementation
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


// BroadcastTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void broadcastTo_a(device const T* src,
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


// ReduceTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
kernel void reduceTo_a(device const T* src,
                       device       T* dst,
                       device const T2* shape,
                       device const T2* newShape,
                       constant T2& shapeSize,
                       constant T2& newShapeSize,
                       uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translationIndex(index, shape, newShape, shapeSize, newShapeSize);
    atomic_fetch_add_explicit((device atomic<T>*)&(dst[originalIndex]), src[index], memory_order_relaxed);

    // NOTE: Metal Framework supports add and sub operations for only atomic_float, atomic_uint and atomic_uint.
}


// nullKernel
// -----------------------------------------------------------------
kernel void nullKernel(uint index [[thread_position_in_grid]])
{
}


// -----------------------------------------------------------------
// TEMPLATE SPECIALIZATIONS
// -----------------------------------------------------------------


// Add
// -----------------------------------------------------------------
template [[ host_name("add_aa_f32") ]]
kernel void add_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_f16") ]]
kernel void add_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_bf16") ]]
kernel void add_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i64") ]]
kernel void add_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i32") ]]
kernel void add_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i16") ]]
kernel void add_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_i8") ]]
kernel void add_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("add_aa_ui8") ]]
kernel void add_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Sub
// -----------------------------------------------------------------
template [[ host_name("sub_aa_f32") ]]
kernel void sub_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_f16") ]]
kernel void sub_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_bf16") ]]
kernel void sub_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i64") ]]
kernel void sub_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i32") ]]
kernel void sub_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i16") ]]
kernel void sub_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_i8") ]]
kernel void sub_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sub_aa_ui8") ]]
kernel void sub_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Mul
// -----------------------------------------------------------------
template [[ host_name("mul_aa_f32") ]]
kernel void mul_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_f16") ]]
kernel void mul_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_bf16") ]]
kernel void mul_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i64") ]]
kernel void mul_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i32") ]]
kernel void mul_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i16") ]]
kernel void mul_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_i8") ]]
kernel void mul_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("mul_aa_ui8") ]]
kernel void mul_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);


// Div
// -----------------------------------------------------------------
template [[ host_name("div_aa_f32") ]]
kernel void div_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_f16") ]]
kernel void div_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_bf16") ]]
kernel void div_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i64") ]]
kernel void div_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i32") ]]
kernel void div_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i16") ]]
kernel void div_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_i8") ]]
kernel void div_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("div_aa_ui8") ]]
kernel void div_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Sqrt
// -----------------------------------------------------------------
template [[ host_name("sqrt_a_f32") ]]
kernel void sqrt_a(device const float4*,
                   constant const float&,
                   constant MatrixSize&,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_f16") ]]
kernel void sqrt_a(device const half4*,
                   constant const float&,
                   constant MatrixSize&,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_bf16") ]]
kernel void sqrt_a(device const bfloat4*,
                   constant const float&,
                   constant MatrixSize&,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i64") ]]
kernel void sqrt_a(device const long4*,
                   constant const float&,
                   constant MatrixSize&,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i32") ]]
kernel void sqrt_a(device const int4*,
                   constant const float&,
                   constant MatrixSize&,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i16") ]]
kernel void sqrt_a(device const short4*,
                   constant const float&,
                   constant MatrixSize&,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_i8") ]]
kernel void sqrt_a(device const char4*,
                   constant const float&,
                   constant MatrixSize&,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("sqrt_a_ui8") ]]
kernel void sqrt_a(device const uchar4*,
                   constant const float&,
                   constant MatrixSize&,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Sin
// -----------------------------------------------------------------
template [[ host_name("sin_a_f32") ]]
kernel void sin_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_f16") ]]
kernel void sin_a(device const half4*,
                  constant const float&,
                  constant MatrixSize&,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_bf16") ]]
kernel void sin_a(device const bfloat4*,
                  constant const float&,
                  constant MatrixSize&,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i64") ]]
kernel void sin_a(device const long4*,
                  constant const float&,
                  constant MatrixSize&,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i32") ]]
kernel void sin_a(device const int4*,
                  constant const float&,
                  constant MatrixSize&,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i16") ]]
kernel void sin_a(device const short4*,
                  constant const float&,
                  constant MatrixSize&,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_i8") ]]
kernel void sin_a(device const char4*,
                  constant const float&,
                  constant MatrixSize&,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("sin_a_ui8") ]]
kernel void sin_a(device const uchar4*,
                  constant const float&,
                  constant MatrixSize&,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);


// Cos
// -----------------------------------------------------------------
template [[ host_name("cos_a_f32") ]]
kernel void cos_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_f16") ]]
kernel void cos_a(device const half4*,
                  constant const float&,
                  constant MatrixSize&,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_bf16") ]]
kernel void cos_a(device const bfloat4*,
                  constant const float&,
                  constant MatrixSize&,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i64") ]]
kernel void cos_a(device const long4*,
                  constant const float&,
                  constant MatrixSize&,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i32") ]]
kernel void cos_a(device const int4*,
                  constant const float&,
                  constant MatrixSize&,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i16") ]]
kernel void cos_a(device const short4*,
                  constant const float&,
                  constant MatrixSize&,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_i8") ]]
kernel void cos_a(device const char4*,
                  constant const float&,
                  constant MatrixSize&,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("cos_a_ui8") ]]
kernel void cos_a(device const uchar4*,
                  constant const float&,
                  constant MatrixSize&,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);



// Tanh
// -----------------------------------------------------------------
template [[ host_name("tanh_a_f32") ]]
kernel void tanh_a(device const float4*,
                   constant const float&,
                   constant MatrixSize&,
                   device float4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_f16") ]]
kernel void tanh_a(device const half4*,
                   constant const float&,
                   constant MatrixSize&,
                   device half4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_bf16") ]]
kernel void tanh_a(device const bfloat4*,
                   constant const float&,
                   constant MatrixSize&,
                   device bfloat4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i64") ]]
kernel void tanh_a(device const long4*,
                   constant const float&,
                   constant MatrixSize&,
                   device long4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i32") ]]
kernel void tanh_a(device const int4*,
                   constant const float&,
                   constant MatrixSize&,
                   device int4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i16") ]]
kernel void tanh_a(device const short4*,
                   constant const float&,
                   constant MatrixSize&,
                   device short4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_i8") ]]
kernel void tanh_a(device const char4*,
                   constant const float&,
                   constant MatrixSize&,
                   device char4*,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("tanh_a_ui8") ]]
kernel void tanh_a(device const uchar4*,
                   constant const float&,
                   constant MatrixSize&,
                   device uchar4*,
                   uint index [[thread_position_in_grid]]);



// Log
// -----------------------------------------------------------------
template [[ host_name("log_a_f32") ]]
kernel void log_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_f16") ]]
kernel void log_a(device const half4*,
                  constant const float&,
                  constant MatrixSize&,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_bf16") ]]
kernel void log_a(device const bfloat4*,
                  constant const float&,
                  constant MatrixSize&,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i64") ]]
kernel void log_a(device const long4*,
                  constant const float&,
                  constant MatrixSize&,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i32") ]]
kernel void log_a(device const int4*,
                  constant const float&,
                  constant MatrixSize&,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i16") ]]
kernel void log_a(device const short4*,
                  constant const float&,
                  constant MatrixSize&,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_i8") ]]
kernel void log_a(device const char4*,
                  constant const float&,
                  constant MatrixSize&,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("log_a_ui8") ]]
kernel void log_a(device const uchar4*,
                  constant const float&,
                  constant MatrixSize&,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);



// Exp
// -----------------------------------------------------------------
template [[ host_name("exp_a_f32") ]]
kernel void exp_a(device const float4*,
                  constant const float&,
                  constant MatrixSize&,
                  device float4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_f16") ]]
kernel void exp_a(device const half4*,
                  constant const float&,
                  constant MatrixSize&,
                  device half4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_bf16") ]]
kernel void exp_a(device const bfloat4*,
                  constant const float&,
                  constant MatrixSize&,
                  device bfloat4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i64") ]]
kernel void exp_a(device const long4*,
                  constant const float&,
                  constant MatrixSize&,
                  device long4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i32") ]]
kernel void exp_a(device const int4*,
                  constant const float&,
                  constant MatrixSize&,
                  device int4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i16") ]]
kernel void exp_a(device const short4*,
                  constant const float&,
                  constant MatrixSize&,
                  device short4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_i8") ]]
kernel void exp_a(device const char4*,
                  constant const float&,
                  constant MatrixSize&,
                  device char4*,
                  uint index [[thread_position_in_grid]]);

template [[ host_name("exp_a_ui8") ]]
kernel void exp_a(device const uchar4*,
                  constant const float&,
                  constant MatrixSize&,
                  device uchar4*,
                  uint index [[thread_position_in_grid]]);


// Pow
// -----------------------------------------------------------------
template [[ host_name("pow_aa_f32") ]]
kernel void pow_aa(device const float4*,
                   device const float4*,
                   device float4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_f16") ]]
kernel void pow_aa(device const half4*,
                   device const half4*,
                   device half4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_bf16") ]]
kernel void pow_aa(device const bfloat4*,
                   device const bfloat4*,
                   device bfloat4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i64") ]]
kernel void pow_aa(device const long4*,
                   device const long4*,
                   device long4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i32") ]]
kernel void pow_aa(device const int4*,
                   device const int4*,
                   device int4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i16") ]]
kernel void pow_aa(device const short4*,
                   device const short4*,
                   device short4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_i8") ]]
kernel void pow_aa(device const char4*,
                   device const char4*,
                   device char4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);

template [[ host_name("pow_aa_ui8") ]]
kernel void pow_aa(device const uchar4*,
                   device const uchar4*,
                   device uchar4*,
                   constant MatrixSize&,
                   constant MatrixSize&,
                   uint index [[thread_position_in_grid]]);


// Sum
// -----------------------------------------------------------------
template [[ host_name("sum_a_f32") ]]
kernel void sum_a(device const float*,
                  constant const float&,
                  constant MatrixSize&,
                  device float*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_f16") ]]
kernel void sum_a(device const half*,
                  constant const float&,
                  constant MatrixSize&,
                  device half*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_bf16") ]]
kernel void sum_a(device const bfloat*,
                  constant const float&,
                  constant MatrixSize&,
                  device bfloat*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i64") ]]
kernel void sum_a(device const long*,
                  constant const float&,
                  constant MatrixSize&,
                  device long*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i32") ]]
kernel void sum_a(device const int*,
                  constant const float&,
                  constant MatrixSize&,
                  device int*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i16") ]]
kernel void sum_a(device const short*,
                  constant const float&,
                  constant MatrixSize&,
                  device short*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_i8") ]]
kernel void sum_a(device const char*,
                  constant const float&,
                  constant MatrixSize&,
                  device char*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);

template [[ host_name("sum_a_ui8") ]]
kernel void sum_a(device const uchar*,
                  constant const float&,
                  constant MatrixSize&,
                  device uchar*,
                  uint li [[thread_position_in_threadgroup]],
                  uint tgi [[threadgroup_position_in_grid]],
                  uint threadsPerThreadgroup [[threads_per_threadgroup]]);


// Matrix_Mul
// -----------------------------------------------------------------
template [[ host_name("matrixMul_aa_f32") ]]
kernel void matrixMul_aa(device const float*,
                         device const float*,
                         device float*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);

template [[ host_name("matrixMul_aa_f16") ]]
kernel void matrixMul_aa(device const half*,
                         device const half*,
                         device half*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);

template [[ host_name("matrixMul_aa_bf16") ]]
kernel void matrixMul_aa(device const bfloat*,
                         device const bfloat*,
                         device bfloat*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);

template [[ host_name("matrixMul_aa_i64") ]]
kernel void matrixMul_aa(device const long*,
                         device const long*,
                         device long*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);

template [[ host_name("matrixMul_aa_i32") ]]
kernel void matrixMul_aa(device const int*,
                         device const int*,
                         device int*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);

template [[ host_name("matrixMul_aa_i16") ]]
kernel void matrixMul_aa(device const short*,
                         device const short*,
                         device short*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);

template [[ host_name("matrixMul_aa_i8") ]]
kernel void matrixMul_aa(device const char*,
                         device const char*,
                         device char*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);

template [[ host_name("matrixMul_aa_ui8") ]]
kernel void matrixMul_aa(device const uchar*,
                         device const uchar*,
                         device uchar*,
                         constant MatrixSize&,
                         constant MatrixSize&,
                         uint2 gid [[thread_position_in_grid]]);


// Transpose2D
// -----------------------------------------------------------------
template [[ host_name("transpose2D_a_f32") ]]
kernel void transpose2D_a(device const float*,
                          device float*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_f16") ]]
kernel void transpose2D_a(device const half*,
                          device half*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_bf16") ]]
kernel void transpose2D_a(device const bfloat*,
                          device bfloat*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i64") ]]
kernel void transpose2D_a(device const long*,
                          device long*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i32") ]]
kernel void transpose2D_a(device const int*,
                          device int*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i16") ]]
kernel void transpose2D_a(device const short*,
                          device short*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_i8") ]]
kernel void transpose2D_a(device const char*,
                          device char*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);

template [[ host_name("transpose2D_a_ui8") ]]
kernel void transpose2D_a(device const uchar*,
                          device uchar*,
                          constant MatrixSize&,
                          uint2 gid,
                          uint2 tid);


// Transpose
// -----------------------------------------------------------------
template [[ host_name("transpose_a_f32") ]]
kernel void transpose_a(device const float* data,
                        device float* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_f16") ]]
kernel void transpose_a(device const half* data,
                        device half* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_bf16") ]]
kernel void transpose_a(device const bfloat* data,
                        device bfloat* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i64") ]]
kernel void transpose_a(device const long* data,
                        device long* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i32") ]]
kernel void transpose_a(device const int* data,
                        device int* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i16") ]]
kernel void transpose_a(device const short* data,
                        device short* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_i8") ]]
kernel void transpose_a(device const char* data,
                        device char* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);

template [[ host_name("transpose_a_ui8") ]]
kernel void transpose_a(device const uchar* data,
                        device uchar* result,
                        constant size_t& dim0,
                        constant size_t& dim1,
                        device const size_t* strides,
                        constant size_t& stridesSize,
                        device const size_t* newStrides,
                        constant size_t& newStridesSize,
                        constant size_t& size,
                        uint index [[thread_position_in_grid]]);


// Copy
// -----------------------------------------------------------------
template [[ host_name("copy_aa_f32_f32") ]]
kernel void copy_aa(device const float4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f32_f16") ]]
kernel void copy_aa(device const float4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f32_bf16") ]]
kernel void copy_aa(device const float4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f32_i64") ]]
kernel void copy_aa(device const float4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f32_i32") ]]
kernel void copy_aa(device const float4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f32_i16") ]]
kernel void copy_aa(device const float4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f32_i8") ]]
kernel void copy_aa(device const float4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f32_ui8") ]]
kernel void copy_aa(device const float4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_f32") ]]
kernel void copy_aa(device const half4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_f16") ]]
kernel void copy_aa(device const half4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_bf16") ]]
kernel void copy_aa(device const half4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_i64") ]]
kernel void copy_aa(device const half4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_i32") ]]
kernel void copy_aa(device const half4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_i16") ]]
kernel void copy_aa(device const half4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_i8") ]]
kernel void copy_aa(device const half4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_f16_ui8") ]]
kernel void copy_aa(device const half4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_f32") ]]
kernel void copy_aa(device const bfloat4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_f16") ]]
kernel void copy_aa(device const bfloat4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_bf16") ]]
kernel void copy_aa(device const bfloat4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_i64") ]]
kernel void copy_aa(device const bfloat4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_i32") ]]
kernel void copy_aa(device const bfloat4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_i16") ]]
kernel void copy_aa(device const bfloat4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_i8") ]]
kernel void copy_aa(device const bfloat4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_bf16_ui8") ]]
kernel void copy_aa(device const bfloat4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_f32") ]]
kernel void copy_aa(device const long4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_f16") ]]
kernel void copy_aa(device const long4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_bf16") ]]
kernel void copy_aa(device const long4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_i64") ]]
kernel void copy_aa(device const long4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_i32") ]]
kernel void copy_aa(device const long4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_i16") ]]
kernel void copy_aa(device const long4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_i8") ]]
kernel void copy_aa(device const long4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i64_ui8") ]]
kernel void copy_aa(device const long4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_f32") ]]
kernel void copy_aa(device const int4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_f16") ]]
kernel void copy_aa(device const int4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_bf16") ]]
kernel void copy_aa(device const int4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_i64") ]]
kernel void copy_aa(device const int4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_i32") ]]
kernel void copy_aa(device const int4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_i16") ]]
kernel void copy_aa(device const int4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_i8") ]]
kernel void copy_aa(device const int4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i32_ui8") ]]
kernel void copy_aa(device const int4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_f32") ]]
kernel void copy_aa(device const short4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_f16") ]]
kernel void copy_aa(device const short4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_bf16") ]]
kernel void copy_aa(device const short4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_i64") ]]
kernel void copy_aa(device const short4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_i32") ]]
kernel void copy_aa(device const short4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_i16") ]]
kernel void copy_aa(device const short4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_i8") ]]
kernel void copy_aa(device const short4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i16_ui8") ]]
kernel void copy_aa(device const short4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_f32") ]]
kernel void copy_aa(device const char4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_f16") ]]
kernel void copy_aa(device const char4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_bf16") ]]
kernel void copy_aa(device const char4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_i64") ]]
kernel void copy_aa(device const char4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_i32") ]]
kernel void copy_aa(device const char4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_i16") ]]
kernel void copy_aa(device const char4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_i8") ]]
kernel void copy_aa(device const char4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_i8_ui8") ]]
kernel void copy_aa(device const char4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_f32") ]]
kernel void copy_aa(device const uchar4*,
                    device float4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_f16") ]]
kernel void copy_aa(device const uchar4*,
                    device half4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_bf16") ]]
kernel void copy_aa(device const uchar4*,
                    device bfloat4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_i64") ]]
kernel void copy_aa(device const uchar4*,
                    device long4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_i32") ]]
kernel void copy_aa(device const uchar4*,
                    device int4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_i16") ]]
kernel void copy_aa(device const uchar4*,
                    device short4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_i8") ]]
kernel void copy_aa(device const uchar4*,
                    device char4*,
                    constant MatrixSize&,
                    uint index);

template [[ host_name("copy_aa_ui8_ui8") ]]
kernel void copy_aa(device const uchar4*,
                    device uchar4*,
                    constant MatrixSize&,
                    uint index);


// Unary
// -----------------------------------------------------------------
template [[ host_name("unary_a_f32") ]]
kernel void unary_a(device const float4*,
                    constant const float&,
                    constant MatrixSize&,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_f16") ]]
kernel void unary_a(device const half4*,
                    constant const float&,
                    constant MatrixSize&,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_bf16") ]]
kernel void unary_a(device const bfloat4*,
                    constant const float&,
                    constant MatrixSize&,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i64") ]]
kernel void unary_a(device const long4*,
                    constant const float&,
                    constant MatrixSize&,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i32") ]]
kernel void unary_a(device const int4*,
                    constant const float&,
                    constant MatrixSize&,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i16") ]]
kernel void unary_a(device const short4*,
                    constant const float&,
                    constant MatrixSize&,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_i8") ]]
kernel void unary_a(device const char4*,
                    constant const float&,
                    constant MatrixSize&,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("unary_a_ui8") ]]
kernel void unary_a(device const uchar4*,
                    constant const float&,
                    constant MatrixSize&,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);


// Fill
// -----------------------------------------------------------------
template [[ host_name("fill_aa_f32_f32") ]]
kernel void fill_aa(device const float4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_f16") ]]
kernel void fill_aa(device const float4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_bf16") ]]
kernel void fill_aa(device const float4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i64") ]]
kernel void fill_aa(device const float4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i32") ]]
kernel void fill_aa(device const float4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i16") ]]
kernel void fill_aa(device const float4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_i8") ]]
kernel void fill_aa(device const float4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f32_ui8") ]]
kernel void fill_aa(device const float4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_f32") ]]
kernel void fill_aa(device const half4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_f16") ]]
kernel void fill_aa(device const half4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_bf16") ]]
kernel void fill_aa(device const half4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i64") ]]
kernel void fill_aa(device const half4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i32") ]]
kernel void fill_aa(device const half4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i16") ]]
kernel void fill_aa(device const half4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_i8") ]]
kernel void fill_aa(device const half4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_f16_ui8") ]]
kernel void fill_aa(device const half4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_f32") ]]
kernel void fill_aa(device const bfloat4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_f16") ]]
kernel void fill_aa(device const bfloat4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_bf16") ]]
kernel void fill_aa(device const bfloat4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i64") ]]
kernel void fill_aa(device const bfloat4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i32") ]]
kernel void fill_aa(device const bfloat4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i16") ]]
kernel void fill_aa(device const bfloat4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_i8") ]]
kernel void fill_aa(device const bfloat4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_bf16_ui8") ]]
kernel void fill_aa(device const bfloat4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_f32") ]]
kernel void fill_aa(device const long4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_f16") ]]
kernel void fill_aa(device const long4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_bf16") ]]
kernel void fill_aa(device const long4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i64") ]]
kernel void fill_aa(device const long4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i32") ]]
kernel void fill_aa(device const long4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i16") ]]
kernel void fill_aa(device const long4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_i8") ]]
kernel void fill_aa(device const long4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i64_ui8") ]]
kernel void fill_aa(device const long4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_f32") ]]
kernel void fill_aa(device const int4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_f16") ]]
kernel void fill_aa(device const int4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_bf16") ]]
kernel void fill_aa(device const int4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i64") ]]
kernel void fill_aa(device const int4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i32") ]]
kernel void fill_aa(device const int4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i16") ]]
kernel void fill_aa(device const int4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_i8") ]]
kernel void fill_aa(device const int4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i32_ui8") ]]
kernel void fill_aa(device const int4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_f32") ]]
kernel void fill_aa(device const short4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_f16") ]]
kernel void fill_aa(device const short4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_bf16") ]]
kernel void fill_aa(device const short4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i64") ]]
kernel void fill_aa(device const short4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i32") ]]
kernel void fill_aa(device const short4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i16") ]]
kernel void fill_aa(device const short4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_i8") ]]
kernel void fill_aa(device const short4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i16_ui8") ]]
kernel void fill_aa(device const short4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_f32") ]]
kernel void fill_aa(device const char4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_f16") ]]
kernel void fill_aa(device const char4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_bf16") ]]
kernel void fill_aa(device const char4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i64") ]]
kernel void fill_aa(device const char4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i32") ]]
kernel void fill_aa(device const char4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i16") ]]
kernel void fill_aa(device const char4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_i8") ]]
kernel void fill_aa(device const char4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_i8_ui8") ]]
kernel void fill_aa(device const char4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_f32") ]]
kernel void fill_aa(device const uchar4*,
                    device float4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_f16") ]]
kernel void fill_aa(device const uchar4*,
                    device half4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_bf16") ]]
kernel void fill_aa(device const uchar4*,
                    device bfloat4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i64") ]]
kernel void fill_aa(device const uchar4*,
                    device long4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i32") ]]
kernel void fill_aa(device const uchar4*,
                    device int4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i16") ]]
kernel void fill_aa(device const uchar4*,
                    device short4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_i8") ]]
kernel void fill_aa(device const uchar4*,
                    device char4*,
                    uint index [[thread_position_in_grid]]);

template [[ host_name("fill_aa_ui8_ui8") ]]
kernel void fill_aa(device const uchar4*,
                    device uchar4*,
                    uint index [[thread_position_in_grid]]);


// BroadcastTo
// -----------------------------------------------------------------
template [[ host_name("broadcastTo_a_f32") ]]
kernel void broadcastTo_a(device const float* src,
                          device       float* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_f16") ]]
kernel void broadcastTo_a(device const half* src,
                          device       half* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_bf16") ]]
kernel void broadcastTo_a(device const bfloat* src,
                          device       bfloat* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i64") ]]
kernel void broadcastTo_a(device const int64_t* src,
                          device       int64_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i32") ]]
kernel void broadcastTo_a(device const int32_t* src,
                          device       int32_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i16") ]]
kernel void broadcastTo_a(device const int16_t* src,
                          device       int16_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_i8") ]]
kernel void broadcastTo_a(device const int8_t* src,
                          device       int8_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);

template [[ host_name("broadcastTo_a_ui8") ]]
kernel void broadcastTo_a(device const uint8_t* src,
                          device       uint8_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]);


// ReduceTo
// -----------------------------------------------------------------
template [[ host_name("reduceTo_a_f32") ]]
kernel void reduceTo_a(device const float* src,
                       device       float* dst,
                       device const size_t* shape,
                       device const size_t* newShape,
                       constant size_t& shapeSize,
                       constant size_t& newShapeSize,
                       uint index [[thread_position_in_grid]]);

template [[ host_name("reduceTo_a_i32") ]]
kernel void reduceTo_a(device const int* src,
                       device       int* dst,
                       device const size_t* shape,
                       device const size_t* newShape,
                       constant size_t& shapeSize,
                       constant size_t& newShapeSize,
                       uint index [[thread_position_in_grid]]);

// IMPORTANT NOTE: The following specialization is just a dummy kernel. Not for use since atomic<ulong> is not supported.

kernel void reduceTo_a_f16(device const half* src,
                           device       half* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_bf16(device const bfloat* src,
                            device       bfloat* dst,
                            device const size_t* shape,
                            device const size_t* newShape,
                            constant size_t& shapeSize,
                            constant size_t& newShapeSize,
                            uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_i64(device const int64_t* src,
                           device       int64_t* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_i16(device const int16_t* src,
                           device       int16_t* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_i8(device const int8_t* src,
                          device       int8_t* dst,
                          device const size_t* shape,
                          device const size_t* newShape,
                          constant size_t& shapeSize,
                          constant size_t& newShapeSize,
                          uint index [[thread_position_in_grid]]) { }

kernel void reduceTo_a_ui8(device const uint8_t* src,
                           device       uint8_t* dst,
                           device const size_t* shape,
                           device const size_t* newShape,
                           constant size_t& shapeSize,
                           constant size_t& newShapeSize,
                           uint index [[thread_position_in_grid]]) { }


)";


}   // namespace