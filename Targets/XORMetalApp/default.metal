//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#include <metal_stdlib>
using namespace metal;

struct MatrixSize
{
    uint rows;
    uint cols;
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
    result[index] = inA[index] + inB[index];
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
    result[index] = inA[index] - inB[index];
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
    result[index] = inA[index] * inB[index];
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
    result[index] = inA[index] / inB[index];
}

// -----------------------------------------------------------------
// Add - Array + Scalar
// -----------------------------------------------------------------
template<typename T>
kernel void add_a_s(device const T* inA,
                    constant const T& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + scalar;
}

// -----------------------------------------------------------------
// Sub - Scalar - Array
// -----------------------------------------------------------------
template<typename T>
kernel void sub_s_a(device const T* inA,
                    constant const T& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    result[index] = scalar - inA[index];
}

// -----------------------------------------------------------------
// Mul - Array * Scalar
// -----------------------------------------------------------------
template<typename T>
kernel void mul_a_s(device const T* inA,
                    constant const T& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * scalar;
}

// -----------------------------------------------------------------
// Div - Array / Scalar
// -----------------------------------------------------------------
template<typename T>
kernel void div_a_s(device const T* inA,
                    constant const T& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] / scalar;
}

// -----------------------------------------------------------------
// Div - Scalar / Array
// -----------------------------------------------------------------
template<typename T>
kernel void div_s_a(device const T* inA,
                    constant const T& scalar,
                    constant MatrixSize& aSize,
                    device T* result,
                    uint index [[thread_position_in_grid]])
{
    result[index] = scalar / inA[index];
}

// -----------------------------------------------------------------
// Sqrt - sqrt(Array)
// -----------------------------------------------------------------
template<typename T>
kernel void sqrt_a(device const T* inA,
                   constant const T& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    result[index] = sqrt(inA[index]);
}

// -----------------------------------------------------------------
// Sin - sin(Array)
// -----------------------------------------------------------------
template<typename T>
kernel void sin_a(device const T* inA,
                  constant const T& scalar,
                  constant MatrixSize& aSize,
                  device T* result,
                  uint index [[thread_position_in_grid]])
{
    result[index] = sin(inA[index]);
}

// -----------------------------------------------------------------
// Cos - cos(Array)
// -----------------------------------------------------------------
template<typename T>
kernel void cos_a(device const T* inA,
                   constant const T& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    result[index] = cos(inA[index]);
}

// -----------------------------------------------------------------
// Tanh - tanh(Array)
// -----------------------------------------------------------------
template<typename T>
kernel void tanh_a(device const T* inA,
                   constant const T& scalar,
                   constant MatrixSize& aSize,
                   device T* result,
                   uint index [[thread_position_in_grid]])
{
    result[index] = tanh(inA[index]);
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
// Matrix Transpose - Naive implementation
// -----------------------------------------------------------------
template<typename T>
kernel void matrix_transpose(constant const T* mat,
                             device T* result,
                             constant MatrixSize& matSize,
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]])
{
    uint ofs1 = gid.y * matSize.rows + gid.x;
    uint ofs2 = gid.x * matSize.cols + gid.y;
    result[ofs1] = mat[ofs2];
}


// Templates


template [[ host_name("add_float") ]]
kernel void add(device const float*,
                device const float*,
                device float*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("sub_float") ]]
kernel void sub(device const float*,
                device const float*,
                device float*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("mul_float") ]]
kernel void mul(device const float*,
                device const float*,
                device float*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("div_float") ]]
kernel void div(device const float*,
                device const float*,
                device float*,
                constant MatrixSize&,
                constant MatrixSize&,
                uint index [[thread_position_in_grid]]);


template [[ host_name("add_a_s_float") ]]
kernel void add_a_s(device const float*,
                    constant const float&,
                    constant MatrixSize&,
                    device float*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("sub_s_a_float") ]]
kernel void sub_s_a(device const float*,
                    constant const float&,
                    constant MatrixSize&,
                    device float*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("mul_a_s_float") ]]
kernel void mul_a_s(device const float*,
                    constant const float&,
                    constant MatrixSize&,
                    device float*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("div_a_s_float") ]]
kernel void div_a_s(device const float*,
                    constant const float&,
                    constant MatrixSize&,
                    device float*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("div_s_a_float") ]]
kernel void div_s_a(device const float*,
                    constant const float&,
                    constant MatrixSize&,
                    device float*,
                    uint index [[thread_position_in_grid]]);


template [[ host_name("sqrt_a_float") ]]
kernel void sqrt_a(device const float*,
                   constant const float&,
                   constant MatrixSize&,
                   device float*,
                   uint index [[thread_position_in_grid]]);


template [[ host_name("sin_a_float") ]]
kernel void sin_a(device const float*,
                  constant const float&,
                  constant MatrixSize&,
                  device float*,
                  uint index [[thread_position_in_grid]]);


template [[ host_name("cos_a_float") ]]
kernel void cos_a(device const float*,
                  constant const float&,
                  constant MatrixSize&,
                  device float*,
                  uint index [[thread_position_in_grid]]);


template [[ host_name("tanh_a_float") ]]
kernel void tanh_a(device const float*,
                   constant const float&,
                   constant MatrixSize&,
                   device float*,
                   uint index [[thread_position_in_grid]]);


template [[ host_name("matrix_mul_float") ]]
kernel void matrix_mul(device const float*,
                       device const float*,
                       device float*,
                       constant MatrixSize&,
                       constant MatrixSize&,
                       uint2 gid [[thread_position_in_grid]]);


template [[ host_name("matrix_transpose_float") ]]
kernel void matrix_transpose(constant const float*,
                             device float*,
                             constant MatrixSize&,
                             uint2 gid,
                             uint2 tid);
