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
    if (gid.x < matBSize.cols && gid.y < matASize.rows)
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
