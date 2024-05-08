//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

namespace mps
{

typedef float DataType;

extern "C"
{
    void* createMPSDevice(int deviceIndex = 0);
    void  releaseMPSDevice(void * device);

    void* allocate(void* device, size_t size);
    void  deallocate(void* device, void* memory);

    void  add_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result);
    void  sub_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result);
    void  mul_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result);
    void  div_a_a(void* device, const DataType * a1, const DataType * a2, size_t size, DataType * result);

    void  add_a_s(void* device, const DataType * a, DataType scalar, size_t size, DataType * result);
    void  sub_s_a(void* device, DataType scalar, const DataType * a, size_t size, DataType * result);
    void  mul_a_s(void* device, const DataType * a, DataType scalar, size_t size, DataType * result);
    void  div_a_s(void* device, const DataType * a, DataType scalar, size_t size, DataType * result);
    void  div_s_a(void* device, DataType scalar, const DataType * a, size_t size, DataType * result);

    void  sqrt_a(void* device, const DataType * a, size_t size, DataType * result);
    void  sin_a(void* device, const DataType * a, size_t size, DataType * result);
    void  cos_a(void* device, const DataType * a, size_t size, DataType * result);
    void  tanh_a(void* device, const DataType * a, size_t size, DataType * result);
    void  log_a(void* device, const DataType * a, size_t size, DataType * result);
    void  exp_a(void* device, const DataType * a, size_t size, DataType * result);

    void  matmul(void* device, const DataType * a1, size_t rows1, size_t cols1, const DataType * a2, size_t rows2, size_t cols2, DataType * result);
    void  transpose(void* device, const DataType * mat, size_t rows, size_t cols, DataType * result);
    void  copy_a_a(void* device, const DataType* src, DataType* dst, size_t size);
    void  copy_s_a(void* device, DataType scalar, size_t size, DataType * result);
}

}   // MPS - Metal Performance Shaders
