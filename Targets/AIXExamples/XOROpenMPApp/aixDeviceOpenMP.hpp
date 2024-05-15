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
#include <aix.hpp>
// External includes
#include <omp.h>
// System includes


namespace aix
{

class DeviceOpenMP : public aix::Device
{
public:
    DeviceType type() const  override { return DeviceType::kCPU_OMP; }

    void * allocate(size_t size) override
    {
        return std::malloc(size);
    }

    void deallocate(void * memory) override
    {
        return std::free(memory);
    }

    void add(const DataType* a1, const DataType* a2, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] + a2[i];
        }
    }

    void sub(const DataType* a1, const DataType* a2, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] - a2[i];
        }
    }

    void mul(const DataType* a1, const DataType* a2, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] * a2[i];
        }
    }

    void div(const DataType* a1, const DataType* a2, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a1[i] / a2[i];
        }
    }

    // Scalar operations

    void add(const DataType* a, DataType scalar, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] + scalar;
        }
    }

    void sub(const DataType* a, DataType scalar, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] - scalar;
        }
    }

    void sub(DataType scalar, const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar - a[i];
        }
    }

    void mul(const DataType* a, DataType scalar, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] * scalar;
        }
    }

    void div(const DataType* a, DataType scalar, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = a[i] / scalar;
        }
    }

    void div(DataType scalar, const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar / a[i];
        }
    }

    void unary(const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = -a[i];
        }
    }

    void fill(DataType scalar, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = scalar;
        }
    }

    void sum(const DataType* a, const size_t size, DataType & result) override
    {
        DataType sum = 0;
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            sum += a[i];
        }
        result = sum;
    }

    void mean(const DataType* a, const size_t size, DataType & result) override
    {
        DataType sum = 0;
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            sum += a[i];
        }
        result = sum / static_cast<DataType>(size);
    }

    void sqrt(const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::sqrt(a[i]);
        }
    }

    void sin(const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::sin(a[i]);
        }
    }

    void cos(const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::cos(a[i]);
        }
    }

    void tanh(const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::tanh(a[i]);
        }
    }

    void log(const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::log(a[i]);
        }
    }

    void exp(const DataType* a, const size_t size, DataType* result) override
    {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = std::exp(a[i]);
        }
    }

    void matmul(const DataType* a1, const Shape & s1, const DataType* a2, const Shape & s2, DataType* result) override
    {
        // NOTE: Since TensorValue validated the parameters, device method do not validate again.
        size_t m = s1[0];        // Rows of the first matrix
        size_t n = s2[1];        // Columns of the second matrix
        size_t inner = s1[1];    // Inner dimension

        // Perform matrix multiplication
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                DataType sum = 0;
                for (size_t k = 0; k < inner; ++k)
                {
                    sum += a1[i * s1[1] + k] * a2[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }

    void transpose(const DataType* a, const Shape & shape, DataType* result) override
    {
        // Perform the transpose operation.
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < shape[0]; ++i)
        {
            for (size_t j = 0; j < shape[1]; ++j)
            {
                // Swap the indices for the transposition.
                result[j * shape[0] + i] = a[i * shape[1] + j];
            }
        }
    }

    void copy(const DataType* src, DataType* dst, size_t size) override
    {
        std::memcpy(dst, src, size * sizeof(DataType));
    }

    void copy_immediate(const DataType* src, DataType* dst, size_t size) override
    {
        std::memcpy(dst, src, size * sizeof(DataType));
    }
};

}   // namespace
