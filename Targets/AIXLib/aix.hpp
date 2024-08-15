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
#include "aixFloat16.hpp"
// External includes
// System includes
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <stack>
#include <utility>


namespace aix
{

enum class DataType : size_t
{
    kFloat64  = 0,
    kFloat32  = 1,
    kFloat16  = 2,
    kBFloat16 = 3,
    kInt64    = 4,
    kInt32    = 5,
    kInt16    = 6,
    kInt8     = 7,
    kUInt8    = 8,
};

enum class DeviceType
{
    kCPU,
    kGPU_METAL,
};

constexpr size_t DataTypeCount   = 9;
constexpr size_t DeviceTypeCount = 2;

// Primary template (default case)
template <typename T> constexpr DataType getDataType();
template <> constexpr DataType getDataType<double>()     { return DataType::kFloat64;  }
template <> constexpr DataType getDataType<float>()      { return DataType::kFloat32;  }
template <> constexpr DataType getDataType<float16_t>()  { return DataType::kFloat16;  }
template <> constexpr DataType getDataType<bfloat16_t>() { return DataType::kBFloat16; }
template <> constexpr DataType getDataType<int64_t>()    { return DataType::kInt64;    }
template <> constexpr DataType getDataType<int32_t>()    { return DataType::kInt32;    }
template <> constexpr DataType getDataType<int16_t>()    { return DataType::kInt16;    }
template <> constexpr DataType getDataType<int8_t>()     { return DataType::kInt8;     }
template <> constexpr DataType getDataType<uint8_t>()    { return DataType::kUInt8;    }


static DataType promoteDataType(DataType dtype1, DataType dtype2)
{
    assert(static_cast<size_t>(dtype1) < DataTypeCount && static_cast<size_t>(dtype2) < DataTypeCount);
    static_assert(static_cast<size_t>(DataType::kFloat64)  == 0);
    static_assert(static_cast<size_t>(DataType::kFloat32)  == 1);
    static_assert(static_cast<size_t>(DataType::kFloat16)  == 2);
    static_assert(static_cast<size_t>(DataType::kBFloat16) == 3);
    static_assert(static_cast<size_t>(DataType::kInt64)    == 4);
    static_assert(static_cast<size_t>(DataType::kInt32)    == 5);
    static_assert(static_cast<size_t>(DataType::kInt16)    == 6);
    static_assert(static_cast<size_t>(DataType::kInt8)     == 7);
    static_assert(static_cast<size_t>(DataType::kUInt8)    == 8);

    static const size_t promotionTable[DataTypeCount][DataTypeCount] =
    {
        //                  F  F  F  B  I  I  I  I  U
        //                  6  3  1  1  6  3  1  8  8
        //                  4  2  6  6  4  2  6
        /*  kFloat64  */  { 0, 0, 0, 0, 0, 0, 0, 0, 0, },
        /*  kFloat32  */  { 0, 1, 1, 1, 1, 1, 1, 1, 1, },
        /*  kFloat16  */  { 0, 1, 2, 1, 2, 2, 2, 2, 2, },
        /*  kBFloat16 */  { 0, 1, 1, 3, 3, 3, 3, 3, 3, },
        /*  kInt64    */  { 0, 1, 2, 3, 4, 4, 4, 4, 4, },
        /*  kInt32    */  { 0, 1, 2, 3, 4, 5, 5, 5, 5, },
        /*  kInt16    */  { 0, 1, 2, 3, 4, 5, 6, 6, 6, },
        /*  kInt8     */  { 0, 1, 2, 3, 4, 5, 6, 7, 6, },
        /*  kUInt8    */  { 0, 1, 2, 3, 4, 5, 6, 6, 8, },
    };

    return static_cast<DataType>(promotionTable[static_cast<size_t>(dtype1)][static_cast<size_t>(dtype2)]);
}

// Promotes a data type to Float32 if the type is an integer type, otherwise it returns the same float data type.
static DataType promoteDataTypeToFloat(DataType dtype)
{
    assert(static_cast<size_t>(dtype) < DataTypeCount);
    static const size_t formatConversionTable[DataTypeCount] =
    {
    //  F  F  F  B  I  I  I  I  U
    //  6  3  1  1  6  3  1  8  8
    //  4  2  6  6  4  2  6
        0, 1, 2, 3, 1, 1, 1, 1, 1,
    };
    return static_cast<DataType>(formatConversionTable[static_cast<size_t>(dtype)]);
}

// Forward declarations
class Tensor;

// Tensor Index, Shape and Stride Types
using Index  = std::vector<size_t>;
using Shape  = std::vector<size_t>;
using Stride = std::vector<size_t>;


class Device
{
public:
    // Constructor.
    explicit Device([[maybe_unused]] size_t deviceIndex = 0) { }

    // Destructor.
    virtual ~Device() = default;

    virtual DeviceType type() const { return DeviceType::kCPU; }
    virtual std::string name() const { return "CPU"; }

    static size_t dataTypeSize(DataType dtype)
    {
        static const size_t dTypeSizeTable[DataTypeCount]
        {
            sizeof(double    ),   // kFloat64
            sizeof(float     ),   // kFloat32
            sizeof(float16_t ),   // kFloat16
            sizeof(bfloat16_t),   // kBFloat16
            sizeof(int64_t   ),   // kInt64
            sizeof(int32_t   ),   // kInt32
            sizeof(int16_t   ),   // kInt16
            sizeof(int8_t    ),   // kInt8
            sizeof(uint8_t   ),   // kUInt8
        };
        return dTypeSizeTable[static_cast<size_t>(dtype)];
    }

    virtual void* allocate(size_t size)
    {
        return std::malloc(size);
    }

    virtual void* allocate(size_t size, DataType dtype)
    {
        return allocate(size * dataTypeSize(dtype));
    }

    virtual void deallocate(void * memory)
    {
        return std::free(memory);
    }

    virtual void add(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            addGeneric<double    >,
            addGeneric<float     >,
            addGeneric<float16_t >,
            addGeneric<bfloat16_t>,
            addGeneric<int64_t   >,
            addGeneric<int32_t   >,
            addGeneric<int16_t   >,
            addGeneric<int8_t    >,
            addGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void sub(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            subGeneric<double    >,
            subGeneric<float     >,
            subGeneric<float16_t >,
            subGeneric<bfloat16_t>,
            subGeneric<int64_t   >,
            subGeneric<int32_t   >,
            subGeneric<int16_t   >,
            subGeneric<int8_t    >,
            subGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void mul(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            mulGeneric<double    >,
            mulGeneric<float     >,
            mulGeneric<float16_t >,
            mulGeneric<bfloat16_t>,
            mulGeneric<int64_t   >,
            mulGeneric<int32_t   >,
            mulGeneric<int16_t   >,
            mulGeneric<int8_t    >,
            mulGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void div(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            divGeneric<double    >,
            divGeneric<float     >,
            divGeneric<float16_t >,
            divGeneric<bfloat16_t>,
            divGeneric<int64_t   >,
            divGeneric<int32_t   >,
            divGeneric<int16_t   >,
            divGeneric<int8_t    >,
            divGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void unary(const void* a1, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            unaryGeneric<double    >,
            unaryGeneric<float     >,
            unaryGeneric<float16_t >,
            unaryGeneric<bfloat16_t>,
            unaryGeneric<int64_t   >,
            unaryGeneric<int32_t   >,
            unaryGeneric<int16_t   >,
            unaryGeneric<int8_t    >,
            unaryGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, size, result);
    }

    virtual void fill(const void* scalar, DataType srcDType, size_t size, void* result, DataType dstDType)
    {
        // Define a function pointer type for the conversion copy functions.
        using fillFunc = void (*)(const void*, void*, size_t);

        // Create a lookup table of the functions.
        static const fillFunc funcTable[DataTypeCount][DataTypeCount] =
        {
            { fillGeneric<double, double>,     fillGeneric<double, float>,     fillGeneric<double, float16_t>,     fillGeneric<double, bfloat16_t>,     fillGeneric<double, int64_t>,     fillGeneric<double, int32_t>,     fillGeneric<double, int16_t>,     fillGeneric<double, int8_t>,     fillGeneric<double, uint8_t>     },
            { fillGeneric<float, double>,      fillGeneric<float, float>,      fillGeneric<float, float16_t>,      fillGeneric<float, bfloat16_t>,      fillGeneric<float, int64_t>,      fillGeneric<float, int32_t>,      fillGeneric<float, int16_t>,      fillGeneric<float, int8_t>,      fillGeneric<float, uint8_t>      },
            { fillGeneric<float16_t, double>,  fillGeneric<float16_t, float>,  fillGeneric<float16_t, float16_t>,  fillGeneric<float16_t, bfloat16_t>,  fillGeneric<float16_t, int64_t>,  fillGeneric<float16_t, int32_t>,  fillGeneric<float16_t, int16_t>,  fillGeneric<float16_t, int8_t>,  fillGeneric<float16_t, uint8_t>  },
            { fillGeneric<bfloat16_t, double>, fillGeneric<bfloat16_t, float>, fillGeneric<bfloat16_t, float16_t>, fillGeneric<bfloat16_t, bfloat16_t>, fillGeneric<bfloat16_t, int64_t>, fillGeneric<bfloat16_t, int32_t>, fillGeneric<bfloat16_t, int16_t>, fillGeneric<bfloat16_t, int8_t>, fillGeneric<bfloat16_t, uint8_t> },
            { fillGeneric<int64_t, double>,    fillGeneric<int64_t, float>,    fillGeneric<int64_t, float16_t>,    fillGeneric<int64_t, bfloat16_t>,    fillGeneric<int64_t, int64_t>,    fillGeneric<int64_t, int32_t>,    fillGeneric<int64_t, int16_t>,    fillGeneric<int64_t, int8_t>,    fillGeneric<int64_t, uint8_t>    },
            { fillGeneric<int32_t, double>,    fillGeneric<int32_t, float>,    fillGeneric<int32_t, float16_t>,    fillGeneric<int32_t, bfloat16_t>,    fillGeneric<int32_t, int64_t>,    fillGeneric<int32_t, int32_t>,    fillGeneric<int32_t, int16_t>,    fillGeneric<int32_t, int8_t>,    fillGeneric<int32_t, uint8_t>    },
            { fillGeneric<int16_t, double>,    fillGeneric<int16_t, float>,    fillGeneric<int16_t, float16_t>,    fillGeneric<int16_t, bfloat16_t>,    fillGeneric<int16_t, int64_t>,    fillGeneric<int16_t, int32_t>,    fillGeneric<int16_t, int16_t>,    fillGeneric<int16_t, int8_t>,    fillGeneric<int16_t, uint8_t>    },
            { fillGeneric<int8_t, double>,     fillGeneric<int8_t, float>,     fillGeneric<int8_t,  float16_t>,    fillGeneric<int8_t,  bfloat16_t>,    fillGeneric<int8_t, int64_t>,     fillGeneric<int8_t, int32_t>,     fillGeneric<int8_t, int16_t>,     fillGeneric<int8_t, int8_t>,     fillGeneric<int8_t, uint8_t>     },
            { fillGeneric<uint8_t, double>,    fillGeneric<uint8_t, float>,    fillGeneric<uint8_t, float16_t>,    fillGeneric<uint8_t, bfloat16_t>,    fillGeneric<uint8_t, int64_t>,    fillGeneric<uint8_t, int32_t>,    fillGeneric<uint8_t, int16_t>,    fillGeneric<uint8_t, int8_t>,    fillGeneric<uint8_t, uint8_t>    },
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(srcDType)][static_cast<size_t>(dstDType)](scalar, result, size);
    }

    virtual void fillMin(DataType dtype, size_t size, void* result)
    {
        // Create a lookup table of the functions.
        static const auto funcTable = std::array
        {
            fillMinGeneric<double    >,
            fillMinGeneric<float     >,
            fillMinGeneric<float16_t >,
            fillMinGeneric<bfloat16_t>,
            fillMinGeneric<int64_t   >,
            fillMinGeneric<int32_t   >,
            fillMinGeneric<int16_t   >,
            fillMinGeneric<int8_t    >,
            fillMinGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](result, size);
    }

    virtual void sum(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sumGeneric<double    >,
            sumGeneric<float     >,
            sumGeneric<float16_t >,
            sumGeneric<bfloat16_t>,
            sumGeneric<int64_t   >,
            sumGeneric<int32_t   >,
            sumGeneric<int16_t   >,
            sumGeneric<int8_t    >,
            sumGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void sqrt(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sqrtGeneric<double    >,
            sqrtGeneric<float     >,
            sqrtGeneric<float16_t >,
            sqrtGeneric<bfloat16_t>,
            sqrtGeneric<int64_t   >,
            sqrtGeneric<int32_t   >,
            sqrtGeneric<int16_t   >,
            sqrtGeneric<int8_t    >,
            sqrtGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void sin(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sinGeneric<double    >,
            sinGeneric<float     >,
            sinGeneric<float16_t >,
            sinGeneric<bfloat16_t>,
            sinGeneric<int64_t   >,
            sinGeneric<int32_t   >,
            sinGeneric<int16_t   >,
            sinGeneric<int8_t    >,
            sinGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void cos(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            cosGeneric<double    >,
            cosGeneric<float     >,
            cosGeneric<float16_t >,
            cosGeneric<bfloat16_t>,
            cosGeneric<int64_t   >,
            cosGeneric<int32_t   >,
            cosGeneric<int16_t   >,
            cosGeneric<int8_t    >,
            cosGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void tanh(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            tanhGeneric<double    >,
            tanhGeneric<float     >,
            tanhGeneric<float16_t >,
            tanhGeneric<bfloat16_t>,
            tanhGeneric<int64_t   >,
            tanhGeneric<int32_t   >,
            tanhGeneric<int16_t   >,
            tanhGeneric<int8_t    >,
            tanhGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void log(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            logGeneric<double    >,
            logGeneric<float     >,
            logGeneric<float16_t >,
            logGeneric<bfloat16_t>,
            logGeneric<int64_t   >,
            logGeneric<int32_t   >,
            logGeneric<int16_t   >,
            logGeneric<int8_t    >,
            logGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void exp(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            expGeneric<double    >,
            expGeneric<float     >,
            expGeneric<float16_t >,
            expGeneric<bfloat16_t>,
            expGeneric<int64_t   >,
            expGeneric<int32_t   >,
            expGeneric<int16_t   >,
            expGeneric<int8_t    >,
            expGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void pow(const void* a, const void* exp, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            powGeneric<double    >,
            powGeneric<float     >,
            powGeneric<float16_t >,
            powGeneric<bfloat16_t>,
            powGeneric<int64_t   >,
            powGeneric<int32_t   >,
            powGeneric<int16_t   >,
            powGeneric<int8_t    >,
            powGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, exp, size, result);
    }

    virtual void max(const void* a, size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            maxGeneric<double    >,
            maxGeneric<float     >,
            maxGeneric<float16_t >,
            maxGeneric<bfloat16_t>,
            maxGeneric<int64_t   >,
            maxGeneric<int32_t   >,
            maxGeneric<int16_t   >,
            maxGeneric<int8_t    >,
            maxGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void argmax(const void* a, size_t size, void* result, DataType dtype, DataType resultDtype)
    {
        if (resultDtype != DataType::kInt32)
        {
            throw std::invalid_argument("Device::argmax supports only int32 data type for its result.");
        }

        static const auto funcTable = std::array
        {
            argmaxGeneric<double    , int32_t>,
            argmaxGeneric<float     , int32_t>,
            argmaxGeneric<float16_t , int32_t>,
            argmaxGeneric<bfloat16_t, int32_t>,
            argmaxGeneric<int64_t   , int32_t>,
            argmaxGeneric<int32_t   , int32_t>,
            argmaxGeneric<int16_t   , int32_t>,
            argmaxGeneric<int8_t    , int32_t>,
            argmaxGeneric<uint8_t   , int32_t>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void argmaxIndices(const void* a, size_t size, void* result, DataType dtype, DataType resultDtype)
    {
        if (resultDtype != DataType::kInt32)
        {
            throw std::invalid_argument("Device::argmaxIndices supports only int32 data type for its result.");
        }

        static const auto funcTable = std::array
        {
            argmaxIndicesGeneric<double    , int32_t>,
            argmaxIndicesGeneric<float     , int32_t>,
            argmaxIndicesGeneric<float16_t , int32_t>,
            argmaxIndicesGeneric<bfloat16_t, int32_t>,
            argmaxIndicesGeneric<int64_t   , int32_t>,
            argmaxIndicesGeneric<int32_t   , int32_t>,
            argmaxIndicesGeneric<int16_t   , int32_t>,
            argmaxIndicesGeneric<int8_t    , int32_t>,
            argmaxIndicesGeneric<uint8_t   , int32_t>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void matmul(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            matmulGeneric<double    >,
            matmulGeneric<float     >,
            matmulGeneric<float16_t >,
            matmulGeneric<bfloat16_t>,
            matmulGeneric<int64_t   >,
            matmulGeneric<int32_t   >,
            matmulGeneric<int16_t   >,
            matmulGeneric<int8_t    >,
            matmulGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, s1, a2, s2, result);
    }

    virtual void transpose(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                           const Stride& strides, const Stride& newStrides, const size_t size, void* result,
                           DataType dtype)
    {
        static const auto funcTable = std::array
        {
            transposeGeneric<double    >,
            transposeGeneric<float     >,
            transposeGeneric<float16_t >,
            transposeGeneric<bfloat16_t>,
            transposeGeneric<int64_t   >,
            transposeGeneric<int32_t   >,
            transposeGeneric<int16_t   >,
            transposeGeneric<int8_t    >,
            transposeGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](dim0, dim1, data, shape, strides, newStrides, size, result);
    }

    virtual void copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
    {
        // Define a function pointer type for the conversion copy functions.
        using copyFunc = void (*)(const void*, void*, size_t);

        // Create a lookup table of the functions.
        static const copyFunc funcTable[DataTypeCount][DataTypeCount] =
        {
            { copyGeneric<double, double>,     copyGeneric<double, float>,     copyGeneric<double, float16_t>,     copyGeneric<double, bfloat16_t>,     copyGeneric<double, int64_t>,     copyGeneric<double, int32_t>,     copyGeneric<double, int16_t>,     copyGeneric<double, int8_t>,     copyGeneric<double, uint8_t>     },
            { copyGeneric<float, double>,      copyGeneric<float, float>,      copyGeneric<float, float16_t>,      copyGeneric<float, bfloat16_t>,      copyGeneric<float, int64_t>,      copyGeneric<float, int32_t>,      copyGeneric<float, int16_t>,      copyGeneric<float, int8_t>,      copyGeneric<float, uint8_t>      },
            { copyGeneric<float16_t, double>,  copyGeneric<float16_t, float>,  copyGeneric<float16_t, float16_t>,  copyGeneric<float16_t, bfloat16_t>,  copyGeneric<float16_t, int64_t>,  copyGeneric<float16_t, int32_t>,  copyGeneric<float16_t, int16_t>,  copyGeneric<float16_t, int8_t>,  copyGeneric<float16_t, uint8_t>  },
            { copyGeneric<bfloat16_t, double>, copyGeneric<bfloat16_t, float>, copyGeneric<bfloat16_t, float16_t>, copyGeneric<bfloat16_t, bfloat16_t>, copyGeneric<bfloat16_t, int64_t>, copyGeneric<bfloat16_t, int32_t>, copyGeneric<bfloat16_t, int16_t>, copyGeneric<bfloat16_t, int8_t>, copyGeneric<bfloat16_t, uint8_t> },
            { copyGeneric<int64_t, double>,    copyGeneric<int64_t, float>,    copyGeneric<int64_t, float16_t>,    copyGeneric<int64_t, bfloat16_t>,    copyGeneric<int64_t, int64_t>,    copyGeneric<int64_t, int32_t>,    copyGeneric<int64_t, int16_t>,    copyGeneric<int64_t, int8_t>,    copyGeneric<int64_t, uint8_t>    },
            { copyGeneric<int32_t, double>,    copyGeneric<int32_t, float>,    copyGeneric<int32_t, float16_t>,    copyGeneric<int32_t, bfloat16_t>,    copyGeneric<int32_t, int64_t>,    copyGeneric<int32_t, int32_t>,    copyGeneric<int32_t, int16_t>,    copyGeneric<int32_t, int8_t>,    copyGeneric<int32_t, uint8_t>    },
            { copyGeneric<int16_t, double>,    copyGeneric<int16_t, float>,    copyGeneric<int16_t, float16_t>,    copyGeneric<int16_t, bfloat16_t>,    copyGeneric<int16_t, int64_t>,    copyGeneric<int16_t, int32_t>,    copyGeneric<int16_t, int16_t>,    copyGeneric<int16_t, int8_t>,    copyGeneric<int16_t, uint8_t>    },
            { copyGeneric<int8_t, double>,     copyGeneric<int8_t, float>,     copyGeneric<int8_t,  float16_t>,    copyGeneric<int8_t,  bfloat16_t>,    copyGeneric<int8_t, int64_t>,     copyGeneric<int8_t, int32_t>,     copyGeneric<int8_t, int16_t>,     copyGeneric<int8_t, int8_t>,     copyGeneric<int8_t, uint8_t>     },
            { copyGeneric<uint8_t, double>,    copyGeneric<uint8_t, float>,    copyGeneric<uint8_t, float16_t>,    copyGeneric<uint8_t, bfloat16_t>,    copyGeneric<uint8_t, int64_t>,    copyGeneric<uint8_t, int32_t>,    copyGeneric<uint8_t, int16_t>,    copyGeneric<uint8_t, int8_t>,    copyGeneric<uint8_t, uint8_t>    },
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(srcDType)][static_cast<size_t>(dstDType)](src, dst, size);
    }

    virtual void copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
    {
        copy(src, srcDType, dst, dstDType, size);
        commitAndWait();    // This call has no effect, but it shows the difference between copy and copyImmediate.
    }

    virtual void broadcastTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                             DataType dtype)
    {
        static const auto funcTable = std::array
        {
            broadcastToGeneric<double    >,
            broadcastToGeneric<float     >,
            broadcastToGeneric<float16_t >,
            broadcastToGeneric<bfloat16_t>,
            broadcastToGeneric<int64_t   >,
            broadcastToGeneric<int32_t   >,
            broadcastToGeneric<int16_t   >,
            broadcastToGeneric<int8_t    >,
            broadcastToGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, size, shape, newShape);
    }

    virtual void reduceTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                          DataType dtype)
    {
        static const auto funcTable = std::array
        {
            reduceToGeneric<double    >,
            reduceToGeneric<float     >,
            reduceToGeneric<float16_t >,
            reduceToGeneric<bfloat16_t>,
            reduceToGeneric<int64_t   >,
            reduceToGeneric<int32_t   >,
            reduceToGeneric<int16_t   >,
            reduceToGeneric<int8_t    >,
            reduceToGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, size, shape, newShape);
    }

    virtual void maxTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                       DataType dtype)
    {
        static const auto funcTable = std::array
        {
            maxToGeneric<double    >,
            maxToGeneric<float     >,
            maxToGeneric<float16_t >,
            maxToGeneric<bfloat16_t>,
            maxToGeneric<int64_t   >,
            maxToGeneric<int32_t   >,
            maxToGeneric<int16_t   >,
            maxToGeneric<int8_t    >,
            maxToGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, size, shape, newShape);
    }

    virtual void argmaxTo(const void* src, void* dst, size_t srcSize, size_t dstSize,
                          const Shape& shape, const Shape& newShape, const Shape& strides, size_t dim,
                          DataType dtype, DataType resultDtype)
    {
        if (resultDtype != DataType::kInt32)
        {
            throw std::invalid_argument("Device::argmaxTo supports only int32 data type for its result.");
        }

        static const auto funcTable = std::array
        {
            argmaxToGeneric<double    , int32_t>,
            argmaxToGeneric<float     , int32_t>,
            argmaxToGeneric<float16_t , int32_t>,
            argmaxToGeneric<bfloat16_t, int32_t>,
            argmaxToGeneric<int64_t   , int32_t>,
            argmaxToGeneric<int32_t   , int32_t>,
            argmaxToGeneric<int16_t   , int32_t>,
            argmaxToGeneric<int8_t    , int32_t>,
            argmaxToGeneric<uint8_t   , int32_t>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, srcSize, dstSize, shape, newShape, strides, dim);
    }

    virtual void argmaxIndicesTo(const void* src, void* dst, size_t srcSize, size_t dstSize,
                                 const Shape& shape, const Shape& newShape, DataType dtype, DataType resultDtype)
    {
        if (resultDtype != DataType::kInt32)
        {
            throw std::invalid_argument("Device::argmaxIndicesTo supports only int32 data type for its result.");
        }

        static const auto funcTable = std::array
        {
            argmaxIndicesToGeneric<double    , int32_t>,
            argmaxIndicesToGeneric<float     , int32_t>,
            argmaxIndicesToGeneric<float16_t , int32_t>,
            argmaxIndicesToGeneric<bfloat16_t, int32_t>,
            argmaxIndicesToGeneric<int64_t   , int32_t>,
            argmaxIndicesToGeneric<int32_t   , int32_t>,
            argmaxIndicesToGeneric<int16_t   , int32_t>,
            argmaxIndicesToGeneric<int8_t    , int32_t>,
            argmaxIndicesToGeneric<uint8_t   , int32_t>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, srcSize, dstSize, shape, newShape);
    }

    virtual void slice(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                       const Shape& strides, size_t dim, size_t start, size_t step, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sliceGeneric<double    >,
            sliceGeneric<float     >,
            sliceGeneric<float16_t >,
            sliceGeneric<bfloat16_t>,
            sliceGeneric<int64_t   >,
            sliceGeneric<int32_t   >,
            sliceGeneric<int16_t   >,
            sliceGeneric<int8_t    >,
            sliceGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, size, shape, newShape, strides, dim, start, step);
    }

    virtual void sliceSet(void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                            const Shape& strides, size_t dim, size_t start, size_t step, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sliceSetGeneric<double    >,
            sliceSetGeneric<float     >,
            sliceSetGeneric<float16_t >,
            sliceSetGeneric<bfloat16_t>,
            sliceSetGeneric<int64_t   >,
            sliceSetGeneric<int32_t   >,
            sliceSetGeneric<int16_t   >,
            sliceSetGeneric<int8_t    >,
            sliceSetGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, size, shape, newShape, strides, dim, start, step);
    }

    virtual void tril(void* dst, size_t size, const Shape& shape, const Shape& strides, ssize_t diagonal, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            trilGeneric<double    >,
            trilGeneric<float     >,
            trilGeneric<float16_t >,
            trilGeneric<bfloat16_t>,
            trilGeneric<int64_t   >,
            trilGeneric<int32_t   >,
            trilGeneric<int16_t   >,
            trilGeneric<int8_t    >,
            trilGeneric<uint8_t   >,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](dst, size, shape, strides, diagonal);
    }

    virtual void commitAndWait()
    {
    }

protected:
    template <typename T>
    static void addGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] + t2[i];
        }
    }

    template <typename T>
    static void subGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] - t2[i];
        }
    }

    template <typename T>
    static void mulGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] * t2[i];
        }
    }

    template <typename T>
    static void divGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] / t2[i];
        }
    }

    template <typename T>
    static void unaryGeneric(const void* a1, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = -t1[i];
        }
    }

    template <typename SrcType, typename DstType>
    static void fillGeneric(const void* src, void* dst, size_t size)
    {
        auto tSrc = static_cast<const SrcType*>(src);
        auto tDst = static_cast<DstType*>(dst);
        for (size_t i=0; i<size; ++i)
        {
            tDst[i] = static_cast<DstType>(tSrc[0]);
        }
    }

    template <typename T>
    static void fillMinGeneric(void* dst, size_t size)
    {
        auto tDst = static_cast<T*>(dst);
        for (size_t i=0; i<size; ++i)
        {
            tDst[i] = std::numeric_limits<T>::lowest();
        }
    }

    template <typename T>
    static void sumGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        T sum = 0;
        for (size_t i = 0; i < size; ++i)
        {
            sum += t1[i];
        }
        *res = sum;
    }

    template <typename T>
    static void sqrtGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::sqrt(t1[i]);
        }
    }

    template <typename T>
    static void sinGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::sin(t1[i]);
        }
    }

    template <typename T>
    static void cosGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::cos(t1[i]);
        }
    }

    template <typename T>
    static void tanhGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::tanh(t1[i]);
        }
    }

    template <typename T>
    static void logGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::log(t1[i]);
        }
    }

    template <typename T>
    static void expGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::exp(t1[i]);
        }
    }

    template <typename T>
    static void powGeneric(const void* a1, const void* exp, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(exp);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::pow(t1[i], t2[i]);
        }
    }

    template <typename T>
    static void maxGeneric(const void* a, const size_t size, void* result)
    {
        auto t  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        res[0] = t[0];
        for (size_t i = 1; i < size; ++i)
        {
            res[0] = std::max<T>(res[0], t[i]);
        }
    }

    template <typename T, typename T2>
    static void argmaxGeneric(const void* a, const size_t size, void* result)
    {
        auto t  = static_cast<const T*>(a);
        auto res = static_cast<T2*>(result);

        T max = t[0];
        res[0] = 0;
        for (size_t i = 1; i < size; ++i)
        {
            if (t[i] > max)
            {
                max = t[i];
                res[0] = i;
            }
        }
    }

    template <typename T, typename T2>
    static void argmaxToGeneric(const void* src, void* dst, size_t srcSize, size_t dstSize,
                                const Shape& shape, const Shape& newShape, const Shape& strides, size_t dim)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T2*>(dst);
        auto tmaxTemp = new T[dstSize];     // Temporary helper buffer to store max values for comparison.

        // Initialize the temp buffer with the lowest value of the data type, T.
        fillMinGeneric<T>(tmaxTemp, dstSize);

        for (size_t index = 0; index < srcSize; ++index)
        {
            auto transIndex = translationIndex(index, newShape, shape);
            if (tSrc[index] > tmaxTemp[transIndex])
            {
                tmaxTemp[transIndex] = tSrc[index];
                tDst[transIndex] = (index / strides[dim]) % shape[dim];
            }
        }
        delete [] tmaxTemp;
    }

    template <typename T, typename T2>
    static void argmaxIndicesGeneric(const void* a, const size_t size, void* result)
    {
        auto t  = static_cast<const T*>(a);
        auto res = static_cast<T2*>(result);

        T max = t[0];
        T2 index = res[0] = 0;
        for (size_t i = 1; i < size; ++i)
        {
            res[i] = 0;
            if (t[i] > max)
            {
                max = t[i];
                index = i;
            }
        }
        res[index] = 1;
    }

    template <typename T, typename T2>
    static void argmaxIndicesToGeneric(const void* src, void* dst, size_t srcSize, size_t dstSize,
                                       const Shape& shape, const Shape& newShape)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T2*>(dst);
        auto tmaxTemp = new T[dstSize];     // Temporary helper buffer to store max values for comparison.

        // Initialize the temp buffer with the lowest value of the data type, T.
        fillMinGeneric<T>(tmaxTemp, dstSize);
        size_t maxElementCount = 1;
        for (auto i : newShape)
        {
            maxElementCount *= i;
        }

        auto tDstTemp = new T2[maxElementCount];   // Temporary helper buffer to store index of max elements.

        for (size_t index = 0; index < srcSize; ++index)
        {
            auto transIndex = translationIndex(index, newShape, shape);
            if (tSrc[index] > tmaxTemp[transIndex])
            {
                tmaxTemp[transIndex] = tSrc[index];
                tDstTemp[transIndex] = index;
            }
        }

        for (size_t i = 0; i < maxElementCount; ++i)
        {
            tDst[tDstTemp[i]] = 1;
        }

        delete [] tmaxTemp;
        delete [] tDstTemp;
    }

    template <typename T>
    static void matmulGeneric(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        // NOTE: Since TensorValue validated the parameters, device method do not validate again.
        size_t m = s1[0];        // Rows of the first matrix
        size_t n = s2[1];        // Columns of the second matrix
        size_t inner = s1[1];    // Inner dimension

        // Perform matrix multiplication
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < inner; ++k)
                {
                    sum += t1[i * s1[1] + k] * t2[k * n + j];
                }
                res[i * n + j] = sum;
            }
        }
    }

    template <typename T>
    static void transposeGeneric(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                                 const Stride& strides, const Stride& newStrides, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(data);
        auto res = static_cast<T*>(result);

        // Perform the generalized transpose operation.
        for (size_t i=0; i<size; ++i)
        {
            auto oldIndices = unflattenIndex(i, strides);
            std::swap(oldIndices[dim0], oldIndices[dim1]);
            size_t newIndex = flattenIndex(oldIndices, newStrides);
            res[newIndex] = t1[i];
        }
    }

    template <typename SrcType, typename DstType>
    static void copyGeneric(const void* src, void* dst, size_t size)
    {
        if constexpr (std::is_same_v<SrcType, DstType>)
        {
            std::memcpy(dst, src, size * sizeof(SrcType));
        }
        else
        {
            auto tSrc = static_cast<const SrcType*>(src);
            auto tDst = static_cast<DstType*>(dst);
            for (size_t i=0; i<size; ++i)
            {
                tDst[i] = static_cast<DstType>(tSrc[i]);
            }
        }
    }

    template <typename T>
    static void broadcastToGeneric(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T*>(dst);

        for (size_t index = 0; index < size; ++index)
        {
            // Copy value from original index to the new index.
            tDst[index] = tSrc[translationIndex(index, shape, newShape)];
        }
    }

    template <typename T>
    static void reduceToGeneric(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T*>(dst);

        // Sum the values from the broadcasted tensor to the original tensor shape. The reduction involves summation
        // because each element of the original tensor is used multiple times in the broadcasted operation.
        // Summing the gradients correctly aggregates these contributions.
        for (size_t index = 0; index < size; ++index)
        {
            tDst[translationIndex(index, newShape, shape)] += tSrc[index];
        }
    }

    template <typename T>
    static void maxToGeneric(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T*>(dst);

        for (size_t index = 0; index < size; ++index)
        {
            auto transIndex = translationIndex(index, newShape, shape);
            tDst[transIndex] = std::max<T>(tDst[transIndex], tSrc[index]);
        }
    }

    template <typename T>
    static void sliceGeneric(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                             const Shape& strides, size_t dim, size_t start, size_t step)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T*>(dst);

        // Iterate over all elements in the destination tensor.
        for (size_t index = 0; index < size; ++index)
        {
            // Translate the flat index into multi-dimensional indices.
            size_t dstIndex = index;
            size_t srcIndex = 0;

            for (ssize_t i = static_cast<ssize_t>(shape.size()) - 1; i >= 0; --i)
            {
                size_t coordinate = dstIndex % newShape[i];
                dstIndex /= newShape[i];

                if (i == static_cast<ssize_t>(dim))   // Handle the slicing dimension.
                    srcIndex += (start + coordinate * step) * strides[i];
                else
                    srcIndex += coordinate * strides[i];
            }

            // Copy the element from the source to the destination.
            tDst[index] = tSrc[srcIndex];
        }
    }

    template <typename T>
    static void sliceSetGeneric(void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                                const Shape& strides, size_t dim, size_t start, size_t step)
    {
        auto tSrc = static_cast<T*>(src);
        auto tDst = static_cast<T*>(dst);

        for (size_t index = 0; index < size; ++index)
        {
            // Translate the flat index into multi-dimensional indices.
            size_t dstIndex = index;
            size_t srcIndex = 0;

            for (ssize_t i = static_cast<ssize_t>(shape.size()) - 1; i >= 0; --i)
            {
                size_t coordinate = dstIndex % newShape[i];
                dstIndex /= newShape[i];

                if (i == static_cast<ssize_t>(dim))   // Handle the slicing dimension.
                    srcIndex += (start + coordinate * step) * strides[i];
                else
                    srcIndex += coordinate * strides[i];
            }

            tDst[srcIndex] = tSrc[index];
        }
    }

    template <typename T>
    static void trilGeneric(void* dst, size_t size, const Shape& shape, const Shape& strides, ssize_t diagonal)
    {
        auto tDst = static_cast<T*>(dst);

        size_t shapeSize = shape.size();
        size_t rows = shape[shapeSize - 2];      // Rows in the last 2-dim tensor.
        size_t cols = shape[shapeSize - 1];      // Columns in the last 2-dim tensor.

        for (size_t i = 0; i < size; ++i)
        {
            // Calculate the row and column indices for the last 2-dim slice.
            size_t row = (i / strides[shapeSize - 2]) % rows;
            size_t col = (i / strides[shapeSize - 1]) % cols;

            // Zero out the elements above the specified diagonal.
            if (static_cast<ssize_t>(col) > static_cast<ssize_t>(row) + diagonal)
            {
                tDst[i] = 0;
            }
        }
    }

    // Helper Methods

    // Calculate the translation index, originalIndex, to copy data from the original index to the new index.
    static size_t translationIndex(size_t index, const Shape& shape, const Shape& newShape)
    {
        size_t originalIndex  = 0;
        size_t targetStride   = 1;
        size_t originalStride = 1;

        for (ssize_t i = newShape.size() - 1, j = shape.size() - 1; i >= 0; --i)
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

    static size_t flattenIndex(const Stride& indices, const Stride& strides)
    {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            index += indices[i] * strides[i];
        }
        return index;
    }

    static Stride unflattenIndex(size_t index, const Stride& strides)
    {
        Stride indices(strides.size());
        for (size_t i = 0; i < strides.size(); ++i)
        {
            indices[i] = index / strides[i];
            index %= strides[i];
        }
        return indices;
    }
};


// TODO: Global parameters needs to move to a global context.
static Device defaultDevice;
static std::random_device randomDevice;
static std::mt19937 randGen(randomDevice());


class TensorValue
{
public:
    // Constructor
    TensorValue(const void* data, size_t size, DataType srcDType, Shape shape, Device* device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_data = device->allocate(size, dType);
        device->copy(data, srcDType, m_data, dType, size);
        m_size = size;
        // Compute the strides for indexing multi-dimensional data.
        m_strides = computeStrides();
    }

    // Constructor
    template<typename T>
    TensorValue(const std::initializer_list<T> & data, Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_data = device->allocate(data.size(), dType);
        device->copy(data.begin(), getDataType<T>(), m_data, dType, data.size());
        m_size = data.size();
        // Compute the strides for indexing multi-dimensional data.
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(float value, Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = device->allocate(m_size, dType);
        // initialize data.
        device->fill(&value, DataType::kFloat32, m_size, m_data, dType);
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = device->allocate(m_size, dType);
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(Shape shape, Device * device, size_t size, Stride strides, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_size(size), m_shape(std::move(shape)), m_strides(std::move(strides)), m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_data = device->allocate(m_size, dType);
    }

    // Constructor
    TensorValue(float value, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape{}, m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_size = 1;
        m_data = device->allocate(m_size, dType);
        device->fill(&value, DataType::kFloat32, m_size, m_data, dType);
        m_strides = computeStrides();
    }

    // Destructor
    ~TensorValue()
    {
        if (m_data) m_device->deallocate(m_data);
        m_data   = nullptr;
        m_device = nullptr;
    }

    // Copy constructor
    TensorValue(const TensorValue& other) noexcept
    {
        m_dType   = other.m_dType;
        m_size    = other.m_size;
        m_shape   = other.m_shape;
        m_strides = other.m_strides;
        m_device  = other.m_device;
        m_data    = m_device->allocate(other.m_size, other.m_dType);
        m_device->copy(other.m_data, other.m_dType, m_data, other.m_dType, other.m_size);
    }

    // Copy assignment operator
    TensorValue& operator=(const TensorValue& other) noexcept
    {
        if (this != &other)     // Protect against self-assignment
        {
            m_device->deallocate(m_data);
            m_dType   = other.m_dType;
            m_size    = other.m_size;
            m_shape   = other.m_shape;
            m_strides = other.m_strides;
            m_device  = other.m_device;
            m_data    = m_device->allocate(other.m_size, other.m_dType);
            m_device->copy(other.m_data, other.m_dType, m_data, other.m_dType, other.m_size);
        }

        return *this;
    }

    // Move constructor
    TensorValue(TensorValue&& other) noexcept
    {
        m_dType   = other.m_dType;
        m_data    = other.m_data;
        m_size    = other.m_size;
        m_shape   = other.m_shape;
        m_strides = other.m_strides;
        m_device  = other.m_device;
        other.m_size   = 0;
        other.m_data   = nullptr;           // Avoid double deletion
        other.m_device = nullptr;
    }

    // Move assignment operator
    TensorValue& operator=(TensorValue&& other) noexcept
    {
        if (this != &other)
        {
            m_device->deallocate(m_data);   // Free existing resource
            m_dType   = other.m_dType;
            m_data    = other.m_data;
            m_size    = other.m_size;
            m_shape   = other.m_shape;
            m_strides = other.m_strides;
            m_device  = other.m_device;
            other.m_size   = 0;
            other.m_data   = nullptr;       // Avoid double deletion
            other.m_device = nullptr;
        }

        return *this;
    }

    // Select operator.
    TensorValue operator[](ssize_t index) const
    {
        return select(0, index);
    }

    // Access element at a specific index (non-const version).
    template<typename T>
    T & getValueAt(const Index & indices)     { return static_cast<T*>(m_data)[getIndex(indices)]; }

    // Access element at a specific index (const version).
    template<typename T>
    T getValueAt(const Index & indices) const { return static_cast<T*>(m_data)[getIndex(indices)]; }

    // Get the data type of the tensor.
    DataType dataType() const      { return m_dType; }

    // Get the shape of the tensor
    const Shape & shape() const    { return m_shape; }

    // Get the strides of the tensor
    const Stride & strides() const  { return m_strides; }

    // Get the raw data of the tensor.
    const void* data() const    { return m_data; }
    void* data()                { return m_data; }

    // Get the raw data of the tensor.
    template<typename T>
    const T* data() const       { return static_cast<T*>(m_data); }
    template<typename T>
    T* data()                   { return static_cast<T*>(m_data); }

    // Get the size of the data
    size_t size() const         { return m_size; }

    // Get the device
    Device * device() const     { return m_device; }

    // Set the device
    TensorValue to(Device * device) const
    {
        if (m_device == device) return *this;
        return {m_data, m_size, m_dType, m_shape, device, m_dType};
    }

    template<typename T>
    T item() const
    {
        if (!m_shape.empty())    // Scalar value must have no dimension.
        {
            throw std::invalid_argument("Tensor is not a scalar.");
        }
        return static_cast<T*>(m_data)[0];
    }

    // Returns a new TensorValue with a new shape.
    TensorValue reshape(const Shape & newShape) const
    {
        size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>());
        if (m_size != newSize)
        {
            throw std::invalid_argument("Reshape error: element count mismatch (" +
                                        std::to_string(m_size) + " vs " + std::to_string(newSize) + ").");
        }
        return {m_data, m_size, m_dType, newShape, m_device, m_dType};
    }

    // Equalize tensor data types by promoting data type of tensors.
    TensorValue to(DataType newDataType) const
    {
        if (dataType() != newDataType)
        {
            return {data(), size(), dataType(), shape(), device(), newDataType};
        }
        return *this;
    }

    // Returns true if two TensorValue shapes are compatible for a broadcast operation.
    static bool checkBroadcastShapes(const Shape& shape1, const Shape& shape2)
    {
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();
        while (it1 != shape1.rend() || it2 != shape2.rend())
        {
            size_t dim1 = (it1 != shape1.rend()) ? *it1++ : 1;
            size_t dim2 = (it2 != shape2.rend()) ? *it2++ : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            {
                return false;
            }
        }
        return true;
    }

    // Returns final shape of a broadcast operation.
    static Shape broadcastShapes(const Shape& shape1, const Shape& shape2)
    {
        Shape resultShape;
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();
        while (it1 != shape1.rend() || it2 != shape2.rend())
        {
            size_t dim1 = (it1 != shape1.rend()) ? *it1++ : 1;
            size_t dim2 = (it2 != shape2.rend()) ? *it2++ : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            {
                throw std::invalid_argument("Shapes are not compatible for broadcasting.");
            }
            resultShape.push_back(std::max(dim1, dim2));
        }
        std::reverse(resultShape.begin(), resultShape.end());
        return resultShape;
    }

    static bool checkBroadcastTo(const Shape& sourceShape, const Shape& targetShape)
    {
        if (sourceShape.size() > targetShape.size()) return false;

        auto itSrc = sourceShape.rbegin();
        auto itTgt = targetShape.rbegin();

        while (itTgt != targetShape.rend())
        {
            size_t dimSrc = (itSrc != sourceShape.rend()) ? *itSrc++ : 1;
            size_t dimTgt = *itTgt++;

            if (dimSrc != dimTgt && dimSrc != 1)
            {
                return false;
            }
        }

        return true;
    }

    // Returns a broadcasted TensorValue with a new shape.
    TensorValue broadcastTo(const Shape& newShape) const
    {
        if (shape() == newShape) return *this;
        if (!checkBroadcastTo(shape(), newShape))
        {
            throw std::invalid_argument("Target TensorValue shape is not broadcastable.");
        }
        Shape resultShape = broadcastShapes(shape(), newShape);
        TensorValue result(resultShape, device(), m_dType);
        device()->broadcastTo(m_data, result.data(), result.size(), shape(), resultShape, m_dType);
        return result;
    }

    // Reduces the TensorValue back to the original shape.
    TensorValue reduceTo(const Shape & originalShape) const
    {
        if (shape() == originalShape) return *this;
        // Ensure tensor values are initialized to zero, as the reduction operation performs a summation.
        TensorValue result(0, originalShape, device(), m_dType);
        device()->reduceTo(m_data, result.data(), m_size, m_shape, originalShape, m_dType);
        return result;
    }

    // Operators

    // Overload the + operator
    TensorValue operator+(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::add, other);
    }

    // Overload the - operator
    TensorValue operator-(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::sub, other);
    }

    // Overload the * operator
    TensorValue operator*(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::mul, other);
    }

    // Overload the / operator
    TensorValue operator/(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::div, other);
    }

    // Overload the += operator - In-place operation.
    TensorValue & operator+=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::add, other);
    }

    // Overload the -= operator - In-place operation.
    TensorValue & operator-=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::sub, other);
    }

    // Overload the *= operator - In-place operation.
    TensorValue & operator*=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::mul, other);
    }

    // Overload the /= operator - In-place operation.
    TensorValue & operator/=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::div, other);
    }

    // Overload the unary - operator
    TensorValue operator-() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->unary(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue operator+(float scalar) const
    {
        return *this + TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue operator-(float scalar) const
    {
        return *this - TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue operator*(float scalar) const
    {
        return *this * TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue operator/(float scalar) const
    {
        return *this / TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue& operator+=(float scalar)
    {
        return *this += TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue& operator-=(float scalar)
    {
        return *this -= TensorValue{scalar, m_shape, m_device, m_dType};
    }

    TensorValue& operator*=(float scalar)
    {
        return *this *= TensorValue{scalar, m_shape, m_device, m_dType};
    }

    TensorValue& operator/=(float scalar)
    {
        return *this /= TensorValue{scalar, m_shape, m_device, m_dType};
    }

    friend TensorValue operator+(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} + tensor;
    }

    friend TensorValue operator-(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} - tensor;
    }

    friend TensorValue operator*(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} * tensor;
    }

    friend TensorValue operator/(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} / tensor;
    }

    void fill(float value) const
    {
        m_device->fill(&value, DataType::kFloat32, m_size, m_data, m_dType);
    }

    TensorValue sum() const
    {
        TensorValue result({}, device(), m_dType);
        m_device->sum(m_data, m_size, result.data(), m_dType);
        return result;
    }

    TensorValue sum(ssize_t dim, bool keepDim=false) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::sum() is out of range.");
        }

        Shape resultShape = m_shape;
        resultShape[dim] = 1;
        auto result = reduceTo(resultShape);
        return keepDim ? result : result.squeeze(dim);
    }

    TensorValue mean() const
    {
        return sum() / size();
    }

    TensorValue mean(ssize_t dim, bool keepDim=false) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        return sum(dim, keepDim) / shape()[dim];
    }

    TensorValue sqrt() const
    {
        return tensorMathFunc(&Device::sqrt);
    }

    TensorValue sin() const
    {
        return tensorMathFunc(&Device::sin);
    }

    TensorValue cos() const
    {
        return tensorMathFunc(&Device::cos);
    }

    TensorValue tanh() const
    {
        return tensorMathFunc(&Device::tanh);
    }

    TensorValue log() const
    {
        return tensorMathFunc(&Device::log);
    }

    TensorValue exp() const
    {
        return tensorMathFunc(&Device::exp);
    }

    TensorValue pow(const TensorValue & exp) const
    {
        if (shape() != exp.shape() || dataType() != exp.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = exp;
            auto result = prepareTensors(lhs, rhs);
            result.device()->pow(lhs.data(), rhs.data(), lhs.size(), result.data(), result.dataType());
            return result;
        }

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->pow(m_data, exp.m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue max() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result({}, m_device, m_dType);
        m_device->max(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue max(ssize_t dim, bool keepDim=false) const
    {
        if (m_shape.empty()) return *this;      // Return itself if it's a scalar tensor.

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::max() is out of range.");
        }

        Shape newShape = m_shape;
        newShape[dim] = 1;

        TensorValue result(newShape, device(), m_dType);            // Zero initialization is not required.
        device()->fillMin(m_dType, result.size(), result.data());   // Initialize the tensor with the lowest value.
        device()->maxTo(m_data, result.data(), m_size, m_shape, newShape, m_dType);
        return keepDim ? result : result.squeeze(dim);
    }

    TensorValue argmax() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result({}, m_device, aix::DataType::kInt32);        // Index is by default in int32 type.
        m_device->argmax(m_data, m_size, result.m_data, m_dType, aix::DataType::kInt32);
        return result;
    }

    TensorValue argmax(ssize_t dim, bool keepDim=false) const
    {
        if (m_shape.empty()) return {0, m_shape, m_device, aix::DataType::kInt32};  // Scalar tensor.

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::argmax() is out of range.");
        }

        Shape newShape = m_shape;
        newShape[dim] = 1;

        TensorValue result(newShape, m_device, aix::DataType::kInt32);        // Index is by default in int32 type.
        m_device->argmaxTo(m_data, result.data(), m_size, result.size(), m_shape, newShape, m_strides, dim,
                           m_dType, aix::DataType::kInt32);
        return keepDim ? result : result.squeeze(dim);
    }

    TensorValue argmaxIndices() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, aix::DataType::kInt32);   // Index is by default in int32 type.
        m_device->argmaxIndices(m_data, m_size, result.m_data, m_dType, aix::DataType::kInt32);
        return result;
    }

    TensorValue argmaxIndices(ssize_t dim) const
    {
        if (m_shape.empty()) return {1, m_shape, m_device, aix::DataType::kInt32};  // Scalar tensor.

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::argmaxIndices() is out of range.");
        }

        Shape newShape = m_shape;
        newShape[dim] = 1;

        TensorValue result(0, m_shape, m_device, aix::DataType::kInt32);        // Index is by default in int32 type.
        m_device->argmaxIndicesTo(m_data, result.data(), m_size, result.size(), m_shape, newShape, m_dType,
                                  aix::DataType::kInt32);
        return result;
    }

    // Matrix multiplication for 2D tensors.
    TensorValue matmul(const TensorValue & b) const
    {
        // Ensure both tensors are 2D or can be treated as such.
        if (m_shape.size() != 2 || b.shape().size() != 2)
        {
            throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
        }

        // Check if the inner dimensions match.
        if (m_shape[1] != b.shape()[0])
        {
            throw std::invalid_argument("The inner dimensions of the tensors do not match.");
        }

        Shape resultShape{m_shape[0], b.shape()[1]};

        // Convert tensors to the promoted data type if necessary
        if (dataType() != b.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = b;
            auto promotedDType = promoteDataType(lhs.dataType(), rhs.dataType());
            lhs = lhs.to(promotedDType);
            rhs = rhs.to(promotedDType);
            TensorValue result(resultShape, lhs.device(), promotedDType);
            result.device()->matmul(lhs.data(), lhs.shape(), rhs.data(), rhs.shape(), result.data(), result.dataType());
            return result;
        }

        // Resultant tensor shape
        TensorValue result(resultShape, m_device, m_dType);
        m_device->matmul(m_data, m_shape, b.m_data, b.m_shape, result.m_data, m_dType);
        return result;
    }

    // Generalized transpose function.
    TensorValue transpose(ssize_t dim0, ssize_t dim1) const
    {
        auto shapeSize = static_cast<ssize_t>(shape().size());
        dim0 = dim0 < 0 ? shapeSize + dim0 : dim0;
        dim1 = dim1 < 0 ? shapeSize + dim1 : dim1;

        // Check dimensions
        if (dim0 < 0 || dim0 >= shapeSize || dim1 < 0 || dim1 >= shapeSize)
        {
            throw std::invalid_argument("Dimension is out of range for transpose.");
        }

        Shape newShape = m_shape;
        std::swap(newShape[dim0], newShape[dim1]);
        TensorValue result(newShape, device(), m_dType);
        m_device->transpose(dim0, dim1, m_data, m_shape, m_strides, result.strides(), result.size(), result.data(), m_dType);
        return result;
    }

    TensorValue squeeze(ssize_t dim) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        if (dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Invalid dimension for squeeze.");
        }

        if (m_shape[dim] == 1)
        {
            auto squeezedShape = m_shape;
            squeezedShape.erase(squeezedShape.begin() + dim);
            return reshape(squeezedShape);
        }
        return *this;
    }

    TensorValue unsqueeze(ssize_t dim) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        if (dim > static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Invalid dimension for unsqueeze.");
        }

        auto unsqueezedShape = m_shape;
        unsqueezedShape.insert(unsqueezedShape.begin() + dim, 1);
        return reshape(unsqueezedShape);
    }

    TensorValue slice(ssize_t dim=0, std::optional<ssize_t> startOpt = std::nullopt,
                      std::optional<ssize_t> endOpt = std::nullopt, ssize_t step=1) const
    {
        if (m_shape.empty())
        {
            throw std::invalid_argument("slice() cannot be applied to a 0-dim, a scalar tensor.");
        }

        if (step < 1)
        {
            throw std::invalid_argument("Slice step must be greater than zero.");
        }

        // Normalize dimension index.
        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of slice() is out of range.");
        }

        // Handle start and end indices.
        ssize_t start = startOpt.value_or(0);
        ssize_t end   = endOpt.value_or(static_cast<ssize_t>(m_shape[dim]));

        // Normalize negative indices.
        start = start < 0 ? static_cast<ssize_t>(m_shape[dim]) + start : start;
        end   = end   < 0 ? static_cast<ssize_t>(m_shape[dim]) + end   : end;

        // Clamp the start and end indices within valid bounds.
        start = std::max<ssize_t>(0, std::min<ssize_t>(start, static_cast<ssize_t>(m_shape[dim])));
        end   = std::max<ssize_t>(0, std::min<ssize_t>(end,   static_cast<ssize_t>(m_shape[dim])));

        if (start >= end)
        {
            throw std::invalid_argument("Start index of slice() must be less than end index.");
        }

        // Calculate the new shape for the sliced tensor.
        Shape newShape = m_shape;
        newShape[dim] = (end - start + step - 1) / step; // This computes the size along the slicing dimension.

        TensorValue result(newShape, device(), m_dType);    // Zero initialization is not required.
        // Slice and copy data to the result tensor.
        device()->slice(m_data, result.data(), result.size(), m_shape, newShape, m_strides,
                        dim, start, step, m_dType);
        return result;
    }

    TensorValue sliceSet(const TensorValue& tensor, ssize_t dim=0, std::optional<ssize_t> startOpt = std::nullopt,
                         std::optional<ssize_t> endOpt = std::nullopt, ssize_t step=1, bool inPlace=false) const
    {
        if (m_shape.empty())
        {
            throw std::invalid_argument("slice() cannot be applied to a 0-dim, a scalar tensor.");
        }

        if (step < 1)
        {
            throw std::invalid_argument("Slice step must be greater than zero.");
        }

        // Normalize dimension index.
        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of slice() is out of range.");
        }

        // Handle start and end indices.
        ssize_t start = startOpt.value_or(0);
        ssize_t end   = endOpt.value_or(static_cast<ssize_t>(m_shape[dim]));

        // Normalize negative indices.
        start = start < 0 ? static_cast<ssize_t>(m_shape[dim]) + start : start;
        end   = end   < 0 ? static_cast<ssize_t>(m_shape[dim]) + end   : end;

        // Clamp the start and end indices within valid bounds.
        start = std::max<ssize_t>(0, std::min<ssize_t>(start, static_cast<ssize_t>(m_shape[dim])));
        end   = std::max<ssize_t>(0, std::min<ssize_t>(end,   static_cast<ssize_t>(m_shape[dim])));

        if (start >= end)
        {
            throw std::invalid_argument("Start index of slice() must be less than end index.");
        }

        // Calculate the new shape for the sliced tensor.
        Shape newShape = m_shape;
        newShape[dim] = (end - start + step - 1) / step; // This computes the size along the slicing dimension.

        if (tensor.shape() != newShape)
        {
            throw std::invalid_argument("The tensor's shape does not match the new shape of sliceSet().");
        }

        if (inPlace)
        {
            // Slice and set tensor's data to the result tensor.
            device()->sliceSet(tensor.m_data, m_data, tensor.size(), m_shape, newShape, m_strides, dim, start, step,
                               m_dType);
            return *this;
        }

        TensorValue result(0, m_shape, device(), m_dType);  // Zero initialization is required.
        // Slice and set tensor's data to the result tensor.
        device()->sliceSet(tensor.m_data, result.data(), tensor.size(), m_shape, newShape, m_strides, dim, start, step,
                           m_dType);
        return result;
    }

    TensorValue select(ssize_t dim, ssize_t index) const
    {
        if (m_shape.empty())
        {
            throw std::invalid_argument("select() cannot be applied to a scalar, zero-dimension, tensor.");
        }
        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        index = index < 0 ? static_cast<ssize_t>(m_shape[dim]) + index : index;
        return slice(dim, index, index + 1, 1).squeeze(dim);
    }

    std::vector<TensorValue> split(ssize_t splitSize, ssize_t dim=0) const
    {
        if (splitSize < 0)
        {
            throw std::invalid_argument("Split size must be a positive number.");
        }

        if (m_shape.empty())
        {
            throw std::invalid_argument("Split operation needs at least a 1-dim tensor.");
        }

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0)
        {
            throw std::invalid_argument("Split dimension is out of range.");
        }

        std::vector<TensorValue> tensors;       // Stores splitted tensors.
        for (size_t i=0; i<m_shape[dim]; i+=splitSize)
        {
            tensors.emplace_back(slice(dim, i, i + splitSize, 1));
        }
        return tensors;
    }

    TensorValue tril(ssize_t diagonal=0) const
    {
        if (m_shape.size() < 2)
        {
            throw std::invalid_argument("Tensor must have at least two dimensions for tril operation.");
        }

        TensorValue result = *this;
        device()->tril(result.data(), result.size(), m_shape, m_strides, diagonal, m_dType);
        return result;
    }

    static TensorValue cat(const std::vector<TensorValue>& tensors, ssize_t dim)
    {
        if (tensors.empty())
        {
            throw std::invalid_argument("cat() operation needs at least one tensor.");
        }

        const auto& tensor = tensors[0];

        if (tensor.shape().empty())
        {
            throw std::invalid_argument("Zero-dimensional tensor cannot be concatenated.");
        }

        if (tensors.size() == 1) return tensor;

        DataType promotedDType = tensor.dataType();
        for (size_t i=0; i<tensors.size()-1; ++i)
        {
            // Tensor shapes must be the same.
            if (tensors[i].shape() != tensors[i+1].shape())
            {
                throw std::invalid_argument("Tensor shapes must be the same for the cat() operation.");
            }
            // Tensor devices must be the same.
            if (tensors[i].device() != tensors[i+1].device())
            {
                throw std::invalid_argument("Tensor devices must be the same for the cat() operation.");
            }
            promotedDType = promoteDataType(promotedDType, tensors[i+1].dataType());
        }

        dim = dim < 0 ? static_cast<ssize_t>(tensor.shape().size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(tensor.shape().size()))
        {
            throw std::invalid_argument("Dimension is out of range for cat() operation.");
        }

        auto newShape = tensor.shape();
        newShape[dim] *= tensors.size();
        size_t dimSize = tensor.shape()[dim];

        TensorValue result(newShape, tensor.device(), promotedDType);
        for (size_t i=0; i<tensors.size(); ++i)
        {
            result.sliceSet(tensors[i].to(promotedDType), dim, i * dimSize, (i + 1) * dimSize, 1, true);
        }
        return result;
    }

    // Friend function to overload operator<<
    inline friend std::ostream& operator<<(std::ostream & os, const TensorValue & tensor);

private:
    template<typename T>
    inline TensorValue arithmeticOpFunc(const T & func, const TensorValue & other) const
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            auto result = prepareTensors(lhs, rhs);
            (result.device()->*func)(lhs.data(), rhs.data(), lhs.size(), result.data(), result.dataType());
            return result;
        }
        TensorValue result(m_shape, m_device, m_dType);
        (m_device->*func)(m_data, other.m_data, m_size, result.m_data, m_dType);
        return result;
    }

    template<typename T>
    inline TensorValue & arithmeticInPlaceOpFunc(const T & func, const TensorValue & other)
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            prepareTensors(lhs, rhs);
            (m_device->*func)(lhs.data(), rhs.data(), lhs.size(), lhs.data(), lhs.dataType());
            *this = TensorValue(lhs.data(), lhs.size(), lhs.dataType(), lhs.shape(), lhs.device(), dataType());
            return *this;
        }
        else
        {
            (m_device->*func)(m_data, other.m_data, m_size, m_data, m_dType);
        }
        return *this;
    }

    template<typename T>
    inline TensorValue tensorMathFunc(const T & func) const
    {
        auto promotedDType = promoteDataTypeToFloat(m_dType);
        if (dataType() != promotedDType)
        {
            // This constructor requires copy operation.
            TensorValue result(m_data, m_size, m_dType, m_shape, m_device, promotedDType);
            (m_device->*func)(result.data(), result.size(), result.data(), promotedDType);
            return result;
        }
        // This constructor does not require copy operation.
        TensorValue result(m_shape, m_device, m_dType);
        (m_device->*func)(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    // Compute the strides based on the shape of the tensor
    Stride computeStrides()
    {
        Stride strides(m_shape.size());
        size_t stride = 1;
        for (int64_t i = strides.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= m_shape[i];
        }
        return strides;
    }

    // Get the flat index from a vector of indices
    size_t getIndex(const Index & indices) const
    {
        assert(indices.size() == m_shape.size());
        return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), 0);
    }

    // Promotes data types and applies broadcasting if necessary.
    static TensorValue prepareTensors(TensorValue & lhs, TensorValue & rhs)
    {
        // TODO: Minimize copy operations.
        auto promotedDType = lhs.dataType();

        if (lhs.dataType() != rhs.dataType())
        {
            // Convert tensors to the promoted data type if necessary
            promotedDType = promoteDataType(lhs.dataType(), rhs.dataType());
            lhs = lhs.to(promotedDType);
            rhs = rhs.to(promotedDType);
        }

        // If shapes are different then try broadcasting.
        if (lhs.shape() != rhs.shape())
        {
            Shape bcShape = broadcastShapes(lhs.shape(), rhs.shape());
            lhs = lhs.broadcastTo(bcShape);
            rhs = rhs.broadcastTo(bcShape);
        }

        return {lhs.shape(), lhs.device(), promotedDType};
    }

    // Print Tensor data
    template<typename T>
    void print(std::ostream & os) const
    {
        os << std::fixed << std::setprecision(4);

        // Print scalar value, a tensor with no dimension.
        if (m_shape.empty())
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
                os << static_cast<int>(item<T>()) << "\n\n";
            else
                os << item<T>() << "\n\n";
        }
        else if (m_shape.size() == 1)
        {
            // Print tensor that has only one dimension.
            for (size_t i = 0; i < m_shape[0]; ++i)
            {
                if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
                    os << "  " << static_cast<int>(getValueAt<T>({i})) << "\n";
                else
                    os << "  " << getValueAt<T>({i}) << "\n";
            }
            os << "\n";
        }
        else
        {
            // Print tensor that has at least two dimensions.
            std::stack<std::pair<Index, size_t>> stack;
            stack.push({Index(), 0});

            while (!stack.empty())
            {
                auto [indices, dim] = stack.top();
                stack.pop();

                if (dim == m_shape.size() - 2)
                {
                    bool isOverTwo = m_shape.size() > 2;
                    if (isOverTwo)
                    {
                        os << "(";
                    }

                    for (size_t i = 0; i < indices.size(); ++i)
                    {
                        os << indices[i];
                        if (i < indices.size() - 1)
                        {
                            os << ",";
                        }
                    }

                    if (isOverTwo)
                    {
                        os << ",.,.) =\n";
                    }

                    size_t rows = m_shape[dim];
                    size_t cols = m_shape[dim + 1];

                    for (size_t i = 0; i < rows; ++i)
                    {
                        for (size_t j = 0; j < cols; ++j)
                        {
                            Index subIndices = indices;
                            subIndices.push_back(i);
                            subIndices.push_back(j);
                            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
                                os << "  " << static_cast<int>(getValueAt<T>(subIndices));
                            else
                                os << "  " << getValueAt<T>(subIndices);
                        }
                        os << '\n';
                    }

                    os << '\n';
                }
                else
                {
                    for (size_t i = m_shape[dim]; i-- > 0;) // Process in reverse order
                    {
                        Index subIndices = indices;
                        subIndices.push_back(i);
                        stack.push({subIndices, dim + 1});
                    }
                }
            }
        }

        auto deviceName = device()->name();
        // Print shape
        switch (dataType())
        {
            case DataType::kFloat64:  os << "[ " << deviceName << " Float64 {";  break;
            case DataType::kFloat32:  os << "[ " << deviceName << " Float32 {";  break;
            case DataType::kFloat16:  os << "[ " << deviceName << " Float16 {";  break;
            case DataType::kBFloat16: os << "[ " << deviceName << " BFloat16 {"; break;
            case DataType::kInt64:    os << "[ " << deviceName << " Int64 {";    break;
            case DataType::kInt32:    os << "[ " << deviceName << " Int32 {";    break;
            case DataType::kInt16:    os << "[ " << deviceName << " Int16 {";    break;
            case DataType::kInt8:     os << "[ " << deviceName << " Int8 {";     break;
            case DataType::kUInt8:    os << "[ " << deviceName << " UInt8 {";    break;
            default:                  os << "[ " << deviceName << " Unknown {";  break;
        }

        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            os << m_shape[i];
            if (i < m_shape.size() - 1)
            {
                os << ",";
            }
        }
        os << "} ]\n";
    }

private:
    DataType  m_dType{DataType::kFloat32};
    void*     m_data{nullptr};  // The flat array of tensor elements.
    size_t    m_size{0};        // Number of elements in DataType.
    Shape     m_shape;          // The shape of the tensor.
    Stride    m_strides;        // The strides for indexing the tensor.
    Device *  m_device{nullptr};
};


class TensorNode
{
public:
    // Constructor
    explicit TensorNode(TensorValue value, bool requireGrad = false) :
        m_value{std::move(value)}, m_grad{m_value.shape(), m_value.device(), m_value.size(), m_value.strides(), m_value.dataType()},
        m_requireGrad{requireGrad}
    {
    }

    // Constructor
    explicit TensorNode(const Shape & shape, Device * device, bool requireGrad = false, DataType dType = DataType::kFloat32) :
        m_value{shape, device, dType}, m_grad{shape, device, m_value.size(), m_value.strides(), dType},
        m_requireGrad{requireGrad}
    {
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(const TensorValue & seed)
    {
        if (m_retainGrad)
        {
            m_grad += seed;
        }
        m_backwardFunc(this, seed);
    }

    Device * device() const          { return m_value.device(); }

    std::string  m_name;
    TensorValue  m_value;
    TensorValue  m_grad;
    bool  m_requireGrad;
    bool  m_retainGrad{false};
    std::shared_ptr<TensorNode>  m_a{nullptr};
    std::shared_ptr<TensorNode>  m_b{nullptr};
    size_t m_dim0{0};
    size_t m_dim1{0};
    bool m_keepDim{false};
    std::optional<ssize_t> m_start;
    std::optional<ssize_t> m_end;
    std::vector<std::shared_ptr<TensorNode>> m_aMulti;
    std::function<void(TensorNode * tensor, const TensorValue & seed)>  m_backwardFunc{nullptr};
};


struct TensorOptions
{
    inline TensorOptions requireGrad(bool state)    { m_requireGrad = state; return *this; }
    inline TensorOptions dtype(DataType dataType)   { m_dtype = dataType;    return *this; }
    inline TensorOptions device(Device* device)     { m_device = device;     return *this; }

    bool m_requireGrad{false};
    aix::DataType m_dtype{aix::DataType::kFloat32};
    aix::Device* m_device{&aix::defaultDevice};
};
inline TensorOptions requireGrad(bool state)    { return { .m_requireGrad=state }; }
inline TensorOptions dtype(DataType dataType)   { return { .m_dtype=dataType    }; }
inline TensorOptions device(Device* device)     { return { .m_device=device     }; }


class Tensor
{
public:
    // Constructor.
    Tensor() = default;

    // Constructor.
    explicit Tensor(const void* data, size_t size, DataType srcDType, const Shape & shape, const TensorOptions & opt = {})
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{data, size, srcDType, shape, opt.m_device, opt.m_dtype},
                                              opt.m_requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(float value, const Shape & shape, const TensorOptions & opt = {})
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{value, shape, opt.m_device, opt.m_dtype}, opt.m_requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(const Shape & shape, const TensorOptions & opt = {})
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(shape, opt.m_device, opt.m_requireGrad, opt.m_dtype);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(float value=1)  { m_data->backward(TensorValue{value, m_data->m_a->m_grad.shape(), device(), dataType()}); }
    void backward(float value, const Shape & gradShape)  { m_data->backward(TensorValue{value, gradShape, device(), dataType()}); }

    // Getters and setters for the tensor's value.
    inline const TensorValue & value() const    { return m_data->m_value; }
    inline TensorValue & value()                { return m_data->m_value; }
    inline const Shape & shape() const          { return m_data->m_value.shape(); }
    inline DataType dataType() const            { return m_data->m_value.dataType(); }

    // Gradient-related methods.
    inline const TensorValue & grad() const
    {
        validateRetainGradientState();
        return m_data->m_grad;
    }

    inline TensorValue & grad()
    {
        validateRetainGradientState();
        return m_data->m_grad;
    }

    inline void zeroGrad()                      { m_data->m_grad.fill(0); }
    inline bool isRequireGrad() const           { return m_data->m_requireGrad; }
    inline void retainGrad() const              { m_data->m_retainGrad = true; m_data->m_grad.fill(0); }
    inline const Tensor& requireGrad(bool state) const
    {
        m_data->m_requireGrad = m_data->m_retainGrad = state;
        return *this;
    }

    inline Device * device() const              { return m_data->device(); }

    inline void name(const std::string& name) const  { m_data->m_name = name; }
    inline const std::string& name() const           { return m_data->m_name; }

    // Returns a new Tensor with a new shape.
    Tensor reshape(const Shape & newShape) const
    {
        size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>());
        if (value().size() != newSize)
        {
            throw std::invalid_argument("Reshape error: element count mismatch (" +
                                        std::to_string(value().size()) + " vs " + std::to_string(newSize) + ").");
        }

        const auto& tv = m_data->m_value;
        TensorOptions opt{ .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() };
        Tensor result{tv.data(), tv.size(), tv.dataType(), newShape, opt};
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = reshapeBackwardFunc;
        return result;
    }

    // Returns a new Tensor with a new shape. This method accepts one inferred dimension.
    Tensor reshape(const std::initializer_list<ssize_t>& newShape) const
    {
        return reshape(shapeWithInferredDimToShape(newShape));
    }

    Tensor broadcastTo(const Shape & newShape) const
    {
        if (shape() == newShape) return *this;
        TensorValue tValue = m_data->m_value.broadcastTo(newShape);
        Tensor result{tValue.data(), tValue.size(), tValue.dataType(), tValue.shape(),
                      { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device()}};
        result.m_data->m_a = m_data;            // Keep the reference to the original tensor node
        result.m_data->m_backwardFunc = broadcastBackwardFunc;
        return result;
    }

    // Set operation device for the tensor.
    inline Tensor to(std::unique_ptr<Device> & device)    { return to(*device); }
    inline Tensor to(Device * device)                     { return to(*device); }
    Tensor to(Device & newDevice)
    {
        if (&newDevice == m_data->device()) return *this;
        Tensor result{shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=&newDevice }};
        result.m_data->m_value = m_data->m_value.to(&newDevice);
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = toDeviceBackwardFunc;
        return result;
    }

    Tensor to(DataType newDataType) const
    {
        if (dataType() == newDataType) return *this;
        TensorOptions opt{ .m_requireGrad=isRequireGrad(), .m_dtype=newDataType, .m_device=device() };
        Tensor result{value().data(), value().size(), value().dataType(), value().shape(), opt};
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = toDataTypeBackwardFunc;
        return result;
    }

    static void defaultBackward(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_requireGrad && !node->m_retainGrad)
        {
            assert(node->m_grad.dataType() == seed.dataType());
            node->m_grad += seed;
        }
    }

    static void reshapeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        node->m_a->backward(seed.reshape(node->m_a->m_value.shape()));
    }

    static void broadcastBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // Accumulate the gradient to the original node by reducing the gradient from the broadcasted shape to the
        // original shape. Summation is used for gradient accumulation when reducing dimensions because each element
        // of the original tensor contributes to multiple elements of the resulting tensor after broadcasting.
        node->m_a->backward(seed.reduceTo(node->m_a->m_value.shape()));
    }

    static void toDeviceBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        if (seed.device() != node->m_a->m_value.device())
        {
            // Synchronize seed to ensure the seed's data is available before copying it to a different device.
            seed.device()->commitAndWait();
        }
        node->m_a->backward(seed.to(node->m_a->m_value.device()));
    }

    static void toDataTypeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // Ensure that the seed gradient is converted back to the data type of the original tensor.
        node->m_a->backward(seed.to(node->m_a->m_value.dataType()));
    }

    static void addBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(seed);
    }

    static void subBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(-seed);
    }

    static void mulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(node->m_b->m_value * seed);
        node->m_b->backward(node->m_a->m_value * seed);
    }

    static void divBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed / node->m_b->m_value);                                               // ∂f/∂a = 1 / b
        node->m_b->backward(-node->m_a->m_value * seed / (node->m_b->m_value * node->m_b->m_value));  // ∂f/∂b = -a / b^2
    }

    static void unaryBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // Calculate gradients.
        node->m_a->backward(-seed);
    }

    static void sqrtBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of sqrt(a) with respect to 'a' is 0.5/sqrt(a).
        // Therefore, the gradient of the input is multiplied by 0.5/sqrt(a).
        node->m_a->backward(0.5 / node->m_a->m_value.sqrt() * seed);   // ∂f/∂a = 0.5/sqrt(a)
    }

    static void sinBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of sin(a) with respect to 'a' is cos(a).
        // Therefore, the gradient of the input is multiplied by cos(a).
        node->m_a->backward(node->m_a->m_value.cos() * seed);   // ∂f/∂a = cos(a)
    }

    static void cosBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of cos(a) with respect to 'a' is -sin(a).
        // Therefore, the gradient of the input is multiplied by -sin(a).
        node->m_a->backward(-node->m_a->m_value.sin() * seed);   // ∂f/∂a = -sin(a)
    }

    static void tanhBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of tanh(a) with respect to 'a' is 1 - tanh^2(a).
        // Therefore, the gradient of the input is multiplied by (1 - tanh^2(a)).
        const auto & tanhValue = node->m_a->m_value.tanh();
        node->m_a->backward((float(1) - tanhValue * tanhValue) * seed);  // ∂f/∂a = (1 - tanh^2(a))
    }

    static void logBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // TODO: Handle division by zero case.
        // The derivative of log(a) with respect to 'a' is 1/a.
        node->m_a->backward(seed / node->m_a->m_value);  // ∂f/∂a = 1/a
    }

    static void expBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of exp(a) with respect to 'a' is exp(a), itself.
        node->m_a->backward(seed * node->m_a->m_value.exp());  // ∂f/∂a = exp(a)
    }

    static void maxBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of max(a) with respect to 'a' is a zero tensor with argmax index set to 1.
        node->m_a->backward(seed * node->m_a->m_value.argmaxIndices());
    }

    static void maxBackwardFunc2(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of max(a) with respect to 'a' is a zero tensor with max indexes set to 1.
        node->m_a->backward(seed * node->m_a->m_value.argmaxIndices(static_cast<ssize_t>(node->m_dim0)));
    }

    static void powBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // The derivative of pow(a, b) with respect to 'a' is b * a^(b-1).
        // ∂f/∂a = b * pow(a, b-1)
        node->m_a->backward(seed * node->m_b->m_value * node->m_a->m_value.pow(node->m_b->m_value - float(1)));
    }

    static void matmulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Assuming m_a and m_b are the input matrices a and b, respectively,
        // and seed is ∂E/∂c, the gradient of the loss with respect to the output matrix c.
        // Compute gradients with respect to a and b

        // Corrected to use matrix multiplication for backward pass calculations
        node->m_a->backward(seed.matmul(node->m_b->m_value.transpose(0, 1)));      // ∂E/∂a = ∂E/∂c * b^T
        node->m_b->backward(node->m_a->m_value.transpose(0, 1).matmul(seed));      // ∂E/∂b = a^T * ∂E/∂c
    }

    static void transposeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        node->m_a->backward(seed.transpose(node->m_dim0, node->m_dim1));
    }

    static void sliceBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        node->m_a->backward(node->m_a->m_value.sliceSet(seed, node->m_dim0, node->m_start, node->m_end, node->m_dim1));
    }

    static void sumBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // For the sum operation, the gradient is simply the seed
        node->m_a->backward(seed);
    }

    static void sumBackwardFunc2(TensorNode* node, const TensorValue& seed)
    {
        if (!node->m_a) return;
        const auto& originalShape = node->m_a->m_value.shape();

        // For keepDim=False case, 1 dimension was squeezed. That dimension needs to be unsqueezed.
        if (!node->m_keepDim)
            node->m_a->backward(seed.unsqueeze(static_cast<ssize_t>(node->m_dim0)).broadcastTo(originalShape));
        else
            node->m_a->backward(seed.broadcastTo(originalShape));
    }

    static void squeezeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        node->m_a->backward(seed.unsqueeze(node->m_dim0));
    }

    static void unsqueezeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        node->m_a->backward(seed.squeeze(node->m_dim0));
    }

    static void trillBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        auto onesLikeSeed = TensorValue(1.0, seed.shape(), seed.device(), seed.dataType());
        node->m_a->backward(seed * onesLikeSeed.tril(static_cast<ssize_t>(node->m_dim0)));      // m_dim0 = diagonal
    }

    static void catBackwardFunc(TensorNode* node, const TensorValue& seed)
    {
        size_t numTensors = node->m_aMulti.size();
        if (numTensors == 0) return;

        // The dimension along which tensors were concatenated.
        auto dim = static_cast<ssize_t>(node->m_dim0);

        // Get the shape of the original tensors.
        size_t dimSize = node->m_aMulti[0]->m_value.shape()[dim];

        // Iterate over each original tensor and propagate the gradient.
        for (size_t i=0; i<numTensors; ++i)
        {
            // Propagate this sliced gradient to the corresponding original tensor.
            node->m_aMulti[i]->backward(seed.slice(dim, i * dimSize, (i + 1) * dimSize, 1));
        }
    }

    // Select operator.
    Tensor operator[](ssize_t index) const
    {
        return select(0, index);
    }

    // Overload the + operator
    Tensor operator+(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(shape(), { .m_requireGrad=isRequireGrad() || other.isRequireGrad(), .m_dtype=dataType(),
                                 .m_device=device()});
        result.m_data->m_value = lhs.m_data->m_value + rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = addBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(shape(), { .m_requireGrad=isRequireGrad() || other.isRequireGrad(), .m_dtype=dataType(),
                                 .m_device=device()});
        result.m_data->m_value = lhs.m_data->m_value - rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = subBackwardFunc;
        return result;
    }

    // Overload the * operator
    Tensor operator*(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(shape(), { .m_requireGrad=isRequireGrad() || other.isRequireGrad(), .m_dtype=dataType(),
                                 .m_device=device()});
        result.m_data->m_value = lhs.m_data->m_value * rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = mulBackwardFunc;
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(bcShape, { .m_requireGrad=isRequireGrad() || other.isRequireGrad(), .m_dtype=dataType(),
                                 .m_device=device() });
        result.m_data->m_value = lhs.m_data->m_value / rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = divBackwardFunc;
        return result;
    }

    Tensor operator-() const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = -m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = unaryBackwardFunc;
        return result;
    }

    Tensor operator+(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        Tensor tensor(scalar, shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=promotedFloatType, .m_device=device() });
        return *this + tensor;
    }

    Tensor operator-(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        Tensor tensor(scalar, shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=promotedFloatType, .m_device=device() });
        return *this - tensor;
    }

    Tensor operator*(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        Tensor tensor(scalar, shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=promotedFloatType, .m_device=device() });
        return *this * tensor;
    }

    Tensor operator/(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        Tensor tensor(scalar, shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=promotedFloatType, .m_device=device() });
        return *this / tensor;
    }

    friend Tensor operator+(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, rhsTensor.shape(), { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor + rhsTensor;
    }

    friend Tensor operator-(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, rhsTensor.shape(), { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor - rhsTensor;
    }

    friend Tensor operator*(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, rhsTensor.shape(), { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor * rhsTensor;
    }

    friend Tensor operator/(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, rhsTensor.shape(), { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor / rhsTensor;
    }

    Tensor sqrt() const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.sqrt();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sqrtBackwardFunc;
        return result;
    };

    Tensor sin() const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.sin();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sinBackwardFunc;
        return result;
    };

    Tensor cos() const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.cos();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = cosBackwardFunc;
        return result;
    };

    Tensor tanh() const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.tanh();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = tanhBackwardFunc;
        return result;
    };

    Tensor log() const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.log();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = logBackwardFunc;
        return result;
    };

    Tensor exp() const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.exp();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = expBackwardFunc;
        return result;
    };

    Tensor sum() const
    {
        Tensor result({}, { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.sum();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sumBackwardFunc;
        return result;
    }

    Tensor sum(ssize_t dim, bool keepDim=false) const
    {
        Tensor result({}, { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.sum(dim, keepDim);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = dim >= 0 ? dim : dim + m_data->m_value.shape().size();
        result.m_data->m_keepDim = keepDim;
        result.m_data->m_backwardFunc = sumBackwardFunc2;
        return result;
    }

    Tensor mean() const
    {
        return sum() / value().size();
    }

    Tensor mean(ssize_t dim, bool keepDim=false) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        return sum(dim, keepDim) / value().shape()[dim];
    }

    Tensor max() const
    {
        Tensor result({}, { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.max();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = maxBackwardFunc;
        return result;
    }

    Tensor max(ssize_t dim, bool keepDim=false) const
    {
        Tensor result({}, { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.max(dim, keepDim);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = dim >= 0 ? dim : dim + m_data->m_value.shape().size();
        result.m_data->m_backwardFunc = maxBackwardFunc2;
        return result;
    }

    Tensor argmax() const
    {
        Tensor result({}, { .m_requireGrad=false, .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.argmax();
        // argmax does not require gradient.
        return result;
    }

    Tensor argmax(ssize_t dim, bool keepDim=false) const
    {
        Tensor result({}, { .m_requireGrad=false, .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.argmax(dim, keepDim);
        // argmax does not require gradient.
        return result;
    }

    Tensor pow(float exp) const
    {
        TensorOptions opt{ .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() };
        Tensor expTensor(exp, shape(), opt);
        Tensor result(shape(), opt);
        result.m_data->m_value = m_data->m_value.pow(expTensor.m_data->m_value);
        result.m_data->m_a = m_data;
        result.m_data->m_b = expTensor.m_data;
        result.m_data->m_backwardFunc = powBackwardFunc;
        return result;
    }

    Tensor pow(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);        // Exponent tensor.

        Tensor result(bcShape, { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = lhs.m_data->m_value.pow(rhs.m_data->m_value);
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = powBackwardFunc;
        return result;
    }

    Tensor matmul(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        auto lhs = to(promotedDType);
        auto rhs = other.to(promotedDType);

        Tensor result({shape()[0], rhs.shape()[1]}, { .m_requireGrad=isRequireGrad() || rhs.isRequireGrad(),
                                                      .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = lhs.m_data->m_value.matmul(rhs.m_data->m_value);
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = matmulBackwardFunc;
        return result;
    }

    Tensor transpose(ssize_t dim0, ssize_t dim1) const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.transpose(dim0, dim1);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = dim0;
        result.m_data->m_dim1 = dim1;
        result.m_data->m_backwardFunc = transposeBackwardFunc;
        return result;
    }

    Tensor slice(ssize_t dim=0, std::optional<ssize_t> startOpt = std::nullopt,
                 std::optional<ssize_t> endOpt = std::nullopt, ssize_t step=1) const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.slice(dim, startOpt, endOpt, step);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = dim;
        result.m_data->m_dim1 = step;
        result.m_data->m_start = startOpt;
        result.m_data->m_end = endOpt;
        result.m_data->m_backwardFunc = sliceBackwardFunc;
        return result;
    }

    Tensor squeeze(ssize_t dim) const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.squeeze(dim);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = dim;
        result.m_data->m_backwardFunc = squeezeBackwardFunc;
        return result;
    }

    Tensor unsqueeze(ssize_t dim) const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.unsqueeze(dim);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = dim;
        result.m_data->m_backwardFunc = unsqueezeBackwardFunc;
        return result;
    }

    Tensor var(bool unbiased=true) const
    {
        auto deviation = *this - mean();
        auto elementCount = unbiased ? deviation.value().size() - 1 : deviation.value().size();
        return (deviation * deviation).sum() / float(elementCount);
    }

    Tensor var(ssize_t dim, bool unbiased=true, bool keepdim=false) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        auto deviation = *this - mean(dim, true);
        auto elementCount = unbiased ? shape()[dim] - 1 : shape()[dim];
        auto var = (deviation * deviation).sum(dim, true) / float(elementCount);
        return keepdim ? var : var.squeeze(dim);
    }

    Tensor select(ssize_t dim, ssize_t index) const
    {
        if (shape().empty())
        {
            throw std::invalid_argument("select() cannot be applied to a scalar, zero-dimension, tensor.");
        }
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        index = index < 0 ? static_cast<ssize_t>(shape()[dim]) + index : index;
        return slice(dim, index, index + 1, 1).squeeze(dim);
    }

    std::vector<Tensor> split(ssize_t splitSize, ssize_t dim=0) const
    {
        if (splitSize < 0)
        {
            throw std::invalid_argument("Split size must be a positive number.");
        }

        if (shape().empty())
        {
            throw std::invalid_argument("Split operation needs at least a 1-dim tensor.");
        }

        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        if (dim < 0)
        {
            throw std::invalid_argument("Split dimension is out of range.");
        }

        std::vector<Tensor> tensors;        // Stores splitted tensors.
        for (size_t i = 0; i < shape()[dim]; i += splitSize)
        {
            tensors.emplace_back(slice(dim, i, i + splitSize, 1));
        }
        return tensors;
    }

    Tensor tril(ssize_t diagonal=0) const
    {
        Tensor result(shape(), { .m_requireGrad=isRequireGrad(), .m_dtype=dataType(), .m_device=device() });
        result.m_data->m_value = m_data->m_value.tril(diagonal);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = diagonal;
        result.m_data->m_backwardFunc = trillBackwardFunc;
        return result;
    }

    static Tensor cat(const std::vector<Tensor>& tensors, ssize_t dim)
    {
        if (tensors.empty())
        {
            throw std::invalid_argument("cat() operation needs at least one tensor.");
        }

        const auto& tensor = tensors[0];

        if (tensor.shape().empty())
        {
            throw std::invalid_argument("Zero-dimensional tensor cannot be concatenated.");
        }

        if (tensors.size() == 1) return tensor;

        bool requireGrad = tensor.isRequireGrad();
        DataType promotedDType = tensor.dataType();
        // Tensor shapes must be the same.
        for (size_t i=0; i<tensors.size()-1; ++i)
        {
            if (tensors[i].shape() != tensors[i+1].shape())
            {
                throw std::invalid_argument("Tensor shapes must be the same for the cat() operation.");
            }
            if (tensors[i].device() != tensors[i+1].device())
            {
                throw std::invalid_argument("Tensor devices must be the same for the cat() operation.");
            }
            requireGrad |= tensors[i+1].isRequireGrad();
            promotedDType = promoteDataType(promotedDType, tensors[i+1].dataType());
        }

        dim = dim < 0 ? static_cast<ssize_t>(tensor.shape().size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(tensor.shape().size()))
        {
            throw std::invalid_argument("Dimension is out of range for cat() operation.");
        }

        auto newShape = tensor.shape();
        newShape[dim] *= tensors.size();
        size_t dimSize = tensor.shape()[dim];

        Tensor result(newShape, { .m_requireGrad=requireGrad, .m_dtype=promotedDType, .m_device=tensor.device() });
        for (size_t i=0; i<tensors.size(); ++i)
        {
            result.value().sliceSet(tensors[i].to(promotedDType).value(), dim, i * dimSize, (i + 1) * dimSize, 1, true);
            // Store original tensors for the back prop.
            result.m_data->m_aMulti.emplace_back(tensors[i].m_data);
        }
        result.m_data->m_dim0 = dim;
        result.m_data->m_backwardFunc = catBackwardFunc;
        return result;
    }

    // Friend function to overload operator<<
    inline friend std::ostream & operator<<(std::ostream& os, const Tensor& tensor);

protected:
    inline Shape broadcastShape(const Shape& otherShape) const
    {
        return shape() == otherShape ? shape() : TensorValue::broadcastShapes(shape(), otherShape);
    }

    inline void validateRetainGradientState() const
    {
        if (!m_data->m_requireGrad && !m_data->m_retainGrad)
        {
            throw std::runtime_error("Gradients for non-leaf tensors won’t be populated during automatic gradient"
                                     " calculation. Use .retainGrad() on the non-leaf tensor if needed, or access"
                                     " the leaf tensor instead.");
        }
    }

    Shape shapeWithInferredDimToShape(const std::initializer_list<ssize_t>& newShape) const
    {
        Shape currShape = shape();
        Shape resultShape(newShape.size());
        ssize_t inferredDimIndex = -1;
        size_t inferredDimCount  = 0;
        size_t invalidDimCount   = 0;
        ssize_t inferredDimSize  = std::accumulate(currShape.begin(), currShape.end(), 1, std::multiplies<>());

        ssize_t i = 0;
        for (const auto dim : newShape)
        {
            if (dim == -1)
            {
                ++inferredDimCount;
                inferredDimIndex = i;
            }
            else
            {
                inferredDimSize /= dim;
                resultShape[i] = dim;
            }
            if (dim == 0 || dim < -1) ++invalidDimCount;
            ++i;
        }

        if (invalidDimCount > 0)
        {
            throw std::invalid_argument("Shape contains invalid dimension.");
        }

        if (inferredDimCount > 1)
        {
            throw std::invalid_argument("Only one dimension can be inferred.");
        }

        if (inferredDimIndex >= 0)
            resultShape[inferredDimIndex] = inferredDimSize;

        return resultShape;
    }

    std::shared_ptr<TensorNode>  m_data{nullptr};
};

// Some convenience method definitions.

inline Tensor tensor(float value, const TensorOptions & opt = {})
{
    return Tensor{value, Shape{}, opt};
}

inline Tensor tensor(const std::initializer_list<double> & data, const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<double>(), shape, opt};
}

inline Tensor tensor(const std::initializer_list<float> & data, const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<float>(), shape, opt};
}

inline Tensor tensor(const std::initializer_list<double> & data, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<double>(), Shape{data.size()}, opt};
}

inline Tensor tensor(const std::initializer_list<float> & data, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<float>(), Shape{data.size()}, opt};
}

inline Tensor tensor(const std::vector<double> & data, const TensorOptions & opt = {})
{
    return Tensor{data.data(), data.size(), getDataType<double>(), Shape{data.size()}, opt};
}

inline Tensor tensor(const std::vector<float> & data, const TensorOptions & opt = {})
{
    return Tensor{data.data(), data.size(), getDataType<float>(), Shape{data.size()}, opt};
}

inline Tensor ones(const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{1, shape, opt};
}

inline Tensor zeros(const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{0, shape, opt};
}

inline Tensor onesLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{1, tensor.shape(), { .m_requireGrad=requireGrad, .m_dtype=tensor.dataType(), .m_device=tensor.device() }};
}

inline Tensor zerosLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{0, tensor.shape(), { .m_requireGrad=requireGrad, .m_dtype=tensor.dataType(), .m_device=tensor.device() }};
}

inline Tensor sqrt(const Tensor & A)   { return A.sqrt(); }
inline Tensor sin(const Tensor & A)    { return A.sin();  }
inline Tensor cos(const Tensor & A)    { return A.cos();  }
inline Tensor tanh(const Tensor & A)   { return A.tanh(); }
inline Tensor log(const Tensor & A)    { return A.log();  }
inline Tensor exp(const Tensor & A)    { return A.exp();  }
inline Tensor sum(const Tensor & A)    { return A.sum();  }
inline Tensor mean(const Tensor & A)   { return A.mean(); }
inline Tensor sum(const Tensor & A, ssize_t dim, bool keepDim=false)    { return A.sum(dim, keepDim);  }
inline Tensor mean(const Tensor & A, ssize_t dim, bool keepDim=false)   { return A.mean(dim, keepDim);  }
inline Tensor pow(const Tensor & A, const Tensor & exp)     { return A.pow(exp); }
inline Tensor max(const Tensor & A)         { return A.max();    }
inline Tensor argmax(const Tensor & A)      { return A.argmax(); }
inline Tensor matmul(const Tensor & A, const Tensor & B)    { return A.matmul(B); }
inline Tensor squeeze(const Tensor & A, ssize_t dim)    { return A.squeeze(dim);    }
inline Tensor unsqueeze(const Tensor & A, ssize_t dim)  { return A.unsqueeze(dim);  }
inline Tensor cat(const std::vector<Tensor>& tensors, ssize_t dim)     {  return Tensor::cat(tensors, dim);  }
inline Tensor hstack(const std::vector<Tensor>& tensors)    { return Tensor::cat(tensors, 1); }
inline Tensor vstack(const std::vector<Tensor>& tensors)    { return Tensor::cat(tensors, 0); }
inline Tensor var(const Tensor & A, bool unbiased=true)     { return A.var(unbiased); }
inline Tensor var(const Tensor & A, ssize_t dim, bool unbiased=true, bool keepdim=false)
{
    return A.var(dim, unbiased, keepdim);
}

static Tensor randn(const Shape & shape, const TensorOptions & opt = {})
{
    std::uniform_real_distribution<float> distr(-1, 1);

    size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<float> rndData(totalSize);

    // Fill rndData with random numbers
    std::generate(rndData.begin(), rndData.end(), [&distr]() -> float { return distr(randGen); });

    return Tensor{rndData.data(), rndData.size(), getDataType<float>(), shape, opt};
}

// Returns evenly spaced values within a given interval. The interval including start but excluding stop. [start, end)
static Tensor arange(float start, float end, float step, const TensorOptions & opt = {})
{
    if (step == 0)
    {
        throw std::invalid_argument("Step must be non-zero.");
    }

    auto range = end - start;
    if ((range > 0 && step < 0) || (range < 0 && step > 0))
    {
        throw std::invalid_argument("Range direction is inconsistent with step sign.");
    }

    auto size = static_cast<size_t>(std::ceil(range / step));
    std::vector<float> data(size);
    std::generate(data.begin(), data.end(), [step,x=start]() mutable -> float { float v=x; x += step; return v; });
    return Tensor{data.data(), data.size(), getDataType<float>(), {size}, opt};
}
inline Tensor arange(float end, const TensorOptions & opt = {})               { return arange(0.0, end, 1.0, opt);   }
inline Tensor arange(float start, float end, const TensorOptions & opt = {})  { return arange(start, end, 1.0, opt); }


// Optimizers Namespace


namespace optim
{

class Optimizer
{
public:
    // Constructor
    Optimizer() = default;
    // Constructor
    explicit Optimizer(const std::vector<Tensor> & parameters) : m_parameters(parameters) { }

    // Destructor
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    virtual void zeroGrad()
    {
        for (auto & param : m_parameters)
        {
            param.zeroGrad();
        }
    }

    inline void setDataType(DataType dtype)
    {
        if (dtype != DataType::kFloat64 && dtype != DataType::kFloat32 &&
            dtype != DataType::kFloat16 && dtype != DataType::kBFloat16)
        {
            throw std::invalid_argument("Optimization has to perform in Float data type to be effective.");
        }
        m_calculationDType = dtype;
    }

protected:
    std::vector<Tensor> m_parameters;
    DataType m_calculationDType{DataType::kFloat32};
};


class SGD : public Optimizer
{
public:
    SGD() = default;
    explicit SGD(const std::vector<Tensor> & parameters, float lr = 0.01f)
        : Optimizer(parameters), m_lr(lr) { }

    void step() final
    {
        for (auto & param : m_parameters)
        {
            if (param.isRequireGrad())
            {
                param.value() -= param.grad() * m_lr;   // w' = w - lr * w_gradient.
            }
        }
    }

private:
    float m_lr{0.01f};      // Learning rate
};


class Adam : public Optimizer
{
public:
    Adam() = default;
    explicit Adam(const std::vector<Tensor> & parameters, float lr = 0.001f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(parameters), m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
    {
        for (const auto & param : m_parameters)
        {
            m_m.emplace_back(0, param.shape(), param.value().device(), m_calculationDType);
            m_v.emplace_back(0, param.shape(), param.value().device(), m_calculationDType);
        }
    }

    void step() final
    {
        ++m_timestep;
        for (size_t i = 0; i < m_parameters.size(); ++i)
        {
            if (m_parameters[i].isRequireGrad())
            {
                // Convert the parameter's data type to the optimizer's internal calculation type.
                auto gradFloat = m_parameters[i].grad().to(m_calculationDType);

                // Update biased first moment estimate.
                m_m[i] = m_beta1 * m_m[i] + float(1.0 - m_beta1) * gradFloat;

                // Update biased second raw moment estimate.
                m_v[i] = m_beta2 * m_v[i] + float(1.0 - m_beta2) * gradFloat * gradFloat;

                // Compute bias-corrected first moment estimate.
                TensorValue mHat = m_m[i] / float(1.0 - std::pow(m_beta1, m_timestep));

                // Compute bias-corrected second raw moment estimate.
                TensorValue vHat = m_v[i] / float(1.0 - std::pow(m_beta2, m_timestep));

                // Update parameter.
                m_parameters[i].value() -= (m_lr * mHat / (vHat.sqrt() + m_epsilon)).to(m_parameters[i].dataType());
            }
        }
    }

private:
    float m_lr{0.001f};         // Learning rate.
    float m_beta1{0.9f};        // Exponential decay rate for the first moment estimates.
    float m_beta2{0.999f};      // Exponential decay rate for the second moment estimates.
    float m_epsilon{1e-8f};     // Small constant for numerical stability.
    size_t m_timestep{0};       // Time step.
    std::vector<TensorValue>    m_m;    // First moment vector.
    std::vector<TensorValue>    m_v;    // Second moment vector.
};

}   // optim namespace


// Neural Network Namespace


namespace nn
{

class Module
{
public:
    virtual ~Module() = default;

    virtual Tensor forward(Tensor x) const = 0;

    void registerParameter(Tensor & tensor)
    {
        m_parameters.emplace_back(tensor);
    }

    void registerModule(const Module & module)
    {
        for (const auto & param : module.parameters())
        {
            m_parameters.emplace_back(param);
        }
    }

    std::vector<Tensor> parameters() const
    {
        return m_parameters;
    }

    // Returns the total number of elements (learnable parameters) in each Tensor.
    size_t learnableParameters() const
    {
        size_t totalParams = 0;
        for (const auto & param: m_parameters)
        {
            if (param.isRequireGrad())
            {
                totalParams += param.value().size();
            }
        }
        return totalParams;
    }

    void to(std::unique_ptr<Device> & device) const   { to(*device); }
    void to(Device * device) const                    { to(*device); }
    void to(Device & device) const
    {
        for (auto & param : parameters())
        {
            param.value() = param.value().to(&device);
            param.grad()  = param.grad().to(&device);
        }
    }

    void to(DataType newDtype) const
    {
        for (auto & param : parameters())
        {
            if (param.isRequireGrad())
            {
                param.value() = param.value().to(newDtype);
                param.grad()  = param.grad().to(newDtype);
            }
        }
    }

private:
    std::vector<Tensor> m_parameters;
};


class Sequential : public Module
{
public:
    // Override the forward function.
    Tensor forward(Tensor x) const override
    {
        for (const auto & module : m_modules)
        {
            x = module->forward(x);
        }
        return x;
    }

    // Function to add modules dynamically if needed.
    void add(Module* module)
    {
        registerModule(*module);
        m_modules.emplace_back(module);     // Use std::unique_ptr to take ownership of the module pointer.
    }

protected:
    // Use std::unique_ptr for polymorphic containment.
    std::vector<std::unique_ptr<Module>>  m_modules;
};


class Linear : public Module
{
public:
    // Constructor
    Linear(size_t numInputs, size_t numOutputs)
    {
        m_w1 = randn({numInputs, numOutputs}, { .m_requireGrad=true });
        m_b1 = randn({1,         numOutputs}, { .m_requireGrad=true });

        // Register learnable parameters.
        registerParameter(m_w1);
        registerParameter(m_b1);
    }

    // Forward
    Tensor forward(Tensor x) const override
    {
        return matmul(x, m_w1) + m_b1;
    }

    Tensor  m_w1;
    Tensor  m_b1;
};


class Tanh : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return tanh(x);
    }
};


class Sigmoid : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 1 / (1 + exp(-x));
    }
};


class Softmax : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        auto expX = x.exp();
        return expX / expX.sum();
    }
};


class LogSoftmax : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        // LogSoftmax(x) = log(e^x / sum(e^x)) = log(e^x) - log(sum(e^x)) = x - log(sum(e^x))
        return x - x.exp().sum().log();
    }
};


class GeLU : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 0.5 * x * (1.0 + tanh(std::sqrtf(2.0 / std::numbers::pi) * (x + 0.044715 * x.pow(3))));
    }
};


class MSELoss
{
public:
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        auto diff = predictions - targets;
        auto loss = mean(diff * diff);
        return loss;
    }
};


class BinaryCrossEntropyLoss
{
public:
    // Prediction values must be in [0..1] range.
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        return -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions));
    }
};

}   // nn namespace


// Auxiliary Features


inline void save(const nn::Module & module, const std::string & filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::ios_base::failure("Failed to open file for writing.");
    }

    const auto params = module.parameters();
    for (auto param : params)
    {
        const auto & value = param.value();
        size_t size = value.size();
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));                       // Save parameter size
        size_t paramDTypeSize = Device::dataTypeSize(param.dataType());
        ofs.write(reinterpret_cast<const char*>(value.data()), size * paramDTypeSize);       // Save parameter data
    }

    ofs.close();
}

inline void load(nn::Module & module, const std::string & filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        throw std::ios_base::failure("Failed to open model parameter file for reading.");
    }

    const auto params = module.parameters();    // Get model parameters
    for (auto param : params)
    {
        size_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size));         // Read size of parameter
        if (size != param.value().size())
        {
            throw std::runtime_error("Invalid parameter size found when loading the model.");
        }
        size_t paramDTypeSize = Device::dataTypeSize(param.dataType());
        ifs.read(reinterpret_cast<char*>(param.value().data()), size * paramDTypeSize); // Read the parameter data
    }

    ifs.close();
}

// Overload the << operator to print TensorValue.
std::ostream & operator<<(std::ostream& os, const TensorValue& tensor)
{
    switch (tensor.dataType())
    {
        case DataType::kFloat64:   tensor.print<double    >(os);   break;
        case DataType::kFloat32:   tensor.print<float     >(os);   break;
        case DataType::kFloat16:   tensor.print<float16_t >(os);   break;
        case DataType::kBFloat16:  tensor.print<bfloat16_t>(os);   break;
        case DataType::kInt64:     tensor.print<int64_t   >(os);   break;
        case DataType::kInt32:     tensor.print<int32_t   >(os);   break;
        case DataType::kInt16:     tensor.print<int16_t   >(os);   break;
        case DataType::kInt8:      tensor.print<int8_t    >(os);   break;
        case DataType::kUInt8:     tensor.print<uint8_t   >(os);   break;
        default:
            throw std::runtime_error("Data type for print is not supported.");
            break;
    }
    return os;
}

// Overload the << operator to print Tensor.
std::ostream & operator<<(std::ostream& os, const Tensor& tensor)
{
    os << tensor.value();
    return os;
}

inline void manualSeed(size_t seed)
{
    randGen.seed(seed);
}

}   // aix namespace
