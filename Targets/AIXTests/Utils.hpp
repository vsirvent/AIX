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
#include <doctest/doctest.h>
// System includes

inline auto Approx(auto value, double epsilon = 1e-5)
{
    return doctest::Approx(value).epsilon(epsilon);
}

inline void CheckVectorApproxValues(const aix::TensorValue & results, const aix::TensorValue & expected)
{
    if (results.dataType() != expected.dataType())
    {
        throw std::invalid_argument("Tensor data types do no match for test result comparison.");
    }

    if (static_cast<size_t>(results.dataType()) >= aix::DataTypeCount)
    {
        throw std::invalid_argument("CheckVectorApproxValues does not support the new data type.");
    }

    if (results.dataType() == aix::DataType::kFloat64)
    {
        for (size_t i=0; i<expected.size(); ++i)
        {
            CHECK(results.data<double>()[i] == Approx(expected.data<double>()[i]));
        }
    }
    else if (results.dataType() == aix::DataType::kFloat32)
    {
        for (size_t i=0; i<expected.size(); ++i)
        {
            CHECK(results.data<float>()[i] == Approx(expected.data<float>()[i]));
        }
    }
    else if (results.dataType() == aix::DataType::kInt64)
    {
        for (size_t i=0; i<expected.size(); ++i)
        {
            CHECK(results.data<int64_t>()[i] == Approx(expected.data<int64_t>()[i]));
        }
    }
    else if (results.dataType() == aix::DataType::kInt32)
    {
        for (size_t i=0; i<expected.size(); ++i)
        {
            CHECK(results.data<int32_t>()[i] == Approx(expected.data<int32_t>()[i]));
        }
    }
    else if (results.dataType() == aix::DataType::kInt16)
    {
        for (size_t i=0; i<expected.size(); ++i)
        {
            CHECK(results.data<int16_t>()[i] == Approx(expected.data<int16_t>()[i]));
        }
    }
    else if (results.dataType() == aix::DataType::kInt8)
    {
        for (size_t i=0; i<expected.size(); ++i)
        {
            CHECK(results.data<int8_t>()[i] == Approx(expected.data<int8_t>()[i]));
        }
    }
    else if (results.dataType() == aix::DataType::kUInt8)
    {
        for (size_t i=0; i<expected.size(); ++i)
        {
            CHECK(results.data<uint8_t>()[i] == Approx(expected.data<uint8_t>()[i]));
        }
    }
}

inline void CheckVectorApproxValues(const aix::Tensor & results, const aix::Tensor & expected)
{
    CheckVectorApproxValues(results.value(), expected.value());
}
