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

#define EPSILON  1e-5

inline auto Approx(auto value)
{
    return doctest::Approx(value).epsilon(EPSILON);
}

inline void CheckVectorApproxValues(const aix::TensorValue & results, const aix::TensorValue & expected)
{
    for (size_t i=0; i<expected.size(); ++i)
    {
        CHECK(results.data()[i] == Approx(expected.data()[i]));
    }
}

inline void CheckVectorApproxValues(const aix::Tensor & results, const aix::Tensor & expected)
{
    CheckVectorApproxValues(results.value(), expected.value());
}
