//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "Utils.hpp"
#include <aix.hpp>
// External includes
#include <doctest/doctest.h>
// System includes

using namespace aix;


TEST_CASE("Activation Func - Softmax")
{
    SUBCASE("Scalar")
    {
        auto input = tensor(0.5);
        auto result = nn::Softmax().forward(input);
        CHECK(result.shape() == input.shape());
        CHECK(result.sum().value().item() == 1.0);
        CheckVectorApproxValues(result.value(), tensor(1.0).value());
    }

    SUBCASE("1 dimension - 1 element")
    {
        auto input = tensor({0.5}, Shape{1});
        auto result = nn::Softmax().forward(input);
        CHECK(result.shape() == input.shape());
        CHECK(result.sum().value().item() == 1.0);
        CheckVectorApproxValues(result.value(), tensor({1.0}, input.shape()).value());
    }

    SUBCASE("1 dimension - n elements")
    {
        auto input = tensor({1.0, 2.0, 3.0, 4.0}, Shape{4});
        auto result = nn::Softmax().forward(input);
        CHECK(result.shape() == input.shape());
        CHECK(result.sum().value().item() == 1.0);
        CheckVectorApproxValues(result.value(), tensor({0.0320586, 0.0871443, 0.236883, 0.643914}, input.shape()).value());
    }

    SUBCASE("1x1 dimension")
    {
        auto input = tensor({0.5}, Shape{1,1});
        auto result = nn::Softmax().forward(input);
        CHECK(result.shape() == input.shape());
        CHECK(result.sum().value().item() == 1.0);
        CheckVectorApproxValues(result.value(), tensor({1.0}, input.shape()).value());
    }

    SUBCASE("2x2 dimension")
    {
        auto input = tensor({1.0, 2.0, 3.0, 4.0}, Shape{2,2});
        auto result = nn::Softmax().forward(input);
        CHECK(result.shape() == input.shape());
        CHECK(result.sum().value().item() == 1.0);
        CheckVectorApproxValues(result.value(), tensor({0.0320586, 0.0871443, 0.236883, 0.643914}, input.shape()).value());
    }

    // Note: Results are consistent with those from PyTorch.
}
