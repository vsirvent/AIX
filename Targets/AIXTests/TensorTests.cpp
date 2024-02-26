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


TEST_CASE("Simple TensorValue 1 dim - Add")
{
    auto x = TensorValue({1, 2, 3}, {1, 3});
    auto y = TensorValue({4, 5, 6}, {1, 3});

    auto z = x + y;

    CHECK(z.shape() == Shape{1, 3});
    CHECK(z.data() == Array{5, 7, 9});
}


TEST_CASE("Simple TensorValue 2 dim - Add")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3});

    auto z = x + y;

    CHECK(z.shape() == Shape{2, 3});
    CHECK(z.data() == Array{8, 10, 12, 14, 16, 18});
}


TEST_CASE("Simple TensorValue 2 dim - Sub")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3});

    auto z = x - y;

    CHECK(z.shape() == Shape{2, 3});
    CHECK(z.data() == Array{-6, -6, -6, -6, -6, -6});
}


TEST_CASE("Simple TensorValue 2 dim - Mul")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3});

    auto z = x * y;

    CHECK(z.shape() == Shape{2, 3});
    CHECK(z.data() == Array{7, 16, 27, 40, 55, 72});
}


TEST_CASE("Simple TensorValue 2 dim - Div")
{
    auto x = TensorValue({-5, 0, 5, 10, 15, 20}, {2, 3});
    auto y = TensorValue({5, 5, 5, 5, -5, -20},  {2, 3});

    auto z = x / y;

    CHECK(z.shape() == Shape{2, 3});
    CHECK(z.data() == Array{-1, 0, 1, 2, -3, -1});
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto z = x;

    CHECK(z.shape() == x.shape());
    CHECK(z.data()  == x.data());
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto copyTensor = [](TensorValue value) { return value; };
    auto z = copyTensor(x);

    CHECK(z.shape() == x.shape());
    CHECK(z.data()  == x.data());
}


TEST_CASE("TensorValue::tanh 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, 0.4}, {2, 2});
    CheckVectorApproxValues(TensorValue::tanh(x).data(), {0.099668, 0.197375, 0.291313, 0.379949});
}


TEST_CASE("TensorValue::matmul 1x1 1x1")
{
    auto a = TensorValue(2, {1, 1});
    auto b = TensorValue(3, {1, 1});
    auto c = TensorValue::matmul(a, b);

    CHECK(c.shape() == Shape{1, 1});
    CheckVectorApproxValues(c.data(), {6});
}


TEST_CASE("TensorValue::matmul 2x4 4x3")
{
    auto a = TensorValue({1,2,3,4,5,6,7,8},            {2, 4});
    auto b = TensorValue({1,2,3,4,5,6,7,8,9,10,11,12}, {4, 3});
    auto c = TensorValue::matmul(a, b);     // Result matrix.

    CHECK(c.shape() == Shape{2, 3});
    CheckVectorApproxValues(c.data(), {70, 80, 90, 158, 184, 210});
}


TEST_CASE("TensorValue::transpose 1x1")
{
    auto a = TensorValue(2, {1, 1}).transpose();
    CHECK(a.shape() == Shape{1, 1});
    CheckVectorApproxValues(a.data(), {2});
}


TEST_CASE("TensorValue::transpose 2x3")
{
    auto a = TensorValue({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    CHECK(a.shape() == Shape{3, 2});
    CheckVectorApproxValues(a.data(), {1, 4, 2, 5, 3, 6});
}
