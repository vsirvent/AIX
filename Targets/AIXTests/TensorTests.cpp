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

    CHECK(z.shape() == std::vector<size_t>{1, 3});
    CHECK(z.data()  == std::vector<float>{5, 7, 9});
}


TEST_CASE("Simple TensorValue 2 dim - Add")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3});

    auto z = x + y;

    CHECK(z.shape() == std::vector<size_t>{2, 3});
    CHECK(z.data()  == std::vector<float>{8, 10, 12, 14, 16, 18});
}


TEST_CASE("Simple TensorValue 2 dim - Sub")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3});

    auto z = x - y;

    CHECK(z.shape() == std::vector<size_t>{2, 3});
    CHECK(z.data()  == std::vector<float>{-6, -6, -6, -6, -6, -6});
}


TEST_CASE("Simple TensorValue 2 dim - Mul")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3});
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3});

    auto z = x * y;

    CHECK(z.shape() == std::vector<size_t>{2, 3});
    CHECK(z.data()  == std::vector<float>{7, 16, 27, 40, 55, 72});
}


TEST_CASE("Simple TensorValue 2 dim - Div")
{
    auto x = TensorValue({-5, 0, 5, 10, 15, 20}, {2, 3});
    auto y = TensorValue({5, 5, 5, 5, -5, -20},  {2, 3});

    auto z = x / y;

    CHECK(z.shape() == std::vector<size_t>{2, 3});
    CHECK(z.data()  == std::vector<float>{-1, 0, 1, 2, -3, -1});
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


TEST_CASE("TensorValue::Tanh 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, 0.4}, {2, 2});
    CheckVectorApproxValues(TensorValue::tanh(x).data(), {0.099668, 0.197375, 0.291313, 0.379949});
}


/*

TEST_CASE("Testing broadcasting support in TensorValue")
{
    SUBCASE("Broadcast scalar to tensor")
    {
        TensorValue tensor({1.0, 2.0, 3.0}, {3});
        TensorValue scalar({10.0}, {1}); // Scalar is a tensor with shape {1}
        TensorValue result = tensor + scalar;
        CHECK(result.data() == std::vector<float>({11.0, 12.0, 13.0}));
    }

    SUBCASE("Broadcast vector to matrix along last dimension")
    {
        TensorValue matrix({{1.0, 2.0}, {3.0, 4.0}}, {2, 2});
        TensorValue vector({10.0, 20.0}, {2}); // Vector is a tensor with shape {2}
        TensorValue result = matrix + vector;
        CHECK(result.data() == std::vector<float>({11.0, 22.0, 13.0, 24.0}));
    }

    SUBCASE("Broadcast vector to matrix along first dimension")
    {
        TensorValue matrix({{1.0, 2.0}, {3.0, 4.0}}, {2, 2});
        TensorValue vector({10.0, 20.0}, {2, 1}); // Vector is a tensor with shape {2, 1}
        TensorValue result = matrix + vector;
        CHECK(result.data() == std::vector<float>({11.0, 12.0, 23.0, 24.0}));
    }

    SUBCASE("Broadcast matrix to 3D tensor")
    {
        TensorValue tensor3d({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}, {2, 2, 2});
        TensorValue matrix({{10.0, 20.0}, {30.0, 40.0}}, {2, 2}); // Matrix is a tensor with shape {2, 2}
        TensorValue result = tensor3d + matrix;
        CHECK(result.data() == std::vector<float>({11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0}));
    }

    SUBCASE("Incompatible shapes for broadcasting")
    {
        TensorValue tensor1({1.0, 2.0, 3.0}, {3});
        TensorValue tensor2({10.0, 20.0}, {2});
        CHECK_THROWS_WITH(tensor1 + tensor2, "Shapes are not compatible for broadcasting.");
    }
}

*/
