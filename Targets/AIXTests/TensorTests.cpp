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
#include <sstream>


using namespace aix;

Device  testDevice;     // Default CPU device.

TEST_CASE("Simple TensorValue 1 dim - Add")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);
    auto y = TensorValue({4.0, 5.0, 6.0}, {1, 3}, &testDevice);

    auto z = x + y;

    CHECK(z.shape() == Shape{1, 3});
    CheckVectorApproxValues(z, TensorValue({5.0, 7.0, 9.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Add")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},    {2, 3}, &testDevice);
    auto y = TensorValue({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {2, 3}, &testDevice);

    auto z = x + y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({8.0, 10.0, 12.0, 14.0, 16.0, 18.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Sub")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},    {2, 3}, &testDevice);
    auto y = TensorValue({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {2, 3}, &testDevice);

    auto z = x - y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({-6.0, -6.0, -6.0, -6.0, -6.0, -6.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Mul")
{
    auto x = TensorValue({1.0, 2.0, 3.0,  4.0,  5.0,  6.0}, {2, 3}, &testDevice);
    auto y = TensorValue({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {2, 3}, &testDevice);

    auto z = x * y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({7.0, 16.0, 27.0, 40.0, 55.0, 72.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Div")
{
    auto x = TensorValue({-5.0, 0.0, 5.0, 10.0, 15.0, 20.0}, {2, 3}, &testDevice);
    auto y = TensorValue({5.0, 5.0, 5.0, 5.0, -5.0, -20.0},  {2, 3}, &testDevice);

    auto z = x / y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({-1.0, 0.0, 1.0, 2.0, -3.0, -1.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, &testDevice);
    auto z = x;
    z.data<float>()[0] = 1;

    CHECK(z.shape() == x.shape());
    CheckVectorApproxValues(z, x);
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, &testDevice);
    auto copyTensor = [](TensorValue value) { return value; };
    auto z = copyTensor(x);

    CHECK(z.shape() == x.shape());
    CheckVectorApproxValues(z, x);
}


TEST_CASE("TensorValue - item()")
{
    SUBCASE("Must fail due to a non-scalar tensor")
    {
        TensorValue input = TensorValue({1.0, 2.0}, {2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input.item<float>(), std::invalid_argument);

        TensorValue input2 = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input2.item<float>(), std::invalid_argument);
    }

    SUBCASE("Should return a scalar value")
    {
        TensorValue input = TensorValue(1.0, Shape{}, &testDevice);
        CHECK(input.item<float>() == 1.0);
    }
}


TEST_CASE("TensorValue - matmul()")
{
    SUBCASE("Must fail due to an inner dimension mismatch")
    {
        TensorValue A = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
        TensorValue B = TensorValue({1.0, 2.0, 3.0, 4.0}, {1, 4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("Must fail due to dimension size mismatch")
    {
        TensorValue A = TensorValue({1.0, 2.0, 3.0, 4.0}, {4}, &testDevice);
        TensorValue B = TensorValue({1.0, 2.0, 3.0, 4.0}, {1, 4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("A[1x1] B[1x1] multiplication")
    {
        TensorValue A = TensorValue(2.0, Shape{1, 1}, &testDevice);
        TensorValue B = TensorValue(3.0, Shape{1, 1}, &testDevice);
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, TensorValue(6.0, Shape{1, 1}, &testDevice));
    }

    SUBCASE("A[2x3] B[3x4] multiplication")
    {
        TensorValue A = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3}, &testDevice);
        TensorValue B = TensorValue({12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}, Shape{3, 4}, &testDevice);
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{2,4});
        CHECK(result.size() == 8);
        CheckVectorApproxValues(result, TensorValue({40.0, 34.0, 28.0, 22.0, 112.0, 97.0, 82.0, 67.0}, Shape{2, 4}, &testDevice));
    }
}


TEST_CASE("TensorValue - transpose()")
{
    SUBCASE("Must fail if dimension size is higher")
    {
        TensorValue input1 = TensorValue({1.0,2.0,3.0,4.0}, {2,1,2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input1.transpose(1, 3), std::invalid_argument);
        TensorValue input2 = TensorValue({1.0,2.0,3.0,4.0}, {4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input2.transpose(0, 1), std::invalid_argument);
    }

    SUBCASE("{} transpose")
    {
        TensorValue A = TensorValue(1.0, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.transpose(0, 0), std::invalid_argument);
    }

    SUBCASE("1x1 transpose")
    {
        TensorValue A = TensorValue(1.0, {1,1}, &testDevice);
        auto result = A.transpose(0, 1);
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, TensorValue(1.0, {1,1}, &testDevice));
    }

    SUBCASE("3x2 transpose")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(0, 1);
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}, &testDevice));
    }

    SUBCASE("3x2 transpose - dims(0,0)")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(0, 0);
        CHECK(result.shape() == Shape{3,2});
        CheckVectorApproxValues(result, TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice));
    }

    SUBCASE("3x2 transpose - dims(1,1)")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(1, 1);
        CHECK(result.shape() == Shape{3,2});
        CheckVectorApproxValues(result, TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice));
    }

    SUBCASE("3x2 transpose - dims(1,0)")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(1, 0);
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}, &testDevice));
    }

    SUBCASE("3x2x2 transpose(0,1) -> 2x3x2")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}, {3,2,2}, &testDevice);
        auto result1 = A.transpose(0, 1);
        CHECK(result1.shape() == Shape{2,3,2});
        CheckVectorApproxValues(result1, TensorValue({1.0,2.0,5.0,6.0,9.0,10.0,
                                                      3.0,4.0,7.0,8.0,11.0,12.0}, {2,3,2}, &testDevice));

        auto result2 = A.transpose(1, 0);
        CHECK(result2.shape() == Shape{2,3,2});
        CheckVectorApproxValues(result2, TensorValue({1.0,2.0,5.0,6.0,9.0,10.0,
                                                      3.0,4.0,7.0,8.0,11.0,12.0}, {2,3,2}, &testDevice));
    }

    SUBCASE("3x2x2 transpose(0,2) -> 2x2x3")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}, {3,2,2}, &testDevice);
        auto result1 = A.transpose(0, 2);
        CHECK(result1.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result1, TensorValue({1.0,5.0,9.0,3.0,7.0,11.0,
                                                      2.0,6.0,10.0,4.0,8.0,12.0}, {2,2,3}, &testDevice));

        auto result2 = A.transpose(2, 0);
        CHECK(result2.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result2, TensorValue({1.0,5.0,9.0,3.0,7.0,11.0,
                                                      2.0,6.0,10.0,4.0,8.0,12.0}, {2,2,3}, &testDevice));
    }

    SUBCASE("2x3x2 transpose(1,2) -> 2x2x3")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}, {2,3,2}, &testDevice);
        auto result1 = A.transpose(1, 2);
        CHECK(result1.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result1, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0,
                                                      7.0,9.0,11.0,8.0,10.0,12.0}, {2,2,3}, &testDevice));

        auto result2 = A.transpose(2, 1);
        CHECK(result2.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result2, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0,
                                                      7.0,9.0,11.0,8.0,10.0,12.0}, {2,2,3}, &testDevice));
    }
}


TEST_CASE("TensorValue::tanh 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, 0.4}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.tanh(), TensorValue({0.099668, 0.197375, 0.291313, 0.379949}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue::log 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, 0.4}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.log(), TensorValue({-2.30259, -1.60944, -1.20397, -0.916291}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue::exp 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, -0.4}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.exp(), TensorValue({1.10517, 1.2214, 1.34986, 0.67032}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue::pow 2x2")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
    auto exp = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.pow(exp), TensorValue({1.0, 4.0, 27.0, 256.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Add with Scalar")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);
    float scalar = 5;
    x += scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({6.0, 7.0, 8.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Sub with Scalar")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    float scalar = 5;
    x -= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({1.0, 2.0, 3.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Mul with Scalar")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);
    float scalar = 5;
    x *= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({5.0, 10.0, 15.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Div with Scalar")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    float scalar = 2;
    x /= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({3.0, 3.5, 4.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Add with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({1.0, 2.0, -1.0}, {1, 3}, &testDevice);
    x += y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({7.0, 9.0, 7.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Sub with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({1.0, 2.0, -1.0}, {1, 3}, &testDevice);
    x -= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({5.0, 5.0, 9.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Mul with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({2.0, 3.0, -2.0}, {1, 3}, &testDevice);
    x *= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({12.0, 21.0, -16.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Div with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({2.0, 2.0, -2.0}, {1, 3}, &testDevice);
    x /= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({3.0, 3.5, -4.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - Unary Minus")
{
    auto x = TensorValue({1.0, -2.0, 3.0}, {1, 3}, &testDevice);
    auto y = -x;

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({-1.0, 2.0, -3.0}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Fill")
{
    auto x = TensorValue({1.0, 1.0, 1.0}, {1, 3}, &testDevice);
    float fillValue = 7;
    x.fill(fillValue);

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({7.0, 7.0, 7.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - Sum")
{
    auto x = TensorValue({-1.0, 2.2, 0.0, 4.8}, {2, 2}, &testDevice);

    CheckVectorApproxValues(x.sum(), TensorValue(6.0, &testDevice));
}


TEST_CASE("TensorValue - Mean")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);

    CheckVectorApproxValues(x.mean(), TensorValue(2.5, &testDevice));
}


TEST_CASE("TensorValue - Sqrt")
{
    auto x = TensorValue({4.0, 9.0, 16.0}, {1, 3}, &testDevice);
    auto y = x.sqrt();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({2.0, 3.0, 4.0}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Sin")
{
    auto x = TensorValue({0.5, 0.0, -0.5}, {1, 3}, &testDevice);
    auto y = x.sin();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({float(std::sin(0.5)),
                                            float(std::sin(0)),
                                            float(std::sin(-0.5))}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Cos")
{
    auto x = TensorValue({0.5, 0.0, -0.5}, {1, 3}, &testDevice);
    auto y = x.cos();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({float(std::cos(0.5)),
                                            float(std::cos(0)),
                                            float(std::cos(-0.5))}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Device Switch")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);

    Device  newDevice;
    x.device(&newDevice);

    CHECK(x.device() == &newDevice);
    CheckVectorApproxValues(x, TensorValue({1.0, 2.0, 3.0}, x.shape(), &newDevice));
}


TEST_CASE("TensorValue - Reshape")
{
    SUBCASE("scalar -> 1 dimension")
    {
        auto input = TensorValue(5.0, {}, &testDevice);
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("scalar -> 1x1 dimension")
    {
        auto input = TensorValue(5.0, {}, &testDevice);
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1 dimension -> 1x1 dimension")
    {
        auto input = TensorValue(5.0, {1}, &testDevice);
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1 dimension -> scalar")
    {
        auto input = TensorValue(5.0, {1}, &testDevice);
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1x1 dimension -> scalar")
    {
        auto input = TensorValue(5.0, {1,1}, &testDevice);
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1x1 dimension -> 1 dimension")
    {
        auto input = TensorValue(5.0, {1,1}, &testDevice);
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1x3 dimension -> 3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0};
        auto input = TensorValue(data, {1,3}, &testDevice);
        auto newShape = Shape{3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 6 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{6};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("6 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {6}, &testDevice);
        auto newShape = Shape{2, 3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 3x1x2 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{3, 1, 2};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("Invalid reshape throws invalid_argument")
    {
        TensorValue inputs = TensorValue({1.0,2.0,3.0,4.0,5.0}, {1,5}, &testDevice);
        // Check that reshape throws an invalid_argument exception
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({2,2}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({1,6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({4}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({}), std::invalid_argument);
    }
}


TEST_CASE("TensorValue - broadcastTo")
{
    SUBCASE("[1] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue(1.0, {1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0,1.0,1.0,1.0,1.0,1.0}, newShape, &testDevice));
    }

    SUBCASE("[2x1] to [2x2x3]")
    {
        Shape newShape{2,2,3};
        auto bct = TensorValue({1.0,2.0}, {2,1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 12);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0,1.0,1.0,
                                                  2.0,2.0,2.0,
                                                  1.0,1.0,1.0,
                                                  2.0,2.0,2.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3] to [1x3]")
    {
        Shape newShape{1, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 3);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3] to [3x3]")
    {
        Shape newShape{3, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 9);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0,
                                                  1.0, 2.0, 3.0,
                                                  1.0, 2.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[2x1] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue({1.0, 2.0}, {2, 1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 1.0, 1.0, 2.0, 2.0, 2.0}, newShape, &testDevice));
    }

    SUBCASE("[2x3] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape, &testDevice));
    }

    SUBCASE("[3x1] to [3x3]")
    {
        Shape newShape{3, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {3, 1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 9);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 1.0, 1.0,
                                                  2.0, 2.0, 2.0,
                                                  3.0, 3.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3x1] to [1x2x3x2]")
    {
        Shape newShape{1,2,3,2};
        auto bct = TensorValue({1.0,2.0,3.0}, {1, 3, 1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 12);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 1.0,
                                                  2.0, 2.0,
                                                  3.0, 3.0,
                                                  1.0, 1.0,
                                                  2.0, 2.0,
                                                  3.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[4] to [2]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0, 4.0}, {4}, &testDevice).broadcastTo({2}), std::invalid_argument);
    }

    SUBCASE("[2] to [4]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0}, {2}, &testDevice).broadcastTo({4}), std::invalid_argument);
    }

    SUBCASE("[2x3] to [6]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo({6}), std::invalid_argument);
    }

    SUBCASE("[2x3] to [1x3]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo({1, 3}), std::invalid_argument);
    }

    SUBCASE("[2x3] to [3x2]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo({3, 2}), std::invalid_argument);
    }

    SUBCASE("[2x1x3] to [2x3]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 1, 3}, &testDevice).broadcastTo({2, 3}), std::invalid_argument);
    }
}


TEST_CASE("Tensor - ones")
{
    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::ones({2, 3});
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == false);
        CheckVectorApproxValues(t, Tensor(1.0, {2, 3}));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::ones({2, 3}, true);
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == true);
        CheckVectorApproxValues(t, Tensor(1.0, {2, 3}));
    }
}


TEST_CASE("Tensor - zeros")
{
    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::zeros({2, 3});
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == false);
        CheckVectorApproxValues(t, Tensor(0.0, {2, 3}));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::zeros({2, 3}, true);
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == true);
        CheckVectorApproxValues(t, Tensor(0.0, {2, 3}));
    }
}


TEST_CASE("Tensor - onesLike")
{
    aix::Tensor src(1.5, {2, 3});

    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::onesLike(src);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == false);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(1.0, src.shape()));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::onesLike(src, true);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == true);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(1.0, src.shape()));
    }
}


TEST_CASE("Tensor - zerosLike")
{
    aix::Tensor src(1.5, {2, 3});

    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::zerosLike(src);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == false);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(0.0, src.shape()));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::zerosLike(src, true);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == true);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(0.0, src.shape()));
    }
}


TEST_CASE("Tensor - matmul()")
{
    SUBCASE("Must fail due to an inner dimension mismatch")
    {
        Tensor A = tensor({1.0,2.0,3.0,4.0}, {2,2});
        Tensor B = tensor({1.0,2.0,3.0,4.0}, {1,4});
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("Must fail due to dimension size mismatch")
    {
        Tensor A = tensor({1.0,2.0,3.0,4.0}, Shape{4});
        Tensor B = tensor({1.0,2.0,3.0,4.0}, Shape{1,4});
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("A[1x1] B[1x1] multiplication")
    {
        Tensor A = tensor({2.0}, Shape{1,1});
        Tensor B = tensor({3.0}, Shape{1,1});
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, tensor({6.0}, Shape{1,1}));
    }

    SUBCASE("A[2x3] B[3x4] multiplication")
    {
        Tensor A = tensor({1.0,2.0,3.0,4.0,5.0,6.0}, Shape{2,3});
        Tensor B = tensor({12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0}, Shape{3,4});
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{2,4});
        CHECK(result.value().size() == 8);
        CheckVectorApproxValues(result, tensor({40.0,34.0,28.0,22.0,112.0,97.0,82.0,67.0}, Shape{2,4}));
    }
}


TEST_CASE("Tensor - print")
{
    SUBCASE("scalar tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor(1.0);

        ss << input;
        std::string expected = R"(1

[ Float{} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1 dimension tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, Shape{1});

        ss << input;
        std::string expected = R"(  1

[ Float{1} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1 dimension tensor - three elements")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0, 2.0, 3.0}, Shape{3});

        ss << input;
        std::string expected = R"(  1
  2
  3

[ Float{3} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1x1 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, {1,1});

        ss << input;
        std::string expected = R"(  1

[ Float{1,1} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("2x2 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({
                                 1.0, 2.0,
                                 2.0, 3.0,
                                 }, {2, 2});

        ss << input;
        std::string expected = R"(  1  2
  2  3

[ Float{2,2} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1x2x2 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({
                                 1.0, 2.0,
                                 2.0, 3.0,
                                 }, {1, 2, 2});

        ss << input;
        std::string expected = R"((0,.,.) =
  1  2
  2  3

[ Float{1,2,2} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("2x2x2 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({
                                 1.0, 2.0,
                                 2.0, 3.0,
                                 3.0, 4.0,
                                 4.0, 5.0,
                                 }, {2, 2, 2});

        ss << input;
        std::string expected = R"((0,.,.) =
  1  2
  2  3

(1,.,.) =
  3  4
  4  5

[ Float{2,2,2} ]
)";
        CHECK(ss.str() == expected);
    }

}


TEST_CASE("Tensor - Tensor OP Scalar")
{
    float scalar = 5;
    Shape shape{2, 2};
    std::vector<float> testData = {1.0, -2.0, 0.0, 3.0};

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input + scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x += scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input - scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x -= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input * scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x *= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input / scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x /= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }
}


TEST_CASE("Tensor - Scalar OP Tensor")
{
    float scalar = 5;
    Shape shape{2, 2};
    std::vector<float> testData = {1.0, -2.0, 0.5, 3.0};

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar + input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar + x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar - input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar - x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar * input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar * x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar / input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar / x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }
}


TEST_CASE("Tensor - Tensor OP Scalar Tensor")
{
    float scalar = 5;
    Tensor scalarTensor = tensor(scalar);
    Shape shape{2, 2};
    std::vector<float> testData = {1.0, -2.0, 0.0, 3.0};

    // Scalar tensor has no dimension.
    CHECK(scalarTensor.shape().size() == 0);

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input + scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x += scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input - scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x -= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input * scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x *= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input / scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x /= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }
}


TEST_CASE("Tensor - Reshape")
{
    SUBCASE("scalar -> 1 dimension")
    {
        auto input = Tensor(5.0, {});
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("scalar -> 1x1 dimension")
    {
        auto input = Tensor(5.0, {});
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1 dimension -> 1x1 dimension")
    {
        auto input = Tensor(5.0, {1});
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1 dimension -> scalar")
    {
        auto input = Tensor(5.0, {1});
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1x1 dimension -> scalar")
    {
        auto input = Tensor(5.0, {1,1});
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1x1 dimension -> 1 dimension")
    {
        auto input = Tensor(5.0, {1,1});
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1x3 dimension -> 3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {1,3});
        auto newShape = Shape{3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("2x3 dimension -> 6 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{6};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("6 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {6});
        auto newShape = Shape{2, 3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("2x3 dimension -> 3x1x2 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{3, 1, 2};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("2x3 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("Size mismatch")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("Invalid reshape throws invalid_argument")
    {
        Tensor inputs = tensor({1.0,2.0,3.0,4.0,5.0}, {1,5});
        // Check that reshape throws an invalid_argument exception
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({2,2}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({1,6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({4}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({}), std::invalid_argument);
    }
}


TEST_CASE("Tensor - broadcastTo")
{
    // NOTE: Since TensorValue tests cover the broadcast tests, Tensor does not need exhaustive broadcastTo tests.

    SUBCASE("[1x3] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = tensor({1.0, 2.0, 3.0}, {1, 3}).broadcastTo(newShape);
        CHECK(bct.value().size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, tensor({1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, newShape));
    }

    SUBCASE("[2x3] to [3x2]")
    {
        CHECK_THROWS_AS(tensor({1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0}, {2, 3}).broadcastTo({3, 2}), std::invalid_argument);
    }
}


TEST_CASE("TensorValue - broadcast")
{
    SUBCASE("([],[1],[1,1],[1,3],[2,3]) op [2x3]")
    {
        std::vector<Shape> shapes{{}, {1}, {1,1}, {1,3}, {2,3}};
        for (const auto& shape : shapes)
        {
            Shape newShape{2,3};
            size_t newSize = 6;
            auto x = TensorValue(2.0, shape, &testDevice);
            auto y = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape, &testDevice);

            auto a1 = x + y;
            CHECK(a1.size() == newSize);
            CHECK(a1.shape() == newShape);
            CheckVectorApproxValues(a1, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            // Try reverse order
            auto a2 = y + x;
            CHECK(a2.size() == newSize);
            CHECK(a2.shape() == newShape);
            CheckVectorApproxValues(a2, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            auto s1 = x - y;
            CHECK(s1.size() == newSize);
            CHECK(s1.shape() == newShape);
            CheckVectorApproxValues(s1, TensorValue({1.0, 0.0, -1.0, -2.0, -3.0, -4.0}, newShape, &testDevice));

            // Try reverse order
            auto s2 = y - x;
            CHECK(s2.size() == newSize);
            CHECK(s2.shape() == newShape);
            CheckVectorApproxValues(s2, TensorValue({-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, newShape, &testDevice));

            auto m1 = x * y;
            CHECK(m1.size() == newSize);
            CHECK(m1.shape() == newShape);
            CheckVectorApproxValues(m1, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            // Try reverse order
            auto m2 = y * x;
            CHECK(m2.size() == newSize);
            CHECK(m2.shape() == newShape);
            CheckVectorApproxValues(m2, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            auto d1 = x / y;
            CHECK(d1.size() == newSize);
            CHECK(d1.shape() == newShape);
            CheckVectorApproxValues(d1, TensorValue({2.0, 1.0, 0.666667, 0.5, 0.4, 0.333334}, newShape, &testDevice));

            // Try reverse order
            auto d2 = y / x;
            CHECK(d2.size() == newSize);
            CHECK(d2.shape() == newShape);
            CheckVectorApproxValues(d2, TensorValue({0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, newShape, &testDevice));
        }
    }

    SUBCASE("([],[1],[1,1],[1,3],[2,3]) op [2x3] - In-place operations")
    {
        std::vector<Shape> shapes{{}, {1}, {1,1}, {1,3}, {2,3}};
        for (const auto& shape : shapes)
        {
            Shape newShape{2,3};
            size_t newSize = 6;
            auto x = TensorValue(2.0, shape, &testDevice);
            auto y = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape, &testDevice);

            auto a1 = x;
            a1 += y;
            CHECK(a1.size() == newSize);
            CHECK(a1.shape() == newShape);
            CheckVectorApproxValues(a1, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            // Try reverse order
            auto a2 = y;
            a2 += x;
            CHECK(a2.size() == newSize);
            CHECK(a2.shape() == newShape);
            CheckVectorApproxValues(a2, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            auto s1 = x;
            s1 -= y;
            CHECK(s1.size() == newSize);
            CHECK(s1.shape() == newShape);
            CheckVectorApproxValues(s1, TensorValue({1.0, 0.0, -1.0, -2.0, -3.0, -4.0}, newShape, &testDevice));

            // Try reverse order
            auto s2 = y;
            s2 -= x;
            CHECK(s2.size() == newSize);
            CHECK(s2.shape() == newShape);
            CheckVectorApproxValues(s2, TensorValue({-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, newShape, &testDevice));

            auto m1 = x;
            m1 *= y;
            CHECK(m1.size() == newSize);
            CHECK(m1.shape() == newShape);
            CheckVectorApproxValues(m1, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            // Try reverse order
            auto m2 = y;
            m2 *= x;
            CHECK(m2.size() == newSize);
            CHECK(m2.shape() == newShape);
            CheckVectorApproxValues(m2, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            auto d1 = x;
            d1 /= y;
            CHECK(d1.size() == newSize);
            CHECK(d1.shape() == newShape);
            CheckVectorApproxValues(d1, TensorValue({2.0, 1.0, 0.666667, 0.5, 0.4, 0.333334}, newShape, &testDevice));

            // Try reverse order
            auto d2 = y;
            d2 /= x;
            CHECK(d2.size() == newSize);
            CHECK(d2.shape() == newShape);
            CheckVectorApproxValues(d2, TensorValue({0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, newShape, &testDevice));
        }
    }

    SUBCASE("[2x3] [3x2]")
    {
        std::initializer_list<float> data{1.0, 2.0, 3.0,4.0, 5.0, 6.0};
        Shape shape1{2,3};
        Shape shape2{3,2};
        auto tensorVal1 = TensorValue(data, shape1, &testDevice);
        auto tensorVal2 = TensorValue(data, shape2, &testDevice);

        // Add
        CHECK_THROWS_AS({ auto t = tensorVal1; t += tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t += tensorVal1; }, std::invalid_argument);

        // Sub
        CHECK_THROWS_AS({ auto t = tensorVal1; t -= tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t -= tensorVal1; }, std::invalid_argument);

        // Mul
        CHECK_THROWS_AS({ auto t = tensorVal1; t *= tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t *= tensorVal1; }, std::invalid_argument);

        // Div
        CHECK_THROWS_AS({ auto t = tensorVal1; t /= tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t /= tensorVal1; }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue - Data Type Conversion")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f64Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }
}


TEST_CASE("TensorValue - Data Type Conversion wit .to()")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f64Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }
}


TEST_CASE("TensorValue - Data Type Promotion")
{
    auto f32Data = std::initializer_list<float>{2.0, 4.0, 6.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    // Add

    SUBCASE("Add - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 3.0f);
        CHECK(tv.data<float>()[1] == 6.0f);
        CHECK(tv.data<float>()[2] == 9.0f);
    }

    SUBCASE("Add - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    // Sub

    SUBCASE("Sub - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 1.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 3.0f);
    }

    SUBCASE("Sub - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    // Mul

    SUBCASE("Mul - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==  2.0f);
        CHECK(tv.data<float>()[1] ==  8.0f);
        CHECK(tv.data<float>()[2] == 18.0f);
    }

    SUBCASE("Mul - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    // Div

    SUBCASE("Div - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 2.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 2.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 2.0f);
    }

    SUBCASE("Div - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 2.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  2.0);
        CHECK(tv.data<double>()[2] ==  2.0);
    }

    // Pow

    SUBCASE("Pow - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==   2.0);
        CHECK(tv.data<double>()[1] ==  16.0);
        CHECK(tv.data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==   2.0f);
        CHECK(tv.data<float>()[1] ==  16.0f);
        CHECK(tv.data<float>()[2] == 216.0f);
    }

    SUBCASE("Pow - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==   2.0);
        CHECK(tv.data<double>()[1] ==  16.0);
        CHECK(tv.data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==   2.0);
        CHECK(tv.data<double>()[1] ==  16.0);
        CHECK(tv.data<double>()[2] == 216.0);
    }

    // Matmul

    SUBCASE("Matmul - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat64);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == 1);
        CHECK(tv.data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat32);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == 1);
        CHECK(tv.data<float>()[0] == 28.0f);
    }

    SUBCASE("Matmul - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat64);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == 1);
        CHECK(tv.data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat32);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == 1);
        CHECK(tv.data<double>()[0] == 28.0);
    }
}


TEST_CASE("TensorValue - Data Type Promotion (In-Place")
{
    auto f32Data = std::initializer_list<float>{2.0, 4.0, 6.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    // Add - In-place

    SUBCASE("Add - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 3.0f);
        CHECK(tv.data<float>()[1] == 6.0f);
        CHECK(tv.data<float>()[2] == 9.0f);
    }

    SUBCASE("Add - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 3.0);
        CHECK(tv.data<float>()[1] == 6.0);
        CHECK(tv.data<float>()[2] == 9.0);
    }

    SUBCASE("Add - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    // Sub - In-place

    SUBCASE("Sub - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 1.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 3.0f);
    }

    SUBCASE("Sub - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 1.0);
        CHECK(tv.data<float>()[1] == 2.0);
        CHECK(tv.data<float>()[2] == 3.0);
    }

    SUBCASE("Sub - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    // Mul - In-place

    SUBCASE("Mul - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==  2.0f);
        CHECK(tv.data<float>()[1] ==  8.0f);
        CHECK(tv.data<float>()[2] == 18.0f);
    }

    SUBCASE("Mul - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==  2.0);
        CHECK(tv.data<float>()[1] ==  8.0);
        CHECK(tv.data<float>()[2] == 18.0);
    }

    SUBCASE("Mul - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    // Div - In-place

    SUBCASE("Div - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 2.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 2.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 2.0f);
    }

    SUBCASE("Div - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 2.0);
        CHECK(tv.data<float>()[1] == 2.0);
        CHECK(tv.data<float>()[2] == 2.0);
    }

    SUBCASE("Div - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  2.0);
        CHECK(tv.data<double>()[2] ==  2.0);
    }
}




TEST_CASE("Tensor - Data Type Conversion")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f64Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }
}


TEST_CASE("Tensor - Data Type Conversion wit .to()")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = aix::tensor(f64Data, shape, false).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = aix::tensor(f64Data, shape).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f64Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }
}


TEST_CASE("Tensor - Data Type Promotion")
{
    auto f32Data = std::initializer_list<float>{2.0, 4.0, 6.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    // Add

    SUBCASE("Add - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 3.0f);
        CHECK(tv.value().data<float>()[1] == 6.0f);
        CHECK(tv.value().data<float>()[2] == 9.0f);
    }

    SUBCASE("Add - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    // Sub

    SUBCASE("Sub - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 1.0f);
        CHECK(tv.value().data<float>()[1] == 2.0f);
        CHECK(tv.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Sub - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    // Mul

    SUBCASE("Mul - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] ==  2.0f);
        CHECK(tv.value().data<float>()[1] ==  8.0f);
        CHECK(tv.value().data<float>()[2] == 18.0f);
    }

    SUBCASE("Mul - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    // Div

    SUBCASE("Div - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 2.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 2.0f);
        CHECK(tv.value().data<float>()[1] == 2.0f);
        CHECK(tv.value().data<float>()[2] == 2.0f);
    }

    SUBCASE("Div - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 2.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  2.0);
        CHECK(tv.value().data<double>()[2] ==  2.0);
    }

    // Pow

    SUBCASE("Pow - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] ==   2.0f);
        CHECK(tv.value().data<float>()[1] ==  16.0f);
        CHECK(tv.value().data<float>()[2] == 216.0f);
    }

    SUBCASE("Pow - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat64);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, shape, false, DataType::kFloat32);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    // Matmul

    SUBCASE("Matmul - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, {3,1}, false, DataType::kFloat64);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, {3,1}, false, DataType::kFloat32);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<float>()[0] == 28.0f);
    }

    SUBCASE("Matmul - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, false, DataType::kFloat32);
        auto tv2 = aix::tensor(f64Data, {3,1}, false, DataType::kFloat64);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, false, DataType::kFloat64);
        auto tv2 = aix::tensor(f64Data, {3,1}, false, DataType::kFloat32);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }
}
