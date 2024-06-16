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
    auto x = TensorValue({1, 2, 3}, {1, 3}, &testDevice);
    auto y = TensorValue({4, 5, 6}, {1, 3}, &testDevice);

    auto z = x + y;

    CHECK(z.shape() == Shape{1, 3});
    CheckVectorApproxValues(z, TensorValue({5, 7, 9}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Add")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3}, &testDevice);
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3}, &testDevice);

    auto z = x + y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({8, 10, 12, 14, 16, 18}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Sub")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3}, &testDevice);
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3}, &testDevice);

    auto z = x - y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({-6, -6, -6, -6, -6, -6}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Mul")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3}, &testDevice);
    auto y = TensorValue({7, 8, 9, 10, 11, 12}, {2, 3}, &testDevice);

    auto z = x * y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({7, 16, 27, 40, 55, 72}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Div")
{
    auto x = TensorValue({-5, 0, 5, 10, 15, 20}, {2, 3}, &testDevice);
    auto y = TensorValue({5, 5, 5, 5, -5, -20},  {2, 3}, &testDevice);

    auto z = x / y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({-1, 0, 1, 2, -3, -1}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3}, &testDevice);
    auto z = x;
    z.data()[0] = 1;

    CHECK(z.shape() == x.shape());
    CheckVectorApproxValues(z, x);
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1, 2, 3, 4, 5, 6},    {2, 3}, &testDevice);
    auto copyTensor = [](TensorValue value) { return value; };
    auto z = copyTensor(x);

    CHECK(z.shape() == x.shape());
    CheckVectorApproxValues(z, x);
}


TEST_CASE("TensorValue - item()")
{
    SUBCASE("Must fail due to a non-scalar tensor")
    {
        TensorValue input = TensorValue({1.0,2.0}, {2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input.item(), std::invalid_argument);

        TensorValue input2 = TensorValue({1.0,2.0,3.0,4.0}, {2, 2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input2.item(), std::invalid_argument);
    }

    SUBCASE("Should return a scalar value")
    {
        TensorValue input = TensorValue(1.0, Shape{}, &testDevice);
        CHECK(input.item() == 1.0);
    }
}


TEST_CASE("TensorValue - matmul()")
{
    SUBCASE("Must fail due to an inner dimension mismatch")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0}, {2,2}, &testDevice);
        TensorValue B = TensorValue({1.0,2.0,3.0,4.0}, {1,4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("Must fail due to dimension size mismatch")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0}, {4}, &testDevice);
        TensorValue B = TensorValue({1.0,2.0,3.0,4.0}, {1,4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("A[1x1] B[1x1] multiplication")
    {
        TensorValue A = TensorValue(2.0, Shape{1,1}, &testDevice);
        TensorValue B = TensorValue(3.0, Shape{1,1}, &testDevice);
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, TensorValue(6.0, Shape{1,1}, &testDevice));
    }

    SUBCASE("A[2x3] B[3x4] multiplication")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, Shape{2,3}, &testDevice);
        TensorValue B = TensorValue({12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0}, Shape{3,4}, &testDevice);
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{2,4});
        CHECK(result.size() == 8);
        CheckVectorApproxValues(result, TensorValue({40.0,34.0,28.0,22.0,112.0,97.0,82.0,67.0,}, Shape{2,4}, &testDevice));
    }
}


TEST_CASE("TensorValue - transpose()")
{
    SUBCASE("Must fail if dimension size is different than 2")
    {
        TensorValue input1 = TensorValue({1.0,2.0,3.0,4.0}, {2,1,2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input1.transpose(), std::invalid_argument);
        TensorValue input2 = TensorValue({1.0,2.0,3.0,4.0}, {4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input2.transpose(), std::invalid_argument);
    }

    SUBCASE("1x1 transpose")
    {
        TensorValue A = TensorValue(1.0, {1,1}, &testDevice);
        auto result = A.transpose();
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, TensorValue(1.0, {1,1}, &testDevice));
    }

    SUBCASE("3x2 transpose")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose();
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}, &testDevice));
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


TEST_CASE("TensorValue::transpose 1x1")
{
    auto a = TensorValue(2, {1, 1}, &testDevice).transpose();
    CHECK(a.shape() == Shape{1, 1});
    CheckVectorApproxValues(a, TensorValue(2, a.shape(), &testDevice));
}


TEST_CASE("TensorValue::transpose 2x3")
{
    auto a = TensorValue({1, 2, 3, 4, 5, 6}, {2, 3}, &testDevice).transpose();
    CHECK(a.shape() == Shape{3, 2});
    CheckVectorApproxValues(a, TensorValue({1, 4, 2, 5, 3, 6}, a.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Add with Scalar")
{
    auto x = TensorValue({1, 2, 3}, {1, 3}, &testDevice);
    DataType scalar = 5;
    x += scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({6, 7, 8}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Sub with Scalar")
{
    auto x = TensorValue({6, 7, 8}, {1, 3}, &testDevice);
    DataType scalar = 5;
    x -= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({1, 2, 3}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Mul with Scalar")
{
    auto x = TensorValue({1, 2, 3}, {1, 3}, &testDevice);
    DataType scalar = 5;
    x *= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({5, 10, 15}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Div with Scalar")
{
    auto x = TensorValue({6, 7, 8}, {1, 3}, &testDevice);
    DataType scalar = 2;
    x /= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({3, 3.5, 4}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Add with TensorValue")
{
    auto x = TensorValue({6, 7, 8}, {1, 3}, &testDevice);
    auto y = TensorValue({1, 2, -1}, {1, 3}, &testDevice);
    x += y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({7, 9, 7}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Sub with TensorValue")
{
    auto x = TensorValue({6, 7, 8}, {1, 3}, &testDevice);
    auto y = TensorValue({1, 2, -1}, {1, 3}, &testDevice);
    x -= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({5, 5, 9}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Mul with TensorValue")
{
    auto x = TensorValue({6, 7, 8}, {1, 3}, &testDevice);
    auto y = TensorValue({2, 3, -2}, {1, 3}, &testDevice);
    x *= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({12, 21, -16}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Div with TensorValue")
{
    auto x = TensorValue({6, 7, 8}, {1, 3}, &testDevice);
    auto y = TensorValue({2, 2, -2}, {1, 3}, &testDevice);
    x /= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({3, 3.5, -4}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - Unary Minus")
{
    auto x = TensorValue({1, -2, 3}, {1, 3}, &testDevice);
    auto y = -x;

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({-1, 2, -3}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Fill")
{
    auto x = TensorValue({1, 1, 1}, {1, 3}, &testDevice);
    DataType fillValue = 7;
    x.fill(fillValue);

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({7, 7, 7}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - Sum")
{
    auto x = TensorValue({-1, 2.2, 0, 4.8}, {2, 2}, &testDevice);

    CHECK(x.sum() == Approx(6));
}


TEST_CASE("TensorValue - Mean")
{
    auto x = TensorValue({1, 2, 3, 4}, {2, 2}, &testDevice);

    CHECK(x.mean() == Approx(2.5));
}


TEST_CASE("TensorValue - Sqrt")
{
    auto x = TensorValue({4, 9, 16}, {1, 3}, &testDevice);
    auto y = x.sqrt();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({2, 3, 4}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Sin")
{
    auto x = TensorValue({0.5, 0.0, -0.5}, {1, 3}, &testDevice);
    auto y = x.sin();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({DataType(std::sin(0.5)),
                                            DataType(std::sin(0)),
                                            DataType(std::sin(-0.5))}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Cos")
{
    auto x = TensorValue({0.5, 0.0, -0.5}, {1, 3}, &testDevice);
    auto y = x.cos();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({DataType(std::cos(0.5)),
                                            DataType(std::cos(0)),
                                            DataType(std::cos(-0.5))}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Device Switch")
{
    auto x = TensorValue({1, 2, 3}, {1, 3}, &testDevice);

    Device  newDevice;
    x.device(&newDevice);

    CHECK(x.device() == &newDevice);
    CheckVectorApproxValues(x, TensorValue({1, 2, 3}, x.shape(), &newDevice));
}


TEST_CASE("TensorValue - Reshape")
{
    SUBCASE("scalar -> 1 dimension")
    {
        auto input = TensorValue(5, {}, &testDevice);
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5, newShape, &testDevice));
    }

    SUBCASE("scalar -> 1x1 dimension")
    {
        auto input = TensorValue(5, {}, &testDevice);
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5, newShape, &testDevice));
    }

    SUBCASE("1 dimension -> 1x1 dimension")
    {
        auto input = TensorValue(5, {1}, &testDevice);
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5, newShape, &testDevice));
    }

    SUBCASE("1 dimension -> scalar")
    {
        auto input = TensorValue(5, {1}, &testDevice);
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5, newShape, &testDevice));
    }

    SUBCASE("1x1 dimension -> scalar")
    {
        auto input = TensorValue(5, {1,1}, &testDevice);
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5, newShape, &testDevice));
    }

    SUBCASE("1x1 dimension -> 1 dimension")
    {
        auto input = TensorValue(5, {1,1}, &testDevice);
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5, newShape, &testDevice));
    }

    SUBCASE("1x3 dimension -> 3 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0};
        auto input = TensorValue(data, {1,3}, &testDevice);
        auto newShape = Shape{3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 6 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{6};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("6 dimension -> 2x3 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {6}, &testDevice);
        auto newShape = Shape{2, 3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 3x1x2 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{3, 1, 2};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 2x3 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
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


TEST_CASE("Tensor - ones")
{
    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::ones({2, 3});
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == false);

        aix::TensorValue expected(1.0, {2, 3}, t.value().device());
        CheckVectorApproxValues(t.value(), expected);
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::ones({2, 3}, true);
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == true);

        aix::TensorValue expected(1.0, {2, 3}, t.value().device());
        CheckVectorApproxValues(t.value(), expected);
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

        aix::TensorValue expected(0.0, {2, 3}, t.value().device());
        CheckVectorApproxValues(t.value(), expected);
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::zeros({2, 3}, true);
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == true);

        aix::TensorValue expected(0.0, {2, 3}, t.value().device());
        CheckVectorApproxValues(t.value(), expected);
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

        aix::TensorValue expected(1.0, src.shape(), src.value().device());
        CheckVectorApproxValues(t.value(), expected);
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::onesLike(src, true);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == true);
        CHECK(t.value().device() == src.value().device());

        aix::TensorValue expected(1.0, src.shape(), src.value().device());
        CheckVectorApproxValues(t.value(), expected);
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

        aix::TensorValue expected(0.0, src.shape(), src.value().device());
        CheckVectorApproxValues(t.value(), expected);
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::zerosLike(src, true);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == true);
        CHECK(t.value().device() == src.value().device());

        aix::TensorValue expected(0.0, src.shape(), src.value().device());
        CheckVectorApproxValues(t.value(), expected);
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
        CheckVectorApproxValues(result.value(), tensor({6.0}, Shape{1,1}).value());
    }

    SUBCASE("A[2x3] B[3x4] multiplication")
    {
        Tensor A = tensor({1.0,2.0,3.0,4.0,5.0,6.0}, Shape{2,3});
        Tensor B = tensor({12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0}, Shape{3,4});
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{2,4});
        CHECK(result.value().size() == 8);
        CheckVectorApproxValues(result.value(), tensor({40.0,34.0,28.0,22.0,112.0,97.0,82.0,67.0,}, Shape{2,4}).value());
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

[ Shape{} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1 dimension tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, Shape{1});

        ss << input;
        std::string expected = R"(  1

[ Shape{1} ]
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

[ Shape{3} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1x1 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, {1,1});

        ss << input;
        std::string expected = R"(  1

[ Shape{1,1} ]
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

[ Shape{2,2} ]
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

[ Shape{1,2,2} ]
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

[ Shape{2,2,2} ]
)";
        CHECK(ss.str() == expected);
    }

}


TEST_CASE("Tensor - Tensor OP Scalar")
{
    DataType scalar = 5;
    Shape shape{2, 2};
    std::vector<DataType> testData = {1.0, -2.0, 0.0, 3.0};

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input + scalar;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x += scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input - scalar;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x -= scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input * scalar;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x *= scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input / scalar;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x /= scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }
}


TEST_CASE("Tensor - Scalar OP Tensor")
{
    DataType scalar = 5;
    Shape shape{2, 2};
    std::vector<DataType> testData = {1.0, -2.0, 0.5, 3.0};

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = scalar + input;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x = scalar + x; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = scalar - input;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x = scalar - x; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = scalar * input;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x = scalar * x; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = scalar / input;

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x = scalar / x; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }
}


TEST_CASE("Tensor - Tensor OP Scalar Tensor")
{
    DataType scalar = 5;
    Tensor scalarTensor = tensor(scalar);
    Shape shape{2, 2};
    std::vector<DataType> testData = {1.0, -2.0, 0.0, 3.0};

    // Scalar tensor has no dimension.
    CHECK(scalarTensor.shape().size() == 0);

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input + scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x += scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input - scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x -= scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input * scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x *= scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::tensor(data, shape);
        auto result = input / scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](DataType & x) { x /= scalar; });
        CheckVectorApproxValues(result.value(), tensor(data, shape).value());
    }
}


TEST_CASE("Tensor - Reshape")
{
    SUBCASE("scalar -> 1 dimension")
    {
        auto input = Tensor(5, {});
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(5, newShape).value());
    }

    SUBCASE("scalar -> 1x1 dimension")
    {
        auto input = Tensor(5, {});
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(5, newShape).value());
    }

    SUBCASE("1 dimension -> 1x1 dimension")
    {
        auto input = Tensor(5, {1});
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(5, newShape).value());
    }

    SUBCASE("1 dimension -> scalar")
    {
        auto input = Tensor(5, {1});
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(5, newShape).value());
    }

    SUBCASE("1x1 dimension -> scalar")
    {
        auto input = Tensor(5, {1,1});
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(5, newShape).value());
    }

    SUBCASE("1x1 dimension -> 1 dimension")
    {
        auto input = Tensor(5, {1,1});
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(5, newShape).value());
    }

    SUBCASE("1x3 dimension -> 3 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0};
        auto input = Tensor(data.data(), data.size(), {1,3});
        auto newShape = Shape{3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(data.data(), data.size(), newShape).value());
    }

    SUBCASE("2x3 dimension -> 6 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.data(), data.size(), {2,3});
        auto newShape = Shape{6};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(data.data(), data.size(), newShape).value());
    }

    SUBCASE("6 dimension -> 2x3 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.data(), data.size(), {6});
        auto newShape = Shape{2, 3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(data.data(), data.size(), newShape).value());
    }

    SUBCASE("2x3 dimension -> 3x1x2 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.data(), data.size(), {2,3});
        auto newShape = Shape{3, 1, 2};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(data.data(), data.size(), newShape).value());
    }

    SUBCASE("2x3 dimension -> 2x3 dimension")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.data(), data.size(), {2,3});
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(data.data(), data.size(), newShape).value());
    }

    SUBCASE("Size mismatch")
    {
        auto data = std::vector<DataType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.data(), data.size(), {2,3});
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x.value(), Tensor(data.data(), data.size(), newShape).value());
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
