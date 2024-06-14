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


TEST_CASE("TensorValue::matmul 1x1 1x1")
{
    auto a = TensorValue(2, {1, 1}, &testDevice);
    auto b = TensorValue(3, {1, 1}, &testDevice);
    auto c = a.matmul(b);

    CHECK(c.shape() == Shape{1, 1});
    CheckVectorApproxValues(c, TensorValue(6, c.shape(), &testDevice));
}


TEST_CASE("TensorValue::matmul 2x4 4x3")
{
    auto a = TensorValue({1,2,3,4,5,6,7,8},            {2, 4}, &testDevice);
    auto b = TensorValue({1,2,3,4,5,6,7,8,9,10,11,12}, {4, 3}, &testDevice);
    auto c = a.matmul(b);     // Result matrix.

    CHECK(c.shape() == Shape{2, 3});
    CheckVectorApproxValues(c, TensorValue({70, 80, 90, 158, 184, 210}, c.shape(), &testDevice));
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
