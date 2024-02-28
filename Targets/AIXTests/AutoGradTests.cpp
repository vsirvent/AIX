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


struct TestModel : public aix::nn::Module
{
    TestModel(const aix::Array & xData,
              const aix::Array & yData,
              const aix::Array & tData,
              const aix::Array & uData,
              const aix::Shape & shape)
    {
        m_x = aix::tensor(xData, shape, true);
        m_y = aix::tensor(yData, shape, true);
        m_t = aix::tensor(tData, shape, true);
        m_u = aix::tensor(uData, shape, true);

        registerParameter(m_x);
        registerParameter(m_y);
        registerParameter(m_t);
        registerParameter(m_u);
    }

    Tensor forward([[maybe_unused]] Tensor x) const final
    {
        auto z = m_x * (m_x + m_y) / m_t - Tensor::tanh(m_y * m_y);
        auto m = m_x * z + Tensor::sin(m_u) * m_u;
        return m;
    }

    Tensor  m_x;
    Tensor  m_y;
    Tensor  m_t;
    Tensor  m_u;
};


TEST_CASE("Auto Grad - Module Test - 1x1 Tensor")
{
    auto shape = std::vector<size_t>{1, 1};

    auto tm = TestModel({2},   // x
                        {3},   // y
                        {4},   // t
                        {5},   // u
                        shape);

    auto m = tm.forward({});
    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward(1);      // ∂m/∂m = [1]  1x1 tensor.

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CHECK(tm.m_x.grad().data()[0] == Approx(5));
    CHECK(tm.m_y.grad().data()[0] == Approx(0.999999));
    CHECK(tm.m_t.grad().data()[0] == Approx(-1.25));
    CHECK(tm.m_u.grad().data()[0] == Approx(0.459387));
    CHECK(m.value().data()[0]     == Approx(-1.79462));
}


TEST_CASE("Auto Grad - Module Test - 1x2 Tensor")
{
    auto shape = std::vector<size_t>{1, 2};

    auto tm = TestModel({1, 2},   // x
                        {3, 4},   // y
                        {5, 6},   // t
                        {7, 8},   // u
                        shape);

    auto m = tm.forward({});
    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward();   // ∂m/∂m = [1, 1]  1x2 tensor

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CheckVectorApproxValues(tm.m_x.grad().data(), {0.8, 3.66667});
    CheckVectorApproxValues(tm.m_y.grad().data(), {0.199999, 0.666667});
    CheckVectorApproxValues(tm.m_t.grad().data(), {-0.16, -0.666667});
    CheckVectorApproxValues(tm.m_u.grad().data(), {5.9343, -0.174642});
    CheckVectorApproxValues(m.value().data(),     {4.39891, 9.91487});
}


TEST_CASE("Auto Grad - Module Test - 2x3 Tensor")
{
    auto shape = std::vector<size_t>{2, 3};

    auto tm = TestModel({ 1,  2,  3,  4,  5,  6},   // x
                        { 7,  8,  9, 10, 11, 12},   // y
                        {13, 14, 15, 16, 17, 18},   // t
                        {19, 20, 21, 22, 23, 24},   // u
                        shape);

    auto m = tm.forward({});
    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward();   // ∂m/∂m = [1,1,1,1,1,1]  2x3 tensor

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CheckVectorApproxValues(tm.m_x.grad().data(), {0.307692, 2.14286, 4.4, 7, 9.88235, 13});
    CheckVectorApproxValues(tm.m_y.grad().data(), {0.0769231, 0.285714, 0.6, 1, 1.47059, 2});
    CheckVectorApproxValues(tm.m_t.grad().data(), {-0.0473373, -0.204082, -0.48, -0.875, -1.38408, -2});
    CheckVectorApproxValues(tm.m_u.grad().data(), {18.9353, 9.07459, -10.6657, -22.008, -13.1014, 9.27472});
    CheckVectorApproxValues(m.value().data(),     {2.46305, 19.116, 21.7698, 9.80527, -0.933655, 8.26612});
}
