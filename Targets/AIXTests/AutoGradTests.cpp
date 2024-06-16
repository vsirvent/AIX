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
    TestModel(const std::vector<DataType> & xData,
              const std::vector<DataType> & yData,
              const std::vector<DataType> & tData,
              const std::vector<DataType> & uData,
              const Shape & shape)
    {
        m_x = tensor(xData, shape, true);
        m_y = tensor(yData, shape, true);
        m_t = tensor(tData, shape, true);
        m_u = tensor(uData, shape, true);

        registerParameter(m_x);
        registerParameter(m_y);
        registerParameter(m_t);
        registerParameter(m_u);
    }

    Tensor forward([[maybe_unused]] Tensor x) const final
    {
        auto z = m_x * (m_x + m_y) / m_t - tanh(m_y * m_y);
        auto m = m_x * z + sin(m_u) * m_u;
        return m;
    }

    Tensor  m_x;
    Tensor  m_y;
    Tensor  m_t;
    Tensor  m_u;
};


TEST_CASE("Auto Grad - Module Test - 1x1 Tensor")
{
    auto shape = std::vector<size_t>{};     // Scalar has no dimensions.

    auto tm = TestModel({2},   // x
                        {3},   // y
                        {4},   // t
                        {5},   // u
                        shape);

    auto m = tm.forward({});

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward();

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

    CHECK(tm.m_x.grad().item() == Approx(5));
    CHECK(tm.m_y.grad().item() == Approx(0.999999));
    CHECK(tm.m_t.grad().item() == Approx(-1.25));
    CHECK(tm.m_u.grad().item() == Approx(0.459387));
    CHECK(m.value().item()     == Approx(-1.79462));
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

    CheckVectorApproxValues(tm.m_x.grad(), tensor({0.8, 3.66667},       shape).value());
    CheckVectorApproxValues(tm.m_y.grad(), tensor({0.199999, 0.666667}, shape).value());
    CheckVectorApproxValues(tm.m_t.grad(), tensor({-0.16, -0.666667},   shape).value());
    CheckVectorApproxValues(tm.m_u.grad(), tensor({5.9343, -0.174642},  shape).value());
    CheckVectorApproxValues(m,             tensor({4.39891, 9.91487},   shape));
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

    CheckVectorApproxValues(tm.m_x.grad(), tensor({0.307692, 2.14286, 4.4, 7, 9.88235, 13}, shape).value());
    CheckVectorApproxValues(tm.m_y.grad(), tensor({0.0769231, 0.285714, 0.6, 1, 1.47059, 2}, shape).value());
    CheckVectorApproxValues(tm.m_t.grad(), tensor({-0.0473373, -0.204082, -0.48, -0.875, -1.38408, -2}, shape).value());
    CheckVectorApproxValues(tm.m_u.grad(), tensor({18.9353, 9.07459, -10.6657, -22.008, -13.1014, 9.27472}, shape).value());
    CheckVectorApproxValues(m,             tensor({2.46305, 19.116, 21.7698, 9.80527, -0.933655, 8.26612}, shape));
}


TEST_CASE("Auto Grad - log Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, true);
    auto z = log(x);
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({10, 5, 3.33333, 2.5}, shape).value());
}


TEST_CASE("Auto Grad - exp Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, true);
    auto z = exp(x);
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({1.10517, 1.2214, 1.34986, 1.49182}, shape).value());
}


TEST_CASE("Auto Grad - sum Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, true);
    auto z = x.sum();
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({1, 1, 1, 1}, shape).value());
}


TEST_CASE("Auto Grad - sigmoid Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, true);
    auto z = aix::nn::Sigmoid().forward(x);
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({0.249376, 0.247517, 0.244458, 0.240261}, shape).value());
}
