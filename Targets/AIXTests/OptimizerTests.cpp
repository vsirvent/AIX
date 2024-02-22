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


TEST_CASE("Simple optimizer test")
{
    auto x = Tensor(2, true);
    auto y = Tensor(3, true);
    auto t = Tensor(4, true);
    auto u = Tensor(5, true);

    // z = x * (x + y) / t - y * y
    // m = x * z + Sin(u) * u
    // Decompose all operations into atomic calculations to construct the graph.
    // Assume that all operations are scalar and applied element-wise.
    // Execute calculations as though they were Assembly instructions. :)
    Add  p1(x, y);    // p1 = x + y
    Mul  m1(x, p1);   // m1 = x * p1
    Div  d1(m1, t);   // d1 = m1 / t
    Mul  m2(y, y);    // m2 = y * y
    Sub  z(d1, m2);   // z = d1 - m2
    Mul  r(x, z);     // r = x * z
    Sin  s1(u);       // s1 = Sin(u)
    Mul  s2(s1, u);   // s2 = s1 * u
    Add  m(r, s2);    // m = r + s2
    // TODO: Implement MatMull when Tensor can handle multi-dimensions.

    // Traverse the graph (starting from the end) to calculate all expression values.
    // This approach is known as lazy evaluation, meaning that values are not calculated
    // until the 'evaluate' function is called.
    m.evaluate();

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward(1);      // ∂m/∂m = 1.

    // Define the tensors that require gradients. These are learnable parameters.
    std::vector<Tensor*> parameters = {&x, &y, &t, &u};

    // Create an instance of the optimizer with a specified learning rate.
    SGDOptimizer optimizer(parameters, 0.01f);

    // Perform an optimization step.
    optimizer.step();

    // Print the updated values of x, y, t, u to see the effect of the single optimization step.
    CHECK(x.value() == Approx(2.03));
    CHECK(y.value() == Approx(3.11));
    CHECK(t.value() == Approx(4.0125));
    CHECK(u.value() == Approx(4.99541));
    // Note: Results are consistent with those from PyTorch.
}
