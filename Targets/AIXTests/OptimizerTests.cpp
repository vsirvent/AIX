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
    auto x = tensor({2}, {1,1}, true);
    auto y = tensor({3}, {1,1}, true);
    auto t = tensor({4}, {1,1}, true);
    auto u = tensor({5}, {1,1}, true);

    auto z = x * (x + y) / t - y * y;
    auto m = x * z + Tensor::sin(u) * u;



    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward(1);      // ∂m/∂m = 1.

    // Create a vector list of learnable parameters for optimizations.
    auto parameters = {x, y, t, u};

    // Create an instance of the optimizer with a specified learning rate.
    optim::SGDOptimizer optimizer(parameters, 0.01f);

    // Perform an optimization step.
    optimizer.step();

    // Print the updated values of x, y, t, u to see the effect of the single optimization step.
    CHECK(x.value().data()[0] == Approx(2.03));
    CHECK(y.value().data()[0] == Approx(3.11));
    CHECK(t.value().data()[0] == Approx(4.0125));
    CHECK(u.value().data()[0] == Approx(4.99541));
    // Note: Results are consistent with those from PyTorch.
}
