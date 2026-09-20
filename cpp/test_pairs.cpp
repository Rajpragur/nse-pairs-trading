#include "pairs_math.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x, y;
    for (int i = 0; i < 500; ++i) {
        x.push_back(std::sin(i * 0.03) * 10.0 + i * 0.01);
        y.push_back(2.0 + 1.5 * x.back() + std::sin(i * 0.17) * 0.03);
    }
    const auto result = pairs::engle_granger(y, x);
    if (std::abs(result.beta - 1.5) >= 0.01 || !std::isfinite(result.adf_t_stat)) {
        std::cerr << "beta=" << result.beta << " adf_t=" << result.adf_t_stat << "\n";
        return 1;
    }
    return 0;
}
