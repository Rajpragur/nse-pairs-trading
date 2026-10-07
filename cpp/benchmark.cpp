#include "pairs_math.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t n = 100000;
    std::vector<double> x(n), y(n);
    double walk = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        walk += std::sin(static_cast<double>(i) * 0.017) * 0.01 + 0.001;
        x[i] = walk;
        y[i] = 1.5 + 1.7 * walk + std::sin(static_cast<double>(i) * 0.071) * 0.02;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto result = pairs::engle_granger(y, x);
    const auto finish = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    const double adf_t_stat = pairs::engle_granger_adf_t_stat(result);
    std::cout << "intercept=" << result.intercept << " beta=" << result.beta << " adf_t=" << adf_t_stat << "\n";
    std::cout << "observations=" << n << " elapsed_ms=" << elapsed << "\n";
}
