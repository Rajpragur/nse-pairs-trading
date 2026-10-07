#include "pairs_math.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    std::vector<double> x, y;
    for (int i = 0; i < 500; ++i) {
        x.push_back(std::sin(i * 0.03) * 10.0 + i * 0.01);
        y.push_back(2.0 + 1.5 * x.back() + std::sin(i * 0.17) * 0.03);
    }
    const auto result = pairs::engle_granger(y, x);
    const double cxx_adf = pairs::engle_granger_adf_t_stat(result);
    if (std::abs(result.beta - 1.5) >= 0.01 || !std::isfinite(cxx_adf)) {
        std::cerr << "beta=" << result.beta << " adf_t=" << cxx_adf << "\n";
        return 1;
    }
    const double cpp_half_life = pairs::half_life(result.residual);
    if (!std::isfinite(cpp_half_life) || cpp_half_life <= 0.0) return 2;

    std::vector<double> fixture_x, fixture_y;
    for (int i = 0; i < 80; ++i) {
        const double walk = 0.15 * i + std::sin(i * 0.19) * 1.1;
        fixture_x.push_back(50.0 + walk);
        fixture_y.push_back(3.25 + 1.27 * walk + std::sin(i * 0.73) * 0.12);
    }
    const auto parity = pairs::engle_granger(fixture_y, fixture_x);
    if (std::abs(parity.intercept - (-60.1894031564033)) > 1e-9 ||
        std::abs(parity.beta - 1.2689508618001466) > 1e-10 ||
        std::abs(pairs::engle_granger_adf_t_stat(parity) - (-3.2348351594757765)) > 1e-7 ||
        std::abs(pairs::half_life(parity.residual) - 2.768876852350559) > 1e-8) {
        std::cerr.precision(17);
        std::cerr << "Python/C++ parity mismatch intercept=" << parity.intercept
                  << " beta=" << parity.beta
                  << " adf=" << pairs::engle_granger_adf_t_stat(parity)
                  << " half_life=" << pairs::half_life(parity.residual) << "\n";
        return 3;
    }
    bool rejected = false;
    fixture_x[10] = std::numeric_limits<double>::quiet_NaN();
    try { (void)pairs::engle_granger(fixture_y, fixture_x); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 4;
    return 0;
}
