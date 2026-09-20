#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace pairs {

struct EngleGrangerResult {
    double intercept{};
    double beta{};
    double adf_t_stat{};
};

inline EngleGrangerResult engle_granger(const std::vector<double>& y, const std::vector<double>& x) {
    if (y.size() != x.size() || y.size() < 30) {
        throw std::invalid_argument("equal vectors with at least 30 observations required");
    }
    const double n = static_cast<double>(x.size());
    double sx = 0.0, sy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { sx += x[i]; sy += y[i]; }
    const double mx = sx / n, my = sy / n;
    double xx = 0.0, xy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { const double dx = x[i] - mx; xx += dx * dx; xy += dx * (y[i] - my); }
    const double beta = xy / xx;
    const double intercept = my - beta * mx;
    std::vector<double> residual(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) residual[i] = y[i] - intercept - beta * x[i];

    const double observations = static_cast<double>(y.size() - 1);
    double sum_lag = 0.0, sum_delta = 0.0;
    for (std::size_t i = 1; i < residual.size(); ++i) { sum_lag += residual[i - 1]; sum_delta += residual[i] - residual[i - 1]; }
    const double mean_lag = sum_lag / observations, mean_delta = sum_delta / observations;
    double denom = 0.0, numer = 0.0;
    for (std::size_t i = 1; i < residual.size(); ++i) { const double lag = residual[i - 1] - mean_lag; const double delta = residual[i] - residual[i - 1] - mean_delta; denom += lag * lag; numer += lag * delta; }
    const double gamma = numer / denom;
    const double adf_intercept = mean_delta - gamma * mean_lag;
    double sse = 0.0;
    for (std::size_t i = 1; i < residual.size(); ++i) { const double fitted = adf_intercept + gamma * residual[i - 1]; const double error = residual[i] - residual[i - 1] - fitted; sse += error * error; }
    const double variance = sse / (observations - 2.0);
    const double covariance_beta = variance * (1.0 / denom + mean_lag * mean_lag / (observations * denom));
    return {intercept, beta, gamma / std::sqrt(covariance_beta)};
}

}  // namespace pairs
