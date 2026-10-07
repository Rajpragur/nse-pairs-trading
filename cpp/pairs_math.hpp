#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pairs {

struct EngleGrangerResult {
    double intercept{};
    double beta{};
    std::size_t observations{};
    std::vector<double> residual;
};

inline EngleGrangerResult engle_granger(const std::vector<double>& y, const std::vector<double>& x) {
    if (y.size() != x.size() || y.size() < 30) {
        throw std::invalid_argument("equal vectors with at least 30 observations required");
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
            throw std::invalid_argument("observations must be finite");
    }
    const double n = static_cast<double>(x.size());
    double sx = 0.0, sy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { sx += x[i]; sy += y[i]; }
    const double mx = sx / n, my = sy / n;
    double xx = 0.0, xy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { const double dx = x[i] - mx; xx += dx * dx; xy += dx * (y[i] - my); }
    const double beta = xy / xx;
    if (xx <= 0.0 || !std::isfinite(beta)) throw std::invalid_argument("x must have positive finite variance");
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
    const double adf_t_stat = gamma / std::sqrt(covariance_beta);
    if (!std::isfinite(adf_t_stat)) throw std::invalid_argument("ADF statistic is not finite");
    return {intercept, beta, x.size(), std::move(residual)};
}

inline double engle_granger_adf_t_stat(const EngleGrangerResult& result) {
    const auto& residual = result.residual;
    const double observations = static_cast<double>(residual.size() - 1);
    double sum_lag = 0.0, sum_delta = 0.0;
    for (std::size_t i = 1; i < residual.size(); ++i) {
        sum_lag += residual[i - 1];
        sum_delta += residual[i] - residual[i - 1];
    }
    const double mean_lag = sum_lag / observations, mean_delta = sum_delta / observations;
    double denom = 0.0, numer = 0.0;
    for (std::size_t i = 1; i < residual.size(); ++i) {
        const double lag = residual[i - 1] - mean_lag;
        const double delta = residual[i] - residual[i - 1] - mean_delta;
        denom += lag * lag;
        numer += lag * delta;
    }
    const double gamma = numer / denom;
    const double adf_intercept = mean_delta - gamma * mean_lag;
    double sse = 0.0;
    for (std::size_t i = 1; i < residual.size(); ++i) {
        const double error = residual[i] - residual[i - 1] - adf_intercept - gamma * residual[i - 1];
        sse += error * error;
    }
    const double variance = sse / (observations - 2.0);
    const double covariance_beta = variance * (1.0 / denom + mean_lag * mean_lag / (observations * denom));
    return gamma / std::sqrt(covariance_beta);
}

inline double half_life(const std::vector<double>& residual) {
    if (residual.size() < 4) throw std::invalid_argument("at least four residuals required");
    double sx = 0.0, sy = 0.0;
    const double n = static_cast<double>(residual.size() - 1);
    for (std::size_t i = 1; i < residual.size(); ++i) {
        sx += residual[i - 1];
        sy += residual[i] - residual[i - 1];
    }
    const double mx = sx / n, my = sy / n;
    double xx = 0.0, xy = 0.0;
    for (std::size_t i = 1; i < residual.size(); ++i) {
        const double dx = residual[i - 1] - mx;
        xx += dx * dx;
        xy += dx * (residual[i] - residual[i - 1] - my);
    }
    const double gamma = xy / xx;
    if (!std::isfinite(gamma) || gamma >= 0.0) return INFINITY;
    return -std::log(2.0) / gamma;
}

}  // namespace pairs
