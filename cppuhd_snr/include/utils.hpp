#pragma once
#include <vector>
#include <complex>
#include <cmath>

inline float compute_power(const std::vector<std::complex<float>>& samples) {
    float power = 0.0f;
    for (auto& s : samples) power += std::norm(s);
    return power / samples.size();
}

inline float compute_snr_db(float signal_power, float noise_power) {
    return 10.0f * std::log10(signal_power / noise_power);
}
