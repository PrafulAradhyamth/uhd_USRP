#pragma once
#include <vector>
#include <complex>

inline std::vector<std::complex<float>> generate_barker_bpsk() {
    std::vector<int> b13 = {+1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,-1,+1};
    std::vector<std::complex<float>> bpsk;
    for (int s : b13) {
        bpsk.emplace_back((float)s, 0.0f);
    }
    return bpsk;
}
