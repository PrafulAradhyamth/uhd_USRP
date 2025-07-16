#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <fstream>
#include <cmath>

using Complex = std::complex<double>;

class BPSKModulator {
public:
    std::vector<Complex> generateBarker() {
        std::vector<int> barker = {1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1};
        std::vector<Complex> signal;
        for (int bit : barker) {
            signal.emplace_back(static_cast<double>(bit), 0.0);  // I component, Q = 0
        }
        return signal;
    }
};

class AWGNChannel {
public:
    AWGNChannel(double noiseStd) : dist(0.0, noiseStd) {}

    std::vector<Complex> transmit(const std::vector<Complex>& tx) {
        std::vector<Complex> rx;
        for (const auto& s : tx) {
            Complex noise(dist(rng), dist(rng));
            rx.push_back(s + noise);
        }
        return rx;
    }

private:
    std::default_random_engine rng{std::random_device{}()};
    std::normal_distribution<double> dist;
};

class SNRCalculator {
public:
    static double estimateSNR(const std::vector<Complex>& tx, const std::vector<Complex>& rx) {
        if (tx.size() != rx.size()) {
            std::cerr << "TX and RX size mismatch!\n";
            return -1;
        }

        double signalPower = 0.0;
        double noisePower = 0.0;

        for (size_t i = 0; i < tx.size(); ++i) {
            signalPower += std::norm(tx[i]);
            Complex noise = rx[i] - tx[i];
            noisePower += std::norm(noise);
        }

        signalPower /= tx.size();
        noisePower /= tx.size();

        double snr = 10 * std::log10(signalPower / noisePower);
        return snr;
    }
};

class CSVWriter {
public:
    static void saveIQ(const std::vector<Complex>& iq, const std::string& filename) {
        std::ofstream file(filename);
        for (const auto& s : iq) {
            file << s.real() << "," << s.imag() << "\n";
        }
        file.close();
    }
};

int main() {
    double noiseStdDev = 0.3;  // Change this to simulate different noise levels

    // 1. Generate signal
    BPSKModulator mod;
    auto tx = mod.generateBarker();

    // 2. Simulate channel
    AWGNChannel channel(noiseStdDev);
    auto rx = channel.transmit(tx);

    // 3. Estimate SNR
    double snr = SNRCalculator::estimateSNR(tx, rx);
    std::cout << "Estimated SNR: " << snr << " dB\n";

    // 4. Save received samples
    CSVWriter::saveIQ(rx, "rx_iq.csv");

    std::cout << "IQ samples saved to 'rx_iq.csv'\n";
    return 0;
}
