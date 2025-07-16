import numpy as np
import matplotlib.pyplot as plt

# Load RX IQ samples from CSV
data = np.loadtxt("rx_iq.csv", delimiter=",")
i = data[:, 0]
q = data[:, 1]
rx = i + 1j * q

# (Optional) If you know the transmitted Barker sequence
def generate_barker_bpsk():
    barker_bits = [1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1]
    tx = np.array(barker_bits, dtype=np.float64)
    return tx + 0j  # make it complex

# Estimate SNR (optional)
def estimate_snr(tx, rx):
    signal_power = np.mean(np.abs(tx)**2)
    noise_power = np.mean(np.abs(rx - tx)**2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

# Plot IQ constellation
plt.figure(figsize=(6, 6))
# plt.plot(i, q, 'o', markersize=6, label="Received")
plt.plot(np.real(rx), label="Received")
plt.plot(np.real(generate_barker_bpsk()), label="Transmitted (BPSK)")
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)
plt.title("IQ Constellation (BPSK)")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

# Optional SNR estimation in Python
try:
    tx = generate_barker_bpsk()
    snr_est = estimate_snr(tx, rx)
    print(f"Estimated SNR (Python): {snr_est:.2f} dB")
except Exception as e:
    print(f"SNR estimation failed: {e}")
