import numpy as np
import pytest

from ofdm_sim.channel import awgn_channel, example_channel, frequency_response, multipath
from ofdm_sim.dft import dft, fft_radix2, idft
from ofdm_sim.link import qam_ber_awgn, run_link
from ofdm_sim.ofdm import (OFDMParams, ofdm_demodulate, ofdm_modulate, qam_demodulate, qam_modulate)
from ofdm_sim.orthogonality import gram_matrix, oscillator_bank_synthesis, subcarriers
from ofdm_sim.sampling import alias_frequency, quantize, satisfies_nyquist, sqnr_db, sqnr_theory_db, spectral_resolution

RNG = np.random.default_rng(0)


# --- Adquisición de señales ---------------------------------------------------
def test_nyquist_and_alias():
    assert satisfies_nyquist(200, 40) and not satisfies_nyquist(60, 40)
    assert alias_frequency(40, 60) == pytest.approx(20)
    assert alias_frequency(40, 200) == pytest.approx(40)


def test_spectral_resolution():
    assert spectral_resolution(1000, 200) == pytest.approx(5.0)


def test_quantization_sqnr_close_to_theory():
    x = np.sin(2 * np.pi * 137 * np.arange(48000) / 48000)
    assert sqnr_db(x, quantize(x, 8)) == pytest.approx(sqnr_theory_db(8), abs=1.0)


# --- DFT / FFT ------------------------------------------------------------------
@pytest.mark.parametrize("n", [8, 64, 100])
def test_dft_matches_numpy(n):
    x = RNG.standard_normal(n) + 1j * RNG.standard_normal(n)
    assert np.allclose(dft(x), np.fft.fft(x))
    assert np.allclose(idft(dft(x)), x)


def test_fft_radix2_matches_numpy_and_rejects_non_power_of_two():
    x = RNG.standard_normal(128) + 1j * RNG.standard_normal(128)
    assert np.allclose(fft_radix2(x), np.fft.fft(x))
    with pytest.raises(ValueError):
        fft_radix2(np.ones(12))


# --- Ortogonalidad ----------------------------------------------------------------
def test_orthogonality_only_when_spacing_is_one_over_T():
    _, car = subcarriers(8, 1.0, 1.0, 1024)
    assert np.allclose(gram_matrix(car), np.eye(8), atol=1e-12)
    _, car = subcarriers(8, 0.7, 1.0, 1024)
    assert np.max(np.abs(gram_matrix(car) - np.eye(8))) > 0.1


def test_ifft_equals_oscillator_bank():
    X = RNG.standard_normal(32) + 1j * RNG.standard_normal(32)
    assert np.allclose(np.fft.ifft(X), oscillator_bank_synthesis(X))


# --- QAM -------------------------------------------------------------------------
@pytest.mark.parametrize("m", [4, 16, 64, 256])
def test_qam_roundtrip_and_unit_power(m):
    bits = RNG.integers(0, 2, 24 * 1000)
    s = qam_modulate(bits, m)
    assert np.mean(np.abs(s) ** 2) == pytest.approx(1.0, rel=0.02)
    assert np.array_equal(qam_demodulate(s, m), bits)


def test_gray_mapping_neighbours_differ_by_one_bit():
    m = 16
    pts = qam_modulate(np.array([int(b) for i in range(m) for b in format(i, "04b")]), m)
    bits = np.array([[int(b) for b in format(i, "04b")] for i in range(m)])
    dmin = np.min(np.abs(pts[:, None] - pts[None, :]) + np.eye(m) * 10)
    for i in range(m):
        for j in range(m):
            if i != j and abs(pts[i] - pts[j]) < dmin * 1.01:
                assert np.sum(bits[i] != bits[j]) == 1


def test_params_validation():
    with pytest.raises(ValueError):
        OFDMParams(mod_order=8)
    p = OFDMParams(n_fft=64, bandwidth=1e6)
    assert p.subcarrier_spacing * p.useful_duration == pytest.approx(1.0)   # Δf = 1/T


# --- Cadena OFDM ---------------------------------------------------------------------
def test_ofdm_roundtrip_identity():
    p = OFDMParams(64, 16, 16)
    grid = qam_modulate(RNG.integers(0, 2, 5 * p.bits_per_ofdm_symbol), 16).reshape(5, 64)
    assert np.allclose(ofdm_demodulate(ofdm_modulate(grid, p), p), grid)


def test_cyclic_prefix_is_copy_of_tail():
    p = OFDMParams(64, 16, 4)
    grid = qam_modulate(RNG.integers(0, 2, p.bits_per_ofdm_symbol), 4).reshape(1, 64)
    x = ofdm_modulate(grid, p)
    assert np.allclose(x[:16], x[-16:])


def test_cp_makes_multipath_a_scalar_channel_per_subcarrier():
    p = OFDMParams(64, 16, 4)
    h = example_channel()
    grid = qam_modulate(RNG.integers(0, 2, 4 * p.bits_per_ofdm_symbol), 4).reshape(4, 64)
    rx = multipath(ofdm_modulate(grid, p), h)
    y = ofdm_demodulate(rx, p)
    assert np.allclose(y[1:], grid[1:] * frequency_response(h, 64), atol=1e-10)


def test_noiseless_link_is_error_free_with_cp_but_not_without():
    p = OFDMParams(64, 16, 16)
    h = example_channel()
    assert run_link(p, h, None, np.random.default_rng(1), 50, use_cp=True).n_errors == 0
    assert run_link(p, h, None, np.random.default_rng(1), 50, use_cp=False).n_errors > 0


def test_simulated_awgn_ber_matches_theory():
    p = OFDMParams(64, 16, 16)
    res = run_link(p, awgn_channel(), 12.0, np.random.default_rng(2), n_ofdm=400)
    assert res.ber == pytest.approx(float(qam_ber_awgn(16, 12.0)), rel=0.15)
