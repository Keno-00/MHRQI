"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images            ║
║  Image Quality Assessment: OMQDI, Wavelet Energy, Speckle Analysis          ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pywt
from scipy.ndimage import convolve


def getCDF97(weight=1):
    """
    Return the CDF 9/7 biorthogonal wavelet used by OMQDI.

    Args:
        weight: Optional scalar weight applied to all filter coefficients.

    Returns:
        pywt.Wavelet object configured with CDF 9/7 filter banks.
    """
    analysis_LP = np.array([0, 0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,
                            0.602949018236, 0.266864118443, -0.078223266529, -0.016864118443, 0.026748757411])
    analysis_LP *= weight

    analysis_HP = np.array([0, 0.091271763114, -0.057543526229, -0.591271763114, 1.11508705,
                           -0.591271763114, -0.057543526229, 0.091271763114, 0, 0])
    analysis_HP *= weight

    synthesis_LP = np.array([0, -0.091271763114, -0.057543526229, 0.591271763114, 1.11508705,
                             0.591271763114, -0.057543526229, -0.091271763114, 0, 0])
    synthesis_LP *= weight

    synthesis_HP = np.array([0, 0.026748757411, 0.016864118443, -0.078223266529, -0.266864118443,
                             0.602949018236, -0.266864118443, -0.078223266529, 0.016864118443, 0.026748757411])
    synthesis_HP *= weight

    return pywt.Wavelet('CDF97', [analysis_LP, analysis_HP, synthesis_LP, synthesis_HP])


def sbEn(coeffs: np.array) -> float:
    """
    Compute the energy of a wavelet sub-band.

    Args:
        coeffs: 2D array of wavelet coefficients for the sub-band.

    Returns:
        Sub-band energy as a scalar.
    """
    I, J = coeffs.shape
    return np.log(1 + np.sum(np.square(coeffs)) / (I * J))


def En(lvlCoeffs: tuple, alpha=0.8) -> float:
    """
    Compute the weighted energy for one level of wavelet decomposition.

    Args:
        lvlCoeffs: Tuple of (LHn, HLn, HHn) coefficient arrays.
        alpha: Weight for the HH sub-band relative to LH and HL.
               Recommended value 0.8 per ISBN 0139353224.

    Returns:
        Weighted energy scalar.
    """
    LHn, HLn, HHn = lvlCoeffs
    return (1 - alpha) * (sbEn(LHn) + sbEn(HLn)) / 2 + alpha * sbEn(HHn)


def S(decompCoeffs: list, alpha=0.8) -> float:
    """
    Compute the cumulative multi-level wavelet energy.

    Args:
        decompCoeffs: List of (LHn, HLn, HHn) tuples for each decomposition level,
                      ordered from coarsest to finest.
        alpha: Weight for the HH sub-band. See En().

    Returns:
        Cumulative energy scalar.
    """
    energy = 0
    for i, lvlCoeffs in enumerate(decompCoeffs):
        n = i + 1
        energy += 2**(3 - n) * En(lvlCoeffs, alpha)
    return energy


def local_mean(img: np.array, window=3, pad_mode='reflect') -> np.array:
    """
    Compute the local mean of pixel intensities using a uniform kernel.

    Args:
        img: Input image of shape (M, N).
        window: Kernel size.
        pad_mode: Padding mode for convolution.

    Returns:
        Local mean array of shape (M, N).
    """
    return convolve(img, np.full((window, window), 1 / window**2), mode=pad_mode)


def local_variance(img: np.array) -> np.array:
    """
    Compute the local variance of an image.

    Args:
        img: Input image of shape (M, N).

    Returns:
        Local variance array of shape (M, N).
    """
    mu_sq = np.square(local_mean(img))
    return local_mean(np.square(img)) - mu_sq


def noise_power(img: np.array) -> float:
    """
    Estimate the noise power (σ̂) of an image as the mean local variance.

    Args:
        img: Input image of shape (M, N).

    Returns:
        Estimated noise power scalar.
    """
    return np.mean(local_variance(img))


def OMQDI(X: np.array, Y: np.array, C=1e-10) -> tuple:
    """
    Compute the Objective Measure of Quality of Denoised Images.

    Reference: DOI 10.1016/j.bspc.2021.102962

    Args:
        X: Noisy input image of shape (M, N).
        Y: Denoised output image of shape (M, N).
        C: Small constant to avoid division by zero.

    Returns:
        Tuple (OMQDI, Q1, Q2):
            - OMQDI: Combined metric Q1 + Q2, ideal value 2, range [1, 2].
            - Q1: Edge-Preservation Factor, ideal value 1, range [0, 1].
            - Q2: Noise-Suppression Factor, ideal value 1, range [0, 1].
    """
    CDF97 = getCDF97()

    coeffX = pywt.wavedec2(X, CDF97, level=3)
    coeffY = pywt.wavedec2(Y, CDF97, level=3)

    SX = S(coeffX[1:])
    SY = S(coeffY[1:])

    npX = noise_power(X)
    npY = noise_power(Y)

    Q1 = (2 * SX * SY + C) / (SX**2 + SY**2 + C)
    Q2 = (npX - npY)**2 / (npX**2 + npY**2 + C)
    return (Q1 + Q2, Q1, Q2)
