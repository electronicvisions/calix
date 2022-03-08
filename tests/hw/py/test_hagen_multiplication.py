"""
Test the multiply-accumulate functionality.
"""

from typing import Optional
import unittest

import numpy as np
import matplotlib.pyplot as plt

from dlens_vx_v3 import halco, logger

from calix.hagen import multiplication
import calix.hagen

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class MultiplyAccumulateTest(ConnectionSetup):
    """
    Test different MAC operations and assert the results change
    in the expected way.

    :cvar log: Logger used to log messages.
    :cvar calib_result: Result of calibration.
    :cvar vectors: Vectors to send during testing of multiplication.
    :cvar matrices: Matrices to use during testing of multiplication.
    """

    log = logger.get("calix.tests.hw.py.MultiplyAccumulateTest")
    calib_result: Optional[calix.hagen.HagenSyninCalibrationResult] = None
    vectors: np.ndarray
    matrices: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # initialize some vectors at different weights
        vector_length = halco.SynapseRowOnSynram.size
        cls.vectors = np.array([                     # expected:
            np.zeros(vector_length, dtype=int),      # 0: zero amps
            np.ones(vector_length, dtype=int) * 31,  # 1: maximum amps
            np.ones(vector_length, dtype=int) * 15,  # 2: still near max
            np.ones(vector_length, dtype=int) * 7,   # 3: medium amps
            np.ones(vector_length, dtype=int) * 3,   # 4: lower amps
            np.ones(vector_length, dtype=int) * 1,   # 5: close to zero amps
            np.arange(vector_length) % 16,               # 6: like 3
            np.arange(vector_length) % 8,                # 7: like 4
            np.arange(vector_length - 1, -1, -1) % 16])  # 8: like 3

        # initialize matrices with linearly increasing and decreasing weights
        cls.matrices = np.zeros((2, halco.SynapseOnSynapseRow.size // 2,
                                 vector_length), dtype=int)
        cls.matrices[0, :127] = np.repeat(
            np.arange(-63, 64, 1)[:, np.newaxis], vector_length, axis=1)
        cls.matrices[1, :127] = np.repeat(
            np.arange(-63, 64, 1)[::-1, np.newaxis], vector_length, axis=1)

    # pylint: disable=too-many-locals
    def check_results(self, results: np.ndarray) -> None:
        """
        Asserts the given results are compatible with expectations.

        :param results: Results obtained by multiplication on chip,
            as generated in the test_01_multiplication() function.
        """

        mean_vector_entries = np.mean(self.vectors, axis=1)

        for synram_name, synram_results in zip(["top", "bottom"], results):
            for matrix, matrix_results in zip(self.matrices, synram_results):
                mean_results = np.mean(matrix_results, axis=0)
                expected_slope = np.mean(np.diff(matrix[:127, 0]))  # 1 or -1

                for vector_id, _ in enumerate(mean_results):
                    slope = np.mean(np.diff(
                        mean_results[vector_id, :127]))
                    noise = np.mean(
                        np.std(matrix_results[:, vector_id], axis=0))

                    self.log.DEBUG(
                        f"Synram {synram_name}, matrix "
                        + ('inverted' if expected_slope < 0
                           else 'not inverted')
                        + f", results for vector {vector_id} "
                        + f"(mean entry: {mean_vector_entries[vector_id]}):"
                        + f"result slope: {slope}, "
                        + f"result noise: {noise}")

                    self.assertLess(
                        noise, 5,
                        f"Noise of synram {synram_name} vector "
                        + f"{vector_id} is too large.")

                    # check amplitudes only if entries are at least 3 on
                    # average, amplitudes may still be zero otherwise.
                    minimum_reliable_amplitude = 3
                    if mean_vector_entries[vector_id] >= \
                            minimum_reliable_amplitude:
                        self.assertGreater(
                            slope * expected_slope, 0.02,
                            f"Slope of synram {synram_name} vector "
                            + f"{vector_id} is too small.")

                        # assert amplitudes are correct with respect to
                        # other vectors with different entries:
                        # If difference between mean vector entry is
                        # significant, we assert the amplitudes differ too.
                        significantly_less = np.all([
                            minimum_reliable_amplitude <= mean_vector_entries,
                            mean_vector_entries
                            < 0.66 * mean_vector_entries[vector_id]], axis=0)
                        significantly_more = np.all([
                            minimum_reliable_amplitude <= mean_vector_entries,
                            mean_vector_entries
                            > 1.5 * mean_vector_entries[vector_id]], axis=0)
                        success_less = \
                            np.abs(mean_results[significantly_less]) \
                            < np.abs(mean_results[vector_id])
                        success_more = \
                            np.abs(mean_results[significantly_more]) \
                            > np.abs(mean_results[vector_id])

                        # assert 60% of columns behave in the expected way:
                        # criterium is so weak since near the middle of the
                        # matrix, we have low weights and hence a bad signal
                        # to noise ratio.
                        np.testing.assert_array_less(
                            0.6 * halco.SynapseOnSynapseRow.size // 2,
                            np.sum(success_less, axis=-1),
                            "Result amplitudes do not follow ordering of "
                            + "vector entries (mismatch to lesser vectors).")
                        np.testing.assert_array_less(
                            0.6 * halco.SynapseOnSynapseRow.size // 2,
                            np.sum(success_more, axis=-1),
                            "Result amplitudes do not follow ordering of "
                            + "vector entries (mismatch to greater vectors).")

    # pylint: disable=too-many-locals
    def plot_results(self, results: np.ndarray,
                     filename: str = "multiplication_synin.png") -> None:
        """
        Plot the obtained results.

        :param results: Array of results, as obtained during
            test_01_multiplication().
        :param filename: Filename to save the plot to.
        """

        # labels for each vector:
        labels = [
            "no event",
            "31",
            "15",
            "7",
            "3",
            "1",
            "% 16",
            "% 8",
            "% 16 reversed"]

        fig, axes = plt.subplots(
            nrows=halco.SynramOnDLS.size, ncols=len(self.matrices),
            figsize=(9, 6), tight_layout=True)

        for synram_id, synram_results in enumerate(results):
            for matrix_id, matrix_results in enumerate(synram_results):
                ax = axes[synram_id][matrix_id]
                means = np.mean(matrix_results, axis=0)
                noise = np.std(matrix_results, axis=0)

                for result_id, (result, noise) in enumerate(zip(means, noise)):
                    ax.errorbar(np.arange(len(result)), result,
                                fmt='.-', yerr=noise,
                                linewidth=0.2, markersize=1, elinewidth=0.3,
                                label=labels[result_id])

                expected_slope = np.mean(np.diff(
                    self.matrices[matrix_id][:127, 0]))  # 1 or -1
                name = "top " if synram_id == 0 else "bottom "
                name += "inverted" if expected_slope < 0 else "not inverted"
                ax.set_title(name)

        axes[0, 0].legend()
        fig.supxlabel('Matrix column')
        fig.supylabel('Synin amplitudes [LSB]')

        fig.savefig(filename, dpi=300)
        plt.close()

    def test_00_calibrate(self):
        """
        Loads calibration for integration on synaptic input lines.
        """

        self.__class__.calib_result = self.apply_calibration("hagen_synin")

    def test_01_multiplication(self, n_runs: int = 30):
        """
        Send different vectors with the synapse matrix configured at
        weights -63...0...63, constant in columns. Repeat the experiment
        n_runs times and with inverted weights.
        Assert the results and deviations are roughly as expected.

        :param n_runs: Number of identical runs to execute.
        """

        # results shape: synram, matrix_id, run, vector_id, column
        results = np.empty(
            (halco.SynramOnDLS.size, 2, n_runs,
             len(self.vectors), halco.SynapseOnSynapseRow.size // 2))

        for synram in halco.iter_all(halco.SynramOnDLS):
            mpl = multiplication.Multiplication(synram=synram)
            mpl.preconfigure(self.connection)
            for matrix_id, matrix in enumerate(self.matrices):
                for run in range(n_runs):
                    results[int(synram.toEnum()), matrix_id, run] = \
                        mpl.multiply(self.connection, self.vectors, matrix)

        self.plot_results(results)
        self.check_results(results)

    def test_02_auto_reshape(self):
        """
        Test multiplication with longer matrices than a synram.

        We don't assert results here, just that the code for splitting
        the long matrix doesn't fail.
        """

        # test a long matrix that will be split
        vectors = np.ones((10, 1000), dtype=int)
        matrix = np.ones((halco.SynapseOnSynapseRow.size, 1000), dtype=int)

        mpl = multiplication.Multiplication(signed_mode=False)
        mpl.preconfigure(self.connection)

        mpl.auto_multiply(self.connection, vectors, matrix)
        mpl.auto_multiply(self.connection, vectors, matrix, n_row_repeats=2)

        # test a too wide matrix (splitting is not supported here)
        with self.assertRaises(ValueError):
            matrix = np.ones(
                (halco.SynapseOnSynapseRow.size + 1, 1000), dtype=int)
            mpl.auto_multiply(self.connection, vectors, matrix)


if __name__ == "__main__":
    unittest.main()
