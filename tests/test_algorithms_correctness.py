import logging
import random
import sys
import unittest
import tqdm

import numpy as np

from app.algorithms import compute_stm
from app.data import generate_synthetic_data


class AlgorithmsTestCase(unittest.TestCase):
    """Test algorithms correctness with synthetic data."""
    def setUp(self) -> None:
        logger = logging.getLogger()
        logger.level = logging.DEBUG
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    def test_stm(self):
        # this should converge quickly.

        # arguments for generating the data

        M = 70  # data dimension
        UNIFORM = False  # True = model has uniform residual variances; False = models has arbitrary residual variances
        OBJECTIVE = False  # False = independent objective; True = aligned objective
        scan_N = (np.array([0.25, 0.5, 1, 2]) * M).astype(np.int)  # the sizes of datasets
        default_K = np.array([5, 9, 15, 20])  # the corresponding K to use
        default_lambda = np.array([2.6, 1.4, 1, .9]) * M  # the corresponding lambda to use
        default_gamma = np.array([8, 6, 4, 2]) * M
        TRIAL = 1  # number of simulation trials
        mu_f = -1
        sigma_f = 2  # magnitude factor loadings
        sigma_p = 4
        sigma_r = 0.6  # magnitude of variation among residual variances

        ## log likelihood record keeper
        STM_llh = np.zeros((TRIAL, len(scan_N)))
        EM_llh = np.zeros((TRIAL, len(scan_N)))
        wrong_llh = np.zeros((TRIAL, len(scan_N)))

        STM_obj = np.zeros((TRIAL, len(scan_N)))
        EM_obj = np.zeros((TRIAL, len(scan_N)))
        Oracle_obj = np.zeros((TRIAL, len(scan_N)))

        ## set random seed for data generation; can be safely ignored
        rand_seed = random.randint(0, 10000)
        rand_seed = 42
        print("Seed: %d" % rand_seed)
        logging.info("Test")
        np.random.seed(rand_seed)

        # begin of simulation
        for trial in tqdm.tqdm(range(TRIAL)):
            X, Sigma_s, c = generate_synthetic_data(M, scan_N[-1], UNIFORM, OBJECTIVE, mu_f, sigma_f, sigma_p,
                                                    sigma_r)  # X=data set; Sigma_s = true covariance matrix

            for index_N, (N, train_K, train_lambda, train_gamma) in enumerate(
                    zip(scan_N, default_K, default_lambda, default_gamma)):
                # compute sample covaraince matrix
                Sigma_SAM = np.zeros((M, M))
                for n in range(N):
                    Sigma_SAM += np.outer(X[:, n], X[:, n])
                Sigma_SAM /= N

                if not UNIFORM:
                    # STM
                    Sigma_STM = compute_stm(Sigma_SAM, train_lambda, N)
                    U_STM = 0.5 * np.linalg.lstsq(Sigma_STM, c, rcond=None)[0]
                    STM_llh[trial, index_N] = -0.5 * (
                                M * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma_STM)) + np.trace(
                            np.linalg.lstsq(Sigma_STM, Sigma_s, rcond=None)[0]))
                    STM_obj[trial, index_N] = c.dot(U_STM) - U_STM.dot(Sigma_s).dot(U_STM)


if __name__ == '__main__':
    unittest.main()
