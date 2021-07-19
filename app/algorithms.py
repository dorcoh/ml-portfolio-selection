import logging
from copy import copy
from enum import Enum

import numpy as np

from app.algorithms_utils import ordered_eig, trans


class AlgorithmType(Enum):
    URM = 1
    UTM = 2
    STM = 3
    EM = 4


def compute_urm(Sigma_SAM, K):
    logging.info("compute_urm")
    M = Sigma_SAM.shape[0]
    eigval, eigvec = ordered_eig(Sigma_SAM)
    sigma_URM = eigval[K:].mean()
    F_URM = eigvec[:, :K].dot(np.diag(eigval[:K] - sigma_URM)).dot(eigvec[:, :K].T)
    R_URM = sigma_URM * np.eye(M)
    Sigma_URM = F_URM + R_URM
    return Sigma_URM


def compute_utm(Sigma_SAM, lamb, N):
    logging.info("compute_utm")
    lambda_p = 2 * lamb / N
    Sigma_UTM, _, _, _ = trans(Sigma_SAM, lambda_p)
    return Sigma_UTM


def compute_stm(Sigma_SAM, lamb, N, max_iter=None, epsilon=0.001, solve_t_max_iter=None, solve_t_epsilon=0.0001):
    logging.info("compute_stm")
    M = Sigma_SAM.shape[0]
    lambda_p = lamb / (N / 2)
    init_R = np.ones((M, 1))
    init_Vhalf = init_R ** -0.5
    T = init_Vhalf / np.exp(np.mean(np.log(init_Vhalf)))

    iter = 0
    while True:
        T_Sigma = np.zeros((M, M))
        for d1 in range(M):
            for d2 in range(d1, M):
                T_Sigma[d1, d2] = T[d1] * Sigma_SAM[d1, d2] * T[d2]
                T_Sigma[d2, d1] = T_Sigma[d1, d2]

        T_eigval, T_eigvec = ordered_eig(T_Sigma)

        # UTM
        sum_eigval = np.trace(T_Sigma)
        i = 1
        for i in range(1, M+1):
            beta = (i * lambda_p + sum_eigval - np.sum(T_eigval[:i])) / (M - (i-1))
            if beta > T_eigval[i-1] - lambda_p:
                break

        K = i - 1
        logging.debug("K: {}".format(K))
        chosen_beta = (K * lambda_p + sum_eigval - np.sum(T_eigval[:K])) / (M - (K-1))
        T_eigval = np.array(T_eigval[:K] - lambda_p)

        # evaluate inverse of hat{Sigma_SAM}; i.e., vI-G
        inv_hat_Sigma = np.eye(M) / chosen_beta
        for d in range(1, K+1):
            inv_hat_Sigma = inv_hat_Sigma - ((1 / chosen_beta) - (1 / T_eigval[d-1])) * np.outer( T_eigvec[:, d-1], T_eigvec[:, d-1].T)

        # update T
        new_T = solve_T(T, inv_hat_Sigma * Sigma_SAM, True, epsilon=solve_t_epsilon, max_iter=solve_t_max_iter)
        _curr = np.max(np.abs(np.divide(new_T, T) - 1))
        logging.debug("compute_stm (first) curr: {}".format(_curr))
        if _curr < epsilon or (max_iter is not None and iter > max_iter):
            logging.debug("compute_stm curr: {}".format(_curr))
            break
        else:
            T = new_T
            iter += 1

    _T_diag = np.diag(np.squeeze(T ** -1))
    F_STM = _T_diag @ T_eigvec[:, :K] @ np.diag(T_eigval - chosen_beta).T @ T_eigvec[:, :K].T @ _T_diag
    R_STM = _T_diag * chosen_beta @ np.eye(M) @ _T_diag
    Sigma_STM = F_STM + R_STM

    return Sigma_STM


def solve_T(init_T, A, normalize, epsilon=0.0001, max_iter=50):
    scale_a = 0.2
    scale_b = 0.5
    T = init_T
    value = T.T @ A @ T - 2 * np.sum(np.log(T))
    i = 0
    while True:
        gradient = 2 * A @ T - 2 * (T ** -1)
        hessian = 2 * A + 2 * np.diag((T ** -2).squeeze())
        dt, _, _, _ = np.linalg.lstsq(-hessian, gradient)
        _curr = np.divide(-gradient.T @ dt, 2)
        logging.debug("solve_T curr (first) grad: {}".format(_curr))
        if _curr <= epsilon or (max_iter is not None and i > max_iter):
            logging.debug("solve_T curr grad: {}".format(_curr))
            break
        # backtrack
        r = 1
        next_t = T + r * dt
        if np.min(next_t) > 0:
            next_value = next_t.T @ A @ next_t - 2 * np.sum(np.log(next_t))
        else:
            next_value = np.inf

        while next_value > value + scale_a * r * gradient.T @ dt:
            r = r * scale_b
            next_t = T + r * dt
            if np.min(next_t) > 0:
                next_value = next_t.T @ A @ next_t - 2 * np.sum(np.log(next_t))
            else:
                next_value = np.inf

        T = copy(next_t)
        value = copy(next_value)
        i += 1

    if normalize:
        T = T / np.exp(np.mean(np.log(T)))

    return T


def compute_em(Sigma_SAM: np.array, K):
    logging.info("compute_em")
    M = Sigma_SAM.shape[0]
    eigval, eigvec = ordered_eig(Sigma_SAM)
    Fhalf = np.zeros((M, K))
    sigma2 = eigval[K + 1:M].mean()
    for i in range(K):
        Fhalf[:, i] = np.sqrt(eigval[i] - sigma2) * eigvec[:, i]

    R = np.diag(Sigma_SAM)
    for i in range(K):
        R = R - Fhalf[:, i] ** 2

    if K == 0:
        Sigma_EM = np.diag(R)
        F_EM = np.zeros((M, M))
        R_EM = np.diag(R)
        return Sigma_EM

    while True:
        big = np.linalg.inv(np.eye(K) + np.matmul(np.matmul(Fhalf.T, np.diag(R ** -1)), Fhalf))
        # term1 = np.zeros((M, K))
        # term2 = np.zeros((K, K))

        temp = np.matmul(np.matmul(big, Fhalf.T), np.diag(R ** -1))
        term1 = np.matmul(Sigma_SAM, temp.T)
        term2 = big + temp @ Sigma_SAM @ temp.T

        newFhalf = term1 @ np.linalg.inv(term2)
        newR = np.diag(Sigma_SAM) - np.diag(newFhalf @ term1.T)

        condition = np.max(np.divide(np.abs((R - newR)), R))
        if condition < 0.001:
            break
        else:
            logging.debug(f"condition value: {condition}")
            R = newR
            Fhalf = newFhalf

    R_EM = np.diag(newR)
    F_EM = newFhalf @ newFhalf.T
    Sigma_EM = F_EM + R_EM

    return Sigma_EM
