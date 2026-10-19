"""Синтетическая модель для демонстрации. НЕ калибровка реального детектора."""
from pathlib import Path

import numpy as np

from xemat.model import build_joint_matrix


def main():
    rng = np.random.default_rng(20261014)
    # Семь ROI, пять компонент: радон и четыре изотопа ксенона.
    R_c = rng.uniform(.02, .15, (7, 5))
    R_c[:5] += np.eye(5)
    R_g1 = R_c * rng.uniform(.8, 1.2, (7, 5))
    R_g2 = R_c * rng.uniform(.7, 1.3, (7, 5))
    F = np.diag([.8, .5, .9, .95, .85])
    kappa_s, kappa_g = .1, .2  # DETBK измеряется дольше пробы и памяти.
    M = build_joint_matrix(R_c, R_g1, R_g2, F, kappa_s=kappa_s, kappa_g=kappa_g)
    H = np.diag([.02, .01, .012, .015, .011])
    Gamma = np.eye(5)
    Gamma[2, 4] = -.4  # Вычитание дочернего вклада: beta>=0 недостаточно.
    G_p, G_s, V0 = np.eye(5) * 1.1, np.eye(5) * 1.2, 2.
    T_V = G_s @ G_p @ Gamma @ H / V0
    theta = np.r_[[80., 30., 60., 10., 20., 20., 10., 12., 3., 8.], np.full(7, 40.)]
    # ζ — среднее DETBK, последнее слагаемое theta. Разовый измеренный
    # DETBK разыгрывается вместе с SAMPLE/GASBK, из x ничего не вычитается.
    mu = M @ theta
    x = rng.poisson(mu)
    path = Path(__file__).with_name('demo_model.npz')
    np.savez(path, M=M, x=x, n_sample_parameters=5, n_detector_parameters=7,
             coordinates='beta_gamma_zeta', T_V=T_V, H=H, Gamma=Gamma,
             G_p=G_p, G_a=G_s, V0=V0, kappa_s=kappa_s, kappa_g=kappa_g,
             isotope_names=['Rn222', 'Xe135', 'Xe133', 'Xe131m', 'Xe133m'],
             parameter_names=['beta_Rn222', 'beta_Xe135', 'beta_Xe133',
                              'beta_Xe131m', 'beta_Xe133m', 'gamma_Rn222',
                              'gamma_Xe135', 'gamma_Xe133', 'gamma_Xe131m',
                              'gamma_Xe133m'] + [f'zeta:ROI{i}' for i in [1, 2, 3, 5, 6, 7, 8]],
             roi_numbers=[1, 2, 3, 5, 6, 7, 8],
             mdc_component_indices=[1, 2, 3, 4],
             true_theta=theta, true_mu=mu)
    print(path)


if __name__ == '__main__':
    main()
