# 메인 클래스
# 작성일: 2025-12-10

import numpy as np
from scipy.special import digamma
import bsds_utils as utils
import bsds_inference as infer
import bsds_learning as learn

class BSDSModel:
    def __init__(self, n_states, max_ldim, n_iter=50, tol=1e-3):
        self.n_states = n_states
        self.max_ldim = max_ldim
        self.n_iter = n_iter
        self.tol = tol
        self.psii = None
        self.Lm, self.Lcov, self.Xm, self.Xcov, self.b = [], [], [], [], []
        self.Wa, self.Wpi, self.stran, self.sprior = None, None, None, None
        self.Fhist = []

    def fit(self, data_list):
        dim = data_list[0].shape[0]
        n_samples_list = [d.shape[1] for d in data_list]
        
        # 초기화
        print("[*] Initializing Parameters...")
        self.psii = np.ones((dim, 1)) * 10
        k = self.max_ldim + 1
        self.Lm = [np.random.randn(dim, k)*0.1 for _ in range(self.n_states)]
        self.Lcov = [np.tile(np.eye(k)[:,:,np.newaxis], (1,1,dim)) for _ in range(self.n_states)]
        self.Xm = [np.zeros((k, sum(n_samples_list))) for _ in range(self.n_states)] # Simplified storage
        self.Xcov = [np.tile(np.eye(k)[:,:,np.newaxis], (1,1,sum(n_samples_list))) for _ in range(self.n_states)]
        
        # K-Means Init
        Qns_list, self.Wa = utils.init_posteriors_kmeans(data_list, self.n_states)
        self._update_probs()
        
        print(f"[*] Starting Loop (Iter={self.n_iter})...")
        for it in range(self.n_iter):
            # E-Step
            log_emissions = infer.compute_log_out_probs(data_list, self.Lm, self.Lcov, self.psii, self.Xm, self.Xcov, self.n_states)
            log_trans = np.log(self.stran + 1e-10)
            log_start = np.log(self.sprior + 1e-10)
            log_lik, gamma_list, xi_list = infer.vbhmm_estep(log_emissions, log_trans, log_start, self.n_states)
            
            # M-Step
            wa_new, wpi_new = learn.update_transition_counts(xi_list, gamma_list, self.n_states)
            self.Wa = wa_new + (1.0/self.n_states)
            self.Wpi = wpi_new + (1.0/self.n_states)
            self._update_probs()
            self.b = learn.infer_q_nu(self.Lm, self.Lcov, 1.0, self.n_states)
            
            print(f"Iter {it+1}: Log Likelihood = {log_lik:.2f}")
            self.Fhist.append(log_lik)
            
        return self

    def _update_probs(self):
        self.stran = np.exp(digamma(self.Wa) - digamma(np.sum(self.Wa, axis=1, keepdims=True)))
        self.sprior = np.exp(digamma(self.Wpi) - digamma(np.sum(self.Wpi)))