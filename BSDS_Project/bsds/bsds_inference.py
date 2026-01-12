# 핵심 추론 엔진
# 작성일: 2025-12-10

import numpy as np
from scipy.special import logsumexp
from scipy.linalg import cholesky

def compute_log_out_probs(data_list, Lm, Lcov, psii, Xm, Xcov, n_states):
    """관측 데이터가 각 상태에서 발생했을 로그 확률 계산 (computeLogOutProbs.m)"""
    log_out_probs = []
    p = data_list[0].shape[0] # ROI 개수
    psi_diag = np.diag(psii.flatten() + 1e-10)
    
    start_col = 0
    for Y in data_list:
        n_samples = Y.shape[1]
        log_qns = np.zeros((n_states, n_samples))
        end_col = start_col + n_samples
        
        for s in range(n_states):
            k_t = Lm[s].shape[1]
            
            # 1. 사전 계산
            lm_psii_lm = Lm[s].T @ psi_diag @ Lm[s]
            lcov_flat = Lcov[s].reshape(k_t * k_t, p)
            lcov_term = (lcov_flat @ psii).reshape(k_t, k_t)
            temp_mat = lm_psii_lm + lcov_term 
            
            # 2. Latent Variable 가져오기
            Xm_subj = Xm[s][:, start_col:end_col]
            Xcov_subj = Xcov[s][:, :, start_col:end_col]
            
            # 3. 수식 항 계산 (Vectorized)
            recon_diff = Y - 2 * (Lm[s] @ Xm_subj)
            term_a = np.sum(Y * (psi_diag @ recon_diff), axis=0)
            term_b = np.einsum('ij,ijt->t', temp_mat, Xcov_subj)
            term_c = np.sum(Xm_subj * (temp_mat @ Xm_subj), axis=0)
            term_d = np.trace(Xcov_subj[1:, 1:, :], axis1=0, axis2=1)
            term_e = np.sum(Xm_subj[1:, :]**2, axis=0)
            
            # Log Determinant (Cholesky)
            term_f = np.zeros(n_samples)
            for t in range(n_samples):
                try:
                    L = cholesky(Xcov_subj[1:, 1:, t] + np.eye(k_t-1)*1e-10)
                    term_f[t] = -2 * np.sum(np.log(np.diag(L)))
                except: term_f[t] = 0
            
            log_qns[s, :] = -0.5 * (term_a + term_b + term_c + term_d + term_e + term_f)
        
        log_out_probs.append(log_qns)
        start_col += n_samples
        
    return log_out_probs

def vbhmm_estep(log_emissions, log_trans, log_start, n_states):
    """HMM Forward-Backward Algorithm (Log-Space)"""
    gamma_list, xi_list = [], []
    total_log_lik = 0
    
    for log_emit in log_emissions:
        T = log_emit.shape[1]
        log_alpha = np.zeros((n_states, T))
        log_alpha[:, 0] = log_start + log_emit[:, 0]
        
        # Forward
        for t in range(1, T):
            for j in range(n_states):
                log_alpha[j, t] = logsumexp(log_alpha[:, t-1] + log_trans[:, j]) + log_emit[j, t]
        
        # Backward
        log_beta = np.zeros((n_states, T))
        for t in range(T-2, -1, -1):
            for i in range(n_states):
                term = log_trans[i, :] + log_emit[:, t+1] + log_beta[:, t+1]
                log_beta[i, t] = logsumexp(term)
        
        # Posteriors
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=0)
        
        xi = np.zeros((n_states, n_states, T-1))
        for t in range(T-1):
            for i in range(n_states):
                for j in range(n_states):
                    xi[i, j, t] = (log_alpha[i, t] + log_trans[i, j] + log_emit[j, t+1] + log_beta[j, t+1])
            xi[:, :, t] -= logsumexp(xi[:, :, t])
        
        gamma_list.append(np.exp(log_gamma))
        xi_list.append(np.exp(xi))
        total_log_lik += logsumexp(log_alpha[:, -1])
        
    return total_log_lik, gamma_list, xi_list