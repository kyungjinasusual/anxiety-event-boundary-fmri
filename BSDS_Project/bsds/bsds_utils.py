# 수학 및 초기화
# 작성일: 2025-12-10

import numpy as np
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans

def kl_dirichlet(alpha_p, alpha_q):
    """Dirichlet 분포 간의 KL Divergence"""
    alpha_p, alpha_q = np.array(alpha_p), np.array(alpha_q)
    sum_p, sum_q = np.sum(alpha_p), np.sum(alpha_q)
    return (gammaln(sum_p) - gammaln(sum_q) - 
            np.sum(gammaln(alpha_p)) + np.sum(gammaln(alpha_q)) + 
            np.sum((alpha_p - alpha_q) * (digamma(alpha_p) - digamma(sum_p))))

def init_posteriors_kmeans(data_list, n_states, n_init=10):
    """K-Means를 이용한 초기 상태 확률(Qns) 설정"""
    # 모든 피험자의 데이터를 시간축으로 이어붙임 (Time x Feature)
    X_concat = np.concatenate([d.T for d in data_list], axis=0)
    
    # K-Means 실행
    kmeans = KMeans(n_clusters=n_states, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(X_concat)
    
    # One-hot Encoding
    n_total = X_concat.shape[0]
    Qns_init = np.zeros((n_total, n_states))
    Qns_init[np.arange(n_total), labels] = 1.0
    
    # 피험자별로 다시 쪼개기
    Qns_list = []
    start = 0
    for data in data_list:
        n_t = data.shape[1]
        Qns_list.append(Qns_init[start:start+n_t, :])
        start += n_t
        
    # 전이 행렬(Wa) 초기값 계산
    Wa_init = np.ones((n_states, n_states))
    for i in range(len(labels)-1):
        Wa_init[labels[i], labels[i+1]] += 1
        
    return Qns_list, Wa_init