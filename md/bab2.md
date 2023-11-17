# pemeringkatan teks
$\mathcal{D} = \{d_1, d_2, \dots, d_n\}$

retreive list 
udah keurut dengan baik.

$D_k = (d_{i_1}, d_{i_2}, \dots, d_{i_k})$

# Recall

$$
\begin{align}
\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
D_k &= (d_{i_1}, d_{i_2}, \dots, d_{i_k}) \\
\text{recall}(q, D_k)\text{@k} &= \frac{\sum_{d \in D_k} \text{rel}(q, d)}{\sum_{d \in \mathcal{D}} \text{rel}(q, d)} \\
\text{dengan } \text{rel}(q, d) &= \begin{cases}
1 & \text{jika } r > 1 \\
0 & \text{jika } r = 0
\end{cases}
\end{align}
$$

# Precision
$$
\begin{align}
\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
D_k &= (d_{i_1}, d_{i_2}, \dots, d_{i_k}) \\
\text{precision}(q, D_k)\text{@k} &= \frac{\sum_{d \in D_k} \text{rel}(q, d)}{|D_k|} \\
\end{align}
$$

## RR

$$
\begin{align}
\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
D_k &= (d_{i_1}, d_{i_2}, \dots, d_{i_k}) \\

\text{RR}(q, D_k)\text{@k} &= \begin{cases}

\frac{1}{\text{FirstRank}(q, D_k)} & \text{jika } \exists d \in D_k \text{ dengan } \text{rel}(q, d) = 1 \\

0 & \text{jika } \forall d \in D_k, \text{ rel}(q, d) = 0 \\

\end{cases} \\


\text{FirstRank}(q,D_k) &= \text{posisi dokumen relevan pertama } d\in D_k \text{ dengan } \text{rel}(q, d) = 1 \\
 
\end{align}
$$

# nDCG
$$
\begin{align}

\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
D_k &= (d_{i_1}, d_{i_2}, \dots, d_{i_k}) \\

\text{nDCG}(q, D_k)\text{@k} &= \frac{\text{DCG}(q, D_k)\text{@k}}{\text{DCG}(q, D_k^{\text{ideal}})\text{@k}} \\
\text{DCG}(q, D_k)\text{@k} &= \sum_{d \in D_k, i \in \mathbb{N}} \frac{2^{\text{rel}(q, d)} - 1}{\log_2(\text{rank}(d, D_k) + 1)} \\
\text{rank}(d,D_k) &= \text{Posisi } d \text{ dalam } D_k \\
\end{align}
$$


jika relevansi biner
$$
\begin{align}
\text{DCG}(q, D_k)\text{@k} &= \sum_{d \in D_k, i \in \mathbb{N}} \frac{\text{rel}(q, d)}{\log_2(\text{rank}(d, D_k) + 1)} \\
\end{align}
$$

# tf-idf
$$
\begin{align}
\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
T_q &= \{t_1, t_2, \dots, t_{L_1}\} \\
T_d &= \{t_1, t_2, \dots, t_{L_2}\} \\

\text{tf}(t, d) &= \frac{\text{Count}(t, d)}{|d|} \\

\text{Count}(t, d) &= \text{jumlah kemunculan } t \text{ dalam } d \\
\text{df}(t, \mathcal{D}) &= \text{jumlah dokumen yang mengandung } t \text{ dalam } \mathcal{D} \\

\text{idf}(t, \mathcal{D}) &= \log_2\left(\frac{|\mathcal{D}|}{\text{df}(t, \mathcal{D})}\right) \\

\text{tf-idf}(t, d, \mathcal{D}) &= \log(1 + \text{tf}(t, d)) \times \text{idf}(t, \mathcal{D}) \\

\text{score}(q,d,\mathcal{D}) &= \sum_{t \in T_q \cap T_d} \text{tf-idf}(t, d, \mathcal{D}) \\

\end{align}
$$

# BM25

$$
\begin{align}
\text{idf}_{\text{BM25}}(t, \mathcal{D}) &= \log\left(1+\frac{|\mathcal{D}| - \text{df}(t, \mathcal{D}) + 0.5}{\text{df}(t, \mathcal{D}) + 0.5}\right) \\
\text{score}_{\text{BM25}}(q,d,\mathcal{D}) &= \frac{\text{tf}(t, d) \times (k_1 + 1)}{\text{tf}(t, d) + k_1 \times (1 - b + b \times \frac{|d|}{\text{avgdl}})} \\
\text{BM25}(t, d, \mathcal{D}) &= \text{idf}(t, \mathcal{D}) \times \text{score}_{\text{BM25}}(q,d,\mathcal{D}) \\
\text{avgdl} &= \text{rata-rata panjang dokumen pada koleksi } \mathcal{D} \\
\text{score}(q,d,\mathcal{D}) &= \sum_{t \in T_q \cap T_d} \text{BM25}(t, d, \mathcal{D}) \\
\end{align}
$$

# Deep learning
# MLP
$$
\begin{align}
f_l(\mathbf{x};\mathbf{W}_l, b_l) &= \phi( \mathbf{x} \mathbf{W}_l + \mathbf{b}_l) \in \mathbb{R}^{d_l}, \quad l = 1, 2, \dots, L-1 \\
f_L(\mathbf{x}) &= \mathbf{x} \mathbf{W}_L + \mathbf{b}_L \in \mathbb{R}^{d_y} \\
f_{\text{model}}(\mathbf{x};\bm{\theta}) &= f_L(f_{L-1}(\dots f_1(\mathbf{x})) \dots) \\
\phi(\mathbf{x}) &= \text{fungsi aktivitasi non-linear} \\
\bm{\theta} &= \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \dots, \mathbf{W}_L, \mathbf{b}_L\} \\
\mathbf{W}_l &= \text{matriks bobot}  \in \mathbb{R}^{d_{l-1} \times d_l} \\
\mathbf{b}_l &= \text{vektor bias} \in \mathbb{R}^{d_l} \\
\end{align}
$$

kalau buat binary klasifikasi
$$
\begin{align}
f_{\text{model}}(\mathbf{x};\bm{\theta}) &= \sigma(f_L(f_{L-1}(\dots f_1(\mathbf{x})) \dots)) \\
f_L(\mathbf{x}) &= \mathbf{x} \mathbf{W}_L + \mathbf{b}_L \in \mathbb{R} \\
\sigma(x) &= \frac{1}{1 + e^{x}} \in [0, 1] \\

\text{decision}(\mathbf{x};\bm{\theta}) &= \begin{cases}
1 & \text{jika } f(\mathbf{x};\bm{\theta}) \geq \text{threshold} \\
0 & \text{jika } f(\mathbf{x};\bm{\theta}) < \text{threshold} \\
\end{cases} \\

\text{threshold}&\in [0, 1] \\
\end{align}
$$

# loss function

$$
\{(\mathbf{x}_i, y_i)\}_{i=1}^N
$$

$$
\begin{align}
y_i \mid \mathbf{x} &\overset{\text{iid}}{\sim} \text{Bernoulli}(\hat y(\mathbf{x})), \quad i = 1, 2, \dots, N \\

p(y \mid \mathbf{x}) &= \mu(\mathbf{x})^y (1 - \mu(\mathbf{x}))^{1-y} \\
\mu(\mathbf{x})& = f_{\text{model}}(\mathbf{x};\bm{\theta}) \\
\end{align} 

$$
fungsi likelihoodnya
$$
\begin{align}
\mathcal{L}(\bm{\theta}) &= \prod_{i=1}^N p(y_i \mid \mathbf{x}_i; \bm{\theta})
\end{align}
$$

maksimum likelihood estimation


$$
\begin{align}
\bm{\theta}_{\text{MLE}} &= \arg\max_{\bm{\theta}} \mathcal{L}(\bm{\theta}) \\
\end{align}
$$

wawaawawa

$$



$$

$$
\begin{align}
\ell{(\bm{\theta})} &= -\log\mathcal{L}(\bm{\theta}) \\
&= -\sum_{i=1}^N \log\left(p(y_i \mid \mathbf{x}_i; \bm{\theta})\right) \\
\bm{\theta}_{\text{MLE}} &= \arg\min_{\bm{\theta}} \ell(\bm{\theta}) \\
\end{align} 
$$

subtitusi $p(y_i \mid \mathbf{x}_i; \bm{\theta})$ dengan $f_{\text{model}}(\mathbf{x}_i; \bm{\theta})$

$$
\begin{align}
\ell(\bm{\theta}) &= \sum_{i=1}^N \log\left(\mu(\mathbf{x}_i)^{y_i} \left(1 - \mu(\mathbf{x}_i)\right)^{1-y_i}\right) \\
&=\sum_{i=1}^{N} \left(y_i \log\left(\mu(\mathbf{x}_i)\right) + (1-y_i) \log\left(1 - \mu(\mathbf{x}_i)\right)\right) \\
\end{align}
$$

$$
\begin{align}
\bm{\theta}_{\text{MLE}} &= \arg\min_{\bm{\theta}}\sum_{i=1}^{N}\underbrace{\left( \left(-y_i \log\left(\mu(\mathbf{x}_i)\right)-(1-y_i) \log\left(1 - \mu(\mathbf{x}_i)\right)\right)\right)}_{\text{Binary Cross Entropy Loss } L(y_i, \mu(\mathbf{x}_i))} \\
\end{align}
$$

---

## system ideal

$$
\frac{2^3-1}{\log_2(1+1)} = 7
$$

$$
\frac{2^1-1}{\log_2(2+1)} = 0.63
$$

## SYSTEM A
 $$
 \frac{2^3-1}{\log_2(3+1)} = 3.5
 $$

 $$
\frac{2^1 -1}{\log_2(1+1)} = 1
 $$

 $$
    \text{nDCG} = \frac{3.5 + 1}{7 + 0.63} = 0.59
 $$


## System B

$$
\frac{2^1-1}{\log_2(3+1)} = 0.5
$$

$$
\text{nDCG} = \frac{0.5}{7 + 0.63} = 0.06
$$

## System c

$$
\frac{2^3-1}{\log_2(2+1)} = 5.05
$$

$$
\frac{2^1-1}{\log_2(3+1)} = 0.5
$$