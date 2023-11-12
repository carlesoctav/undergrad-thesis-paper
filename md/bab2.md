# pemeringkatan teks
$\mathcal{D} = \{d_1, d_2, \dots, d_n\}$

retreive list 
udah keurut dengan baik.

$D_k = [d_{i_1}, d_{i_2}, \dots, d_{i_k}]$

# Recall

$$
\begin{align}
\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
D_k &= [d_{i_1}, d_{i_2}, \dots, d_{i_k}] \\
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
D_k &= [d_{i_1}, d_{i_2}, \dots, d_{i_k}] \\
\text{precision}(q, D_k)\text{@k} &= \frac{\sum_{d \in D_k} \text{rel}(q, d)}{|D_k|} \\
\end{align}
$$

## RR

$$
\begin{align}
\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
D_k &= [d_{i_1}, d_{i_2}, \dots, d_{i_k}] \\

\text{RR}(q, D_k)\text{@k} &= \begin{cases}

\frac{1}{\text{FirstRank}(q, D_k)} & \text{jika } \exists d \in D_k \text{ dengan } \text{rel}(q, d) = 1 \\

0 & \text{jika } \forall d \in D_k, \text{ rel}(q, d) = 0 \\

\end{cases} \\


\text{FirstRank}(q,D_k) &= \min\left(\{j | d_{i_j} \in D_k \text{ dan } \text{rel}(q, d_{i_j}) = 1\}\right)
 
\end{align}
$$

# nDCG
$$
\begin{align}

\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\
D_k &= [d_{i_1}, d_{i_2}, \dots, d_{i_k}] \\

\text{nDCG}(q, D_k)\text{@k} &= \frac{\text{DCG}(q, D_k)\text{@k}}{\text{DCG}(q, D_k^{\text{ideal}})\text{@k}} \\
\text{DCG}(q, D_k)\text{@k} &= \sum_{d \in D_k, i \in \mathbb{N}} \frac{{rel(q,d)}}{\log_2(\text{rank}(d,D_k)+1)} \\

\text{rank}(d,D_k) &= \text{Posisi } d \text{ dalam } D_k \\
\end{align}
$$


# tf-idf
$$

\begin{align}
\mathcal{D} &= \{d_1, d_2, \dots, d_n\} \\

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
\text{idf}(t, \mathcal{D}) &= \log\left(1+\frac{|\mathcal{D}| - \text{df}(t, \mathcal{D}) + 0.5}{\text{df}(t, \mathcal{D}) + 0.5}\right) \\
\text{BM25}(t, d, \mathcal{D}) &= \text{idf}(t, \mathcal{D}) \times \frac{\text{tf}(t, d) \times (k_1 + 1)}{\text{tf}(t, d) + k_1 \times (1 - b + b \times \frac{|d|}{\text{avgdl}})} \\
&= \log\left(1+\frac{|\mathcal{D}| - \text{df}(t, \mathcal{D}) + 0.5}{\text{df}(t, \mathcal{D}) + 0.5}\right) \times \frac{\text{tf}(t, d) \times (k_1 + 1)}{\text{tf}(t, d) + k_1 \times (1 - b + b \times \frac{|d|}{\text{avgdl}})} \\

\text{score}(q,d,\mathcal{D}) &= \sum_{t \in T_q \cap T_d} \text{BM25}(t, d, \mathcal{D}) \\
\end{align}
$$




