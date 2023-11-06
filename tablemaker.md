
# Overall Model

| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |
| $\text{IndoBERT}_{\text{MEAN}}$         | .000       | .000           | .000        | .000           | .000         | .000                       |
| $\text{IndoBERT}_{\text{CAT}}$          | .181       | .642           | .447        | .858           | .455         | .971                       |
| $\text{IndoBERT}_{\text{DOT}}$          | .192       | .847           | .378        | .936           | .355         | .920                       |
| $\text{IndoBERT}_{\text{DOTdnegs}}$  | .232       | .847           | .471        | .921           | .397         | .898                       |
| $\text{IndoBERT}_{\text{DOTMargin}}$    | .207       | .799           | .446        | .929           | .387         | .899                       |
| $\text{IndoBERT}_{\text{KD}}$           | -          |  .803          | .300        | .761           | -            | -                          |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            $\text{IndoBERT}_{\text{MEAN}}$ & .000 & .000 & .000 & .000 & .000 & .000\\ \hline
            $\text{IndoBERT}_{\text{CAT}}$ & .181 & .642 & .447 & .858 & .455 & .971\\ \hline
            $\text{IndoBERT}_{\text{DOT}}$ & .192 & .847 & .378 & .936 & .355 & .920\\ \hline
            $\text{IndoBERT}_{\text{DOTdnegs}}$ & .232 & .847 & .471 & .921 & .397 & .898\\ \hline
            $\text{IndoBERT}_{\text{DOTMargin}}$ & .207 & .799 & .446 & .929 & .387 & .899\\ \hline
            $\text{IndoBERT}_{\text{KD}}$ & - & .803 & .300 & .761 & - & -\\ \hline
            
    \end{tabular}
    
    \label{tab:my_label}

# Sampe BM25
| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |


latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            
    \end{tabular}
    
    \label{tab:my_label}
\end{table}
# sampe indobert mean

| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |
| $\text{IndoBERT}_{\text{MEAN}}$         | .000       | .000           | .000        | .000           | .000         | .000                       |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            $\text{IndoBERT}_{\text{MEAN}}$ & .000 & .000 & .000 & .000 & .000 & .000\\ \hline
    \end{tabular}
    
    \label{tab:my_label}
\end{table}

# Sampe indobert cat tapi ada bandingin sota dan dari paper mmarco

| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |
| $\text{IndoBERT}_{\text{CAT}}$          | .181       | .642           | .447        | .858           | .455         | .971                       |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            $\text{IndoBERT}_{\text{CAT}}$ & .181 & .642 & .447 & .858 & .455 & .971\\ \hline
            
    \end{tabular}
    
    \label{tab:my_label}
\end{table}
# sampe indobert dot

| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |
| $\text{IndoBERT}_{\text{DOT}}$          | .192       | .847           | .378        | .936           | .355         | .920                       |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            $\text{IndoBERT}_{\text{DOT}}$ & .192 & .847 & .378 & .936 & .355 & .920\\ \hline
            
    \end{tabular}
    
    \label{tab:my_label}
\end{table}

# sampe indobert dot hard negs
| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |
| $\text{IndoBERT}_{\text{DOTHardnegs}}$   | .232       | .847           | .471        | .921           | .397         | .898                       |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            IndoBERT_{DOTHardnegs} & .232 & .847 & .471 & .921 & .397 & .898\\ \hline
            
    \end{tabular}
    
\end{table}

# sampe indobert dot margin
| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |
| $\text{IndoBERT}_{\text{DOTMargin}}$    | .207       | .799           | .446        | .929           | .387         | .899                       |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            $\text{IndoBERT}_{\text{DOTMargin}}$ & .207 & .799 & .446 & .929 & .387 & .899\\ \hline
            
    \end{tabular}

#

# RETRIEVAL benchmark

| Model                                   | Latensi (ms)|                | Memori(MB) |                |
|-----------------------------------------|-------------|----------------|------------|----------------|
|$\text{BM25 (Elastic Search)}$           |  6.55       |                |  800       |                |
|$\text{IndoBERT}_{\text{DOT}}$           |  9.9        |                |  3072      |                |
|$\text{IndoBERT}_{\text{CAT}}$           |  242        |                |  800       |                |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Latensi (ms)}& 
         \multicolumn{2}{|c|}{Memori (MB)}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            $\text{IndoBERT}_{\text{DOT}}$ & .192 & .847 & .378 & .936 & .355 & .920\\ \hline
            $\text{IndoBERT}_{\text{CAT}}$ & .181 & .642 & .447 & .858 & .455 & .971\\ \hline
            
    \end{tabular}
    
\end{table}



# sampe indobert kd

| Model                                   | Mmarco Dev |                | MrTyDi Test |                | Miracal Test |                            |
|-----------------------------------------|------------|----------------|-------------|----------------|--------------|----------------------------|
|                                         | MRR@10     | R@1000         | MRR@10      | R@1000         | NCDG@10      | R@1K                       | 
| $\text{BM25 (Elastic Search)}$          | .114       | .642           | .279        | .858           | .391         | .971                       |
| $\text{IndoBERT}_{\text{KD}}$           | .176       | .803           | .300        | .761           | .179         | .072                       |


# en eval

| Model                                   | msarco Dev |                |
|-----------------------------------------|------------|----------------|
|                                         | MRR@10     | R@1000         |
| $\text{BM25 (Elastic Search)}$          | .184       | .857           |
| $\text{IndoBERT}_{\text{KD}}$           | .245       | .912           |

latex version

\begin{table}
    \centering
    \caption{Caption}
    \label{}
    \begin{tabular}{|c|cc|cc|cc|} \hline 
         Model&  \multicolumn{2}{|c|}{Mmarco Dev}& 
         \multicolumn{2}{|c|}{MrTyDi Test}&  \multicolumn{2}{|c|}{Miracl Dev}\\ \hline 
            & MRR@10 & R@1000 & MRR@10 & R@1000 & NCDG@10 & R@1K\\ \hline 
            BM25 (Elastic Search) & .114 & .642 & .279 & .858 & .391 & .971\\ \hline
            $\text{IndoBERT}_{\text{KD}}$ & - & .803 & .300 & .761 & - & -\\ \hline
            
    \end{tabular}
\end{table}