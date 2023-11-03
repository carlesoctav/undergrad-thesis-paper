\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{assets/pics/transformers_seq.png}
	\caption{Arsitektur \f{transformer} untuk mesin translasi neural. Arsitektur terdiri dari \f{encoder} dan \f{decoder} yang terdiri dari beberapa blok \f{transformer} \citep{transformerori}.}
	\label{fig:transformer}
\end{figure}

\f{Transformers} merupakan Arsitektur \f{deep learning} yang pertama kali diperkenalkan oleh \cite{transformerori}. Awalnya Transformers merupakan model \f{sequance to sequance} yang diperuntukkan untuk permasalahn mesin translasi neural (\f{neural machine translation}). Namun, sekarang \f{transformers} juga digunakan untuk permasalahan pemrosesan bahasa alami lainnya. model-model yang menjadi \f{state of the art} permasalahan pemrosesan bahasa alami biasanya menggunakan arsitektur \f{transformers}.

Berbeda dengan arsitektur mesin translasi terdahulu \todoCite{a}, transformers tidak mengunakan \f{recurrent neural network} (RNN) atau \f{convolutional neural network} (CNN), melainkan transformers adalah model \f{feed foward network} yang dapat memproses seluruh \f{input} pada barisan secara paralel. Untuk menggantikan kemampuan RNN dalam mempelajari ketergantungan antar \f{input} yang berurutan dan kemampuan CNN dalam mempelajari fitur lokal, transformers bergantung pada mekanisme \f{attention}.

Terdapat tiga jenis \f{attention} yang digunakan dalam model \f{transformers} \citep{transformerori}:
1. \f{Encoder self-attention} menggunakan barisan \f{input} yang berupa kalimat sebagai masukan untuk menghasilkan barisan representasi kontekstual (vektor) dari \f{input}. Setiap representasi token tersebut memiliki ketergantungan dengan token lainnya dalam urutan masukan.

2. \f{Decoder self-attention} menggunakan barisan \f{target} yang berupa kalimat terjemahan parsial sebagai masukan untuk menghasilkan barisan representasi kontekstual (vektor) dari \f{target}. Setiap representasi token tersebut memiliki ketergantungan dengan token sebelumnya dalam urutan masukan.

3. \f{Decoder-encoder attention} menggunakan barisan representasi kontekstual dari \f{input}, dan barisan representasi kontekstual dari \f{target} untuk menghasilkan token berikutnya yang merupakan hasil prediksi dari model. barisan \f{target} yang digabung dengan token hasil prediksi tersebut akan menjadi barisan \f{target} untuk prediksi selanjutnya.



# ATTENTION

Mekanisme \f{attention} menggambarkan rata-rata terbobot dari barisan elemen dengan bobot yang dihitung secara dinamis berdasarkan kueri masukan dan Kunci elemen. Tujuannya adalah untuk mengambil rata-rata dari fitur beberapa elemen. Namun, daripada memberi bobot setiap elemen secara merata, kita ingin memberi bobot tergantung pada "nilai" elemen tersebut. Dengan kata lain, kita ingin secara dinamis memutuskan elemen masukan mana yang ingin kita perhatikan lebih dari yang lain. Mekanisme \f{attention} biasanya memiliki empat bagian yang perlu ditentukan:

1. vektor kueri ($\mathbf{q}$): Kueri adalah vektor yang merepresentasikan apa yang ingin dicari dalam barisan elemen.

2. vektor kunci ($\mathbf{k}$): Untuk setiap elemen dalam barisan, terdapat vektor yang disebut kunci. Vektor fitur ini secara kasar merepresentasikan apa yang ditawarkan elemen tersebut, atau kapan elemen tersebut menjadi penting. Kunci harus dirancang sedemikian rupa sehingga kita dapat mengidentifikasi elemen yang ingin kita perhatikan berdasarkan kueri.

3. vektor nilai ($\mathbf{v}$): Untuk setiap elemen dalam barisan, terdapat vektor yang disebut nilai. vektor nilai ini yang ingin dirata-ratakan.

4. fungsi skor ($f_{\text{attn}}(\mathbf{q}, \mathbf{k})$): Fungsi skor memberikan bobot-bobot pada pada nilai berdasarkan kueri dan kunci. Fungsi skor dapat dihitung dengan berbagai cara, seperti cara sederhana berupa perkalian skalar atau jarak kosinus, dan dapat juga dihitung menggunakan fungsi skor yang lebih kompleks seperti \f{multi-layer perceptron} (MLP).

Biasanya, hasil fungsi skor diterapkan pada fungsi softmax untuk mendapatkan bobot yang dinormalisasi. Bobot tersebut kemudian digunakan untuk menghitung rata-rata terbobot dari nilai

$$
\begin{aligned}
\alpha_i &= \frac{\exp(f_{\text{attn}}(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}\exp(f_{\text{attn}}(\mathbf{q}, \mathbf{k}_j))} \\

\text{Output} &= \sum_{i=1} \alpha_i \mathbf{v}_i
\end{aligned}
$$

# scaled dot product attention
jelek perlu diubah

Mekanimse \f{attention} yang diterapkan pada transformer disebut sebagai \f{self-attention}. Setiap elemen pada barisan mempunyai kueri, kunci, dan nilai, berbeda dengan mekanisme \f{attention} lainnya yang hanya memiliki nilai dan kunci. Untuk setiap elemen pada barisan, \f{attention layer} akan menggunakan kueri-nya dan semua kunci pada elemen yang lain untuk menghasilkan vektor keluaran yang menjadi representasi kontekstual dari elemen tersebut. Implementasi \f{self-attention} pada \f{transformer} tersebut disebut sebagai \f{scaled dot product attention}.

Misalkan $\mathbf{X}\in \mathbb{R}^{n\times d}$ adalah barisan $n$ elemen dengan dimensi $d$. Untuk setiap elemen pada barisan, kita akan menghasilkan vektor keluaran $\mathbf{Y}\in \mathbb{R}^{n\times d_v}$ yang merupakan representasi kontekstual dari elemen tersebut. Untuk menghasilkan vektor keluaran tersebut, kita akan menggunakan kueri $\mathbf{Q}\in \mathbb{R}^{n\times d_k}$, kunci $\mathbf{K}\in \mathbb{R}^{n\times d_k}$, dan nilai $\mathbf{V}\in \mathbb{R}^{n\times d_v}$ yang di hitung dengan

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{X} \mathbf{W}^Q \\
\mathbf{K} &= \mathbf{X} \mathbf{W}^K \\
\mathbf{V} &= \mathbf{X} \mathbf{W}^V,
\end{aligned}
$$

Dengan $\mathbf{W}^&Q\in \mathbb{R}^{d\times d_k}$, $\mathbf{W}^K\in \mathbb{R}^{d\times d_k}$, dan $\mathbf{W}^V\in \mathbb{R}^{d\times d_v}$ adalah matriks bobot yang merupakan paramater. Pada \f{scaled dot product attention}, fungsi skor dihitung dengan:

$$ 
\begin{equation}
\mathbf{Y} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
\end{equation}
$$


<!-- TODO -->
dari persamaan \todocite, fungsi skor yang digunakan adalah hasil kali skalar antara kueri dan kunci yang dibagi dengan $\sqrt{d_k}$. Pembagian dengan $\sqrt{d_k}$ dilakukan untuk menghindari nilai skor yang terlalu besar atau kecil ketika $d_k$ cukup besar. yang mengakibatkan nilai softmax menjadi sangat kecil.


$\mathbf{Y}$ Merupakan matriks yang berisi representasi kontekstual dari setiap elemen pada barisan $\mathbf{X}$. Matriks $Y$ dapat menjadi masukan untuk \f{attention layer} selanjutnya.

# MULTI-HEAD ATTENTION



$$
\begin{aligned}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O \\
\text{head}_i &= \text{Attention}(\mathbf{Q_i}, \mathbf{K_i}, \mathbf{V_i}) \\
\mathbf{Q_i} &= \mathbf{X} \mathbf{W}_i^Q \\
\mathbf{K_i} &= \mathbf{X} \mathbf{W}_i^K \\
\mathbf{V_i} &= \mathbf{X} \mathbf{W}_i^V 
\end{aligned}
$$


## Pointwise Feed Forward Network

$$
\begin{align}
\text{FFN}(\mathbf{X}) &= \max(0, \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 \\
\end{align}
$$

# Positional Encoding

