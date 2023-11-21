

# Bab 3

## soft-attention as a soft lookup table.


$\alpha, \Alpha$ kita sebut bobot atensi.

$f_{attn}$ kita sebut sebagai skor atensi.

pasangan $\mathcal{KV} = \{(\mathbf{k}_1, \mathbf{v}_2), (\mathbf{k}_2, \mathbf{v}_2), \dots, (\mathbf{k}_n, \mathbf{v}_n)\}$. bisa di rewrite sebagai 

$\mathbf{K} = \begin{matrix}
\mathbf{k}_1 \\
\mathbf{k}_2 \\
\vdots \\
\mathbf{k}_n
\end{matrix} \in \mathbb{R}^{n \times d_k}$

$\mathbf{V} = \begin{matrix}
\mathbf{v}_1 \\
\mathbf{v}_2 \\
\vdots \\
\mathbf{v}_n
\end{matrix} \in \mathbb{R}^{n \times d_v}$


lalu untuk suatu kueri $\mathbf{q} \in \mathbb{R}^{d_k}$, kita dapat menghitung skor attention dengan

$$
\text{Attention}(q, \mathbf{K}, \mathbf{V}) = \mathbf{\alpha}\mathbf{V} \in \mathbb{R}^{d_v}
$$

dengan 

$$
\mathbf{\alpha} = [\alpha_{1}, \alpha_{2}, \dots, \alpha_{n}]
$$

$$
\alpha_{i}(\mathbf{q},\mathbf{k}_i) = \text{Softmax}_i(\mathbf{\alpha}) = \frac{f_{attn}(\mathbf{q}, \mathbf{k}_i)}{\sum_{j=1}^{n} f_{attn}(\mathbf{q}, \mathbf{k}_j)}
\begin{align}
\end{align}
$$

$$
\sum_{i=1}^{n} \alpha_{i} = 1 \\
0 \leq \alpha_{i} \leq 1
$$

dengan $f_{attn}$ adalah fungsi yang menghitung skor antara $\mathbf{q}$ dan $\mathbf{k}_i$, dapat berupa fungsi similaritas seperti hasil kali titik atau jarak kosinus, atau juga dapat yang lebih kompleks seperti fungsi neural network.


## hard-attention
serupa dengan persamaan diatas tapi $\alpha_{i}$ adalah one-hot vector, yaitu $\alpha_{i} \in \{0, 1\}$ dan $\sum_{i=1}^{n} \alpha_{i} = 1$.

$$
\alpha_i = 
\begin{cases}
1, & \text{jika } i = \arg\max_{j} f_{attn}(\mathbf{q}, \mathbf{k}_j) \\
0, & \text{lainnya}
\end{cases}
$$

- tldr hard attention itu tidak terturunkan sehingga ga bsa ditrain pake backpropagation.


## Soft attention Kueri Matriks

$$
\mathbf{Q} = \begin{matrix}
\mathbf{q}_1 \\
\mathbf{q}_2 \\
\vdots \\
\mathbf{q}_m
\end{matrix} \in \mathbb{R}^{m \times d_k}
$$

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{A} \mathbf{V} \in \mathbb{R}^{m \times d_v}

$$


$$
\mathbf{\Alpha} = 
\begin{matrix}
\mathbf{\alpha}_1 \\
\mathbf{\alpha}_2 \\
\vdots \\
\mathbf{\alpha}_m
\end{matrix}

=
\begin{matrix}
\alpha_{11} & \alpha_{12} & \dots & \alpha_{1n} \\
\alpha_{21} & \alpha_{22} & \dots & \alpha_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\alpha_{m1} & \alpha_{m2} & \dots & \alpha_{mn} \\
\end{matrix}
\in \mathbb{R}^{m \times n}
$$

$$
\alpha_{ij}(\mathbf{q}_i, \mathbf{k}_j) = \text{Softmax}_j(\mathbf{\alpha}_i) = \frac{f_{attn}(\mathbf{q}_i, \mathbf{k}_j)}{\sum_{k=1}^{n} f_{attn}(\mathbf{q}_i, \mathbf{k}_k)}
$$

## Kernel regression as non-parametric attention
disini bakal tunjukkin attention itu sebenarnya  parametric version of kernel regression.

misalkan kita punya dataset
$\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_n, y_n)\}$
proykesi vektor 
dimana $, oleh bobot 
kalau kita punya titik $\mathbf{x}_{*}$ yang proykesi vektor 
dimana $, oleh bobot 
$$
y_{*} = f(\mathbf{x}_{*}) = \sum_{i=1}^{n} \alpha_{i}(\mathbf{x}_{*},\mathbf{x}_i) y_i
$$

$$
\alpha_{i}(\mathbf{x}_{*},\mathbf{x}_i) = \frac{f_{attn}(\mathbf{x}_{*}, \mathbf{x}_i)}{\sum_{j=1}^{n} f_{attn}(\mathbf{x}_{*}, \mathbf{x}_j)}
$$



kita pilih fungsi $f_{attn}$ sebagai fungsi kernel, misalnya gaussian kernel

$$
f_{attn}(\mathbf{x}_{*}, \mathbf{x}_i) = \mathcal{K}_{\beta}(\mathbf{x}_{*}-\mathbf{x}_i) = \exp\left(-\frac{||\mathbf{x}_{*}-\mathbf{x}_i||^2 \beta^2}{2}\right)

$$

$$
\begin{align}
y_{*} = f(\mathbf{x}_{*}) = \sum_{i=1}^{n} \alpha_{i}(\mathbf{x}_{*},\mathbf{x}_i) y_i = \sum_{i=1}^{n} \frac{f_{attn}(\mathbf{x}_{*}, \mathbf{x}_i)}{\sum_{j=1}^{n} f_{attn}(\mathbf{x}_{*}, \mathbf{x}_j)} y_i \\


=  \sum_{i=1}^{n} \text{Softmax}_i([
\mathcal{K}_{\beta}(\mathbf{x}_{*}-\mathbf{x}_1),
\mathcal{K}_{\beta}(\mathbf{x}_{*}-\mathbf{x}_2),
\dots,
\mathcal{K}_{\beta}(\mathbf{x}_{*}-\mathbf{x}_n)
]) y_i \\

= \sum_{i=1}^{n} \frac{\exp\left(-\frac{||\mathbf{x}_{*}-\mathbf{x}_i||^2 \beta^2}{2}\right)}{\sum_{j=1}^{n} \exp\left(-\frac{||\mathbf{x}_{*}-\mathbf{x}_j||^2 \beta^2}{2}\right)} y_i \\
\end{align}
$$


## Parametric attention

fungsi attention non parametric
$$
f_{attn}(\mathbf{q}, \mathbf{k}) = \text{Kernel}(\mathbf{q}, \mathbf{k})
$$

kita ubah ke parametric dengan kalikan $\mathbf{q} \in \mathbb{R}^{d_q}$ dengan beban $\mathbf{W}_q$ dan $\mathbf{k} \in \mathbb{R}^{d_k}$ dengan beban $\mathbf{W}_k$.

Biasanya kita proyeksikan \mathbf{q} dan \mathbf{k} ke dimensi yang sama, yaitu $d_q = d_k = d_{attn}$.

Gunakan fungsi similaritas instead of pake kernel. beberapa contoh $f_{attn}$:
1. additive attention
$$
f_{attn}(\mathbf{q}, \mathbf{k}) = (\mathbf{q} \mathbf{W}^q  + \mathbf{k} \mathbf{W}^k)  \mathbf{W}_{\text{out}}
$$

1. dot product attention

$$
f_{attn}(\mathbf{q}, \mathbf{k}) = (\mathbf{q} \mathbf{W}^q) (\mathbf{k} \mathbf{W}^k)^{\top}
$$

3. scaled dot product attention

$$

f_{attn}(\mathbf{q}, \mathbf{k}) = \frac{(\mathbf{q} \mathbf{W}^q) (\mathbf{k} \mathbf{W}^k)^{\top}}{\sqrt{d_{attn}}}
$$

dimana $d_{attn}$ adalah dimensi dari proykesi vektor $\mathbf{q}$ dan $\mathbf{k}$, oleh bobot $\mathbf{W}^q$ 
dan $\mathbf{W}^k$.

kumpulan vektor nilai $\mathbf{V}$ juga di proyeksikan oleh matriks bobot $\mathbf{W}^v$.

Dengan begitu untuk kumpulan kueri $\mathbf{Q}$ dan kumpulan kunci $\mathbf{K}$, dan vektor nilai $\mathbf{V}$ kita dapat menghitung skor attention dengan pada contoh 3 sebagai:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}(\frac{\mathbf{Q} \mathbf{W}^q (\mathbf{K} \mathbf{W}^k)^{\top}}{\sqrt{d_{attn}}}) \mathbf{V} \mathbf{W}^v
$$

dengan operasi $\text{Softmax}(\mathbf{X})$ adalah operasi softmax pada setiap baris dari matriks $\mathbf{X}$.


## scaled dot product attention

pembagian dengan $\sqrt{d_{attn}}$ pada scaled dot product attention adalah untuk menghindari beban attention terpusat pada titik sekitar 0 dan 1, karena skor atensi ($\mathbf{q}\mathbf{k}^{\top}$) akan semakin besar atau semakin kecil seiring dengan bertambahnya $d_{attn}$.

$\mathbf{qW}^q \sim \mathcal{N}(0, \sigma^2)$ dan $\mathbf{kW}^k \sim \mathcal{N}(0, \sigma^2)$.

dan selanjutnya:

$
\text{Var}(\mathbf{qW}^q (\mathbf{kW}^k)^{\top}) = \sum_{i=1}^{d_{attn}} \text{Var}\left((\mathbf{qW}^q)_i ((\mathbf{kW}^k)^{\top}_i\right) = \sigma^4 d_{attn}
$

$\sigma^4$ tidak menjadi masalah karena biasnya karena lapisan layernorm bakal buat jadi unit variance sebelum masuk ke scaled dot product attention.

tapi $d_{attn}$ yang besar akan membuat $\text{Var}(\mathbf{qW}^q (\mathbf{kW}^k)^{\top})$ menjadi besar, sehingga $\text{Softmax}(\mathbf{qW}^q (\mathbf{kW}^k)^{\top})$ akan banyak dekat ke 0 dan 1.

hal ini mengakibatkan hasil
fungsi softmax kurang efektif karena lebih cenderung menuju ke daerah dengan gradien
yang sangat kecil

### Multi-head attention

kalau kita tinjau matriks bobot atensi sebagai kernel matriks, banyak kernel always help, to capture different aspects of similarity from data. ini ide dari MHA pada transforrmers.

kalau kita ada $h$ head, maka:

$$
\begin{align}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^o, \\
\text{dengan} \quad \text{head}_i &= \text{Attention}_i(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \\
\end{align} \\

\text{dimana} \quad \text{Attention}_i(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}(\frac{\mathbf{Q} \mathbf{W}^q_i (\mathbf{K} \mathbf{W}^k_i)^{\top}}{\sqrt{d_{attn}/h}}) \mathbf{V} \mathbf{W}^v_i
$$

fungsi $\text{Concat}(\text{head}_1, \dots, \text{head}_h) = [\text{head}_1 | \dots | \text{head}_h]$ adalah fungsi yang menggabungkan hasil dari setiap head menjadi satu matriks.

kalau kita pengen dimensi dari vektor keluaran $d_{attn}$

### self-attention 
$\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$.

$$
\begin{align}
\text{MultiHead}_h(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O, \\
\text{dengan} \quad \text{head}_i &= \text{Attention}_i(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \\
\end{align} \\

\text{dimana} \quad \text{Attention}_i(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}(\frac{\mathbf{X} \mathbf{W}^q_i (\mathbf{X} \mathbf{W}^k_i)^{\top}}{\sqrt{d_{attn}/h}}) \mathbf{X} \mathbf{W}^v_i \\

$$



## real multihead

$$
\begin{align}
\text{MHSA}(\mathbf{E}) = \text{Concat}(\text{head}_i, \dots, \text{head}_h)\mathbf{W}^O \in \mathbb{R}^{L \times d_{\text{token}}} \\
\text{head}_i = \text{Self-Attention}_i(\mathbf{E}) = \text{Softmax}(\frac{\mathbf{E} \mathbf{W}^q_i (\mathbf{E} \mathbf{W}^k_i)^{\top}}{\sqrt{d_{\text{token}}/h}}) \mathbf{E} \mathbf{W}^v_i  \in  \mathbb{R}^{L \times \frac{d_{\text{token}}}{h}} \\
\text{Concat}(\text{head}_1, \dots, \text{head}_h) = [\text{head}_1 | \dots | \text{head}_h] \in \mathbb{R}^{L \times d_{\text{token}}} \\
\text{dengan } \mathbf{W}^q_i, \mathbf{W}^k_i, \mathbf{W}^v_i,\in \mathbb{R}^{\frac{d_{\text{token}}}{h} \times \frac{d_{\text{token}}}{h}}, \mathbf{W}^O \in \mathbb{R}^{d_{\text{token}} \times d_{\text{token}}} \\


\end{align}
$$


## Token Embedding
notasinya masih jelek, sih, nanti di ubah.


input dari transformers itu sentence, alias sequance of word / subword. paling mudah kita bisa bikin representasi vektornya dari word/subword dengan one hot encoding.

 misalkan $\mathcal{T} = \{t_1, t_2, \dots, t_{|\mathcal{V}|}\}$ adalah kumpulan token (word/subword) yang mungkin.

untuk $t_{i_k} \in \mathcal{T}$, representasi vektornya 
$\mathbf{oh}_{t_{i_k}} = [0, \dots, 0, 1, 0, \dots, 0] \in \mathbb{R}^{|\mathcal{V}|}$, dimana 1 ada di index ke $i_k$.

masalah:
1. sparse
2. representasi yang buruk, karena tidak ada informasi apapun tentang hubungan antar token, tidak bisa melakukan operasi pada representasi yang bermakna.

solusi, buat matriks embedding $\mathbf{E}_{\mathcal{V}} \in \mathbb{R}^{|\mathcal{V}| \times d_{model}}$ dimana $d_{model}$ adalah dimensi yang diinginkan dari representasi vektor token.

sehingga

$$
\mathbf{e}_{t_{i_k}} = \mathbf{oh}_{t_{i_k}} \mathbf{E_\mathcal{V}} \in \mathbb{R}^{d_{model}}
$$

take row $i$ dari embedding matrix.
ini learnable parameter, jadi bisa di initaliaze random terus selama di train akan berubah2.


dengan begitu untuk suatu barisan $\mathbf{t} = [t_{i_1}, t_{i_2}, \dots, t_{i_L}]$ kita dapat menghitung representasi vektor dari barisan tersebut dengan

$$
\text{Embed}(\mathbf{t})= \mathbf{E} = \begin{bmatrix}
\mathbf{e}_{t_{i_1}} \\
\mathbf{e}_{t_{i_2}} \\
\vdots \\
\mathbf{e}_{t_{i_L}} \\
\end{bmatrix} \in \mathbb{R}^{n \times d_{model}}
$$

## Positional Embedding
Salah satu permasalahan pengunaan mekanisme \f{attention} untuk pemodelan bahasa adalah mekanisme \f{attention} bersifat permutasi equivarian, artinya, jika kita mengubah urutan dari elemen dalam barisan, maka hasil dari mekanisme \f{attention} tidak akan berubah. Dengan kata lain, mekanisme \f{attention} bekerja pada himpunan $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$, bukan pada barisan $[\mathbf{x}_1, \dots, \mathbf{x}_n]$. Hal ini menyebabkan model tidak dapat mempelajari pentingnya urutan posisi.

Untuk mengatasi masalah tersebut, \cite{transformerori} menambahkan informasi posisi pada barisan elemen dengan menambahkan vektor posisi pada vektor representasi elemen. Vektor posisi,$\textbf{p} \in \mathbb{R}^{d_{input}}$ tersebut disebut sebagai \f{positional encoding}. Persamaan xx menunjukkan bagaimana \f{positional encoding} dihitung.

$$
\text{PE}_{ \text {pos }, i}= \begin{cases}\sin \left(\frac{p o s}{10000^{i / d_{\text {model }}}}\right) & \text {jika } i \bmod 2=0 ,\\ \cos \left(\frac{p o s}{10000^{(i-1) / d_{\text {model }}}}\right) & \text { lainnya. }\end{cases}
$$

dimana $pos \leq L$ merupakan posisi dari elemen dalam barisan, $i\leq d_{\text{model}}$ merupakan indeks dari vektor posisi, dan $d_{\text{model}}$ merupakan banyaknya elemen dalam barisan.

Jika $\mathbf{E} \in \mathbb{R}^{L\times d_{\text{model}}}$ merupakan \f{word embedding} dari barisan elemen masukan, barisan baru yang telah ditambahkan \f{positional encoding} dapat dihitung sebagai berikut:

$$
\begin{equation}
\mathbf{X} = \mathbf{E} + \mathbf{PE} \times \text{parameter skala}
\end{equation}
$$


kasih contoh E+PE yang berbeda.

$$
\begin{align}
\mathbf{X} = 
\underbrace{\begin{bmatrix}
0.2 & 0.1 & 0.3 & 0.4 \\
0.1 & 0.3 & 0.2 & 0.4 \\
0.3 & 0.2 & 0.1 & 0.4 \\
\end{bmatrix}}_{\text{Embed}((t_1,t_2,t_3))}
+
\underbrace{\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}}_{\text{pos}((t_1,t_2,t_3))}

= \begin{bmatrix}
0.3 & 0.1 & 0.3 & 0.4 \\
0.1 & 0.4 & 0.2 & 0.4 \\
0.3 & 0.2 & 0.2 & 0.4 \\
\end{bmatrix} \\
\mathbf{X'} =
\underbrace{\begin{bmatrix}
0.3 & 0.2 & 0.1 & 0.4 \\
0.1 & 0.3 & 0.2 & 0.4 \\
0.2 & 0.1 & 0.3 & 0.4 \\
\end{bmatrix}}_{\text{Embed}((t_3,t_2,t_1))}
+
\underbrace{\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}}_{\text{pos}((t_3,t_2,t_1))}
= \begin{bmatrix}
0.4 & 0.2 & 0.1 & 0.4 \\
0.1 & 0.4 & 0.2 & 0.4 \\
0.2 & 0.1 & 0.4 & 0.4 \\
\end{bmatrix}  \\
\end{align}
$$

if we're just using token embedding  as the input of the transformer, then the model will be permutation equivariant.

$$
\begin{align}
E_{(t_1,t_2,t_3)} &= \text{Embed}((t_1,t_2,t_3)) \\
\text{Attention}( E_{(t_1,t_2,t_3)}, E_{(t_1,t_2,t_3)}, E_{(t_1,t_2,t_3)}) &= \text{Attention}( E_{(t_3,t_2,t_1)}, E_{(t_3,t_2,t_1)}, E_{(t_3,t_2,t_1)}) \\

\end{align}
$$

$$
but 
$$


parameter skalanya adalah parameter yang digunakan untuk mengontrol besarnya skala dari \f{positional encoding}.
paper aslinya pakai $\sqrt{d_{\text{model}}}$, jujur gatau kenapa :D.

Also PE juga bisa learnable instead of fixed. 

$$
\mathbf{pe}_{\text{pos}} = [0, \dots, 1, 0, \dots, 0] \mathbf{W}^{pe} \in \mathbb{R}^{d_{model}}
$$
dengan posisi 1 ada di index ke $\text{pos}$.
$$
\text{Pos}(\mathbf{t})= \text{PE} = \begin{bmatrix}
\text{pe}_1 \\
\text{pe}_2 \\
\vdots \\
\text{pe}_L \\
\end{bmatrix} \in \mathbb{R}^{L \times d_{model}}

$$

## Pointwise Feedforward Network

\f{Position-wise feed forward network} (FFN)adalah \f{feed forward network} yang terdiri dari dua \f{layer linear} dengan fungsi aktivasi \f{ReLU} diantara kedua lapisan tersebut. Persamaan xx menunjukkan bagaimana \f{position-wise feed forward network} dihitung.

$$
\begin{align}
\text{FFN}(\mathbf{X}) &= \max(0, \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 \\
\end{align}
$$

## Layer Normalization

$$
\begin{align}
\text{LayerNorm}(\mathbf{X}) &= (\mathbf{X}-\bm{\mu})\odot \frac{1}{\bm{\sigma}} \in \mathbb{R}^{ L\times d_{\text{token}}}, \\
\bm{\mu} &= \begin{bmatrix}
\mu_1 &\dots & \mu_1 \\
\vdots & \ddots &\vdots \\
\mu_L & \dots & \mu_L
\end{bmatrix} \in \mathbb{R}^{L\times d_{\text{token}}}, \\
\frac{1}{\bm{\sigma}} &= \begin{bmatrix}
\frac{1}{\sigma_1} &\dots & \frac{1}{\sigma_1} \\
\vdots & \ddots &\vdots \\
\frac{1}{\sigma_L} &\dots & \frac{1}{\sigma_L} \\
\end{bmatrix} \in \mathbb{R}^{L\times d_{\text{token}}}, \\
\mu_i &= \frac{1}{d_\text{token}}\sum_{j=1}^{d_{\text{token}}} x_{ij},\quad i=1,\dots,L, \\
\sigma_i &= \sqrt{\frac{1}{d_{\text{token}}} \sum_{j=1}^{d_{\text{token}}} (x_{ij}-\mu_i)^2}, \quad i = 1,\dots, L, \\
\odot &= \text{element-wise product.} 

\end{align}

$$




## Putting it all together



$$
\begin{align}
\text{EncoderBlock}(\mathbf{t}) &= \mathbf{Y} \\
\mathbf{Y} &= \text{FFN}(\text{LayerNorm}(\mathbf{Z})+\mathbf{Z}) \\
\mathbf{Z} &= \text{LayerNorm}(\text{MultiHead}_h(\mathbf{X}, \mathbf{X}, \mathbf{X}) + \mathbf{X}) \\
\mathbf{X}  &= \text{Embed}(\mathbf{t}) + \text{Pos}(\mathbf{t}) \times \gamma \\
\end{align}
$$

dan kita punya transformers encoder adalah sekumpulan encoder block yang di stack. dengan begitu jika kita punya $n$ transformer block, maka:

$$
\begin{align}
\text{Encoder}(\mathbf{t}) &= \text{EncoderBlock}_n(\text{EncoderBlock}_{n-1}(\dots(\text{EncoderBlock}_1(\mathbf{t})))) \\
\end{align}
$$
## BERT stack of encoder

<!-- $$
\begin{align}
\mathcal{L}_{\text{MLM}} &= -\sum_{i=1}^{L} \log p(t_i | \mathbf{t}_{\backslash i}) \\
&= -\sum_{i=1}^{L} \log \text{Softmax}(\mathbf{X}\mathbf{W}^o)_{t_i} \\
\end{align}
$$ -->

$$
\begin{equation}
\text{BERT}(\mathbf{t}) = \text{Encoder}(\mathbf{t}) = \text{EncoderBlock}_n(\text{EncoderBlock}_{n-1}(\dots(\text{EncoderBlock}_1(\mathbf{t}))))
\end{equation}
$$


$$
\text{Seg}([\text{[CLS]},\mathbf{t}_1, \text{[SEP]}, \mathbf{t}_2, \text{[SEP]}]) = \mathbf{s} \in \mathbb{R}^{L_1+L_2+1+2}
$$

$$
\begin{equation}
s_i = \begin{cases}
\mathbf{0}, & \text{jika } i \leq L_1 + 2 \\
\mathbf{1}, & \text{jika } i > L_1 + 2
\end{cases}
\end{equation}
$$

$\mathbf{t} = [\text{[CLS]}, t_{i_1}, t_{i_2}, \dots, t_{i_L} ]$


bert in NSP

$$
p(y|\mathbf{t}_1, \mathbf{t}_2) = \text{Ber}(y|\sigma(\text{BERT}([\mathbf{t}_1, \mathbf{t}_2])_{\text{[CLS]}} \mathbf{W}^o))
$$

$$
p(y=1|\mathbf{t}_1, \mathbf{t}_2) = \sigma(\text{BERT}([\mathbf{t}_1, \mathbf{t}_2])_{\text{[CLS]}} \mathbf{W}^o)
$$

# BERT buat information retreival

BERTCAT

$$

p(\text{relevan}|q, d) = \text{Ber}(y| \sigma(\text{BERT}([\mathbf{q}, \mathbf{d}])_{\text{[CLS]}} \mathbf{W}^o))
$$

$$
p(\text{relevan} = 1|q, d) = \sigma(\text{BERT}([\mathbf{q}, \mathbf{d}])_{\text{[CLS]}} \mathbf{W}^o)
$$
use this as direct skoring.

$$
\text{BERT}_{\text{DOT}}(q,d) = f_{\text{sim}}\left(\text{BERT}(q)_\text{[CLS]}, \text{BERT}(d)_\text{[CLS]}\right)
$$

kalau pakai dot product

$$
\text{BERT}_{\text{DOT}}(q,d) = \text{BERT}(q)_\text{[CLS]}^{\top} \text{BERT}(d)_\text{[CLS]}
$$
indivdual skor ga ada makna, but it's okeay since we just need to rank them.

## Representasion learning


$$
\mathcal{L}\left(\mathbf{\theta}|x, , x^+, \{x^-_k\}_{k=1}^n\right) = - \log \frac{\exp\left(f_{\text{sim}}\left(\mathbf{e}_x, \mathbf{e}_{x^+}\right)\right)}{\exp\left(f_{\text{sim}}\left(\mathbf{e}_x, \mathbf{e}_{x^+}\right)\right) + \sum_{k=1}^{n} \exp\left(f_{\text{sim}}\left(\mathbf{e}_x, \mathbf{e}_{x^-_k}\right)\right)}
$$

if f_sim is dot product, 
$$
\mathcal{L}\left(\mathbf{\theta}|x, , x^+, \{x^-_k\}_{k=1}^n\right) = - \log \frac{\exp\left(\mathbf{e}_x^{\top} \mathbf{e}_{x^+}\right)}{\exp\left(\mathbf{e}_x^{\top} \mathbf{e}_{x^+}\right) + \sum_{k=1}^{n} \exp\left(\mathbf{e}_x^{\top} \mathbf{e}_{x^-_k}\right)}
$$


