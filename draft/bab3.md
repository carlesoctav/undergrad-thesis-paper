# transformers
<!-- verdict good -->
\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{assets/pics/transformers_seq.png}
	\caption{Arsitektur \f{transformer} untuk mesin translasi neural. Arsitektur terdiri dari \f{encoder} dan \f{decoder} yang terdiri dari beberapa blok \f{transformer} \citep{transformerori}.}
	\label{fig:transformer}
\end{figure}

\f{Transformers} merupakan Arsitektur \f{deep learning} yang pertama kali diperkenalkan oleh \cite{transformerori}. Awalnya Transformers merupakan model \f{sequance to sequance} yang diperuntukkan untuk permasalahan mesin translasi neural (\f{neural machine translation}). Namun, sekarang \f{transformers} juga digunakan untuk permasalahan pemrosesan bahasa alami lainnya. model-model yang menjadi \f{state of the art} permasalahan pemrosesan bahasa alami biasanya menggunakan arsitektur \f{transformers}.

Berbeda dengan arsitektur mesin translasi terdahulu \todoCite{a}, transformers tidak mengunakan \f{recurrent neural network} (RNN) atau \f{convolutional neural network} (CNN), melainkan transformers adalah model \f{feed foward network} yang dapat memproses seluruh \f{input} pada barisan secara paralel. Untuk menggantikan kemampuan RNN dalam mempelajari ketergantungan antar \f{input} yang berurutan dan kemampuan CNN dalam mempelajari fitur lokal, transformers bergantung pada mekanisme \f{attention}.

Terdapat tiga jenis \f{attention} yang digunakan dalam model \f{transformers} \citep{transformerori}:
1. \f{Encoder self-attention} menggunakan barisan \f{input} yang berupa barisan token atau kata sebagai masukan untuk menghasilkan barisan representasi kontekstual, berupa vektor, dari \f{input}. Setiap representasi token tersebut memiliki ketergantungan dengan token lainnya dari barisan \f{input}.

2. \f{Decoder self-attention} menggunakan barisan \f{target} yang berupa kalimat terjemahan parsial, barisan token, sebagai masukan untuk menghasilkan barisan representasi kontekstual (vektor) dari \f{target}. Setiap representasi token tersebut memiliki ketergantungan dengan token sebelumnya dalam urutan masukan.

3. \f{Decoder-encoder attention} menggunakan barisan representasi kontekstual dari \f{input}, dan barisan representasi kontekstual dari \f{target} untuk menghasilkan token berikutnya yang merupakan hasil prediksi dari model. barisan \f{target} yang digabung dengan token hasil prediksi tersebut akan menjadi barisan \f{target} untuk prediksi selanjutnya.



# ATTENTION
<!-- verdict good -->

Mekanisme \f{attention} menggambarkan rata-rata terbobot dari barisan elemen dengan bobot yang dihitung secara dinamis berdasarkan kueri dan kunci elemen. Tujuannya adalah untuk mengambil rata-rata dari fitur atau nilai dari beberapa elemen dalam barisan tersebut. Namun, daripada memberi bobot setiap elemen secara merata, pemberian bobot dilakukan secara dinamis bergantung pada "nilai" elemen tersebut. Dengan kata lain, mekanisme \f{attention} secara dinamis memutuskan elemen tertentu yang ingin diperhatikan lebih dari elemen lainnya. Mekanisme \f{attention} biasanya memiliki empat bagian yang perlu ditentukan:

1. vektor kueri ($\mathbf{q}$). Vektor kueri yang merepresentasikan hal yang ingin dicari dalam barisan elemen.

2. vektor kunci ($\mathbf{k}$). Untuk setiap elemen dalam barisan, terdapat vektor kunci. Vektor kunci merepresentasikan apa yang ditawarkan elemen tersebut, atau kapan elemen tersebut menjadi penting.Vektor kunci harus dirancang sedemikian rupa sehingga mekanisme \f{attention} dapat mengidentifikasi elemen yang ingin diperhatikan berdasarkan vektor kueri-nya.

3. vektor nilai ($\mathbf{v}$). Untuk setiap elemen dalam barisan, terdapat vektor nilai. Vektor nilai merupakan fitur yang ingin diambil rata-rata terbobotnya.

4. fungsi skor ($f_{\text{attn}}(\mathbf{q}, \mathbf{k}))$. Fungsi skor memberikan bobot-bobot pada pada nilai berdasarkan kueri dan kunci. Keluaran dari fungsi skor disebut sebagai nilai atensi. Fungsi skor biasanya dihitung dengan metrik-metrik yang menunjukkan similaritas, seperti perkalian skalar atau jarak kosinus. Selain itu, fungsi skor juga dapat menggunakan \f{neural network} untuk menghitung nilai atensi.

Biasanya, hasil fungsi skor diterapkan pada fungsi \f{softmax} untuk mendapatkan bobot yang dinormalisasi. Bobot tersebut kemudian digunakan untuk menghitung rata-rata terbobot dari nilai. 

$$
\begin{aligned}
\alpha_i &= \frac{\exp(f_{\text{attn}}(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}\exp(f_{\text{attn}}(\mathbf{q}, \mathbf{k}_j))} \\

\text{Output} &= \sum_{i=1} \alpha_i \mathbf{v}_i
\end{aligned}
$$

# scaled dot product attention
<!-- verdict bad -->

$$
\mathbf{X} = \begin{smallmatrix}
\mathbf{x}_1 \\
\vdots \\
\mathbf{x}_n
\end{smallmatrix}
\in \mathbb{R}^{n \times d_k}
$$

Mekanimse \f{attention} yang diterapkan pada transformer disebut sebagai \f{scaled dot product attention}.
\f{Scaled dot product attention} menggunakan fungsi skor berupa hasil kali skalar antara kueri dan kunci yang dibagi dengan $\sqrt{d_k}$, dimana $d_k$ merupakan dimensi dari vektor kunci. Untuk Kumpulan vektor kueri $\mathbf{Q}= 
\begin{smallmatrix}
\mathbf{q}_1 \\
\vdots \\
\mathbf{q}_m
\end{smallmatrix}
\in \mathbb{R}^{m \times d_k}$ dan kumpulan vektor kunci $K =
\begin{smallmatrix}
\mathbf{k}_1 \\
\vdots \\
\mathbf{k}_T
\end{smallmatrix}
\in \mathbb{R}^{n \times d_k}$,
$
dengan $n,m,d_k$ merupakan banyaknya elemen dalam barisan, banyaknya kueri, dan dimensi dari vektor kunci secara berturut-turut, \f{Scaled dot product attention} dapat dihitung sebagai berikut:
j
$$
\begin{equation}
\mathbf{Y} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
\end{equation}
$$

Perkalian matriks $\mathbf{Q}\mathbf{K}^T$ menghasilkan matriks $\mathbf{A} \in \mathbb{R}^{m \times n}$, dimana elemen $a_{ij}$ merupakan hasil perkalian skalar antara $\mathbf{q}_i$ dan $\mathbf{k}_j$. Kemudian, matriks $\mathbf{C}$ dibagi dengan $\sqrt{d_k}$ dan diaplikasikan fungsi \f{softmax} untuk mendapatkan bobot $\alpha_{ij}$ yang dinormalisasi. Bobot tersebut kemudian digunakan untuk menghitung rata-rata terbobot dari nilai $\mathbf{V}$.

Pembagian dengan $\sqrt{d_k}$ dilakukan untuk menjaga variansi dari nilai atensi tetap \f{appropritate}.
pada subab 2.3 telah dijelaskan bahwa model \f{deep learning} yang baik harus memiliki variansi yang seragam untuk setiap keluaran dari \f{hidden layer}.
Perkalian skalar antara $\mathbf{q}$ dan $\mathbf{k}$ pada dua buah vektor dengan variansi $\sigma^2$ menghasilkan variansi yang dengan faktor $d_k$ seperti yang ditunjukkan pada persamaan xx.
$$
\begin{equation}
q_i \sim \mathcal{N}\left(0, \sigma^2\right), k_i \sim \mathcal{N}\left(0, \sigma^2\right) \rightarrow \operatorname{Var}\left(\sum_{i=1}^{d_k} q_i \cdot k_i\right)=\sigma^2 (\sigma^2) (d_k) . \quad q_i, k_i \text{ independen}
\end{equation}
$$

faktor $d_k$ pada variansi menyebabkan nilai atensi menjadi sangat besar atau sangat kecil. fungsi \f{softmax} yang digunakan untuk mendapatkan bobot yang dinormalisasi akan menghasilkan nilai yang dekat ke 0 di satu sisi dan dekat ke 1 di sisi lainnya. Hal ini menyebabkan gradien ketika proses pelatihan akan dekat ke 0. Akibatnya model tidak dapat belajar dengan baik.


faktor tambahan $\sigma^2$ pada persamaan xx tidak menjadi masalah karena biasanya, variansi dari $\mathbf{q}$ dan $\mathbf{k}$ mendekati 1, $\sigma^2 \approx 1$.


# Multi-Head Attention

\f{Multi-Head Attention} adalah arsitektur yang melakukan mekanisme \f{attention} sebanyak $h$ kali. Biasanya, terdapat beberapa aspek pada barisan elemen yang ingin diberikan atensi. Satu buah rata-rata terbobot tidak dapat \f{mengcapture} semua aspek tersebut. 
Oleh karena itu, \f{multi-head attention} melakukan mekanisme \f{attention} sebanyak $h$ kali dengan menggunakan $h$ buah vektor kueri, $h$ buah vektor kunci, dan $h$ buah vektor nilai yang berbeda. Untuk mendapatkan vektor kueri, kunci, value yang berbeda, petakan $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ ke dalam $h$ buah ruang fitur yang berbeda dengan menggunakan matriks bobot $\mathbf{W}_i^Q, \mathbf{W}_i^K, \in \mathbb{R}^{d_{\text{model}} \times d_k}$, dan $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ untuk $i=1,\dots,h$, dengan $d_{\text{model}}$ merupakan banyaknya elemen dalam barisan, $d_k$ merupakan dimensi dari vektor kunci, dan $d_v$ merupakan dimensi dari vektor nilai. Kemudian, setiap hasil mekanisme \f{attention} akan digabungkan menjadi satu barisan dengan menggunakan matriks bobot $\mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$. persamaan xx menunjukkan bagaimana \f{multi-head attention} dihitung.

$$
\begin{align}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O, \\
\text{dengan} \quad \text{head}_i &= \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V), \\
\end{align} \\
\mathbf{W}_i^Q, \mathbf{W}_i^K, \in \mathbb{R}^{d_{\text{x}} \times d_k}, \quad \mathbf{W}_i^V \in \mathbb{R}^{d_{\text{x}} \times d_v}, \quad \mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{x}}}
$$






# Positin-wise Feed Forward Network
\f{Position-wise feed forward network} (FFN)adalah \f{feed forward network} yang terdiri dari dua \f{layer linear} dengan fungsi aktivasi \f{ReLU} diantara kedua lapisan tersebut. Persamaan xx menunjukkan bagaimana \f{position-wise feed forward network} dihitung.





$$
\begin{align}
\text{FFN}(\mathbf{X}) &= \max(0, \mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 \\
\end{align}
$$



# Self-attention
\f{self-attention} adalah mekanisme \f{attention} dimana kueri, kunci, dan nilai berasal dari barisan yang sama.Salah satu penerapan \f{self-attention} yang mudah adalah dengan membiarkan $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ sama dengan $\mathbf{X}$,  dimana $\mathbf{X}$ merupakan barisan elemen masukan. Dengan demikian, \f{self-attention} dapat dihitung sebagai berikut:
$$
\begin{align}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O, \\
\text{dengan} \quad \text{head}_i &= \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_i^K, \mathbf{X}\mathbf{W}_i^V), \\
\end{align} \\
$$


# Positional Encoding

Salah satu permasalahan pengunaan mekanisme \f{attention} untuk pemodelan bahasa adalah mekanisme \f{attention} bersifat permutasi equivarian, artinya, jika kita mengubah urutan dari elemen dalam barisan, maka hasil dari mekanisme \f{attention} tidak akan berubah. Dengan kata lain, mekanisme \f{attention} bekerja pada himpunan $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$, bukan pada barisan $[\mathbf{x}_1, \dots, \mathbf{x}_n]$. Hal ini menyebabkan model tidak dapat mempelajari pentingnya urutan posisi.

Untuk mengatasi masalah tersebut, \cite{transformerori} menambahkan informasi posisi pada barisan elemen dengan menambahkan vektor posisi pada vektor representasi elemen. Vektor posisi,$\textbf{p} \in \mathbb{R}^{d_{input}}$ tersebut disebut sebagai \f{positional encoding}. Persamaan xx menunjukkan bagaimana \f{positional encoding} dihitung.


$$
\text{PE}_{ \text {pos }, i}= \begin{cases}\sin \left(\frac{p o s}{10000^{i / d_{\text {model }}}}\right) & \text {jika } i \bmod 2=0 ,\\ \cos \left(\frac{p o s}{10000^{(i-1) / d_{\text {model }}}}\right) & \text { lainnya. }\end{cases}
$$

dimana $pos \leq d_{\text{model}}$ merupakan posisi dari elemen dalam barisan, $i\leq d_x$ merupakan indeks dari vektor posisi, dan $d_{\text{model}}$ merupakan banyaknya elemen dalam barisan.

Jika $\mathbf{X} \in \mathbb{R}^{d_{\text{model}}\times d_x}$ merupakan \f{word embedding} dari barisan elemen masukan, barisan baru yang telah ditambahkan \f{positional encoding} dapat dihitung sebagai berikut:
$$
\begin{equation}
\mathbf{X'} = \mathbf{X} + \text{PE}
\end{equation}
$$

## Embedding Token




