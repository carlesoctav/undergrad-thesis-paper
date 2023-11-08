# Mekanisme Attention
## Atensi
vektor aktivasi pada lapisan tersembunyi dari model *neural network* biasanya merupakan kombinasi linier dari vektor masukan ditambah dengan fungsi aktivasi tak linier, yaitu $\mathbf{H} = \phi(\mathbf{X}\mathbf{W})$, dengan $\mathbf{X} \in \mathbb{R} ^{N \times D}$ adalah matriks masukan, $\mathbf{W} \in \mathbb{R} ^{D \times H}$ adalah matriks bobot yang sudah tetap dan $\phi$ adalah fungsi aktivasi.

jika bobot matriks pada $\mathbf{W}$ bergantung terhadap vektor masukan $\mathbf{X}$, yaitu $\mathbf{H} = \phi(\mathbf{X}\mathbf{W}(\mathbf{X}))$, $\mathbf{W}(\mathbf{X})$  disebut

## Soft Attention
*Soft attention* menggambarkan rata-rata terbobot dari barisan elemen dengan bobot yang dihitung secara dinamis berdasarkan vektor kueri dan vektor kunci. Tujuannya adalah untuk mengambil rata-rata dari fitur atau nilai dari beberapa elemen dalam barisan tersebut. Namun, daripada memberi bobot setiap elemen secara merata, pemberian bobot dilakukan secara dinamis bergantung pada "nilai" elemen tersebut. Dengan kata lain, *soft attention* secara dinamis memutuskan elemen tertentu yang ingin diperhatikan lebih dari elemen lainnya. Tujuan lainnya digunakan rata-rata terbobot dari vektor nilai daripada hanya melihat satu vektor nilai dengan nilai atensi tertinggi (*hard attention*) adalah memastikan operasi *attention* terturunkan (todocite), sehingga memudahkan proses pelatihan.

biasanya operasi *attention* terdiri dari empat komponen utama, yaitu:

1. vektor kueri ($\mathbf{q}$). Vektor kueri yang merepresentasikan hal yang ingin dicari dalam barisan elemen.

2. vektor kunci ($\mathbf{k}$). Untuk setiap elemen dalam barisan, terdapat vektor kunci. Vektor kunci merepresentasikan apa yang ditawarkan elemen tersebut, atau kapan elemen tersebut menjadi penting.Vektor kunci harus dirancang sedemikian rupa sehingga mekanisme \f{attention} dapat mengidentifikasi elemen yang ingin diperhatikan berdasarkan vektor kueri-nya.

3. vektor nilai ($\mathbf{v}$). Untuk setiap elemen dalam barisan, terdapat vektor nilai. Vektor nilai merupakan fitur yang ingin diambil rata-rata terbobotnya.

4. fungsi skor ($f_{\text{attn}}(\mathbf{q}, \mathbf{k}))$. Fungsi skor  kunci digunakan untuk menghitung similaritas antara vektor $\mathbf{q}$ dengan vektor $\mathbf{k}$. Keluaran dari fungsi skor disebut sebagai nilai atensi. Fungsi skor biasanya dihitung dengan metrik-metrik yang menunjukkan similaritas, seperti perkalian skalar atau jarak kosinus. Selain itu, fungsi skor juga dapat menggunakan \f{neural network} untuk menghitung nilai atensi.

Hasil fungsi skor akan diterapkan fungsi \f{softmax} untuk mendapatkan bobot yang dinormalisasi. Bobot tersebut kemudian digunakan untuk menghitung rata-rata terbobot dari nilai. 






