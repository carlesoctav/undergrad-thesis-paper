Pada tanggal 14 September, pertemuan awal telah dilaksanakan untuk menjelaskan perubahan yang akan diimplementasikan dibandingkan dengan proposal skripsi yang telah disusun sebelumnya. Perubahan tersebut disebabkan oleh ketidakjelasan tujuan skripsi, sebagaimana dicatat oleh dosen penguji pada saat penilaian proposal skripsi. Perubahan utama yang dilakukan melibatkan pemilihan dataset yang lebih terbatas dan difokuskan pada permasalahan awal, yaitu pemeringkatan teks. Selain itu, penelitian akan lebih berfokus pada kemampuan model dalam melakukan pemeringkatan teks, daripada kemampuan model dalam menghasilkan representasi vektor yang optimal. Hal ini dapat dibandingkan dengan judul skripsi dan judul proposal skripsi yang telah diajukan sebelumnya.



Pada tanggal 21 September, penjelasan mengenai eksperimen awal yang telah dilakukan telah disampaikan. Hasil eksperimen tersebut menunjukkan performa yang masih kurang memuaskan, disebabkan oleh fakta bahwa model telah dilatih pada permasalahan yang berbeda, yakni natural language inference (NLI) daripada fokus pada pemeringkatan teks. Temuan ini memperkuat keyakinan bahwa lebih optimal untuk melatih model secara langsung pada permasalahan yang bersangkutan, dibandingkan dengan menggunakan model yang telah dilatih pada permasalahan lain, meskipun terkait dengan pemeringkatan teks.

Dalam rangka mengukur kemampuan model yang dihasilkan, diputuskan untuk memilih suatu model baseline, yaitu BM25, sebagai perbandingan. Langkah ini diambil untuk mengevaluasi sejauh mana model yang dikembangkan mampu bersaing atau bahkan melampaui kinerja model baseline dalam konteks pemeringkatan teks. Dengan demikian, penelitian ini menekankan pentingnya pendekatan langsung pada permasalahan yang ingin diselesaikan, serta adopsi model perbandingan yang relevan.




Pada tanggal 12 Oktober, konfirmasi dilakukan mengenai kelanjutan eksperimentasi, yang masih berfokus pada penilaian kemampuan model-model yang sedang dikembangkan. Selain itu, proses pembuatan program (kode) untuk eksperimentasi juga sedang dikerjakan. Dalam update ini, terdapat eksperimentasi yang berhasil dilakukan dengan model IndoBERTKD, meskipun performanya masih menunjukkan hasil yang kurang memuaskan akibat permasalahan konfigurasi pelatihan.

Detail lebih lanjut mengenai log eksperimentasi dari tanggal 14 hingga 12 dapat ditemukan pada laman GitHub berikut (https://github.com/carlesoctav/undergrad-thesis/tree/gpu-model).



Pada tanggal 26 Oktober, penjelasan diberikan mengenai konfigurasi dan performa keempat model yang telah berhasil dikembangkan, yakni IndoBERTCAT, IndoBERTDOT, IndoBERTDOTHardnegs, dan IndoBERTDOTKD. Keempat model tersebut telah melalui tahap eksperimentasi, dan hasilnya dibandingkan dengan model baseline, yaitu BM25. Perlu diperhatikan bahwa eksperimentasi ini telah dilakukan terlebih dahulu sebelum buku skripsi dikerjakan karena menulis bukan passion saya.

Hasil performa dari keempat model di atas akan dibandingkan dengan model baseline BM25. Eksperimentasi ini dapat diakses pada repositori berikut: [https://github.com/carlesoctav/beir-skripsi](https://github.com/carlesoctav/beir-skripsi). Semua eksperimen ini dilakukan sebelum implementasi buku skripsi untuk memastikan kehandalan dan validitas hasil penelitian.




**9 November:**
Pada tanggal 9 November, saya menjelaskan kepada Ibu Sarini mengenai Bab 3 skripsi saya. Fokus pembahasan pada bab tersebut adalah menjelaskan tentang transformers dan BERT, yang digunakan sebagai arsitektur model-model dalam penelitian saya. Saya memaparkan secara rinci konsep-konsep dasar yang melibatkan transformers dan bagaimana BERT, sebagai implementasi spesifik dari model tersebut, bekerja dalam konteks penelitian saya, yaitu pemeringakatan teks.

**16 November:**
Pada tanggal 16 November, penjelasan saya terfokus pada Bab 2 dari skripsi, yang membahas teori di balik pemeringkatan teks. Saya merincikan bentuk umum dataset yang digunakan, metrik evaluasi yang relevan, serta pendekatan pemeringkatan teks menggunakan statistik (BM25) dan menjelaskan konsep deep learning.

**30 November:**
Pada tanggal 30 November, saya melakukan pembaruan pada tulisan saya dalam Bab 3. Proses ini melibatkan penghapusan beberapa subbab yang dianggap terlalu detail dan kurang penting. Selain itu, saya menjelaskan kembali teori dari attention, transformers, dan BERT dengan lebih fokus.


Pada tanggal 14 Desember, saya memberikan update mengenai tulisan pada Bab 4 skripsi saya dengan beberapa perbaikan untuk memastikan komparasi model dapat dilakukan secara adil, meskipun tidak sepenuhnya 100% fair. 

pada tanggal 27 Desember, dilakukan update tambahan sebagai berikut:

1. Menjaga konsistensi ukuran maksimal token pada setiap model. Dilakukan pelatihan ulang dan evaluasi ulang pada model IndoBERTCAT dan IndoBERTDOT dengan mengurangi ukuran token dari 512 menjadi 256. Hal ini bertujuan untuk memastikan bahwa setiap model dapat dibandingkan secara lebih akurat satu sama lain.

2. Mengganti penggunaan Recall@1000 dengan Recall@100. Penjelasan juga ditambahkan mengapa MIRACL menggunakan NDCG@10 daripada RR@10, memberikan landasan praktis dan teoretis bagi pilihan tersebut.

3. Menambahkan bagian baru yang membahas statistik dataset, termasuk panjang rata-rata token, dan justifikasi mengapa panjang token maksimal ditetapkan pada 256. Ini dimaksudkan untuk memberikan pemahaman yang lebih baik terkait karakteristik dataset yang digunakan.

4. Menambahkan lampiran baru yang berisi evaluasi setiap model terhadap setiap dataset untuk setiap metrik yang digunakan. Ini memberikan transparansi dan kemudahan akses terhadap hasil evaluasi yang mendalam.

Semua perbaikan ini dilakukan guna memastikan keakuratan dan konsistensi hasil penelitian. 