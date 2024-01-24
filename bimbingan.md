
21/09 confirm menjelaskan experimentasi awal yang sudah dilakukan. Hasilnya masih jelek karena model dilatih pada permasalahan yang berbeda (dilatih pada natural language inference (NLI) daripada pemeringkatan teks), Hal ini juga menconfirm lebih baik melatih model langsung pada permasalahan dan dibandingkan dengan permasalahan lain (yang dicoba untuk pemeringkatan teks) dan pilih suatu model baseline, yaitu BM25, sebagai pembanding kemampuan model yang dihasilkan.


12/10 confirm, masih mencoba experimentasi dari kemampuan model-model dan mengerjakan programnya. Pada kali ini sudah ada experimentasi yang berhasil, yaitu IndoBERTKD (performa masih buruk karena permasalahan konfigurasi pelatihan), tetapi kode model lain masih belum dibuat dengan baik.
(log experimentasi dari tanggal 14 hingga 12 dapat dilihat pada https://github.com/carlesoctav/undergrad-thesis/tree/gpu-model)



26/10 Menjelaskan konfigurasi dan performa dari keempat model yang sudah dikerjakan, yaitu IndoBERTCAT, IndoBERTDOT, IndoBERTDOTHardnegs, IndoBERTDOTKD. Setiap model dibandingkan performanya dengan model baseline yaitu BM25 (eksperimentasi dapat dilihat pada repo berikut: https://github.com/carlesoctav/beir-skripsi), ya, eksperimen dikerjakan terlebih dahulu dibandingkan buku skripsi.



9/11 Menjelaskan kepada Bu sarini mengenai bab 3. saya menjelaskan kepada bu sarini mengenai transformers dan BERT yang digunakan sebagai arsitektur model-model pada skripsi saya.

16/11 Menjelaskan mengenai teori dibalik pemeringkatan teks yaitu Bab 2 dari skripsi saya, seperti bentuk umum dataset, metrik evaluasi, pemeringakatan teks dengan statistik, dan deep learning.


30/11 mengupdate tulisan saya pada bab 3, membuang beberapa subbab yang terlalu detail dan tidak terlalu penting, dan menjelaskan kembali teori dari attention dan transformers, dan BERT.



14/12 memberikan update mengenai tulisan pada bab 4, ada beberapa hal yang diperbaiki sehingga model-model yang dihasilkan dapat dilakukan komparasi secara fair (walaupun tidak 100% fair). Menjelaskan juga diskusi-diskusi yang ingin saya sampaikan nanti pada saat persidangan.



27/12 Memberikan update mengenai tulisan pada bab4, ada beberapa hal yang diperbaiki sehingga model-model yang dihasilkan dapat dilakukan komparasi secara fair. perbedaan dengan pertemuan sebelumnya diantara lain:
    1. Kosistenin ukuran maksimal token tiap model, train ulang +eval ulang model bertcat, sama bertdot (dari 512 token ke 256 token) sehingga setiap model jauh dapat lebih dilakukan komparasi satu sama lain.

    R= recall
    2. Use R@100 instead of R@1000 (karena jauh lebih praktikal) add explaination why miracl use NDCG@10 dari pada RR@10.

    3. Add section about statistik dataset statistik panjang token rata2 dan justifikasi kenapa panjang token maksimal adalah 256

    4. New lampiran, ada eval setiap model ke setiap dataset di setiap metriknya 



15/01 Pertemuan setelah sidang. Membahasa revisi skripsi dan inisialisasi penulisan untuk sebuah jurnal.



