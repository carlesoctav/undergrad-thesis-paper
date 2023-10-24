1. review apa yang ingin saya lakukin

2. information retreival, banyak metode training/dataset. abis itu bakal coba di dataset mmarco

3.  indobert-vanilla, tanpa training, gabsa dipakai sama sekali untuk mmarco
4.  indobert-snli, di training di snli, terus di test di mmarco (totally no improvement).
5.  indobert-stsb, sama kyk diatas harusnya, almost no improvement di information retrieval.

6.  indo-mmarco seharusnya ada improvement.

7. knowledge distillation

7. kenapa 4-5 ga ada improvement? karena apa yang di training berbeda sama apa yang di test (information retrieveal)

8. belum coba data yang di training pakai dataset mmarco (kmren udah coba cuman saya force karena kelamaan :D, saya bakal coba reduce datasetnya).

9.  saya udah coba yang knowledge distillation, hasilnya lumayan :D

MODEL MACHINE LEARNING -> VECTOR yang bagus.
