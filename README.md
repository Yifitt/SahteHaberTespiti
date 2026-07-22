# Sahte Haber Tespiti

Bu proje, Türkçe haber metinleri üzerinde veri analizi ve sahte haber/yanlış iddia sınıflandırması yapmak amacıyla hazırlanmıştır. Çalışma, İstinye Üniversitesi Yaz Araştırma Stajı kapsamında yapay zekâ destekli siber güvenlik alanındaki uygulamalar için geliştirilmiştir.

## İçerik

- `eda.ipynb`: Haber verisi üzerinde keşifsel veri analizi ve metin incelemeleri.
- `training_evaluation.ipynb`: ANKA haberleri ile DMM tarafından yanlış olarak etiketlenen iddiaları ayırmak için TF-IDF ve Logistic Regression modeli eğitimi ve değerlendirmesi.

Model, haberlerin doğruluğunu dış kaynaklardan doğrulamaz; kullanılan veri kümelerindeki metinsel örüntüler üzerinden sınıflandırma yapar.

## Veri kümeleri

- [Dezenformasyon Bültenleri](https://huggingface.co/datasets/iletisim/dezenformasyon-bultenleri)
- [ANKA Ajansı Haberleri](https://huggingface.co/datasets/momererkoc/anka_ajansi_haberler)
