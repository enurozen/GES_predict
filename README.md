# GES Üretim Tahmin Modeli

Güneş Enerjisi Santrali (GES) için hibrit (fiziksel + ML) saatlik üretim
tahmin modeli, ve bu modeli beslemek için EPİAŞ üretim verisi + Open-Meteo
hava durumu verisini birleştirip eğitim seti üreten bir veri pipeline'ı.

## İçerik

- `ges_uretim_tahmini.py` — fiziksel baz model + Random Forest residual
  düzeltmesiyle çalışan hibrit üretim tahmin modeli.
- `epias.py` — EPİAŞ Transparency Platform client'ı: `get_tgt` (login) ve
  `fetch_generation_range` (saatlik üretim verisi).
- `weather.py` — Open-Meteo historical weather API client'ı:
  `fetch_weather_range` (saatlik GHI, sıcaklık, bulut kapanımı; API key
  gerektirmez).
- `merge.py` — `build_training_dataset`: üretim ve hava verisini timestamp
  üzerinden inner join ile birleştirir.
- `shared.py` — HTTP çağrıları için ortak retry/backoff yardımcı fonksiyonu.
- `main.py` — bu adımları uçtan uca çalıştıran CLI.
- `.github/workflows/daily_datapull.yml` — günlük otomatik veri çekme
  workflow'u.
- `data/<plant_id>/<tarih>.csv` — `daily_datapull.yml`'in her gün ürettiği,
  santral bazında klasörlenmiş eğitim verisi (bkz. aşağıdaki "Günlük
  otomasyon" bölümü).

## Kurulum

```bash
pip install -r requirements.txt
```

## `main.py` kullanımı

```bash
python main.py \
  --plant-id 2579 \
  --lat 39.9 --lon 32.8 \
  --start 2025-06-01 --end 2025-06-30 \
  --output data/training_set.csv
```

Adımlar: EPİAŞ'a giriş yapıp TGT alır -> `--start`/`--end` aralığı için
saatlik üretim verisi çeker -> aynı aralık için `--lat`/`--lon` konumunun
hava durumu verisini çeker -> ikisini timestamp üzerinden birleştirir ->
sonucu `--output` yoluna CSV olarak yazar. Eksik/uyumsuz saatler
(`merge.py`) uyarı olarak loglanıp atlanır.

Kimlik bilgileri **ortam değişkenlerinden** okunur, komut satırı argümanı
olarak verilmez ve hiçbir yere loglanmaz:

```bash
export EPIAS_USERNAME="you@example.com"
export EPIAS_PASSWORD="your-epias-password"   # EPİAŞ'ın statik API key'i yok, hesap şifresi kullanılır
```

## Testler

```bash
pytest
```

## Günlük otomasyon (`daily_datapull.yml`)

`.github/workflows/daily_datapull.yml`, her gün **UTC 03:00**'te (Türkiye
saatiyle 06:00, önceki günün EPİAŞ verisi netleştikten sonra) otomatik
tetiklenir; ayrıca Actions sekmesinden **manuel** (`workflow_dispatch`) de
çalıştırılabilir. Her çalıştırmada `main.py`'ı `--start`/`--end` = **dünün
tarihi** ile çağırır ve sonucu `data/<PLANT_ID>/<tarih>.csv` olarak
kaydeder (örn. `data/2579/2026-07-19.csv`) — santral bazlı klasörleme,
ileride başka santraller eklendiğinde `data/` tek bir düz klasöre
dolmasın diye.

Santral kimliği/konumu (`PLANT_ID`, `PLANT_LAT`, `PLANT_LON`) workflow
dosyasının en üstünde düz env değişkeni olarak tanımlı — gizli bilgi
değildir, farklı bir santral için doğrudan dosyada güncellenebilir.

### Gerekli GitHub Secrets

Repo ayarlarında **Settings -> Secrets and variables -> Actions -> New
repository secret** ile eklenmeli:

| Secret adı        | Açıklama                                    |
|--------------------|----------------------------------------------|
| `EPIAS_USERNAME`  | EPİAŞ hesap e-postası                         |
| `EPIAS_PASSWORD`  | EPİAŞ hesap şifresi (statik API key yoktur)   |

Workflow dosyasında bu değerler asla düz metin olarak yazılmaz, sadece
`${{ secrets.EPIAS_USERNAME }}` / `${{ secrets.EPIAS_PASSWORD }}` referansı
kullanılır ve job loglarına basılmaz.

### Üretilen veri nasıl saklanıyor: commit vs. artifact

İki yöntem de değerlendirildi; şu an workflow **git-auto-commit-action** ile
CSV'yi doğrudan `data/` klasörüne commit'liyor. Kısa artı/eksileri:

**git-auto-commit-action (şu an aktif olan)**
- ✅ Kalıcı: repo'da tarihsel bir veri arşivi birikir, ileride model
  eğitimi/analiz için doğrudan `git clone` ile veya repo üzerinden okunabilir.
- ✅ Başka bir workflow/script ekstra kimlik doğrulama olmadan (repo
  read erişimiyle) veriye ulaşabilir.
- ❌ Repo zamanla büyür (her gün bir CSV + otomatik commit).
- ❌ `git log`, her gün eklenen otomatik commit'lerle kalabalıklaşır.

**GitHub Actions artifact (alternatif)**
- ✅ Repo'ya commit düşmez, repo boyutu/`git log` temiz kalır.
- ✅ Hassas/geçici veri için daha uygun (repo geçmişinde iz bırakmaz).
- ❌ Varsayılan olarak **90 gün sonra silinir** (retention ayarlanabilir
  ama sınırsız değil) — uzun vadeli arşiv için uygun değil.
- ❌ Veriye erişim `gh run download` / GitHub API üzerinden yapılmalı;
  repo'dan doğrudan `git pull` ile veya dosya olarak okunamaz.

Artifact yöntemine geçmek isterseniz, workflow'daki son adımı
(`git-auto-commit-action`) şununla değiştirmek yeterli:

```yaml
      - name: Upload CSV as artifact
        uses: actions/upload-artifact@v4
        with:
          name: santral-${{ env.PLANT_ID }}-${{ steps.date.outputs.yesterday }}
          path: data/**/*.csv
```
