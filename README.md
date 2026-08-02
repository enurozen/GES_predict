# GES Üretim Tahmin Modeli

Güneş Enerjisi Santrali (GES) için hibrit (fiziksel + ML) saatlik üretim
tahmin modeli, ve bu modeli beslemek için EPİAŞ üretim verisi + Open-Meteo
hava durumu verisini birleştirip eğitim seti üreten bir veri pipeline'ı.

## İçerik

- `ges_uretim_tahmini.py` — fiziksel baz model + Random Forest residual
  düzeltmesiyle çalışan hibrit üretim tahmin modeli. Santral geometrisi
  (`tilt_deg`/`tracker_type`) `plants.yaml`'da tanımlıysa DNI/DHI'den
  plane-of-array (POA) ışınımı ve NOCT tabanlı gerçek hücre sıcaklığı
  hesaplar (bkz. aşağıdaki "Panel geometrisi" bölümü); tanımlı değilse eski
  düz-GHI davranışına döner.
- `epias.py` — EPİAŞ Transparency Platform client'ı: `get_tgt` (login) ve
  `fetch_generation_range` (saatlik üretim verisi).
- `weather.py` — Open-Meteo historical weather API client'ı:
  `fetch_weather_range` (saatlik GHI, DNI, DHI, sıcaklık, bulut kapanımı;
  API key gerektirmez).
- `merge.py` — `build_training_dataset`: üretim ve hava verisini timestamp
  üzerinden inner join ile birleştirir.
- `shared.py` — HTTP çağrıları için ortak retry/backoff yardımcı fonksiyonu.
- `plants.yaml` / `plants.py` — santral kayıt defteri: her `plant_id` için
  `lat`/`lon`/`capacity_mw` tek bir yerde tanımlı, `main.py` ve workflow
  buradan okur.
- `main.py` — bu adımları uçtan uca çalıştıran, tek bir CSV üreten CLI.
- `backfill.py` — geçmişe dönük geniş bir tarih aralığını tek seferde
  çekip `data/<plant_id>/<tarih>.csv` yapısına günlük dosyalar halinde
  yazan CLI (bkz. aşağıdaki "Geçmişe dönük veri (backfill)" bölümü).
- `.github/workflows/daily_datapull.yml` — günlük otomatik veri çekme
  workflow'u.
- `.github/workflows/backfill_datapull.yml` — manuel tetiklenen, geçmişe
  dönük toplu veri çekme workflow'u.
- `data/<plant_id>/<tarih>.csv` — `daily_datapull.yml` ve `backfill.py`'ın
  ürettiği, santral bazında klasörlenmiş eğitim verisi.
- `train_fleet.py` — birden fazla santralin verisini havuzlayıp TEK bir
  residual model eğiten CLI; bugün tek santral kayıtlıysa da çalışır, ikinci
  bir santral eklendiğinde otomatik devreye girer (bkz. aşağıdaki "Çoklu-
  santral (fleet) eğitimi" bölümü).
- `imbalance_cost.py` — EPİAŞ dengesizlik maliyetine duyarlı değerlendirme:
  asimetrik ceza biliniyorsa optimal bid kantilini hesaplayan
  `optimal_bid_quantile` (saf matematik, veri gerektirmez) ve gerçek saatlik
  dengesizlik fiyatıyla bir tahminin maliyetini hesaplayan `imbalance_cost`/
  `evaluate_financial`.
- `reforecast_check.py` — iki `predict.py` çalıştırmasını (gün-öncesi bid
  vs. daha sonra üretilen "refresh" tahmin) karşılaştırıp sapma
  belirlenen bir eşiği aşan saatleri işaretler; gün-içi piyasasında pozisyon
  düzeltmesi değerlendirilmesi için insan karar vericiye sinyal — gerçek
  emir gönderimi (trading) KAPSAM DIŞI, bilinçli olarak implement edilmedi.

## Kurulum

```bash
pip install -r requirements.txt
```

## Santral kayıt defteri (`plants.yaml`)

`main.py`'a `--plant-id` dışında `--lat`/`--lon` girilmez; bu değerler
`plants.yaml`'dan okunur, tek doğruluk kaynağı orasıdır (bkz.
[plants.yaml](plants.yaml)). Yeni bir santral eklemek için o dosyaya bir
girdi daha eklemek yeterli:

```yaml
2579:
  name: "Karapınar GES"
  lat: 39.9
  lon: 32.8
  capacity_mw: 10.0
```

`--plant-id` değeri `plants.yaml`'da yoksa `main.py` net bir hata verip
çıkar (yanlış/eksik konumla sessizce çalışmaz).

### Panel geometrisi (opsiyonel, daha isabetli fiziksel model için)

Aşağıdaki alanlar opsiyoneldir; bilinmiyorsa **eklemeyin/`null` bırakın** -
uydurma değer girmeyin, kod bu durumda otomatik olarak düz-GHI (horizontal)
baz modele döner:

```yaml
2579:
  name: "Karapınar GES"
  lat: 39.9
  lon: 32.8
  capacity_mw: 10.0
  dc_capacity_mw: 13.5        # panel (DC) nameplate kapasitesi, sadece bilgi amaçlı
  tilt_deg: 25.0               # panel eğim açısı, derece (0=yatay)
  azimuth_deg: 0.0             # panelin baktığı yön (Güney=0, Batı=+90, Doğu=-90)
  tracker_type: null           # null | "single_axis_horizontal"
  module_noct_c: 45.0          # datasheet NOCT değeri (yoksa varsayılan 45.0)
```

`tilt_deg` veya `tracker_type` verilirse `train.py`/`evaluate.py`/
`predict.py`/`report.py`/`backtest_forecast.py` DNI/DHI'den plane-of-array
(POA) ışınımı hesaplayıp kullanır (izotropik gökyüzü modeli); DNI/DHI
eksikse (örn. bu alanlar eklenmeden önce çekilmiş eski `data/` CSV'leri)
GHI'den Erbs ayrıştırmasıyla tahmin edilir. `tracker_type:
"single_axis_horizontal"` sadece en yaygın konfigürasyonu (yatay,
Kuzey-Güney eksenli tek-eksen tracker) destekler.

### Ensemble hava kaynağı

```bash
python predict.py --plant-id 2579 --start 2026-07-22 --end 2026-07-24 \
  --model models/2579/model.joblib --output predictions/2579.csv --ensemble
```

`--ensemble`, Open-Meteo'nun `models` parametresiyle aynı anda birden fazla
NWP modelinden (varsayılan: ECMWF, GFS, ICON) tahmin çekip ORTALAMASINI
kullanır (`weather.fetch_weather_forecast_ensemble`) - tek modele bağımlı
kalmaktan kaynaklanan sistematik hatayı azaltır, ek kimlik bilgisi/ücret
gerektirmez.

### Çoklu-santral (fleet) eğitimi

```bash
python train_fleet.py --plant-ids 2579,1234 --output models/fleet/model.joblib
```

`train.py`'ın tek-santral eğitimine alternatif: `build_features`'ın ürettiği
özellikler (clear_sky_index, solar_elevation_deg) kapasiteden/coğrafyadan
bağımsız olduğu için, birden fazla santralin verisi TEK bir residual modelde
havuzlanabilir - özellikle az geçmiş verisi olan yeni bir santral için daha
kararlı bir model. Fiziksel kalibrasyon (`efficiency_scale`/`temp_coeff`)
santrale özgü kalır, sadece ML katmanı havuzlanır. Bugün tek santral
kayıtlıysa da çalışır ("fleet of one"); ikinci bir `plant_id` eklendiğinde
kod değişikliği gerekmeden pooling devreye girer. Tahmin tarafı için bkz.
`ges_uretim_tahmini.predict_production_from_fleet` (henüz `predict.py`'a
bağlanmadı - farklı bundle formatı, ayrı bir CLI gerektirir).

### Dengesizlik-maliyet duyarlı değerlendirme

`imbalance_cost.py`, `evaluate.py`'nin simetrik MAE/RMSE'sinden farklı
olarak EPİAŞ dengesizlik cezasının asimetrik olabileceğini hesaba katar:

- `optimal_bid_quantile(cost_under_mwh, cost_over_mwh)` — saf matematik,
  veri gerektirmez, bugün kullanılabilir: taahhüdün altında/üstünde
  kalmanın maliyeti farklıysa, medyan tahmin yerine hangi kantilin
  bidlenmesi gerektiğini hesaplar (newsvendor çözümü).
- `imbalance_cost`/`evaluate_financial` — gerçek saatlik dengesizlik
  fiyatıyla bir tahmin serisinin gerçek maliyetini hesaplar. Bu fiyat verisi
  `epias.fetch_imbalance_price_for_date` (Sistem Marjinal Fiyatı - SMF,
  madde 5.113) üzerinden geliyor - URL/HTTP method/TGT auth **VE response
  alan adları** (`date`, `hour`, `systemMarginalPrice`) canlı çağrıyla
  **doğrulandı** (`check_epias_endpoints.py`, bkz. aşağısı). Dengesizlik
  maliyeti = SMF (sistem geneli) × santralin **kendi** sapması (üretim −
  bid); sapma zaten `data/` (gerçekleşen) ve `predict.py` (tahmin)
  çıktılarından hesaplanıyor, EPİAŞ'tan ayrıca gelmesi gerekmiyor. EPİAŞ'ın
  "Dengesizlik Tutarı" servisi (madde 5.183) bilinçli olarak KULLANILMIYOR -
  Şeffaflık Platformu'nda katılımcı/santral bazlı (Karapınar'a özel) veri
  vermiyor; santrale özel dengesizlik tutarı sadece EPİAŞ'ın kapalı PYS/YS
  hesabında görünür.

### Gün-içi piyasası yeniden-tahmin kontrolü

```bash
python reforecast_check.py --day-ahead predictions/day_ahead.csv \
  --refresh predictions/refresh.csv --capacity-mw 1000 --tolerance-pct 3
```

`predict.py`'yi iki farklı zamanda (gün-öncesi kapanışında ve daha sonra,
daha güncel hava tahminiyle) çalıştırıp çıktıları karşılaştırır, sapması
`--tolerance-pct`'i aşan saatleri işaretler - gün-içi piyasasında pozisyon
düzeltmesi değerlendirilmesi için bir sinyal. **Gerçek emir gönderimi
(trading) kapsam dışı ve bilinçli olarak implement edilmedi** - EPİAŞ
üye/trading API entegrasyonu ve önemli bir finansal risk gerektirir.

## `main.py` kullanımı

```bash
python main.py \
  --plant-id 2579 \
  --start 2025-06-01 --end 2025-06-30 \
  --output data/training_set.csv
```

Adımlar: `--plant-id`'yi `plants.yaml`'da arar -> EPİAŞ'a giriş yapıp TGT
alır -> `--start`/`--end` aralığı için saatlik üretim verisi çeker ->
santralin kayıtlı konumu için hava durumu verisini çeker -> ikisini
timestamp üzerinden birleştirir -> sonucu `--output` yoluna CSV olarak
yazar. Eksik/uyumsuz saatler (`merge.py`) uyarı olarak loglanıp atlanır.

Kimlik bilgileri **ortam değişkenlerinden** okunur, komut satırı argümanı
olarak verilmez ve hiçbir yere loglanmaz:

```bash
export EPIAS_USERNAME="you@example.com"
export EPIAS_PASSWORD="your-epias-password"   # EPİAŞ'ın statik API key'i yok, hesap şifresi kullanılır
```

## Geçmişe dönük veri (backfill)

`daily_datapull.yml`'in gün gün birikmesini beklemek gerekmez - EPİAŞ ve
Open-Meteo ikisi de geçmişe dönük (historical/archive) API'ler, tek seferde
geniş bir aralık çekilebilir. İki yol var:

**1) GitHub Actions üzerinden (yerel kurulum gerekmez):** Actions sekmesi ->
**Backfill EPİAŞ Data** -> **Run workflow** -> `start`/`end` tarihlerini
(ve gerekirse `plant_id`'yi) girip çalıştırın. Zaten eklediğiniz
`EPIAS_USERNAME`/`EPIAS_PASSWORD` secrets'ını kullanır, sonucu
`data/<plant_id>/` altına günlük CSV'ler halinde commit'ler.

**2) Yerelde:**

```bash
export EPIAS_USERNAME="you@example.com"
export EPIAS_PASSWORD="your-epias-password"
python backfill.py --plant-id 2579 --start 2025-01-01 --end 2026-07-19
```

Her iki yol da `data/<plant_id>/<tarih>.csv` yapısına yazar — `daily_datapull.yml`'in
her gün ürettiği dosyalarla aynı formatta, birbirinden ayırt edilemez.
Zaten bir CSV'si olan günler **varsayılan olarak atlanır** (yarıda kalan
bir backfill'i tekrar çalıştırmak güvenlidir, aynı günü tekrar çekmez);
`--overwrite` ile o günleri yeniden çekip üzerine yazdırabilirsiniz.

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

Workflow, hangi santral için çalışacağını `PLANT_ID` env değişkeninden
bilir (dosyanın en üstünde, gizli bilgi değildir) ve konum/kapasite
bilgisini `plants.yaml`'dan okur — farklı bir santral için hem `PLANT_ID`
hem `plants.yaml`'daki ilgili girdi güncellenmeli.

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
