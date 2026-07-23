# Annotation Guideline — Gold-Standard ABSA Etiketleme

Bu doküman, `annotation_batch.xlsx` dosyasındaki yorumları etiketlerken tutarlı
kalman için referans. Amaç: pyabsa'nın tahminlerini gözden geçirip **gerçek
doğruyu** işaretlemek, böylece Faz 2'de "pyabsa ne kadar isabetli" sorusunu
objektif olarak ölçebilelim.

## Aspect nedir?

Üründeki **spesifik bir özelliğe/bileşene** atıfta bulunan, review metninde
**açıkça geçen** bir kelime veya kısa kelime öbeği.

- Evet: "battery", "screen", "price", "customer service", "battery life"
- Hayır: "it", "this product", "thing", "item" (genel/belirsiz referanslar)
- Hayır: metinde geçmeyen, ima edilen özellikler (örn. review hiç "ses"
  demiyor ama sen "muhtemelen sesi de kötüdür" diye tahmin ediyorsan — **ekleme**,
  sadece metinde açıkça yazan aspect'leri işaretle. Bu, pyabsa ile adil bir
  karşılaştırma için önemli; model de sadece metinde geçen kelimeleri
  span olarak işaretliyor.)

## Sentiment nedir?

O **spesifik aspect'e yönelik** ifadenin duygusu — genel review puanı
(overall) değil. Bir review 1 yıldız olabilir ama "battery" için söylenen
şey pozitif olabilir ("battery'si iyiydi ama kutusu kırık geldi" → battery:
Positive, box: Negative).

Etiketler: `Positive`, `Negative`, `Neutral`.

- Neutral: aspect'ten bahsediliyor ama açık bir olumlu/olumsuz yargı yok
  (örn. sadece "iPad ile uyumlu" gibi nötr bir bilgi cümlesi).
- Sarkazm/alay durumunda görünen kelimenin değil, **gerçek niyetin**
  sentiment'ini yaz.

## `human_verdict` sütunu — ne yazılacak

| Değer | Anlamı | corrected_aspects_json dolduruluyor mu? |
|---|---|---|
| `correct` | pyabsa'nın bulduğu TÜM aspect+sentiment çiftleri doğru ve eksiksiz | Hayır, boş bırak |
| `partial` | Bazıları doğru ama en az biri yanlış, eksik veya fazladan (hallucinated) | Evet, doğru/tam listeyi yaz |
| `incorrect` | pyabsa'nın bulduğu her şey yanlış | Evet, doğru listeyi yaz |
| `no_aspect` | pyabsa boş ({} / null) dedi ve sen de gerçekten metinde aspect yok diyorsun | Hayır, boş bırak |
| `missed_all` | pyabsa boş dedi ama aslında metinde aspect(ler) var | Evet, doğru listeyi yaz |

## `corrected_aspects_json` sütunu — format

Sadece `partial`, `incorrect`, `missed_all` durumlarında doldur. Format tam
olarak pyabsa'nın çıktısıyla aynı JSON: `{"aspect": "Sentiment"}`, birden
fazla aspect varsa hepsi aynı JSON içinde, hepsi lowercase:

```json
{"battery": "Negative", "screen": "Positive"}
```

Gerçekten hiç aspect yoksa hücreyi **boş bırak** (yazma).

## Özel durum: Sipariş edilenle gelen ürün farklıysa (yanlış varyant/spec)

Bazı yorumlar üründen memnuniyetsizlik değil, **yanlış varyant/spec gönderilmiş**
olmasından şikayet eder (örn. "16GB sipariş ettim ama 8GB geldi", yanlış renk,
yanlış model). Bu durumda:

- Sentiment her zaman **Negative** (sipariş edilenle gelen uyuşmuyor, sayının
  kendisi "kötü" olduğu için değil).
- Aspect key olarak **sayıyı/değeri değil, özelliğin genel adını** kullan:
  `storage`, `capacity`, `color`, `model` gibi. Metinde bu kelimelerden biri
  geçiyorsa onu kullan; hiçbiri geçmiyorsa cümledeki en yakın genel ismi kullan.
  Yanlış: `{"16gb": "Negative"}` veya `{"8gb": "Negative"}` — bunlar aynı
  özelliğin farklı değerleri, ayrı aspect değil.
  Doğru: `{"storage": "Negative"}`
- `notes` sütununa "yanlış ürün/spec geldi" gibi kısa bir not düş — bu,
  genel amaçlı ABSA modellerinin zayıf olduğu bilinen bir hata kategorisi,
  Faz 2 hata analizinde ayrı grup olarak incelenecek.

## Özel durum: "Bozuk/çalışmıyor geldi" şikayetleri

**Spesifik bir parça adı geçiyorsa** ("ekranı çatlak geldi", "şarj girişi
bozuk", "kutu ezilmişti") → o parça aspect olur, sentiment Negative:
`{"screen": "Negative"}`, `{"box": "Negative"}` gibi. Ambalaj/kutu da geçerli
bir aspect (ürünün kendisi değil ama spesifik ve metinde açık).

**Genel/isimsiz bir şikayetse** ("ürün bozuk geldi", "çalışmıyor", "DOA",
hiçbir spesifik parça adı geçmeden) → bu satır `no_aspect` olarak işaretlenir
(genel "it/item/ürün" referansları aspect sayılmıyor). Ama `notes` sütununa
**"genel bozuk/DOA şikayeti"** yaz — bu, ileride ayrı bir sinyal (örn. "bu
üründe alıcıların %X'i bozuk ürün aldığını yazmış") olarak değerlendirilecek,
sessizce atlanmasın.

## Örnekler

**Review:** "The battery life is amazing but the screen scratches easily."
**pyabsa tahmini:** `{"battery": "Positive"}` (screen'i kaçırmış)
**Senin işaretlemen:** `human_verdict = partial`,
`corrected_aspects_json = {"battery": "Positive", "screen": "Negative"}`

**Review:** "Great product, does what it says."
**pyabsa tahmini:** `null`
**Senin işaretlemen:** `human_verdict = no_aspect` (metinde spesifik bir
özellik adı geçmiyor, "product" genel bir referans — dokunma)

**Review:** "Terrible. Broke after one day."
**pyabsa tahmini:** `{"day": "Negative"}` (yanlış bir kelimeyi aspect sanmış)
**Senin işaretlemen:** `human_verdict = incorrect`,
`corrected_aspects_json` boş bırakılır (metinde gerçek bir aspect adı yok,
"day" geçerli bir aspect değil — bu satır aslında `missed_all` değil, gerçek
aspect hiç yok, o yüzden boş JSON anlamında hücreyi boş bırak)
