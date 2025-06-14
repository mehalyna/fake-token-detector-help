Ось перелік корисних ознак (features), які можна витягти з назви токена та URL, щоб модель краще розрізняла валідні й скам-токени.

---

## 1. Ознаки з назви токена (`name`, `abbr`)

1. **Довжина назви**

   * `name_len = len(name)` — дуже довгі або дуже короткі назви можуть бути ознакою спаму.
2. **Кількість слів**

   * `word_count = name.count(" ") + 1` — скам-токени часто мають у назві кілька ключових слів (e.g. “Binance Token Scam”).
3. **Наявність ключових слів**

   * Бінарні фічі, наприклад:

     * `has_token = int("token" in name.lower())`
     * `has_coin  = int("coin"  in name.lower())`
     * `has_defi  = int("defi"  in name.lower())`
4. **Відсоток цифр у назві**

   * `digit_ratio = sum(ch.isdigit() for ch in name) / len(name)` — занадто багато цифр може вказувати на автоматично згенеровану назву.
5. **Наявність спецсимволів**

   * `special_char_count = sum(not ch.isalnum() for ch in name)` — незвичні символи (“\$”, “#”, “–”) часто вказують на фішинг.
6. **Великі літери (camelCase / ALLCAPS)**

   * `uppercase_ratio = sum(ch.isupper() for ch in name) / len(name)` — надмірні великі літери можуть сигналізувати про спробу привернути увагу.

## 2. Ознаки з URL чи контракту (`crypturl`, `url`, `uri`)

1. **Довжина URL**

   * `url_len = len(url)` — дуже довгі чи дуже короткі URL можуть бути підозрілими.
2. **Домен і піддомен**

   * `domain = urlparse(url).netloc` і бінарні фічі:

     * `is_official = int(domain.endswith("coinmarketcap.com") or domain.endswith("coingecko.com"))`
     * `is_suspicious = int(domain.count("-") > 2)`
3. **Наявність “0x”**

   * `has_0x = int("0x" in url.lower())` — для Ethereum-контрактів, але зазвичай в офіційних лінках воно приховане.
4. **Тип протоколу**

   * `is_https = int(url.lower().startswith("https://"))` — небезпечні HTTP-посилання можуть бути ознакою фішингу.
5. **Кількість слешів**

   * `slash_count = url.count("/")` — занадто багато рівнів у шляху може вказувати на підроблений ресурс.
6. **Наявність цифр у домені**

   * `domain_digit_ratio = sum(ch.isdigit() for ch in domain) / len(domain)` — числові домени (“123token.io”) зустрічаються рідше.
7. **Якість SSL-сертифікату (за допомогою Trust Score API)**

   * `trust_score = get_trust_score(url)` — додаткова зовнішня ознака (проксі-поле).
8. **Частота змін URL**

   * `url_age_days = (today - get_domain_creation_date(domain)).days` — новостворені домени частіше використовують скамери.

---

Ці ознаки можна додати до датафрейму перед навчанням моделі — вони дадуть більше інформації, ніж тільки числові метрики торгів. Наступний крок — імплементувати ці фічі в коді та зберегти оновлений датасет.
