# Story budget pricing snapshot

Checked 2026-09-07. This is a conservative budget estimate, not invoice reconciliation.

- DeepSeek official CNY prices: https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
  - deepseek-v4-pro peak: cache-hit input 0.30, cache-miss input 9, output 27 CNY per million tokens.
  - Off-peak: 0.15, 4.5, 13.5 respectively.
  - Peak is weekdays 09:00–12:00 and 14:00–18:00 Asia/Shanghai.
- Alibaba Beijing: https://help.aliyun.com/zh/model-studio/qwen3-vl-plus
  - Input <=32k: input 1, output 10 CNY per million tokens.
  - 32k<input<=128k: 1.5 and 15.
  - 128k<input<=256k: 3 and 30.

The original Rust PricingCatalog only accepts fixed integer fen rates and aggregate
input/output token counts. Its local config therefore uses peak/cache-miss DeepSeek
rates (900/2700 fen) and highest-tier Qwen rates (300/3000 fen). No FX rate is needed.
Cache savings, off-peak savings and lower input tiers are not reflected. Actual
provider charges can be lower; these values must not be presented as actual bills.

Local runtime configuration: sibling microcodex-short-drama-studio/config/provider-pricing-v1.json.
Catalog ID explicitly identifies the conservative policy. Prices can change;
recheck sources when models, regions or billing policy change. Live invoice matching
requires a richer contract carrying cached tokens and request billing time.
