# 🏡 Property Search NLP

*Natural language to real-estate filters.*

---

## 📌 Overview

Property Search NLP converts plain-English real estate queries into **structured search filters** such as:

* Property type (e.g., apartment, house)
* Location (suburb, city, region keywords)
* Budget (numeric value + intent: max, min, around)
* Bedrooms / bathrooms
* Parking
* Feature keywords (e.g., *luxury*, *pool*, *investment*)

The goal:
Enable real-estate platforms to support **natural language search** like:

> “3 bedroom townhouse in Camberwell under 900k with parking”
> → `{ bedrooms=3, type='Townhouse', suburb='Camberwell', max_budget=900000, parking=True }`

This project combines:
✔ A **fast, zero-cost rule-based parser**
✔ A fallback **LLM-based parser** for ambiguous interpretations
✔ A full **ground-truth evaluation** framework

---

## 🚀 Features

| Feature                           |        Rule-Based       |    LLM-Based    |
| --------------------------------- | :---------------------: | :-------------: |
| Property Type Extraction          |            ✅            |        ✅        |
| Budget Detection                  |            ✅            |        ✅        |
| Location Extraction               |      High accuracy      |     Highest     |
| Bedrooms / Bathrooms              |            ✅            |        ✅        |
| Semantic Features (e.g. “luxury”) |          Basic          |     Stronger    |
| Latency                           | **Ultra-fast** (0.02ms) |   Slow (1–3s)   |
| Cost                              |         **Free**        | API token usage |

---

## 🧩 How It Works

### Pipeline

```
Raw User Query
        ↓
Parser → Rule-Based approach
    ↘ fallback → LLM (OpenAI GPT)
        ↓
Structured Query Object
```

Each parsed output is represented as a `PropertyQuery` dataclass:

```json
{
  "property_type": "House",
  "location": "Richmond",
  "budget": 800000,
  "budget_type": "max",
  "bedrooms": 3,
  "confidence_score": 85.0
}
```

---

## 📦 Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# OR
source .venv/bin/activate # macOS/Linux
```

Install in editable/development mode:

```bash
pip install -e .
```

---

## ✨ Usage

```python
from property_search_nlp.parser import PropertySearchParser

parser = PropertySearchParser()

result = parser.parse("apartment with 2 bedrooms near Melbourne CBD max $500k")
print(result.to_dict())
```

Output:

```json
{
  "property_type": "Apartment",
  "location": "Melbourne CBD",
  "budget": 500000,
  "budget_type": "max",
  "bedrooms": 2,
  "confidence_score": 85.0
}
```

---

## 🧪 Evaluation Framework

Ground-truth queries are stored in:

```
examples/test_cases.json
```

Run full rule-based vs LLM accuracy benchmarking:

```bash
python scripts/detailed_comparison.py
```

Outputs:

* Detailed comparison table
* Accuracy vs ground truth
* Cost + latency metrics per query

📄 Log saved to:

```
detailed_comparison_log.txt
```

---

## 📁 Project Structure

```
property-search-nlp/
│
├─ src/property_search_nlp/
│   ├─ parser.py                 # Rule-based parser
│   ├─ llm_parser.py             # GPT fallback parser
│   └─ __init__.py
│
├─ examples/
│   └─ test_cases.json           # Ground truth dataset
│
├─ tests/
│   └─ test_parser.py            # Pytest suite
│
├─ scripts/
│   ├─ detailed_comparison.py    # Evaluation CLI
│   └─ comparison_framework.py
│
├─ setup.py                      # Packaging config
├─ requirements.txt              # Pinned dependencies
└─ README.md                     # You're reading this 😄
```

---

## 📈 Performance Snapshot

Rule-based vs LLM comparison (latest run):
* Rule-based correct on **~96%** of labeled fields
* LLM correct on **~92%**, better on complex features
* But ~100x slower and API cost incurred

From evaluation logs:
*“Rule-based parser performs extremely well for clear, structured property search queries.”*

---

## 🔮 Roadmap

* Expand location recognition (lowercase suburb names, postcodes)
* Improve feature extraction (renovated, new build, ocean view, etc.)
* Multi-value range support: “between 600k and 800k”
* Deploy optional web API
* Publish package to PyPI

---

## 🤝 Contributing

Pull requests and discussions welcome!
Issue templates coming soon.

---

## 📜 License

MIT License — free for commercial + personal use.

---

## 👤 Author

**Manusha Fernando**

* GitHub: [@ROMANFern](https://github.com/ROMANFern)
* LinkedIn: [Manusha Fernando](https://linkedin.com/in/manusha-fernando)
* Email: [manusha@romanfern.com](mailto:manusha@romanfern.com)

---

## 🙏 Acknowledgments

* Inspired by REA Group's property search challenges
* Built as a portfolio project demonstrating NLP engineering skills
* Thanks to the Python and NLP community for excellent resources