import os
import json
import time
from enum import Enum
from dotenv import load_dotenv
from instructor import from_openai
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd

# ========== Setup ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found in .env")

client = from_openai(OpenAI(api_key=OPENAI_API_KEY))

PARQUET_PATH = "linkedin_job_listings_small.parquet"
OUT_PATH = "outputs/extractions.json"
MODEL = "gpt-4.1-nano"
NUM_SAMPLES = 5000

os.makedirs("outputs", exist_ok=True)

# ========== Enums & Model ==========
class CompensationType(str, Enum):
    fixed = "fixed"
    variable = "variable"
    equity = "equity"
    mixed = "mixed"
    unspecified = "unspecified"

class RemoteType(str, Enum):
    onsite = "onsite"
    hybrid = "hybrid"
    remote = "remote"
    unspecified = "unspecified"

class Extraction(BaseModel):
    job_tasks: list[str] = Field(default_factory=list, description="List of short, verb-led job tasks.")
    compensation: CompensationType
    remote: RemoteType

# ========== Prompt ==========
def build_prompt(description: str, meta: dict) -> str:
    return f"""
You are an expert information extraction model for job postings.
Read the following English job description and metadata.
Infer and return a JSON object with exactly these keys:

- compensation: one of [fixed, variable, equity, mixed, unspecified]
- remote: one of [onsite, hybrid, remote, unspecified]

Follow these labeling rules carefully:

### 💰 Compensation guidelines
- "variable" → choose this if there is *any* indication of bonuses, commissions, incentive pay, OTE, performance pay, targets,
  or other variable components — even if a fixed base salary is also mentioned.
  Examples: "base + bonus", "performance-based pay", "commission structure", "annual bonus", "incentive plan".
- "fixed" → choose only if a single fixed amount (salary/hourly/wage) is given *without* mention of variable components.
  Examples: "hourly rate $25/hr", "annual salary $60,000".
- "equity" → choose if the description mentions stock, stock options, RSUs, profit sharing, ownership, or equity plans.
  Even if a base salary is present, prefer "equity" if no variable bonuses are mentioned.
- "mixed" → choose if both equity **and** cash pay are mentioned (e.g. “salary plus stock options”).
- "unspecified" → only if there is no compensation or pay information at all.

If multiple types apply (for example, base salary + bonus + stock), prioritize:
variable > mixed > equity > fixed > unspecified

### 🏠 Remote work guidelines
- "remote" → explicitly states *remote*, *work-from-home*, *telecommute*, *fully remote*, *anywhere in the US*, etc.
- "hybrid" → mentions both office and remote work, e.g. "hybrid", "3 days onsite, 2 remote", or "partially remote".
- "onsite" → requires physical presence or specific worksite terms such as
  "on-site", "in-person", "in-office", "office-based", "warehouse", "store", "factory", "clinic", "hospital", "school", or "lab".
- "unspecified" → only if there is no mention of work modality at all.

If multiple terms are found (for example, "hybrid remote, must visit office occasionally"), prefer:
remote > hybrid > onsite > unspecified

### 🧠 Use metadata (may contain missing values):
- salary fields → help infer compensation type.
- remote_allowed or location → can help identify remote type if description is unclear.

Metadata (for context):
{json.dumps(meta, indent=2)}

Description:
{description}

Return ONLY valid JSON, no explanation.
""".strip()

# ========== Load Data ==========
df = pd.read_parquet(PARQUET_PATH)
df = df[df["description"].str.len() > 0]
sample = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)
records = sample.to_dict(orient="records")

# ========== Load existing results ==========
if os.path.exists(OUT_PATH):
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

# ========== Time Tracking ==========
t_total_start = time.time()
processed = 0
total_request_sec = 0.0
skipped = 0
errors = 0

# ========== Process ==========
for row in records:
    job_id = str(row.get("job_id"))
    if job_id in results:
    # if job_id not in results:
        skipped += 1
        print(f"[skip] job_id={job_id} already done.")
        # print(f"[skip] job_id={job_id} not found in existing results.")  # <<< CHANGE
        continue

    description = row.get("description", "")
    if not description.strip():
        skipped += 1
        print(f"[skip] job_id={job_id} has empty description.")
        continue

    meta = {
        "min_salary": row.get("min_salary"),
        "max_salary": row.get("max_salary"),
        "currency": row.get("currency"),
        "pay_period": row.get("pay_period"),
        "compensation_type": row.get("compensation_type"),
        "remote_allowed": row.get("remote_allowed"),
        "work_type": row.get("formatted_work_type"),
        "location": row.get("location"),
        "skills_desc": row.get("skills_desc"),
    }

    prompt = build_prompt(description, meta)

    try:
        t_req_start = time.time()
        extraction = client.chat.completions.create(
            model=MODEL,
            response_model=Extraction,
            messages=[
                {"role": "system", "content": "You extract structured info and respond only in JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        t_req = time.time() - t_req_start

        data = extraction.model_dump()
        data["task_count"] = len(data.get("job_tasks", []))
        results[job_id] = data
        
        # Save file
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        processed += 1
        total_request_sec += t_req
        print(f"[ok] job_id={job_id} | {t_req:.2f}s")

    except Exception as e:
        errors += 1
        print(f"[error] job_id={job_id}: {e}")

# ========== Summary ==========
t_total = time.time() - t_total_start
avg_per_post = (total_request_sec / processed) if processed > 0 else 0.0
per_min = (60.0 / avg_per_post) if avg_per_post > 0 else 0.0

print("\n=== SUMMARY ===")
print(f"Processed: {processed}")
print(f"Skipped:   {skipped}")
print(f"Errors:    {errors}")
print(f"Total wall-clock: {t_total:.2f}s")
print(f"Avg request time/post: {avg_per_post:.2f}s")
print(f"Throughput: ~{per_min:.1f} posts/min (model time)")
print(f"Output: {OUT_PATH}")

