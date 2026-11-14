import json
import os
import re

if os.path.exists("test-input.json"):
    INPUT_NAME = "test-input.json"
    OUTPUT_NAME = "test-output.json"
else:
    INPUT_NAME = "train-input.json"
    OUTPUT_NAME = "my-train-output.json"

GOLDEN_NAME = "train-output.json"

with open(INPUT_NAME, encoding="utf-8") as f:
    data_inputs = json.load(f)

documents = data_inputs["documents"]

NAME_PHRASES = re.compile(
    r"(my (?:name is|wife|husband|son|daughter)) \w+ \w+",
    flags=re.IGNORECASE,
)

def replace_names(text: str) -> str:
    return NAME_PHRASES.sub(r"\1 [NAME]", text)

ACCOUNT_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?\d){10,30}\b")

def replace_account(text: str) -> str:
    return ACCOUNT_PATTERN.sub("[ACCOUNT]", text)

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

def replace_email(text: str) -> str:
    return EMAIL_PATTERN.sub("[EMAIL]", text)

ID_PATTERN = re.compile(r"(?<![\w-])\d-\d{4}-\d{4}(?![\w-])")

def replace_id(text: str) -> str:
    return ID_PATTERN.sub("[ID]", text)

CITY_WORD = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"

ADDRESS_PATTERN = re.compile(
    r"\b(?:Calle|Avenida)\s+\d+"                      
    r",\s*[^,.;\n]+"                                  
    r",\s*" + CITY_WORD + r"(?:\s+" + CITY_WORD + r")*" 
)

def replace_address(text: str) -> str:
    return ADDRESS_PATTERN.sub("[ADDRESS]", text)

YEAR = r"(?:19|20)\d{2}"
DAY_NUM = r"(?:0?[1-9]|[12]\d|3[01])"
MONTH_NUM = r"(?:0?[1-9]|1[0-2])"
DAY = DAY_NUM + r"(?:st|nd|rd|th)?"
MONTH_EN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
MONTH_ES = (
    r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto"
    r"|sep?tiembre|octubre|noviembre|diciembre)"
)

DOB_PATTERN = re.compile(
    r"\b(?:"
    + MONTH_EN + r"\s+" + DAY + r",?\s+" + YEAR
    + r"|" + DAY + r"\s+(?:of\s+)?" + MONTH_EN + r",?\s+" + YEAR
    + r"|" + DAY_NUM + r"\s+de\s+" + MONTH_ES + r"(?:\s+de\s+|\s+)" + YEAR
    + r"|" + DAY_NUM + r"[/.-]" + DAY_NUM + r"[/.-]" + YEAR
    + r"|" + YEAR + r"[/.-]" + MONTH_NUM + r"[/.-]" + DAY_NUM
    + r")\b",
    flags=re.IGNORECASE,
)

def replace_dob(text: str) -> str:
    return DOB_PATTERN.sub("[DOB]", text)

PHONE_PATTERN = re.compile(
    r"(?<![\w-])"
    r"(?:(?:\+|00)?\(?506\)?[ -]?)?"
    r"\d{4}[ -]?\d{4}"
    r"(?![\w-])"
)

def replace_phone(text: str) -> str:
    return PHONE_PATTERN.sub("[PHONE]", text)

redacted_documents = {}

for document in documents:
    document_id = document["id"]
    document_text = document["text"]
    lines = document_text.split("\n\n")

    for i in range(len(lines)):
        line = lines[i]
        line = replace_names(line)
        line = replace_account(line)
        line = replace_email(line)
        line = replace_dob(line)
        line = replace_id(line)
        line = replace_phone(line)
        line = replace_address(line)
        lines[i] = line

    signature = lines[len(lines)-1].split("\n")
    signature[len(signature)-1] = "[NAME]"
    lines[len(lines)-1] = "\n".join(signature)

    redacted_documents[document_id] = "\n\n".join(lines)

with open(OUTPUT_NAME, "w", encoding="utf-8") as f:
    json.dump({"redacted_documents": redacted_documents}, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(redacted_documents)} documents to {OUTPUT_NAME}")

if INPUT_NAME == "train-input.json" and os.path.exists(GOLDEN_NAME):
    with open(GOLDEN_NAME, encoding="utf-8") as f:
        golden = json.load(f)["redacted_documents"]
    same = 0
    for document_id, expected in golden.items():
        if redacted_documents.get(document_id) == expected:
            same += 1
        else:
            print("differs:", document_id)
    print(f"Identical to golden: {same}/{len(golden)}")