#!/usr/bin/env python3
"""Curăță exportul stațiilor meteo și convertește temperaturile în Celsius."""

import csv
import io
import re
from pathlib import Path

INPUT_FILE = Path("train-input.csv")
OUTPUT_FILE = Path("train-output.csv")
EXPECTED_COLUMNS = [
    "reading_id",
    "station_name",
    "temperature_raw",
    "unit",
    "reading_date",
]


def read_text(path: Path) -> str:
    """Citește UTF-8 și elimină un eventual BOM."""
    return path.read_text(encoding="utf-8-sig")


def find_header(lines: list[str]) -> int:
    """Găsește antetul, ignorând comentariile și liniile goale din preambul."""
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Fișierul cerut folosește ';', dar acceptăm și ','.
        delimiter = ";" if ";" in stripped else ","
        columns = [part.strip().lstrip("\ufeff").lower()
                   for part in stripped.split(delimiter)]

        if columns == EXPECTED_COLUMNS:
            return index

    raise ValueError("Nu am găsit antetul așteptat în fișierul de intrare.")


def repair_station_name(name: str) -> str:
    """Încearcă să repare text islandez citit accidental ca Windows-1252/Latin-1."""
    candidates = [name]

    # Repară cazuri frecvente de tipul „Ãžingvellir” -> „Þingvellir”.
    for source_encoding in ("cp1252", "latin-1"):
        try:
            candidates.append(name.encode(source_encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    # Dacă textul pare deja corect, nu îl modificăm.
    mojibake_markers = ("Ã", "Â", "Ð", "Þ", "�")
    for candidate in candidates[1:]:
        if not any(marker in candidate for marker in mojibake_markers):
            return candidate

    return name


def parse_temperature(value: str) -> float:
    """Acceptă atât punctul, cât și virgula ca separator zecimal."""
    normalized = value.strip().replace(" ", "").replace(",", ".")
    return float(normalized)


def main() -> None:
    text = read_text(INPUT_FILE)
    lines = text.splitlines()
    header_index = find_header(lines)

    # Eliminăm comentariile și liniile goale de după antet înainte de csv.reader.
    data_lines = [
        line for line in lines[header_index:]
        if line.strip() and not line.lstrip().startswith("#")
    ]

    # Detectăm delimitatorul din antetul găsit.
    header = data_lines[0]
    delimiter = ";" if ";" in header else ","

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)), delimiter=delimiter)
    unique_rows: dict[int, dict[str, object]] = {}

    for row in reader:
        if not row or not row.get("reading_id"):
            continue

        reading_id = int(row["reading_id"].strip())
        temperature = parse_temperature(row["temperature_raw"])
        unit = row["unit"].strip().upper()

        if unit == "F":
            temperature = (temperature - 32) * 5 / 9
        elif unit != "C":
            raise ValueError(f"Unitate necunoscută pentru reading_id={reading_id}: {unit!r}")

        clean_row = {
            "reading_id": reading_id,
            "station_name": repair_station_name(row["station_name"].strip()),
            "temperature_c": f"{temperature:.1f}",
            "reading_date": row["reading_date"].strip(),
        }

        # Dacă un ID apare de mai multe ori, păstrăm prima apariție.
        # Pentru duplicate identice, rezultatul este același.
        unique_rows.setdefault(reading_id, clean_row)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["reading_id", "station_name", "temperature_c", "reading_date"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(unique_rows[key] for key in sorted(unique_rows))


if __name__ == "__main__":
    main()
