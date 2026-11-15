# Task Overview

## 📖 Storyline

If Italians are proud of their pasta, Belgians are head over heels of their *frieten*: fried twice and served in a paper cone from the local *frietkot*. Since 2014, frietkot culture has been recognised in Belgium as intangible cultural heritage. More than half of Belgium's potato area is in Flanders.

A Flemish growers' co-op wants to fence all its potato fields, and the contractor charges by the metre of edge. Walking every edge with a GPS would take all season, so Mariette, the co-op's surveyor, measured only a sample of fields. The co-op's analyst built a model to estimate the rest from what farmers already declare about each field. The model explained 99% of the variation in edge length, and the co-op accepted the estimates. Still, the fields Mariette picked for her survey seem oddly organised.

## 📌 Task Title

Fry-ndship

## 🔥 Difficulty

Easy

## 🎯 Objective

Model the fields' edge length from their dimensions and find the word Mariette observed in her measurements.

## 🚩 FLAG

The flag format is `SIGMOID_<WORD>`, where `<WORD>` is the observations Mariette did on data.

## 📂 Task Input

You are provided with the [dataset](https://drive.google.com/drive/folders/1qpARslPf8JWOCUHjo2ZIBV1X4Nm-BEdk?usp=sharing), with one row per surveyed potato field.

| **Column**        | **Meaning**                                                                                                                                                                                                                              | **Source**                   |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| `field_id`        | Survey identifier                                                                                                                                                                                                                        | Survey                       |
| `crop_code`       | Crop code in the Flemish agricultural declaration (901, 904, 907, 908)                                                                                                                                                                   | Farmer's declaration         |
| `crop`            | Crop as declared: ware potatoes harvested from 1 Sept (`geplande oogst vanaf 1/9`), early potatoes harvested before 1 Sept (`voor 1/9`), certified seed potatoes (`gecertificeerd pootgoed`), farm-saved seed potatoes (`hoevepootgoed`) | Farmer's declaration         |
| `farm_type`       | Specialisation of the farm that declared the field: Arable, Livestock, Mixed, Horticulture, Other                                                                                                                                        | Farmer's declaration         |
| `lat`, `lon`      | Approximate location of the field (rounded to 0.01°)                                                                                                                                                                                     | Farmer's declaration         |
| `area_ha`         | Area of the field, hectares                                                                                                                                                                                                              | Farmer's declaration         |
| `long_side_m`     | Long side of the field, metres                                                                                                                                                                                                           | Farmer's declaration         |
| `ditch_m`         | Length of the ditch along the field, metres                                                                                                                                                                                              | Farmer's declaration         |
| `hedge_m`         | Length of the hedge along the field, metres                                                                                                                                                                                              | Noted by Mariette on site    |
| `edge_m`          | Edge length, metres                                                                                                                                                                                                                      | Measured by Mariette on site |
| `fence_quote_eur` | The contractor's quote for fencing the field, EUR                                                                                                                                                                                        | Contractor, after the survey |

## ✅ Expected Output

Submit on the platform:

1. The flag, `SIGMOID_<WORD>`.
2. A URL to your fitting code to show how the flag was found.

## 🏆 Scoring Criteria

The flag is checked, and organizers review the linked code and figure by hand.

- ✅ Correct flag and the supporting materials: 100 points (PASS)
- ❌ Incorrect flag, or a correct flag without materials that support it: 0 points (FAIL)
