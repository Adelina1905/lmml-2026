"""Self-contained submission for task_cote_divoire (The Abidjan Transit Ledger).

Same I/O contract as main_template.py: reads test-input.json, writes predictions.json ({"id", "label"} per row), both by
filename in the current directory, no arguments needed. The upload accepts a single .py file, so this file carries its
own training data (OLD-task replay sample + public set covering both tasks). On each run it trains the recovered model
jointly on all ten labels (full replay, the cure for catastrophic forgetting), saves it to
checkpoints/recovered_model.joblib, and predicts. Training takes about a second on CPU.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DEFAULT_MODEL_PATH = Path("checkpoints") / "recovered_model.joblib"

LABELS = ["OLD_JOIN_CUSTOMER_ORDERS", "OLD_GROUPBY_REVENUE_BY_CATEGORY", "OLD_CRUD_INSERT_ORDER", "OLD_CRUD_UPDATE_INVENTORY", "OLD_CRUD_DELETE_ACCOUNT", "NEW_SPATIAL_NEAREST_WAREHOUSE", "NEW_SPATIAL_WITHIN_RADIUS", "NEW_WINDOW_LAG_DELIVERY_DELAY", "NEW_WINDOW_LEAD_NEXT_SHIPMENT", "NEW_WINDOW_PARTITION_RANK_FLEET"]

TRAIN_ROWS = [
    ("Aminata wants their account closed for good", 4),
    ("Fatou is leaving, delete their account", 4),
    ("I need Mariam's order list alongside their contact info", 0),
    ("I need Yao's order list alongside their contact info", 0),
    ("I need to enter a new order for Ibrahim", 2),
    ("I need to enter a new order for Konan", 2),
    ("I want a category breakdown of revenue, starting with cashew", 1),
    ("I want a category breakdown of revenue, starting with cocoa", 1),
    ("I want a category breakdown of revenue, starting with electronics", 1),
    ("Ibrahim is leaving, delete their account", 4),
    ("Konan is leaving, delete their account", 4),
    ("Konan wants their account closed for good", 4),
    ("Mariam just placed an order, add it to the system", 2),
    ("Yao is leaving, delete their account", 4),
    ("add a new order record for Ibrahim", 2),
    ("add a new order record for Konan", 2),
    ("add a new order record for Kouassi", 2),
    ("add a new order record for Yao", 2),
    ("adjust how much textiles stock we have on record", 3),
    ("aggregate order totals per category and highlight produce", 1),
    ("aggregate order totals per category and highlight textiles", 1),
    ("amend the warehouse quantity recorded for cashew", 3),
    ("amend the warehouse quantity recorded for textiles", 3),
    ("break down revenue by category with a group by on cocoa", 1),
    ("calculate the difference from the last shipment for truck-12 using lag", 7),
    ("calculate the difference from the last shipment for truck-47 using lag", 7),
    ("can you input a new order for Fatou right now", 2),
    ("can you input a new order for Konan right now", 2),
    ("can you match Adjoa's orders back to their customer entry", 0),
    ("can you match Fatou's orders back to their customer entry", 0),
    ("cancel and remove Fatou's account record", 4),
    ("capture a fresh order record on behalf of Konan", 2),
    ("change the quantity on hand for cashew in inventory", 3),
    ("close and delete the customer account for Ibrahim", 4),
    ("close and delete the customer account for Kouassi", 4),
    ("combine the customer table with orders to see Adjoa's purchase history", 0),
    ("combine the customer table with orders to see Ibrahim's purchase history", 0),
    ("combine the customer table with orders to see Konan's purchase history", 0),
    ("compare truck-88's current shipment timing to its prior one", 7),
    ("compare van-21's current shipment timing to its prior one", 7),
    ("compare van-3's current shipment timing to its prior one", 7),
    ("compute a partitioned rank over the fleet for truck-88", 9),
    ("compute a partitioned rank over the fleet for van-21", 9),
    ("compute how much truck-47's delivery slipped versus the prior run", 7),
    ("compute how much truck-88's delivery slipped versus the prior run", 7),
    ("compute how much van-21's delivery slipped versus the prior run", 7),
    ("compute the delivery delay compared to the previous shipment for truck-47", 7),
    ("compute the nearest warehouse location relative to Koumassi", 5),
    ("compute total income for the textiles category bucket", 1),
    ("compute truck-88's relative rank inside its fleet partition", 9),
    ("compute van-21's relative rank inside its fleet partition", 9),
    ("compute van-3's relative rank inside its fleet partition", 9),
    ("correct the on-hand quantity for electronics in the warehouse", 3),
    ("correct the on-hand quantity for textiles in the warehouse", 3),
    ("create a new row in orders for Mariam's purchase", 2),
    ("cross-reference orders with customer records for Aminata", 0),
    ("cross-reference orders with customer records for Fatou", 0),
    ("cross-reference orders with customer records for Konan", 0),
    ("delete the account belonging to Mariam", 4),
    ("delete the account belonging to Yao", 4),
    ("determine the proximity-ranked closest warehouse to Koumassi", 5),
    ("determine the proximity-ranked closest warehouse to Treichville", 5),
    ("drop the account row for Aminata", 4),
    ("drop the account row for Fatou", 4),
    ("edit the stored stock level for cashew", 3),
    ("erase the customer profile belonging to Ibrahim", 4),
    ("erase the customer profile belonging to Konan", 4),
    ("erase the customer profile belonging to Kouassi", 4),
    ("fetch Fatou's transaction history together with their account details", 0),
    ("fetch the lead shipment date for van-21", 8),
    ("filter sites to those within 15 km of Yopougon", 6),
    ("filter sites to those within 5 km of Vridi", 6),
    ("find every depot within a 10 kilometer radius of Vridi", 6),
    ("find every depot within a 20 kilometer radius of Vridi", 6),
    ("find the closest warehouse geometry to the Marcory point", 5),
    ("find the lag between this delivery and the one before it for truck-12", 7),
    ("find the lag between this delivery and the one before it for van-3", 7),
    ("find the lead value for the shipment after this one for truck-12", 8),
    ("find the lead value for the shipment after this one for truck-88", 8),
    ("find the lead value for the shipment after this one for van-21", 8),
    ("find the lead value for the shipment after this one for van-3", 8),
    ("find the nearest warehouse to the Koumassi depot using spatial distance", 5),
    ("find the nearest warehouse to the Treichville depot using spatial distance", 5),
    ("find the nearest warehouse to the Yopougon depot using spatial distance", 5),
    ("get rid of Fatou's entry in the accounts table", 4),
    ("get rid of Yao's entry in the accounts table", 4),
    ("get the order history for customer Adjoa with account info", 0),
    ("give me a full rundown of orders tied to Kouassi's account", 0),
    ("give me a full rundown of orders tied to Yao's account", 0),
    ("give me a per-category sales summary covering cashew", 1),
    ("give me a per-category sales summary covering textiles", 1),
    ("give me the next-in-line delivery for truck-47", 8),
    ("give me the next-in-line delivery for truck-88", 8),
    ("give me the warehouse nearest to Vridi as the crow flies", 5),
    ("give me the warehouse nearest to Yopougon as the crow flies", 5),
    ("give me truck-12's delivery delay versus the shipment before it", 7),
    ("give me van-21's delivery delay versus the shipment before it", 7),
    ("give truck-47's standing when the fleet is partitioned and ranked", 9),
    ("give van-3's standing when the fleet is partitioned and ranked", 9),
    ("group orders by category and show total revenue for cashew", 1),
    ("group orders by category and show total revenue for cocoa", 1),
    ("how are earnings distributed across categories like cashew", 1),
    ("how are earnings distributed across categories like electronics", 1),
    ("how much later did truck-47's delivery run versus the previous one", 7),
    ("how much money did cashew bring in overall", 1),
    ("how much money did cocoa bring in overall", 1),
    ("identify the warehouse geographically closest to Koumassi", 5),
    ("identify the warehouse geographically closest to Vridi", 5),
    ("insert a new order for customer Aminata into the orders table", 2),
    ("insert a new order for customer Fatou into the orders table", 2),
    ("insert a new order for customer Yao into the orders table", 2),
    ("insert into orders a fresh purchase made by Fatou", 2),
    ("insert into orders a fresh purchase made by Mariam", 2),
    ("join customers and orders to list all orders for Aminata", 0),
    ("join customers and orders to list all orders for Ibrahim", 0),
    ("join customers and orders to list all orders for Konan", 0),
    ("link the orders table to customers and filter on Adjoa", 0),
    ("link the orders table to customers and filter on Fatou", 0),
    ("list all orders and customer names for Konan", 0),
    ("list all warehouses within 5 km of the Koumassi depot", 6),
    ("list all warehouses within 5 km of the Yopougon depot", 6),
    ("list depots bounded by a 10 kilometer buffer near Marcory", 6),
    ("log a brand new order from Ibrahim", 2),
    ("measure the gap between truck-12's current and preceding shipment", 7),
    ("measure the gap between van-3's current and preceding shipment", 7),
    ("modify inventory levels after restocking cocoa", 3),
    ("modify inventory levels after restocking produce", 3),
    ("modify inventory levels after restocking textiles", 3),
    ("our cocoa inventory numbers are stale, update them", 3),
    ("our produce inventory numbers are stale, update them", 3),
    ("partition the fleet and rank truck-47 by number of deliveries", 9),
    ("partition vehicles by fleet and tell me truck-12's rank", 9),
    ("partition vehicles by fleet and tell me truck-88's rank", 9),
    ("permanently delete Adjoa's account record", 4),
    ("permanently delete Konan's account record", 4),
    ("permanently delete Mariam's account record", 4),
    ("please file Adjoa's new order into the database", 2),
    ("please file Fatou's new order into the database", 2),
    ("please file Yao's new order into the database", 2),
    ("please record Kouassi's latest purchase as a new order", 2),
    ("please refresh the inventory figures for electronics", 3),
    ("please refresh the inventory figures for textiles", 3),
    ("produce a report of Mariam's purchases with customer metadata attached", 0),
    ("pull every order row joined to the customer record for Adjoa", 0),
    ("pull every order row joined to the customer record for Ibrahim", 0),
    ("pull every order row joined to the customer record for Kouassi", 0),
    ("pull every order row joined to the customer record for Mariam", 0),
    ("purge Adjoa from the customer accounts", 4),
    ("purge Aminata from the customer accounts", 4),
    ("purge Fatou from the customer accounts", 4),
    ("query all sites within a 10 kilometer buffer of Yopougon", 6),
    ("query all sites within a 15 kilometer buffer of Koumassi", 6),
    ("query all sites within a 25 kilometer buffer of Vridi", 6),
    ("rank the fleet group containing truck-47 by deliveries", 9),
    ("rank truck-88 against other vehicles in the same fleet partition", 9),
    ("rank vehicles by delivery count partitioned by fleet including truck-12", 9),
    ("rank vehicles by delivery count partitioned by fleet including truck-47", 9),
    ("rank warehouses by proximity to Vridi", 5),
    ("rank warehouses by proximity to Yopougon", 5),
    ("rank-order the fleet partition that truck-12 belongs to", 9),
    ("rank-order the fleet partition that truck-47 belongs to", 9),
    ("rank-order the fleet partition that van-21 belongs to", 9),
    ("register a new order placed by Adjoa", 2),
    ("register a new order placed by Aminata", 2),
    ("register a new order placed by Konan", 2),
    ("register a new order placed by Kouassi", 2),
    ("remove Kouassi from the accounts table", 4),
    ("revise the stock count for cashew after the delivery", 3),
    ("revise the stock count for cocoa after the delivery", 3),
    ("roll up all the income figures by category, starting with cocoa", 1),
    ("select warehouses inside a 15 km buffer around Vridi", 6),
    ("select warehouses inside a 20 km buffer around Marcory", 6),
    ("set the new stock quantity for electronics items in inventory", 3),
    ("show cumulative sales figures per category for electronics", 1),
    ("show cumulative sales figures per category for produce", 1),
    ("show me every order placed by Kouassi along with their customer details", 0),
    ("show truck-88's next shipment relative to the current row", 8),
    ("show van-3's next shipment relative to the current row", 8),
    ("shut down and remove Fatou's account", 4),
    ("sort warehouses by how close they are to Marcory", 5),
    ("sort warehouses by how close they are to Treichville", 5),
    ("sum the sales for each product category including textiles", 1),
    ("sync the recorded electronics stock with what's actually on the shelf", 3),
    ("sync the recorded produce stock with what's actually on the shelf", 3),
    ("sync the recorded textiles stock with what's actually on the shelf", 3),
    ("tally up earnings bucketed by category for cocoa", 1),
    ("tell me van-3's upcoming shipment following this one", 8),
    ("terminate Aminata's account and delete their record", 4),
    ("the cocoa stock count needs to be updated in the system", 3),
    ("the produce stock count needs to be updated in the system", 3),
    ("the textiles stock count needs to be updated in the system", 3),
    ("total revenue grouped by cocoa category this month", 1),
    ("total revenue grouped by electronics category this month", 1),
    ("track truck-12's delay relative to its earlier shipment", 7),
    ("track truck-88's delay relative to its earlier shipment", 7),
    ("update the inventory count for the cocoa warehouse stock", 3),
    ("update the inventory count for the electronics warehouse stock", 3),
    ("update the inventory count for the textiles warehouse stock", 3),
    ("update the inventory table's quantity column for textiles", 3),
    ("use a lag window function to find the prior delivery time for truck-12", 7),
    ("use a lag window function to find the prior delivery time for van-3", 7),
    ("use a lead window function to get the upcoming shipment for truck-12", 8),
    ("use a lead window function to get the upcoming shipment for van-21", 8),
    ("use a lead window function to get the upcoming shipment for van-3", 8),
    ("use a partitioned window to rank truck-12 within its fleet group", 9),
    ("use a partitioned window to rank van-3 within its fleet group", 9),
    ("using lag, show how van-21's delivery time shifted from before", 7),
    ("using lead, show truck-12's subsequent shipment date", 8),
    ("using lead, show van-21's subsequent shipment date", 8),
    ("using lead, show van-3's subsequent shipment date", 8),
    ("what depots are inside a 15 km circle drawn around Marcory", 6),
    ("what depots are inside a 15 km circle drawn around Treichville", 6),
    ("what depots are inside a 20 km circle drawn around Koumassi", 6),
    ("what has Fatou bought, matched against their profile", 0),
    ("what has Konan bought, matched against their profile", 0),
    ("what is the aggregated revenue per category, especially electronics", 1),
    ("what is the next shipment scheduled right after van-3's current one", 8),
    ("what position does truck-88 hold within its partitioned fleet", 9),
    ("what shipment comes right after truck-47's current one", 8),
    ("what warehouse would serve Koumassi with the least travel distance", 5),
    ("what warehouse would serve Vridi with the least travel distance", 5),
    ("what was the time difference from truck-88's previous delivery", 7),
    ("what's our category-level revenue split, specifically for cashew", 1),
    ("what's our category-level revenue split, specifically for cocoa", 1),
    ("what's our category-level revenue split, specifically for electronics", 1),
    ("what's our category-level revenue split, specifically for produce", 1),
    ("what's our category-level revenue split, specifically for textiles", 1),
    ("what's the delay relative to van-3's last delivery", 7),
    ("what's the shortest-distance warehouse from Marcory", 5),
    ("what's the shortest-distance warehouse from Treichville", 5),
    ("what's the upcoming delivery slot for truck-12", 8),
    ("what's the upcoming delivery slot for truck-88", 8),
    ("what's the upcoming delivery slot for van-21", 8),
    ("where does truck-12 rank among its fleet peers", 9),
    ("where does truck-47 rank among its fleet peers", 9),
    ("which depot is the minimum distance away from Vridi", 5),
    ("which facility sits nearest to Marcory on the map", 5),
    ("which locations fall within 10 km of Marcory", 6),
    ("which locations fall within 15 km of Vridi", 6),
    ("which locations fall within 15 km of Yopougon", 6),
    ("which locations fall within 25 km of Treichville", 6),
    ("which sites lie inside the 10 km zone around Treichville", 6),
    ("which sites lie inside the 20 km zone around Treichville", 6),
    ("which sites lie inside the 5 km zone around Yopougon", 6),
    ("which warehouse is closest to Treichville by geographic distance", 5),
    ("which warehouses sit within a 15 kilometer range of Marcory", 6),
    ("which warehouses sit within a 20 kilometer range of Treichville", 6),
    ("which warehouses sit within a 20 kilometer range of Vridi", 6),
    ("wipe Aminata's account out of the system", 4),
    ("write a new entry to the orders table for Adjoa", 2),
    ("write a new entry to the orders table for Fatou", 2),
    ("write a new entry to the orders table for Ibrahim", 2),
]

def build_model() -> Pipeline:
    features = FeatureUnion([("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
    return Pipeline([("features", features), ("clf", LinearSVC(C=1.0, random_state=0))])

def train() -> Pipeline:
    return build_model().fit([t for t, _ in TRAIN_ROWS], [LABELS[i] for _, i in TRAIN_ROWS])

def save_model(model: Pipeline, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    except OSError as exc:
        print(f"could not save checkpoint to {path}: {exc}", file=sys.stderr)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-input.json", help="filename only, no path")
    parser.add_argument("--output", default="predictions.json", help="filename only, no path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="where the recovered checkpoint is saved")
    args = parser.parse_args()
    input_path = Path(args.input)
    if(not input_path.exists()):
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2
    model = train()
    save_model(model, Path(args.model))
    with open(input_path, encoding="utf-8") as fh: records = json.load(fh)
    labels = model.predict([r["text"] for r in records])
    predictions = [{"id": r["id"], "label": str(label)} for r, label in zip(records, labels)]
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
