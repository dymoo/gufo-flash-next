#!/usr/bin/env python3
"""Deterministic document-vision eval: invoices, bar charts, tables and order screens rendered with
Pillow, each with a question whose answer is exact. Writes PNGs and questions.jsonl into OUT."""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(4747)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
f = lambda s: ImageFont.truetype(FONT, s)
ITEMS = ["Widget", "Gasket", "Bracket", "Sensor", "Cable", "Adapter", "Valve", "Bearing", "Hinge", "Switch", "Filter", "Pump"]
NAMES = ["Alice", "Bruno", "Chen", "Dara", "Elif", "Farid", "Greta", "Hugo", "Imani", "Jonas", "Keiko", "Luis"]
CITIES = ["Lisbon", "Oslo", "Lagos", "Denver", "Hanoi", "Quito", "Perth", "Leeds", "Turin", "Busan", "Cairo", "Tampa"]
FRUIT = ["Apples", "Pears", "Plums", "Kiwis", "Limes", "Mangos", "Grapes", "Figs"]
qs = []


def save(img, name, question, answer, kind):
    img.save(out / name)
    qs.append({"image": name, "question": question, "answer": str(answer), "kind": kind})


for i in range(14):  # invoices
    img = Image.new("RGB", (900, 620), "white"); d = ImageDraw.Draw(img)
    inv = f"INV-{rng.randrange(10000, 99999)}"
    d.text((40, 30), "ACME Industrial Supply", font=f(34), fill="black")
    d.text((40, 80), f"Invoice {inv}", font=f(24), fill="#333")
    rows = rng.sample(ITEMS, rng.randrange(3, 6)); y = 150; total = 0
    d.text((40, y), "Item", font=f(22), fill="black"); d.text((400, y), "Qty", font=f(22), fill="black"); d.text((560, y), "Unit", font=f(22), fill="black"); d.text((720, y), "Amount", font=f(22), fill="black")
    d.line((40, y + 34, 860, y + 34), fill="black", width=2); y += 50
    qty_of = {}
    for it in rows:
        q, u = rng.randrange(1, 40), rng.randrange(2, 90)
        qty_of[it] = q; total += q * u
        for x, t in ((40, it), (400, str(q)), (560, f"{u}.00"), (720, f"{q * u}.00")):
            d.text((x, y), t, font=f(22), fill="#111")
        y += 40
    d.line((40, y + 6, 860, y + 6), fill="black", width=2)
    d.text((560, y + 20), "TOTAL", font=f(24), fill="black"); d.text((720, y + 20), f"{total}.00", font=f(24), fill="black")
    if i % 3 == 0:
        save(img, f"invoice{i}.png", "What is the invoice total? Reply with the number only.", total, "invoice-total")
    elif i % 3 == 1:
        it = rng.choice(rows)
        save(img, f"invoice{i}.png", f"How many units of {it} are on this invoice? Reply with the number only.", qty_of[it], "invoice-qty")
    else:
        save(img, f"invoice{i}.png", "What is the invoice number? Reply with the invoice number only.", inv, "invoice-id")

for i in range(12):  # bar charts
    img = Image.new("RGB", (900, 560), "white"); d = ImageDraw.Draw(img)
    cats = rng.sample(FRUIT, rng.randrange(4, 7)); vals = rng.sample(range(5, 95), len(cats))
    d.text((40, 20), "Monthly sales (crates)", font=f(28), fill="black")
    bw = 700 // len(cats)
    for k, (c, v) in enumerate(zip(cats, vals)):
        x0 = 80 + k * bw; h = v * 4
        d.rectangle((x0 + 10, 480 - h, x0 + bw - 10, 480), fill=["#3b6ea8", "#c75b2a", "#3e9a6a", "#9a7a2a", "#7a4ab0", "#2a8a9a"][k % 6])
        d.text((x0 + 14, 488), c, font=f(20), fill="black"); d.text((x0 + 16, 450 - h), str(v), font=f(20), fill="black")
    d.line((70, 480, 850, 480), fill="black", width=2)
    if i % 2 == 0:
        save(img, f"chart{i}.png", "Which category has the highest value? Reply with the category name only.", cats[vals.index(max(vals))], "chart-max")
    else:
        c = rng.choice(cats)
        save(img, f"chart{i}.png", f"What is the value shown for {c}? Reply with the number only.", vals[cats.index(c)], "chart-value")

for i in range(12):  # tables
    img = Image.new("RGB", (900, 440), "white"); d = ImageDraw.Draw(img)
    people = rng.sample(NAMES, 6); cities = rng.sample(CITIES, 6); ages = [rng.randrange(19, 70) for _ in people]
    for x, t in ((40, "Name"), (300, "City"), (560, "Age")):
        d.text((x, 30), t, font=f(24), fill="black")
    d.line((40, 66, 760, 66), fill="black", width=2)
    for r, (p, c, a) in enumerate(zip(people, cities, ages)):
        y = 80 + r * 50
        for x, t in ((40, p), (300, c), (560, str(a))):
            d.text((x, y), t, font=f(22), fill="#111")
        d.line((40, y + 40, 760, y + 40), fill="#bbb", width=1)
    p = rng.choice(people)
    if i % 2 == 0:
        save(img, f"table{i}.png", f"Which city does {p} live in? Reply with the city only.", cities[people.index(p)], "table-city")
    else:
        save(img, f"table{i}.png", f"How old is {p}? Reply with the number only.", ages[people.index(p)], "table-age")

for i in range(10):  # order confirmation screens
    img = Image.new("RGB", (760, 420), "#f4f6f8"); d = ImageDraw.Draw(img)
    d.rectangle((30, 30, 730, 390), fill="white", outline="#cfd5db", width=2)
    order = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randrange(100000, 999999)}"
    day = rng.randrange(1, 28); month = rng.choice(["March", "April", "May", "June", "July", "August"])
    d.text((60, 60), "Thanks for your order!", font=f(30), fill="#1b2a3a")
    d.text((60, 130), f"Order number: {order}", font=f(24), fill="#333")
    d.text((60, 180), f"Estimated delivery: {day} {month}", font=f(24), fill="#333")
    d.text((60, 230), f"Items: {rng.randrange(1, 9)}", font=f(24), fill="#333")
    if i % 2 == 0:
        save(img, f"order{i}.png", "What is the order number? Reply with the order number only.", order, "order-id")
    else:
        save(img, f"order{i}.png", "What month is the estimated delivery? Reply with the month only.", month, "order-month")

(out / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
print(f"wrote {len(qs)} items to {out}")
