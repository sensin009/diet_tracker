import csv
import os
import random
import argparse
from typing import Dict, List


HEADER = [
    'food_id','food_name','calories','protein_g','carbs_g','fats_g',
    'diabetic_friendly','vegetarian','vegan',
    'contains_dairy','contains_nuts','contains_gluten','contains_soy'
]


PROTEIN_SOURCES = [
    ('Chicken Breast', False, False, False, False),
    ('Turkey', False, False, False, False),
    ('Salmon', False, False, False, False),
    ('Tuna', False, False, False, False),
    ('Egg', False, False, False, False),
    ('Paneer', True, False, True, False),
    ('Tofu', True, True, False, True),
    ('Chickpeas', True, True, False, False),
    ('Lentils', True, True, False, False),
    ('Greek Yogurt', True, False, True, False),
]

CARB_BASES = [
    ('Rice', True), ('Brown Rice', True), ('Quinoa', True), ('Whole Wheat', True),
    ('Multigrain', True), ('Millet', True), ('Pasta', True), ('Potato', True),
]

VEG_EXTRAS = [
    'Spinach', 'Broccoli', 'Bell Pepper', 'Onion', 'Tomato', 'Cucumber', 'Carrot', 'Zucchini', 'Mushroom'
]

CUISINES = ['Indian', 'Mediterranean', 'American', 'Mexican', 'Italian', 'Asian']


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def write_rows(path: str, rows: List[Dict[str, str]]):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def synthesize_food_name() -> str:
    prot_name, _, _, _, _ = random.choice(PROTEIN_SOURCES)
    carb_name, _ = random.choice(CARB_BASES)
    veg = random.choice(VEG_EXTRAS)
    cuisine = random.choice(CUISINES)
    pattern = random.choice([
        f"{cuisine} {prot_name} with {veg} and {carb_name}",
        f"{prot_name} {carb_name} Bowl with {veg}",
        f"Grilled {prot_name} & {veg} over {carb_name}",
        f"{veg} {prot_name} Wrap on {carb_name}",
        f"{prot_name} {veg} Salad with {carb_name}",
    ])
    return pattern


def synthesize_macros(goal: str) -> Dict[str, float]:
    # Choose a realistic calorie target per item
    base_cal = random.randint(180, 650)
    if goal == 'muscle_gain':
        base_cal += random.randint(40, 120)
    if 'weight_loss' in goal:
        base_cal -= random.randint(30, 80)
    base_cal = clamp(base_cal, 150, 800)

    # Allocate macros
    # Start with protein grams
    protein_g = random.uniform(12, 40)
    if goal == 'muscle_gain':
        protein_g += random.uniform(6, 12)
    # Carbs vary with endurance/maintenance
    carbs_g = random.uniform(18, 80)
    if goal == 'endurance':
        carbs_g += random.uniform(10, 25)
    # Fats moderate
    fats_g = random.uniform(5, 24)

    # Adjust to match calories roughly
    kcal = 4 * protein_g + 4 * carbs_g + 9 * fats_g
    scale = clamp(base_cal / max(kcal, 1.0), 0.6, 1.6)
    protein_g *= scale
    carbs_g *= scale
    fats_g *= scale
    calories = int(round(4 * protein_g + 4 * carbs_g + 9 * fats_g))

    return {
        'calories': calories,
        'protein_g': round(protein_g, 1),
        'carbs_g': round(carbs_g, 1),
        'fats_g': round(fats_g, 1)
    }


def synthesize_flags() -> Dict[str, str]:
    # Dietary flags with light correlation
    veg_choice = random.random() < 0.55
    vegan_choice = veg_choice and random.random() < 0.35
    contains_dairy = veg_choice and (not vegan_choice) and random.random() < 0.5
    contains_nuts = random.random() < 0.12
    contains_gluten = random.random() < 0.35
    contains_soy = random.random() < 0.15
    diabetic_friendly = random.random() < 0.5

    return {
        'diabetic_friendly': str(bool(diabetic_friendly)),
        'vegetarian': str(bool(veg_choice)),
        'vegan': str(bool(vegan_choice)),
        'contains_dairy': str(bool(contains_dairy)),
        'contains_nuts': str(bool(contains_nuts)),
        'contains_gluten': str(bool(contains_gluten)),
        'contains_soy': str(bool(contains_soy)),
    }


def generate_rows(n_rows: int, start_id: int, goal: str) -> List[Dict[str, str]]:
    out = []
    for i in range(n_rows):
        name = synthesize_food_name()
        macros = synthesize_macros(goal)
        flags = synthesize_flags()
        row = {
            'food_id': str(start_id + i),
            'food_name': name,
            'calories': str(macros['calories']),
            'protein_g': str(macros['protein_g']),
            'carbs_g': str(macros['carbs_g']),
            'fats_g': str(macros['fats_g']),
            **flags,
        }
        out.append(row)
    return out


def main(target_total: int, goal: str):
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'food_database.csv')
    rows = read_rows(csv_path)

    # Determine current max id and count
    current_count = len(rows)
    if current_count == 0:
        raise RuntimeError('food_database.csv is empty or missing header')
    max_id = max(int(r['food_id']) for r in rows if r.get('food_id', '').isdigit())

    if current_count >= target_total:
        print(f"No generation needed. Current rows: {current_count} >= target {target_total}")
        return

    to_add = target_total - current_count
    print(f"Generating {to_add} new rows (goal={goal})...")
    new_rows = generate_rows(to_add, max_id + 1, goal)
    rows.extend(new_rows)

    # Write back
    write_rows(csv_path, rows)
    print(f"Done. New total rows: {len(rows)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment food_database.csv with synthetic but plausible entries')
    parser.add_argument('--target', type=int, default=1000, help='Target total number of rows after augmentation')
    parser.add_argument('--goal', type=str, default='maintenance', choices=['maintenance','weight_loss','muscle_gain','endurance'], help='Bias macro generation by goal')
    args = parser.parse_args()
    main(args.target, args.goal)


