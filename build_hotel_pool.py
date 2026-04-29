"""
Build an expanded hotel pool (~200 hotels) for conjoint task generation.
Combines the manually curated 67-hotel pool with systematically added
chain and independent properties from NYC.

All hotels are real NYC properties with realistic attribute ranges
calibrated from Booking.com, Expedia, KAYAK, and TripAdvisor data.

Output: hotel_pool.json (overwrites existing)
"""

import json
import os

# ── Helper to build a hotel entry ──────────────────────────────────────────

_next_id = 68  # continue from existing pool

def h(name, stars, neighborhood, review_base, review_count,
      price_min, price_max, rooms, canc_prob=0.5, bkfst_prob=0.1,
      amenities=None):
    global _next_id
    entry = {
        "id": _next_id,
        "name": name,
        "stars": stars,
        "neighborhood": neighborhood,
        "review_score_base": review_base,
        "review_count_base": review_count,
        "price_min": price_min,
        "price_max": price_max,
        "room_types": rooms,
        "cancellation_free_prob": canc_prob,
        "breakfast_prob": bkfst_prob,
        "amenities": amenities or ["Free WiFi"]
    }
    _next_id += 1
    return entry

# ── Load existing 67 hotels ───────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "hotel_pool.json"), "r") as f:
    pool_data = json.load(f)

existing = pool_data["hotels"]

# ── Additional hotels (chain + independent) ───────────────────────────────

# Standard room type templates
KING = ["King Room (1 king bed)"]
QUEEN = ["Queen Room (1 queen bed)"]
KQ = ["King Room (1 king bed)", "Queen Room (1 queen bed)"]
KD = ["King Room (1 king bed)", "Double Room (2 double beds)"]
KQD = ["King Room (1 king bed)", "Queen Room (1 queen bed)", "Double Room (2 double beds)"]
SUITE_K = ["King Suite (1 king bed)", "Double Suite (2 double beds)"]

# Standard amenity sets
BASIC = ["Free WiFi"]
STANDARD = ["Free WiFi", "Fitness center"]
FULL = ["Free WiFi", "Fitness center", "Restaurant"]
FULL_BAR = ["Free WiFi", "Fitness center", "Restaurant", "Bar"]
LUXURY = ["Free WiFi", "Fitness center", "Restaurant", "Bar", "Spa"]
LUXURY_POOL = ["Free WiFi", "Fitness center", "Restaurant", "Bar", "Spa", "Pool"]

additional = []

# ── HAMPTON INN (Hilton) — 3-star, breakfast included ─────────────────────
hampton_base = dict(stars=3, canc_prob=0.6, bkfst_prob=0.85, amenities=["Free WiFi", "Breakfast included", "Fitness center"])
additional += [
    h("Hampton Inn Manhattan-Chelsea", neighborhood="Chelsea", review_base=8.2, review_count=2800, price_min=155, price_max=310, rooms=KD, **hampton_base),
    h("Hampton Inn Manhattan/Times Square South", neighborhood="Times Square", review_base=8.3, review_count=3200, price_min=165, price_max=330, rooms=KD, **hampton_base),
    h("Hampton Inn Manhattan Grand Central", neighborhood="Midtown East", review_base=8.4, review_count=2600, price_min=170, price_max=340, rooms=KD, **hampton_base),
    h("Hampton Inn Manhattan-35th St/Empire State Bldg", neighborhood="Midtown", review_base=8.2, review_count=2400, price_min=160, price_max=320, rooms=KD, **hampton_base),
    h("Hampton Inn Manhattan-Seaport/Financial District", neighborhood="Financial District", review_base=8.1, review_count=1800, price_min=145, price_max=290, rooms=KD, **hampton_base),
    h("Hampton Inn Manhattan/Downtown-Financial District", neighborhood="Financial District", review_base=8.0, review_count=1600, price_min=140, price_max=280, rooms=KD, **hampton_base),
    h("Hampton Inn Manhattan-SoHo", neighborhood="SoHo", review_base=8.3, review_count=1400, price_min=170, price_max=340, rooms=KD, **hampton_base),
]

# ── HILTON GARDEN INN — 3-star ────────────────────────────────────────────
hgi_base = dict(stars=3, canc_prob=0.6, bkfst_prob=0.3, amenities=["Free WiFi", "Restaurant", "Fitness center", "Business center"])
additional += [
    h("Hilton Garden Inn New York/Midtown Park Avenue", neighborhood="Midtown East", review_base=8.1, review_count=2200, price_min=160, price_max=320, rooms=KD, **hgi_base),
    h("Hilton Garden Inn New York/Manhattan-Chelsea", neighborhood="Chelsea", review_base=7.9, review_count=1800, price_min=150, price_max=300, rooms=KD, **hgi_base),
    h("Hilton Garden Inn New York/Central Park South", neighborhood="Midtown", review_base=8.3, review_count=2000, price_min=180, price_max=360, rooms=KD, **hgi_base),
    h("Hilton Garden Inn New York/Times Square", neighborhood="Times Square", review_base=8.0, review_count=2400, price_min=170, price_max=340, rooms=KD, **hgi_base),
]

# ── DOUBLETREE (Hilton) — 4-star ─────────────────────────────────────────
dt_base = dict(stars=4, canc_prob=0.6, bkfst_prob=0.15, amenities=["Free WiFi", "Restaurant", "Fitness center"])
additional += [
    h("DoubleTree by Hilton New York Times Square West", neighborhood="Times Square", review_base=7.8, review_count=3200, price_min=175, price_max=350, rooms=KD, **dt_base),
    h("DoubleTree by Hilton New York Times Square South", neighborhood="Times Square", review_base=7.7, review_count=2800, price_min=170, price_max=340, rooms=KD, **dt_base),
    h("DoubleTree by Hilton NYC Financial District", neighborhood="Financial District", review_base=8.0, review_count=2100, price_min=155, price_max=310, rooms=KD, **dt_base),
    h("DoubleTree by Hilton New York Midtown Fifth Ave", neighborhood="Midtown", review_base=7.9, review_count=1900, price_min=180, price_max=360, rooms=KD, **dt_base),
]

# ── HOLIDAY INN (IHG) — 3-star ───────────────────────────────────────────
hi_base = dict(stars=3, canc_prob=0.5, bkfst_prob=0.2, amenities=["Free WiFi", "Fitness center"])
additional += [
    h("Holiday Inn New York City - Times Square", neighborhood="Times Square", review_base=7.6, review_count=3800, price_min=140, price_max=280, rooms=KD, **hi_base),
    h("Holiday Inn NYC - Lower East Side", neighborhood="Lower East Side", review_base=7.8, review_count=1600, price_min=120, price_max=240, rooms=KD, **hi_base),
    h("Holiday Inn New York City - Wall Street", neighborhood="Financial District", review_base=7.7, review_count=2000, price_min=125, price_max=250, rooms=KD, **hi_base),
    h("Holiday Inn Express Manhattan Midtown West", neighborhood="Midtown West", review_base=8.0, review_count=2200, price_min=135, price_max=270, rooms=KD, stars=3, canc_prob=0.5, bkfst_prob=0.85, amenities=["Free WiFi", "Breakfast included", "Fitness center"]),
    h("Holiday Inn Express Manhattan Times Square", neighborhood="Times Square", review_base=7.9, review_count=2600, price_min=145, price_max=290, rooms=KD, stars=3, canc_prob=0.5, bkfst_prob=0.85, amenities=["Free WiFi", "Breakfast included", "Fitness center"]),
]

# ── CROWNE PLAZA (IHG) — 4-star ──────────────────────────────────────────
additional += [
    h("Crowne Plaza HY36 Midtown Manhattan", stars=4, neighborhood="Midtown", review_base=8.5, review_count=2200, price_min=190, price_max=380, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
    h("Crowne Plaza Times Square Manhattan", stars=4, neighborhood="Times Square", review_base=7.5, review_count=4500, price_min=175, price_max=350, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
]

# ── INTERCONTINENTAL (IHG) — 5-star ──────────────────────────────────────
additional += [
    h("InterContinental New York Times Square", stars=5, neighborhood="Times Square", review_base=8.4, review_count=2800, price_min=280, price_max=560, rooms=["King Room (1 king bed)", "Premium King (1 king bed)"], canc_prob=0.6, bkfst_prob=0.15, amenities=LUXURY),
    h("InterContinental New York Barclay", stars=5, neighborhood="Midtown East", review_base=8.7, review_count=1600, price_min=350, price_max=700, rooms=["Deluxe King (1 king bed)", "Premium King (1 king bed)"], canc_prob=0.6, bkfst_prob=0.2, amenities=LUXURY),
]

# ── KIMPTON (IHG) — 4-star boutique ──────────────────────────────────────
additional += [
    h("Kimpton Hotel Eventi", stars=4, neighborhood="Chelsea", review_base=8.6, review_count=1800, price_min=200, price_max=400, rooms=KQ, canc_prob=0.6, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Fitness center", "Wine hour"]),
    h("Kimpton Muse Hotel", stars=4, neighborhood="Times Square", review_base=8.4, review_count=1200, price_min=210, price_max=420, rooms=KQ, canc_prob=0.6, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar", "Wine hour"]),
]

# ── EVEN HOTEL (IHG) — 3-star wellness ───────────────────────────────────
additional += [
    h("EVEN Hotel New York - Times Square South", stars=3, neighborhood="Midtown", review_base=8.5, review_count=1400, price_min=145, price_max=290, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Fitness center", "In-room training zone"]),
]

# ── COURTYARD BY MARRIOTT — 3-4 star ─────────────────────────────────────
cy_base = dict(stars=3, canc_prob=0.6, bkfst_prob=0.2, amenities=FULL)
additional += [
    h("Courtyard by Marriott New York Manhattan/Midtown East", neighborhood="Midtown East", review_base=8.0, review_count=2000, price_min=170, price_max=340, rooms=KD, **cy_base),
    h("Courtyard by Marriott New York Manhattan/SoHo", neighborhood="SoHo", review_base=8.2, review_count=1400, price_min=185, price_max=370, rooms=KD, **cy_base),
    h("Courtyard by Marriott New York Downtown Manhattan", neighborhood="Financial District", review_base=7.9, review_count=1800, price_min=155, price_max=310, rooms=KD, **cy_base),
    h("Courtyard by Marriott New York Manhattan/Upper East Side", neighborhood="Upper East Side", review_base=8.1, review_count=1200, price_min=175, price_max=350, rooms=KD, **cy_base),
]

# ── FAIRFIELD INN (Marriott) — 3-star ────────────────────────────────────
ff_base = dict(stars=3, canc_prob=0.6, bkfst_prob=0.8, amenities=["Free WiFi", "Breakfast included", "Fitness center"])
additional += [
    h("Fairfield Inn & Suites New York Manhattan/Times Square", neighborhood="Times Square", review_base=7.8, review_count=2200, price_min=150, price_max=300, rooms=KD, **ff_base),
    h("Fairfield Inn & Suites New York Manhattan/Chelsea", neighborhood="Chelsea", review_base=7.9, review_count=1600, price_min=140, price_max=280, rooms=KD, **ff_base),
    h("Fairfield Inn & Suites New York Brooklyn", neighborhood="Brooklyn", review_base=7.7, review_count=1200, price_min=110, price_max=220, rooms=KD, **ff_base),
    h("Fairfield Inn & Suites New York Manhattan/Downtown East", neighborhood="Lower East Side", review_base=8.0, review_count=900, price_min=135, price_max=270, rooms=KD, **ff_base),
]

# ── RESIDENCE INN (Marriott) — 3-star extended stay ──────────────────────
ri_base = dict(stars=3, canc_prob=0.6, bkfst_prob=0.8, amenities=["Free WiFi", "Breakfast included", "Kitchen", "Fitness center"])
additional += [
    h("Residence Inn New York Manhattan/Times Square", neighborhood="Times Square", review_base=8.2, review_count=1800, price_min=195, price_max=390, rooms=["Studio Suite (1 king bed)", "One-Bedroom Suite (1 king bed)"], **ri_base),
    h("Residence Inn New York Manhattan/Midtown East", neighborhood="Midtown East", review_base=8.1, review_count=1400, price_min=185, price_max=370, rooms=["Studio Suite (1 king bed)", "One-Bedroom Suite (1 king bed)"], **ri_base),
    h("Residence Inn New York Downtown Manhattan", neighborhood="Financial District", review_base=8.0, review_count=1100, price_min=170, price_max=340, rooms=["Studio Suite (1 king bed)", "One-Bedroom Suite (1 king bed)"], **ri_base),
]

# ── AC HOTEL (Marriott) — 4-star lifestyle ───────────────────────────────
additional += [
    h("AC Hotel New York Times Square", stars=4, neighborhood="Times Square", review_base=8.3, review_count=2000, price_min=190, price_max=380, rooms=KQ, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
    h("AC Hotel New York Downtown", stars=4, neighborhood="Financial District", review_base=8.1, review_count=1200, price_min=170, price_max=340, rooms=KQ, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
]

# ── W HOTEL (Marriott) — 4-star ──────────────────────────────────────────
additional += [
    h("W New York - Times Square", stars=4, neighborhood="Times Square", review_base=7.8, review_count=2800, price_min=220, price_max=440, rooms=["Wonderful King (1 king bed)", "Spectacular King (1 king bed)"], canc_prob=0.6, bkfst_prob=0.1, amenities=FULL_BAR),
    h("W New York - Union Square", stars=4, neighborhood="Union Square", review_base=8.2, review_count=1600, price_min=240, price_max=480, rooms=["Wonderful King (1 king bed)", "Spectacular King (1 king bed)"], canc_prob=0.6, bkfst_prob=0.1, amenities=FULL_BAR),
]

# ── WESTIN (Marriott) — 4-star ───────────────────────────────────────────
additional += [
    h("The Westin New York Grand Central", stars=4, neighborhood="Midtown East", review_base=8.0, review_count=3200, price_min=210, price_max=420, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=["Free WiFi", "Restaurant", "Fitness center", "Heavenly Bed"]),
    h("The Westin New York at Times Square", stars=4, neighborhood="Times Square", review_base=7.9, review_count=4100, price_min=220, price_max=440, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=["Free WiFi", "Restaurant", "Fitness center", "Bar"]),
]

# ── SHERATON (Marriott) — 4-star ─────────────────────────────────────────
additional += [
    h("Sheraton New York Times Square Hotel", stars=4, neighborhood="Times Square", review_base=7.5, review_count=5200, price_min=195, price_max=390, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
    h("Sheraton Tribeca New York Hotel", stars=4, neighborhood="TriBeCa", review_base=8.0, review_count=1600, price_min=180, price_max=360, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
    h("Sheraton Brooklyn New York Hotel", stars=4, neighborhood="Downtown Brooklyn", review_base=7.8, review_count=1400, price_min=150, price_max=300, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=["Free WiFi", "Restaurant", "Fitness center", "Pool"]),
]

# ── HYATT — various tiers ────────────────────────────────────────────────
additional += [
    h("Hyatt Grand Central New York", stars=4, neighborhood="Midtown East", review_base=7.8, review_count=3500, price_min=200, price_max=400, rooms=KD, canc_prob=0.6, bkfst_prob=0.1, amenities=FULL),
    h("Hyatt Centric Midtown 5th Avenue New York", stars=4, neighborhood="Midtown", review_base=8.4, review_count=1800, price_min=210, price_max=420, rooms=KQ, canc_prob=0.6, bkfst_prob=0.1, amenities=FULL_BAR),
    h("Hyatt Centric Times Square New York", stars=4, neighborhood="Times Square", review_base=8.3, review_count=2200, price_min=220, price_max=440, rooms=KQ, canc_prob=0.6, bkfst_prob=0.1, amenities=FULL_BAR),
    h("Hyatt Centric Wall Street New York", stars=4, neighborhood="Financial District", review_base=8.5, review_count=1200, price_min=180, price_max=360, rooms=KQ, canc_prob=0.6, bkfst_prob=0.1, amenities=FULL_BAR),
    h("Hyatt Place New York Midtown South", stars=3, neighborhood="Midtown", review_base=8.1, review_count=1600, price_min=150, price_max=300, rooms=KD, canc_prob=0.6, bkfst_prob=0.6, amenities=["Free WiFi", "Breakfast", "Fitness center"]),
    h("Hyatt Place New York/Chelsea", stars=3, neighborhood="Chelsea", review_base=8.0, review_count=1200, price_min=140, price_max=280, rooms=KD, canc_prob=0.6, bkfst_prob=0.6, amenities=["Free WiFi", "Breakfast", "Fitness center"]),
    h("Hyatt Regency Times Square", stars=4, neighborhood="Times Square", review_base=7.9, review_count=2600, price_min=210, price_max=420, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
    h("Park Hyatt New York", stars=5, neighborhood="Midtown", review_base=9.1, review_count=800, price_min=650, price_max=1300, rooms=["Park King (1 king bed)", "Park Deluxe King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=LUXURY_POOL),
    h("Thompson Central Park New York", stars=4, neighborhood="Midtown West", review_base=8.0, review_count=2400, price_min=280, price_max=560, rooms=KQ, canc_prob=0.6, bkfst_prob=0.1, amenities=FULL_BAR),
    h("The Beekman, A Thompson Hotel", stars=5, neighborhood="Financial District", review_base=8.8, review_count=1400, price_min=320, price_max=640, rooms=["King Room (1 king bed)", "Turret Suite (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=["Free WiFi", "Restaurant", "Bar", "Fitness center", "Historic atrium"]),
]

# ── WYNDHAM / BUDGET CHAINS — 2-3 star ───────────────────────────────────
additional += [
    h("Wyndham New Yorker Hotel", stars=3, neighborhood="Midtown", review_base=7.4, review_count=4200, price_min=120, price_max=240, rooms=KQD, canc_prob=0.4, bkfst_prob=0.1, amenities=STANDARD),
    h("Ramada by Wyndham New York/Eastside", stars=2, neighborhood="Midtown East", review_base=6.8, review_count=1800, price_min=80, price_max=160, rooms=KQ, canc_prob=0.3, bkfst_prob=0.1, amenities=BASIC),
    h("Days Inn by Wyndham NYC Midtown 45", stars=2, neighborhood="Midtown", review_base=6.5, review_count=2200, price_min=75, price_max=150, rooms=KQ, canc_prob=0.3, bkfst_prob=0.1, amenities=BASIC),
    h("La Quinta Inn & Suites New York Times Square South", stars=2, neighborhood="Midtown", review_base=7.2, review_count=1600, price_min=90, price_max=180, rooms=KD, canc_prob=0.4, bkfst_prob=0.5, amenities=["Free WiFi", "Breakfast"]),
    h("Best Western Plus Hospitality House Suites", stars=3, neighborhood="Midtown East", review_base=7.8, review_count=1400, price_min=130, price_max=260, rooms=["Studio Suite (1 queen bed)", "One-Bedroom Suite (1 king bed)"], canc_prob=0.5, bkfst_prob=0.3, amenities=["Free WiFi", "Kitchen", "Fitness center"]),
]

# ── INDEPENDENT / BOUTIQUE — various tiers ───────────────────────────────
additional += [
    # Budget / Economy boutique
    h("The Jane Hotel", stars=2, neighborhood="West Village", review_base=7.3, review_count=2400, price_min=80, price_max=160, rooms=["Standard Cabin (1 twin bed)", "Captain's Cabin (1 queen bed)"], canc_prob=0.3, bkfst_prob=0.0, amenities=["Free WiFi", "Bar", "Ballroom"]),
    h("Freehand New York", stars=3, neighborhood="Flatiron", review_base=8.0, review_count=1800, price_min=130, price_max=260, rooms=["Queen Room (1 queen bed)", "King Room (1 king bed)"], canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar", "Rooftop"]),
    h("The Local NYC", stars=2, neighborhood="Long Island City, Queens", review_base=8.2, review_count=900, price_min=70, price_max=140, rooms=["Queen Room (1 queen bed)", "Bunk Room (2 bunk beds)"], canc_prob=0.3, bkfst_prob=0.0, amenities=["Free WiFi", "Shared kitchen"]),
    h("HI New York City Hostel", stars=1, neighborhood="Upper West Side", review_base=7.0, review_count=3200, price_min=45, price_max=90, rooms=["Dorm Bed (1 bunk bed)", "Private Room (1 double bed)"], canc_prob=0.2, bkfst_prob=0.0, amenities=["Free WiFi", "Shared kitchen", "Lounge"]),
    h("Pod Brooklyn", stars=3, neighborhood="Williamsburg, Brooklyn", review_base=8.1, review_count=800, price_min=85, price_max=170, rooms=["Queen Pod (1 queen bed)", "King Pod (1 king bed)"], canc_prob=0.4, bkfst_prob=0.0, amenities=["Free WiFi", "Rooftop bar"]),

    # Mid-range boutique
    h("The Ludlow Hotel", stars=4, neighborhood="Lower East Side", review_base=8.6, review_count=1100, price_min=230, price_max=460, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar", "Garden"]),
    h("The Bowery Hotel", stars=4, neighborhood="East Village", review_base=8.8, review_count=850, price_min=310, price_max=620, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar", "Fireplace lounge"]),
    h("The Marlton Hotel", stars=4, neighborhood="Greenwich Village", review_base=8.5, review_count=950, price_min=200, price_max=400, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar"]),
    h("The Standard East Village", stars=4, neighborhood="East Village", review_base=8.3, review_count=1200, price_min=220, price_max=440, rooms=KQ, canc_prob=0.5, bkfst_prob=0.05, amenities=["Free WiFi", "Restaurant", "Bar"]),
    h("The Standard High Line", stars=4, neighborhood="Meatpacking District", review_base=8.5, review_count=2000, price_min=290, price_max=580, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar", "Beer garden"]),
    h("NoMo SoHo", stars=4, neighborhood="SoHo", review_base=8.2, review_count=1100, price_min=240, price_max=480, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar"]),
    h("The Greenwich Hotel", stars=5, neighborhood="TriBeCa", review_base=9.3, review_count=500, price_min=600, price_max=1200, rooms=["King Room (1 king bed)", "Deluxe King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=["Free WiFi", "Restaurant", "Spa", "Pool", "Fitness center"]),
    h("The Mercer", stars=5, neighborhood="SoHo", review_base=9.0, review_count=650, price_min=550, price_max=1100, rooms=["Mercer King (1 king bed)", "Penthouse Loft (1 king bed)"], canc_prob=0.5, bkfst_prob=0.15, amenities=["Free WiFi", "Restaurant", "Bar"]),
    h("1 Hotel Brooklyn Bridge", stars=4, neighborhood="Brooklyn Heights", review_base=8.6, review_count=1800, price_min=310, price_max=620, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Rooftop pool", "Restaurant", "Spa"]),
    h("1 Hotel Central Park", stars=5, neighborhood="Midtown", review_base=8.7, review_count=1200, price_min=400, price_max=800, rooms=KQ, canc_prob=0.5, bkfst_prob=0.15, amenities=["Free WiFi", "Restaurant", "Spa", "Fitness center"]),
    h("The Langham New York Fifth Avenue", stars=5, neighborhood="Midtown", review_base=9.0, review_count=700, price_min=480, price_max=960, rooms=["Deluxe King (1 king bed)", "Premier King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=LUXURY),
    h("Mandarin Oriental New York", stars=5, neighborhood="Upper West Side", review_base=8.9, review_count=1000, price_min=700, price_max=1400, rooms=["King Room (1 king bed)", "Hudson River View King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=LUXURY_POOL),
    h("The Pierre, A Taj Hotel", stars=5, neighborhood="Upper East Side", review_base=8.8, review_count=1400, price_min=420, price_max=840, rooms=["Superior King (1 king bed)", "Deluxe King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=LUXURY),
    h("The Carlyle, A Rosewood Hotel", stars=5, neighborhood="Upper East Side", review_base=9.1, review_count=600, price_min=750, price_max=1500, rooms=["Deluxe King (1 king bed)", "Superior King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=["Free WiFi", "Spa", "Restaurant", "Live jazz", "Fitness center"]),
    h("Lotte New York Palace", stars=5, neighborhood="Midtown East", review_base=8.4, review_count=2200, price_min=360, price_max=720, rooms=["Deluxe King (1 king bed)", "Palace King (1 king bed)"], canc_prob=0.6, bkfst_prob=0.2, amenities=LUXURY),
    h("The Peninsula New York", stars=5, neighborhood="Midtown", review_base=9.2, review_count=900, price_min=800, price_max=1600, rooms=["Superior King (1 king bed)", "Deluxe King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=LUXURY_POOL),
    h("The St. Regis New York", stars=5, neighborhood="Midtown", review_base=9.0, review_count=1100, price_min=700, price_max=1400, rooms=["Superior King (1 king bed)", "Grand Deluxe King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=["Free WiFi", "Butler service", "Restaurant", "Bar", "Fitness center"]),

    # Mid-range / Upper mid-range
    h("Arlo Midtown", stars=3, neighborhood="Times Square", review_base=8.0, review_count=1800, price_min=130, price_max=260, rooms=KQ, canc_prob=0.5, bkfst_prob=0.05, amenities=["Free WiFi", "Rooftop bar", "Restaurant"]),
    h("The Renwick Hotel", stars=4, neighborhood="Midtown East", review_base=8.3, review_count=900, price_min=190, price_max=380, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Bar"]),
    h("Walker Hotel Greenwich Village", stars=4, neighborhood="Greenwich Village", review_base=8.4, review_count=700, price_min=210, price_max=420, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant"]),
    h("Walker Hotel Tribeca", stars=4, neighborhood="TriBeCa", review_base=8.2, review_count=500, price_min=200, price_max=400, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant"]),
    h("The Hoxton Williamsburg", stars=4, neighborhood="Williamsburg, Brooklyn", review_base=8.5, review_count=1000, price_min=180, price_max=360, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Rooftop bar"]),
    h("Hotel Hugo", stars=4, neighborhood="SoHo", review_base=8.1, review_count=1200, price_min=200, price_max=400, rooms=KQ, canc_prob=0.5, bkfst_prob=0.05, amenities=["Free WiFi", "Rooftop bar", "Restaurant"]),
    h("The James New York NoMad", stars=4, neighborhood="NoMad", review_base=8.3, review_count=800, price_min=220, price_max=440, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Rooftop bar", "Fitness center"]),
    h("Smyth Tribeca", stars=4, neighborhood="TriBeCa", review_base=8.4, review_count=700, price_min=230, price_max=460, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant"]),
    h("The Rockaway Hotel & Spa", stars=4, neighborhood="Rockaway Beach, Queens", review_base=8.3, review_count=500, price_min=180, price_max=360, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Pool", "Spa", "Restaurant"]),
    h("Wythe Hotel", stars=4, neighborhood="Williamsburg, Brooklyn", review_base=8.6, review_count=1100, price_min=210, price_max=420, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant", "Rooftop bar"]),
    h("McCarren Hotel & Pool", stars=4, neighborhood="Williamsburg, Brooklyn", review_base=8.0, review_count=800, price_min=160, price_max=320, rooms=KQ, canc_prob=0.5, bkfst_prob=0.05, amenities=["Free WiFi", "Pool", "Restaurant"]),
    h("The Williamsburg Hotel", stars=4, neighborhood="Williamsburg, Brooklyn", review_base=8.2, review_count=900, price_min=190, price_max=380, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Pool", "Restaurant", "Bar"]),

    # Budget Brooklyn/Queens
    h("Even Hotel Brooklyn", stars=3, neighborhood="Downtown Brooklyn", review_base=8.3, review_count=700, price_min=120, price_max=240, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Fitness center"]),
    h("Hotel Le Bleu", stars=3, neighborhood="Park Slope, Brooklyn", review_base=8.0, review_count=500, price_min=110, price_max=220, rooms=KQ, canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Restaurant"]),
    h("Z Hotel New York", stars=3, neighborhood="Long Island City, Queens", review_base=7.9, review_count=1400, price_min=95, price_max=190, rooms=KQ, canc_prob=0.4, bkfst_prob=0.1, amenities=["Free WiFi", "Rooftop bar"]),
    h("Paper Factory Hotel", stars=3, neighborhood="Long Island City, Queens", review_base=8.3, review_count=600, price_min=100, price_max=200, rooms=KQ, canc_prob=0.4, bkfst_prob=0.1, amenities=["Free WiFi", "Bar", "Restaurant"]),
    h("The Boro Hotel", stars=4, neighborhood="Long Island City, Queens", review_base=8.5, review_count=800, price_min=130, price_max=260, rooms=KQ, canc_prob=0.5, bkfst_prob=0.05, amenities=["Free WiFi", "Bar", "Manhattan views"]),
    h("Henry Norman Hotel", stars=4, neighborhood="Greenpoint, Brooklyn", review_base=8.4, review_count=500, price_min=140, price_max=280, rooms=["King Suite (1 king bed)", "Loft Suite (1 king bed)"], canc_prob=0.5, bkfst_prob=0.1, amenities=["Free WiFi", "Garden"]),

    # Renaissance / Marriott upper tier
    h("Renaissance New York Midtown Hotel", stars=4, neighborhood="Midtown", review_base=8.1, review_count=2200, price_min=220, price_max=440, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
    h("New York Marriott Marquis", stars=4, neighborhood="Times Square", review_base=7.8, review_count=8500, price_min=250, price_max=500, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=["Free WiFi", "Restaurant", "Bar", "Fitness center", "Theater"]),
    h("New York Marriott Downtown", stars=4, neighborhood="Financial District", review_base=8.0, review_count=3200, price_min=200, price_max=400, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL),
    h("The Lexington Hotel, Autograph Collection", stars=4, neighborhood="Midtown East", review_base=8.1, review_count=2800, price_min=210, price_max=420, rooms=KD, canc_prob=0.6, bkfst_prob=0.15, amenities=FULL_BAR),
    h("Moxy NYC Chelsea", stars=3, neighborhood="Chelsea", review_base=8.1, review_count=1400, price_min=130, price_max=260, rooms=KQ, canc_prob=0.5, bkfst_prob=0.05, amenities=["Free WiFi", "Rooftop bar", "Restaurant"]),

    # Conrad / Canopy (Hilton upper)
    h("Conrad New York Midtown", stars=5, neighborhood="Midtown", review_base=8.5, review_count=1600, price_min=320, price_max=640, rooms=["King Suite (1 king bed)", "Deluxe Suite (1 king bed)"], canc_prob=0.6, bkfst_prob=0.15, amenities=LUXURY),
    h("Conrad New York Downtown", stars=5, neighborhood="Financial District", review_base=8.6, review_count=2000, price_min=290, price_max=580, rooms=["King Suite (1 king bed)", "Deluxe Suite (1 king bed)"], canc_prob=0.6, bkfst_prob=0.15, amenities=LUXURY),
    h("Tempo by Hilton New York Times Square", stars=3, neighborhood="Times Square", review_base=8.2, review_count=600, price_min=175, price_max=350, rooms=KD, canc_prob=0.6, bkfst_prob=0.3, amenities=["Free WiFi", "Fitness center", "Kitchen"]),

    # Loews / Others
    h("Loews Regency Hotel New York", stars=5, neighborhood="Upper East Side", review_base=8.7, review_count=1100, price_min=380, price_max=760, rooms=["Deluxe King (1 king bed)", "Park Avenue King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=LUXURY),
    h("The Ritz-Carlton New York, NoMad", stars=5, neighborhood="NoMad", review_base=9.0, review_count=400, price_min=800, price_max=1600, rooms=["Deluxe King (1 king bed)", "Premier King (1 king bed)"], canc_prob=0.5, bkfst_prob=0.2, amenities=LUXURY_POOL),
]

# ── Combine and write ─────────────────────────────────────────────────────

all_hotels = existing + additional

pool_data["hotels"] = all_hotels
pool_data["description"] = (
    f"Pool of {len(all_hotels)} realistic NYC hotel profiles for conjoint task generation. "
    "Each hotel has fixed attributes (name, stars, neighborhood, amenities) and variable "
    "attributes with realistic ranges (price, room type, cancellation, breakfast, review "
    "score/count). Variable attributes are randomized per task appearance within the "
    "specified ranges to reflect real-world variation."
)

out_path = os.path.join(script_dir, "hotel_pool.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(pool_data, f, indent=2, ensure_ascii=False)

# Summary stats
from collections import Counter
star_counts = Counter(h["stars"] for h in all_hotels)
print(f"Total hotels: {len(all_hotels)}")
print(f"By star rating: {dict(sorted(star_counts.items()))}")
print(f"Price range: ${min(h['price_min'] for h in all_hotels)} - ${max(h['price_max'] for h in all_hotels)}")
neighborhoods = set(h["neighborhood"] for h in all_hotels)
print(f"Neighborhoods: {len(neighborhoods)} unique")
print(f"Written to: {out_path}")
