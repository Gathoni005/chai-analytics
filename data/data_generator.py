"""
ChaiMetrics — Synthetic KTDA Farm Data Generator v2
=====================================================
Based on two REAL KTDA factories:

  1. Weru Tea Factory
     County    : Tharaka-Nithi
     Location  : Muthambi, 2km from Marima, between Chuka and Chogoria
     Altitude  : ~1,750m (eastern slopes of Mt Kenya, Maara/Chuka highlands)
     Zone      : Only tea factory in Tharaka-Nithi County
     Catchment : Maara and Chuka divisions

  2. Rukuriri Tea Factory
     County    : Embu
     Location  : Kyeni North, Runyenjes Division, Embu East District
     GPS       : -0.37S, 37.54E
     Altitude  : 1,703m (south-eastern slopes of Mt Kenya)
     Founded   : 1984
     Growers   : ~10,161
     KTDA Zone : 6 (East of Rift)
     Bonus 2024: KES 57.50/kg (18.75M kg processed)
     Certs     : Fairtrade (2008), Rainforest Alliance, Ethical Tea Partnership

Collection centres derived from the real geographic catchments of each factory.
Rainfall, altitude, and yield parameters calibrated to real regional data.

Agronomic rules:
  - Season runs July–June (season label = ending year)
  - Pruning in June (season_idx 11) and August (season_idx 1)
  - Post-pruning recovery curve: 45% → 75% → 105% → 100%
  - Rainfall profile follows long rains (Mar–May) + short rains (Oct–Dec)
  - Minibonus paid: Jul, Aug, Sep, Nov, Dec (season_idx 0,1,2,4,5)
  - Annual bonus paid: Jun (season_idx 11)
  - Fertiliser/manure: 0–2 applications per season, logged with date and qty

Output:
  farms.json, ktda_pricing.json, synthetic_metadata.json
"""

import json, random, math, os

random.seed(2025)

# ═══════════════════════════════════════════════════════════════
# FACTORY DEFINITIONS  (real data)
# ═══════════════════════════════════════════════════════════════
FACTORIES = [
    {
        "factory_code":        "WRU-01",
        "factory_name":        "Weru Tea Factory Company Limited",
        "county":              "Tharaka-Nithi",
        "region":              "Maara/Chuka Highlands",
        "division":            "Maara",
        "ktda_zone":           "Zone 5 (East of Rift)",
        "factory_lat":        -0.338,
        "factory_lng":         37.637,
        "altitude_m":          1750,
        # Rainfall: Maara/Chuka highlands avg ~1,400mm/yr → ~117mm/month
        # Long rains Mar–May dominant; short rains Oct–Dec moderate
        "rainfall_mean_mm":    117,
        "rainfall_std_mm":     45,
        # Base yield calibrated to eastern Mt Kenya smallholder (~380 kg/ha/month peak)
        "base_yield_kg_per_ha": 375,
        "optimal_rainfall_mm": 120,
        # Weru is the only factory in Tharaka-Nithi; large catchment
        "n_farms":             120,
        # Collection centres: real sub-locations in Maara/Chuka catchment
        "collection_centres": [
            {
                "name": "Marima",
                "lat": -0.325, "lng": 37.625,
                "altitude_m": 1780,
                "rainfall_offset": 1.05,   # slightly wetter, higher
                "notes": "Closest to factory; Nithi River watershed"
            },
            {
                "name": "Chuka",
                "lat": -0.338, "lng": 37.651,
                "altitude_m": 1720,
                "rainfall_offset": 1.00,
                "notes": "Chuka town catchment; B6 Embu-Meru road corridor"
            },
            {
                "name": "Chogoria",
                "lat": -0.298, "lng": 37.667,
                "altitude_m": 1800,
                "rainfall_offset": 1.08,   # higher altitude, more rainfall
                "notes": "Upper highland zone; Mt Kenya forest edge"
            },
            {
                "name": "Marimba",
                "lat": -0.355, "lng": 37.618,
                "altitude_m": 1700,
                "rainfall_offset": 0.95,
                "notes": "Lower Maara slopes; slightly drier"
            },
            {
                "name": "Nthangaari",
                "lat": -0.312, "lng": 37.642,
                "altitude_m": 1760,
                "rainfall_offset": 1.02,
                "notes": "Central Maara; mixed tea and coffee zone"
            },
            {
                "name": "Kiriani",
                "lat": -0.348, "lng": 37.660,
                "altitude_m": 1730,
                "rainfall_offset": 0.98,
                "notes": "Chuka north; river Tana upper tributaries"
            },
        ],
    },
    {
        "factory_code":        "RKR-01",
        "factory_name":        "Rukuriri Tea Factory Company Limited",
        "county":              "Embu",
        "region":              "Kyeni North, Embu East",
        "division":            "Runyenjes",
        "ktda_zone":           "Zone 6 (East of Rift)",
        "factory_lat":        -0.374,
        "factory_lng":         37.543,
        "altitude_m":          1703,      # exact from Aloeus/KTDA records
        # Rainfall: Embu east ~1,200–1,500mm/yr → ~110mm/month
        # South-eastern Mt Kenya slopes; reliable bimodal rainfall
        "rainfall_mean_mm":    112,
        "rainfall_std_mm":     42,
        # Rukuriri is Fairtrade; good management → slightly higher productivity
        "base_yield_kg_per_ha": 395,
        "optimal_rainfall_mm": 115,
        # 10,161 real growers; 6 electoral zones → ~6 collection centres
        "n_farms":             120,
        # Collection centres: Kyeni North / Runyenjes division sub-locations
        "collection_centres": [
            {
                "name": "Rukuriri",
                "lat": -0.374, "lng": 37.543,
                "altitude_m": 1703,
                "rainfall_offset": 1.00,
                "notes": "Factory gate centre; Kyeni North sub-location"
            },
            {
                "name": "Kyeni",
                "lat": -0.385, "lng": 37.531,
                "altitude_m": 1680,
                "rainfall_offset": 0.97,
                "notes": "Kyeni North; Embu-Meru highway corridor"
            },
            {
                "name": "Runyenjes",
                "lat": -0.363, "lng": 37.558,
                "altitude_m": 1720,
                "rainfall_offset": 1.03,
                "notes": "Runyenjes division centre; upper Kyeni"
            },
            {
                "name": "Mbeti",
                "lat": -0.390, "lng": 37.520,
                "altitude_m": 1660,
                "rainfall_offset": 0.94,
                "notes": "Lower Mbeti sub-location; drier edge of catchment"
            },
            {
                "name": "Kagaari",
                "lat": -0.358, "lng": 37.572,
                "altitude_m": 1740,
                "rainfall_offset": 1.06,
                "notes": "Upper Kagaari; forest edge zone, highest yields"
            },
            {
                "name": "Ena",
                "lat": -0.380, "lng": 37.548,
                "altitude_m": 1695,
                "rainfall_offset": 1.01,
                "notes": "Ena River watershed; fertile alluvial soils"
            },
        ],
    },
]

# ═══════════════════════════════════════════════════════════════
# SEASON / CALENDAR CONSTANTS
# ═══════════════════════════════════════════════════════════════
# season_idx: 0=Jul, 1=Aug, 2=Sep, 3=Oct, 4=Nov, 5=Dec,
#             6=Jan, 7=Feb, 8=Mar, 9=Apr, 10=May, 11=Jun
SEASON_MONTHS    = ["Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun"]
MINIBONUS_IDX    = {0, 1, 2, 4, 5}    # Jul, Aug, Sep, Nov, Dec
PRUNING_IDX      = {11, 1}             # Jun, Aug
ANNUAL_BONUS_IDX = 11                  # Jun = season end

# Seasonal rainfall profile (relative weight per month)
# Long rains: Mar(idx8)–May(idx10) → dominant peak
# Short rains: Oct(idx3)–Dec(idx5) → secondary peak
# Dry: Jan(idx6)–Feb(idx7), Jul(idx0)–Sep(idx2) moderate
RAINFALL_PROFILE = [
#  Jul   Aug   Sep   Oct   Nov   Dec   Jan   Feb   Mar   Apr   May   Jun
   0.70, 0.62, 0.80, 1.15, 1.35, 1.20, 0.85, 0.65, 1.55, 1.80, 1.60, 0.82
]

# Seasonal yield profile (relative productivity per month)
# Tea flushes strongly with long rains (Mar–May)
# Moderate during short rains; lowest during pruning months
YIELD_PROFILE = [
#  Jul   Aug   Sep   Oct   Nov   Dec   Jan   Feb   Mar   Apr   May   Jun
   0.78, 0.42, 0.82, 1.05, 1.08, 1.00, 0.88, 0.72, 1.22, 1.38, 1.28, 0.48
# Note: Aug and Jun depressed by pruning even in base profile
]

# Post-pruning recovery curve (months_since_pruning → yield multiplier)
PRUNING_RECOVERY = {0: 0.45, 1: 0.78, 2: 1.08, 3: 1.00}

# Kenyan names pool
FIRST_NAMES = [
    "Joseph","Mary","James","Grace","Peter","Faith","John","Rose","Paul","Agnes",
    "Stephen","Jane","David","Esther","Daniel","Lucy","Michael","Hannah","Patrick",
    "Beatrice","Samuel","Alice","George","Florence","Francis","Naomi","Charles","Lydia",
    "Simon","Mercy","Robert","Dorothy","Thomas","Eunice","Henry","Priscilla","Edward",
    "Tabitha","Philip","Miriam","Benjamin","Rebecca","Joshua","Leah","Emmanuel","Sarah",
    "Solomon","Ruth","Elijah","Deborah","Isaiah","Judith","Jeremiah","Martha","Nathan",
    "Rachael","Aaron","Dorcas","Abel","Salome","Moses","Purity","Caleb","Charity","Mark",
    "Gladys","Boniface","Teresia","Cyprian","Wanjiku","Mwangi","Kamau","Njoroge","Wambui",
]
SURNAMES = [
    "Mwangi","Kamau","Njoroge","Kimani","Kariuki","Gitau","Waweru","Gicheru","Muigai",
    "Ndungu","Kinyua","Murimi","Muthoni","Waithera","Wanjiku","Nyambura","Njeri","Wangari",
    "Gatimu","Gatheru","Macharia","Kabiru","Ndirangu","Karanja","Kiragu","Irungu","Maina",
    "Muturi","Gichuki","Kuria","Muchai","Mureithi","Wainaina","Muriuki","Gitahi","Kung'u",
    "Ngugi","Wambugu","Mwenda","Mutua","Mwangangi","Kilonzo","Muema","Mutiso","Musyoka",
    "Muthama","Nzomo","Kavata","Ndeti","Wambua","Mwau","Mumo","Kitonga","Ndolo","Nthiani",
    # Tharaka-Nithi / Embu specific surnames
    "Njeru","Nthiga","Munyi","Gitonga","Mugambi","Muriithi","Nkirote","Gacheri","Muthee",
    "Muchiri","Gatobu","Gichure","Mutegi","Kirimi","Mbae","Njogu","Thuranira","Gichohi",
    "Ngari","Muchangi","Kirira","Nkonge","Thuku","Mutinda","Kitheka","Nganga","Munyiri",
]

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def season_idx_to_period(season_year, idx):
    """Returns 'YYYY-MM' string for a season index within a season year."""
    if idx <= 5:   # Jul–Dec → prior calendar year
        return f"{season_year - 1}-{idx + 7:02d}"
    else:          # Jan–Jun → season year
        return f"{season_year}-{idx - 5:02d}"

def rainfall_yield_factor(mm, optimal):
    """Parabolic yield response to rainfall. Peak at optimal, falls off either side."""
    if mm < 20:   return 0.45
    if mm < 50:   return 0.55 + (mm - 20) / 30 * 0.18
    if mm < optimal:
        return 0.73 + (mm - 50) / (optimal - 50) * 0.30
    if mm < optimal + 70:
        return 1.03 - (mm - optimal) / 70 * 0.08
    return 0.90   # waterlogging

def pruning_factor(idx, last_prune_idx):
    """Yield multiplier based on distance from most recent pruning event."""
    if last_prune_idx is None:
        return 1.0
    dist = idx - last_prune_idx
    # Wrap around season boundary
    if dist < 0:
        dist += 12
    return PRUNING_RECOVERY.get(dist, 1.0)

# ═══════════════════════════════════════════════════════════════
# RAINFALL GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_centre_rainfall(factory, centre, seasons):
    """
    Generates monthly rainfall for a collection centre across all seasons.
    Centre rainfall = factory base × centre offset × year shock × monthly profile × noise.
    All farms in same centre share this rainfall (correlated neighbours).
    Returns {season_year: [12 monthly mm values]}
    """
    result = {}
    base    = factory["rainfall_mean_mm"] * centre["rainfall_offset"]
    std     = factory["rainfall_std_mm"]
    for yr in seasons:
        year_shock = random.gauss(1.0, 0.13)   # drought/flood year affects whole season
        monthly = []
        for idx in range(12):
            mm = max(0.0, random.gauss(
                base * RAINFALL_PROFILE[idx] * year_shock,
                std * 0.45
            ))
            monthly.append(round(mm, 1))
        result[yr] = monthly
    return result

# ═══════════════════════════════════════════════════════════════
# YIELD GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_farm_seasons(farm_meta, centre_rainfall, seasons):
    """
    Generates historical_seasons list for one farm.
    Incorporates: base productivity, pruning curve, rainfall, fertiliser, trend, noise.
    """
    ha        = farm_meta["hectares"]
    base_kgha = farm_meta["base_kg_per_ha"]
    pf        = farm_meta["productivity_factor"]
    trend     = farm_meta["trend_per_season"]   # e.g. +0.008 = +0.8%/yr
    optimal   = farm_meta["optimal_rainfall_mm"]

    result = []
    for s_num, season_year in enumerate(seasons):
        trend_mult = 1.0 + trend * s_num
        rain       = centre_rainfall.get(season_year, [optimal * 0.9] * 12)

        # Fertiliser / manure: 0–2 applications this season
        n_fert = random.choices([0, 1, 2], weights=[0.10, 0.55, 0.35])[0]
        avail  = [i for i in range(12) if i not in PRUNING_IDX]
        fert_events = []
        for fm in random.sample(avail, min(n_fert, len(avail))):
            qty  = round(random.uniform(20, 90), 1)
            kind = random.choices(["CAN", "NPK", "organic_manure"],
                                  weights=[0.35, 0.40, 0.25])[0]
            fert_events.append({
                "season_month_idx": fm,
                "season_month":     SEASON_MONTHS[fm],
                "input_type":       kind,
                "quantity_kg":      qty,
            })

        monthly_kg   = []
        monthly_earn = []   # filled later with pricing

        # Track last pruning event index within the season
        # Jun (idx 11) is pruned first in season order, Aug (idx 1) second
        for m_idx in range(12):
            # Determine last pruning before this month
            if m_idx == 0:      # Jul — no pruning yet this season; carry Jun from prior
                last_prune = None
                p_factor   = 1.0
            elif m_idx == 1:    # Aug — pruning happens this month
                last_prune = 1
                p_factor   = PRUNING_RECOVERY[0]
            elif 2 <= m_idx <= 4:
                last_prune = 1
                p_factor   = PRUNING_RECOVERY.get(m_idx - 1, 1.0)
            elif m_idx == 11:   # Jun — pruning
                last_prune = 11
                p_factor   = PRUNING_RECOVERY[0]
            else:
                # Jun pruning recovery for rest of season (idx 12 → idx 0 next)
                p_factor   = 1.0

            # Rainfall effect
            r_factor = rainfall_yield_factor(rain[m_idx], optimal)

            # Fertiliser lag effect (+1 and +2 months after application)
            fert_factor = 1.0
            for fe in fert_events:
                dist = m_idx - fe["season_month_idx"]
                if dist == 1:
                    fert_factor += 0.11 * (fe["quantity_kg"] / 50)
                elif dist == 2:
                    fert_factor += 0.06 * (fe["quantity_kg"] / 50)

            # Base yield
            s_factor = YIELD_PROFILE[m_idx]
            noise    = random.gauss(1.0, 0.055)
            kg = max(0.0, (
                ha * base_kgha * pf *
                trend_mult * p_factor * r_factor *
                fert_factor * s_factor * noise
            ))
            monthly_kg.append(round(kg, 1))

        result.append({
            "season_year":             season_year,
            "monthly_kg":              monthly_kg,
            "monthly_earn":            [],    # filled by compute_earnings
            "yearly_bonus":            0.0,   # filled by compute_earnings
            "fertiliser_applications": fert_events,
            "season_rainfall_mm":      rain,
        })

    return result

# ═══════════════════════════════════════════════════════════════
# PRICING GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_ktda_pricing(seasons):
    """
    Generates ktda_pricing collection.
    Calibrated to real Rukuriri data:
      - Bonus 2024: KES 57.50/kg  → annual_bonus ~57.50
      - Monthly rates grew from ~18 KES/kg (2011) to ~32 KES/kg (2024)
      - Minibonus: Jul/Aug/Sep/Nov/Dec only
      - Slight factory-level premium for Rukuriri (Fairtrade)
    """
    records = []

    # Base rates at start (2011)
    base = {
        "monthly":  18.50,
        "minibonus": 2.80,
        "annual":   14.50,
    }
    # Annual growth rates derived from real trajectory
    growth = {
        "monthly":  0.042,   # ~4.2% per year → ~32 by 2024
        "minibonus": 0.032,
        "annual":   0.090,   # faster bonus growth (auction price recovery)
    }

    n = len(seasons)
    for s_num, season_year in enumerate(seasons):
        yr_shock = random.gauss(1.0, 0.025)   # small year-level noise

        monthly_rate  = round(base["monthly"]  * (1 + growth["monthly"]  * s_num) * yr_shock, 2)
        minibonus_rate= round(base["minibonus"] * (1 + growth["minibonus"]* s_num) * yr_shock, 2)
        annual_rate   = round(base["annual"]    * (1 + growth["annual"]   * s_num) * yr_shock, 2)

        for factory in FACTORIES:
            fc = factory["factory_code"]
            # Rukuriri Fairtrade premium ~2–4%
            fp = 1.03 if fc == "RKR-01" else 1.00
            fp *= random.gauss(1.0, 0.015)

            for m_idx in range(12):
                period = season_idx_to_period(season_year, m_idx)
                is_mb  = m_idx in MINIBONUS_IDX
                is_ab  = (m_idx == ANNUAL_BONUS_IDX)
                records.append({
                    "period":                      period,
                    "season_year":                 season_year,
                    "season_month_idx":            m_idx,
                    "season_month":                SEASON_MONTHS[m_idx],
                    "factory_code":                fc,
                    "monthly_rate_kes_per_kg":     round(monthly_rate  * fp, 2),
                    "minibonus_rate_kes_per_kg":   round(minibonus_rate * fp, 2) if is_mb else 0.0,
                    "annual_bonus_rate_kes_per_kg":round(annual_rate   * fp, 2) if is_ab else 0.0,
                    "is_minibonus_month":          is_mb,
                    "is_annual_bonus_month":       is_ab,
                })

    return records

# ═══════════════════════════════════════════════════════════════
# EARNINGS COMPUTATION
# ═══════════════════════════════════════════════════════════════
def compute_earnings(farm_seasons, pricing_lookup, factory_code):
    for season in farm_seasons:
        sy = season["season_year"]
        monthly_earn = []
        yearly_bonus = 0.0
        for m_idx, kg in enumerate(season["monthly_kg"]):
            key = (sy, m_idx, factory_code)
            pr  = pricing_lookup.get(key, {})
            monthly_rate   = pr.get("monthly_rate_kes_per_kg",    25.0)
            minibonus_rate = pr.get("minibonus_rate_kes_per_kg",    5.0)
            annual_rate    = pr.get("annual_bonus_rate_kes_per_kg", 50.0)
            earn = round(kg * (monthly_rate + minibonus_rate), 2)
            monthly_earn.append(earn)
            if m_idx == ANNUAL_BONUS_IDX:
                yearly_bonus = round(kg * annual_rate, 2)
        season["monthly_earn"]  = monthly_earn
        season["yearly_bonus"]  = yearly_bonus
    return farm_seasons

# ═══════════════════════════════════════════════════════════════
# FARM GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_farms(pricing_records, start_season=2010, end_season=2024):
    pricing_lookup = {
        (r["season_year"], r["season_month_idx"], r["factory_code"]): r
        for r in pricing_records
    }
    all_seasons  = list(range(start_season, end_season + 1))
    used_members = set()
    all_farms    = []

    for factory in FACTORIES:
        fc      = factory["factory_code"]
        centres = factory["collection_centres"]
        n_total = factory["n_farms"]
        base_per_centre = n_total // len(centres)
        extras  = n_total % len(centres)

        # Generate centre-level rainfall (shared by all farms in centre)
        centre_rainfalls = {}
        for c in centres:
            centre_rainfalls[c["name"]] = generate_centre_rainfall(
                factory, c, all_seasons
            )

        for c_idx, centre in enumerate(centres):
            n_this_centre = base_per_centre + (1 if c_idx < extras else 0)
            c_rain = centre_rainfalls[centre["name"]]

            for _ in range(n_this_centre):
                # Unique member number
                while True:
                    mn = f"KTD-{random.randint(10000, 99999)}"
                    if mn not in used_members:
                        used_members.add(mn)
                        break

                first = random.choice(FIRST_NAMES)
                last  = random.choice(SURNAMES)

                # Hectarage: Tharaka-Nithi farms slightly larger on average
                if fc == "WRU-01":
                    ha = round(random.choices(
                        [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                        weights=[4, 7, 18, 22, 20, 12, 8, 5, 4]
                    )[0], 2)
                else:
                    ha = round(random.choices(
                        [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                        weights=[6, 10, 22, 24, 18, 10, 6, 4]
                    )[0], 2)

                registered_year = random.randint(1988, 2020)
                actual_start    = max(registered_year, start_season)
                farm_seasons    = [y for y in all_seasons if y >= actual_start]

                # Individual farm variation
                base_kg_ha = (
                    factory["base_yield_kg_per_ha"] *
                    centre["rainfall_offset"] *
                    random.gauss(1.0, 0.14)
                )
                pf      = random.gauss(1.0, 0.08)
                trend   = random.gauss(0.007, 0.011)
                optimal = factory["optimal_rainfall_mm"] * random.gauss(1.0, 0.09)

                farm_meta = {
                    "hectares":            ha,
                    "base_kg_per_ha":      base_kg_ha,
                    "productivity_factor": pf,
                    "trend_per_season":    trend,
                    "optimal_rainfall_mm": optimal,
                }

                raw_seasons    = generate_farm_seasons(farm_meta, c_rain, farm_seasons)
                costed_seasons = compute_earnings(raw_seasons, pricing_lookup, fc)

                historical_seasons = []
                for s in costed_seasons:
                    historical_seasons.append({
                        "season_year":             s["season_year"],
                        "monthly_kg":              s["monthly_kg"],
                        "monthly_earn":            s["monthly_earn"],
                        "yearly_bonus":            s["yearly_bonus"],
                        "fertiliser_applications": s["fertiliser_applications"],
                        "season_rainfall_mm":      s["season_rainfall_mm"],
                    })

                farm_doc = {
                    "ktda_member_no":       mn,
                    "name":                 f"{last} {first} Farm",
                    "owner_name":           f"{first} {last}",
                    "factory_code":         fc,
                    "factory_name":         factory["factory_name"],
                    "county":               factory["county"],
                    "region":               factory["region"],
                    "ktda_zone":            factory["ktda_zone"],
                    "collection_centre":    centre["name"],
                    "division":             factory["division"],
                    "hectares":             ha,
                    "altitude_m":           centre["altitude_m"] + random.randint(-60, 60),
                    "registered_year":      registered_year,
                    "latitude":             round(centre["lat"] + random.gauss(0, 0.018), 6),
                    "longitude":            round(centre["lng"] + random.gauss(0, 0.018), 6),
                    "fairtrade_certified":  (fc == "RKR-01"),
                    "historical_seasons":   historical_seasons,
                    "current_season_daily": [],
                }
                all_farms.append(farm_doc)

    return all_farms

# ═══════════════════════════════════════════════════════════════
# METADATA
# ═══════════════════════════════════════════════════════════════
def generate_metadata():
    return {
        "generated_by": "ChaiMetrics synthetic data generator v2",
        "based_on":     "Real KTDA factory data — Weru (Tharaka-Nithi) and Rukuriri (Embu)",
        "factories":    [
            {**{k: v for k, v in f.items() if k != "collection_centres"},
             "rule_thresholds": {
                 "yield_gap_warn_pct":   15.0,
                 "yield_gap_alert_pct":  30.0,
                 "forecast_drop_pct":    20.0,
                 "pricing_rise_pct":     5.0,
                 "pricing_drop_pct":     5.0,
                 "gnn_anomaly_high":     0.75,
                 "gnn_anomaly_medium":   0.50,
             }}
            for f in FACTORIES
        ],
        "collection_centres": {
            f["factory_code"]: [
                {k: v for k, v in c.items()}
                for c in f["collection_centres"]
            ]
            for f in FACTORIES
        },
        "season_structure": {
            "anchor":           "July–June",
            "label":            "ending calendar year (2024 = Jul 2023 – Jun 2024)",
            "months":           SEASON_MONTHS,
            "minibonus_months": ["Jul","Aug","Sep","Nov","Dec"],
            "pruning_months":   ["Jun","Aug"],
            "annual_bonus_months": ["Jul","Aug","Sep","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun"],
            "season_month_index": {m: i for i, m in enumerate(SEASON_MONTHS)},
        },
        "agronomic_rules": {
            "pruning_suppression":   "45% yield in pruning month",
            "recovery_month_1":      "78% of normal",
            "recovery_month_2":      "108% (flush)",
            "recovery_month_3plus":  "100% (normalised)",
            "fertiliser_lag":        "+11% at month+1, +6% at month+2 per 50kg applied",
            "rainfall_optimum_mm":   "varies by collection centre (~112–120mm/month)",
        },
        "pricing_calibration": {
            "source":              "Calibrated to Rukuriri real bonus data",
            "rukuriri_bonus_2024": "KES 57.50/kg (18,751,307 kg processed)",
            "monthly_rate_2011":   "~KES 18.50/kg",
            "monthly_rate_2024":   "~KES 32.00/kg",
            "rukuriri_fairtrade_premium": "~3%",
        },
    }

# SUMMARY PRINTER

def print_summary(farms, pricing):
    print("\n" + "="*65)
    print("  ChaiMetrics — Synthetic Dataset v2")
    print("  Weru Tea Factory (Tharaka-Nithi) + Rukuriri Tea Factory (Embu)")
    print("="*65)
    print(f"  Total farms         : {len(farms)}")
    print(f"  Total pricing docs  : {len(pricing)}")
    print(f"  Seasons             : 2010 – 2024")

    for factory in FACTORIES:
        fc     = factory["factory_code"]
        f_farms= [f for f in farms if f["factory_code"] == fc]
        print(f"\n  [{fc}] {factory['factory_name']}")
        print(f"    County / Region : {factory['county']} — {factory['region']}")
        print(f"    KTDA Zone       : {factory['ktda_zone']}")
        print(f"    Altitude        : {factory['altitude_m']}m")
        print(f"    Farms generated : {len(f_farms)}")
        by_cc = {}
        for f in f_farms:
            by_cc[f["collection_centre"]] = by_cc.get(f["collection_centre"], 0) + 1
        for cc, n in sorted(by_cc.items()):
            print(f"      {cc:<18}: {n} farms")

    # Agronomic validation on one farm
    sample = next(f for f in farms if f["factory_code"] == "RKR-01")
    s2024  = next((s for s in sample["historical_seasons"] if s["season_year"] == 2024), None)
    if s2024:
        print(f"\n  Sample farm (Rukuriri): {sample['name']}")
        print(f"    Centre: {sample['collection_centre']} | {sample['hectares']}ha | reg {sample['registered_year']}")
        print(f"    Seasons on record: {len(sample['historical_seasons'])}")
        print(f"\n  2024 season — month by month:")
        print(f"  {'Mo':<5} {'KG':>8}  {'Earn KES':>11}  {'Rain mm':>8}  {'Note'}")
        notes = {1:"← pruning", 2:"← recovery", 3:"← flush", 11:"← pruning"}
        for i, m in enumerate(SEASON_MONTHS):
            fert = "fert" if any(fe["season_month_idx"]==i for fe in s2024["fertiliser_applications"]) else ""
            note = notes.get(i, "") + ("  "+fert if fert else "")
            print(f"  {m:<5} {s2024['monthly_kg'][i]:>8.1f}  {s2024['monthly_earn'][i]:>11.2f}  "
                  f"{s2024['season_rainfall_mm'][i]:>8.1f}  {note}")
        print(f"  Annual bonus: KES {s2024['yearly_bonus']:,.2f}")
        tot_earn = sum(s2024['monthly_earn']) + s2024['yearly_bonus']
        tot_kg   = sum(s2024['monthly_kg'])
        print(f"  Total yield:  {tot_kg:,.1f} kg  |  Total income: KES {tot_earn:,.2f}")

    # Pricing check
    print(f"\n  Pricing check — RKR-01 season 2024:")
    rkr = [r for r in pricing if r["factory_code"]=="RKR-01" and r["season_year"]==2024]
    print(f"  {'Mo':<5} {'Monthly':>10}  {'Minibonus':>10}  {'AnnBonus':>10}")
    for r in sorted(rkr, key=lambda x: x["season_month_idx"]):
        print(f"  {r['season_month']:<5} {r['monthly_rate_kes_per_kg']:>10.2f}  "
              f"{r['minibonus_rate_kes_per_kg']:>10.2f}  "
              f"{r['annual_bonus_rate_kes_per_kg']:>10.2f}")
    print("="*65 + "\n")

# MAIN
if __name__ == "__main__":
    OUT = "/home/mogaka/projects/Chai_Analytics"
    os.makedirs(OUT, exist_ok=True)

    print("Generating KTDA pricing history (2010–2024)...")
    pricing = generate_ktda_pricing(list(range(2010, 2025)))

    print("Generating farms (Weru + Rukuriri)...")
    farms = generate_farms(pricing, start_season=2010, end_season=2024)

    metadata = generate_metadata()

    farms_path   = os.path.join(OUT, "farms.json")
    pricing_path = os.path.join(OUT, "ktda_pricing.json")
    meta_path    = os.path.join(OUT, "synthetic_metadata.json")

    with open(farms_path,   "w") as f: json.dump(farms,    f, indent=2, default=str)
    with open(pricing_path, "w") as f: json.dump(pricing,  f, indent=2, default=str)
    with open(meta_path,    "w") as f: json.dump(metadata, f, indent=2, default=str)

    print_summary(farms, pricing)

    print(f"  Files written to: {OUT}")
    print(f"    farms.json            : {len(farms)} documents")
    print(f"    ktda_pricing.json     : {len(pricing)} documents")
    print(f"    synthetic_metadata.json")
    print()
    print("  MongoDB import:")
    print(f"    mongoimport --db chaimterics --collection farms      --file {farms_path} --jsonArray")
    print(f"    mongoimport --db chaimterics --collection ktda_pricing --file {pricing_path} --jsonArray")
    print()