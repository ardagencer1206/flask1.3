# -*- coding: utf-8 -*-
import os
import json
import base64
import traceback
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals, Objective, minimize,
    Constraint, Any as PyAny, value, SolverFactory
)
from pyomo.opt import TerminationCondition
from pathlib import Path

from openai import AzureOpenAI

# Dataset yolu (env ile özelleştirilebilir)
DATASET_PATH = Path(os.environ.get("DATASET_PATH", str(Path(__file__).parent / "dataset.json")))

MAX_SOLVE_SECONDS = 26  # işlemciye verdiğimiz süre
app = Flask(__name__, static_url_path="", static_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB görsel limiti

# ---------------- Azure OpenAI Ayarları ----------------
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "d0167637046c4443badc4920cc612abb")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-fnss.openai.azure.com")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

try:
    aoai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception:
    aoai_client = None
# -------------------------------------------------------


# ------------------------------
# Yardımcılar
# ------------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_dataset_from_file() -> Dict[str, Any]:
    """
    dataset.json dosyasını okur ve dict döner.
    """
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # minimum alan kontrolü
        required = [
            "cities", "main_depot", "periods", "vehicle_types",
            "vehicle_count", "distances", "packages", "minutil_penalty"
        ]
        for k in required:
            if k not in data:
                raise KeyError(f"dataset.json alan eksik: {k}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"dataset.json bulunamadı: {DATASET_PATH}")
    except Exception as e:
        raise RuntimeError(f"dataset.json okunamadı: {e}")


def _normalize_initial_locations(
    vehicles: List[str],
    cities: List[str],
    main_depot: str,
    payload: Dict[str, Any]
) -> Dict[str, str]:
    """
    vehicle_initial_locations aşağıdaki iki formdan birini alabilir:
      A) { "Küçük_1": "Ankara", "Orta_1": "İzmir", ... }  # doğrudan araç adına göre
      B) { "Küçük": "Ankara", "Orta": "İzmir" }           # tipe göre, o tipteki tüm araçlara uygular
    Verilmezse tüm araçlar main_depot'ta başlatılır.
    Şehir doğrulaması yapar; hatalı şehirler sessizce main_depot'a döner.
    """
    raw = payload.get("vehicle_initial_locations", {}) or {}
    init = {v: main_depot for v in vehicles}

    if isinstance(raw, dict):
        # araç adına göre eşleşenler
        for k, city in raw.items():
            if k in vehicles and city in cities:
                init[k] = city
        # tip adına göre eşleşenler
        for k, city in raw.items():
            if k not in vehicles and city in cities:
                prefix = f"{k}_"
                for v in vehicles:
                    if v.startswith(prefix):
                        init[v] = city

    # son güvenlik
    for v in vehicles:
        if init.get(v) not in cities:
            init[v] = main_depot

    return init


def pick_solver():
    """
    Önce Pyomo APPsi-HiGHS (highspy) denenir.
    Ardından klasik arayüzler (highs/cbc/glpk/cplex).
    Geriye (solver_adı, solver_nesnesi, is_appsi_bool) döner.
    """
    # 1) APPsi-HiGHS (Python wrapper)
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs as AppsiHighs
        s = AppsiHighs()
        try:
            s.config.time_limit = MAX_SOLVE_SECONDS
        except Exception:
            pass
        return "appsi_highs", s, True
    except Exception:
        pass

    # 2) Klasik arayüzler
    for cand in ["highs", "cbc", "glpk", "cplex"]:
        try:
            s = SolverFactory(cand)
            if s is not None and s.available():
                try:
                    if cand == "highs":
                        s.options["time_limit"] = MAX_SOLVE_SECONDS
                    elif cand == "cbc":
                        s.options["seconds"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "glpk":
                        s.options["tmlim"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "cplex":
                        s.options["timelimit"] = MAX_SOLVE_SECONDS
                        s.options["mipgap"] = 0.05
                        s.options["threads"] = 2
                except Exception:
                    pass
                return cand, s, False
        except Exception:
            continue

    return None, None, False


# ------------------------------
# Model Kurulumu
# ------------------------------
def build_model(payload: Dict[str, Any]) -> Tuple[ConcreteModel, Dict[str, Any]]:
    """
    Beklenen payload anahtarları:
      cities, main_depot, periods, vehicle_types, vehicle_count,
      distances, packages, minutil_penalty
      (opsiyonel) vehicle_initial_locations
    """
    # --- Girdiler
    cities: List[str] = payload["cities"]
    main_depot: str = payload["main_depot"]
    periods: int = int(payload["periods"])
    Tmin, Tmax = 1, periods
    periods_list = list(range(Tmin, Tmax + 1))

    vehicle_types: Dict[str, Dict[str, Any]] = payload["vehicle_types"]
    vehicle_count: Dict[str, int] = payload["vehicle_count"]

    # Araç isimleri: Tip_index
    vehicles: List[str] = [f"{vt}_{i}" for vt, cnt in vehicle_count.items() for i in range(1, int(cnt) + 1)]

    # Başlangıç konumları
    init_loc = _normalize_initial_locations(vehicles, cities, main_depot, payload)

    # Mesafeler (simetrik tamamla)
    distances: Dict[Tuple[str, str], float] = {}
    for i, j, d in payload["distances"]:
        distances[(i, j)] = float(d)
        distances[(j, i)] = float(d)
    for c in cities:
        distances[(c, c)] = 0.0

    # Paketler
    packages_input: List[Dict[str, Any]] = payload["packages"]
    packages = {}
    for rec in packages_input:
        pid = str(rec["id"])
        packages[pid] = {
            "baslangic": rec["baslangic"],
            "hedef": rec["hedef"],
            "agirlik": float(rec["agirlik"]),
            "baslangic_periyot": int(rec["ready"]),
            "teslim_suresi": int(rec["deadline_suresi"]),
            "ceza_maliyeti": float(rec["ceza"]),
        }

    MINUTIL_PENALTY = safe_float(payload.get("minutil_penalty", 10.0), 10.0)

    # --- Pyomo Modeli
    model = ConcreteModel()

    # Kümeler
    model.Cities = Set(initialize=cities)
    model.Periods = Set(initialize=periods_list)
    model.Vehicles = Set(initialize=vehicles)
    model.Packages = Set(initialize=list(packages.keys()))

    def vtype(v):  # "Küçük_1" -> "Küçük"
        return v.rsplit("_", 1)[0]

    # Parametreler
    model.Distance = Param(model.Cities, model.Cities, initialize=lambda m, i, j: distances[(i, j)])
    model.VehicleCapacity = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["kapasite"])
    model.TransportCost = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["maliyet_km"])
    model.FixedCost = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["sabit_maliyet"])
    model.MinUtilization = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["min_doluluk"])

    model.PackageWeight = Param(model.Packages, initialize=lambda m, p: packages[p]["agirlik"])
    model.PackageOrigin = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["baslangic"])
    model.PackageDest = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["hedef"])
    model.PackageReady = Param(model.Packages, initialize=lambda m, p: packages[p]["baslangic_periyot"])
    model.PackageDeadline = Param(model.Packages, initialize=lambda m, p: packages[p]["teslim_suresi"])
    model.LatePenalty = Param(model.Packages, initialize=lambda m, p: packages[p]["ceza_maliyeti"])

    # Değişkenler
    model.x = Var(model.Vehicles, model.Cities, model.Cities, model.Periods, domain=Binary)  # araç hareketi
    model.y = Var(model.Packages, model.Vehicles, model.Cities, model.Cities, model.Periods, domain=Binary)  # paket hareketi
    model.z = Var(model.Vehicles, model.Periods, domain=Binary)  # araç kullanımı
    model.loc = Var(model.Vehicles, model.Cities, model.Periods, domain=Binary)  # araç konumu
    model.pkg_loc = Var(model.Packages, model.Cities, model.Periods, domain=Binary)  # paket konumu
    model.lateness = Var(model.Packages, domain=NonNegativeReals)  # gecikme
    model.minutil_shortfall = Var(model.Vehicles, model.Periods, domain=NonNegativeReals)  # doluluk açığı (kg)

    # Amaç
    def objective_rule(m):
        transport = sum(
            m.TransportCost[v] * m.Distance[i, j] * m.x[v, i, j, t]
            for v in m.Vehicles for i in m.Cities for j in m.Cities for t in m.Periods if i != j
        )
        fixed = sum(m.FixedCost[v] * m.z[v, t] for v in m.Vehicles for t in m.Periods)
        late = sum(m.LatePenalty[p] * m.lateness[p] for p in m.Packages)
        minutil = MINUTIL_PENALTY * sum(m.minutil_shortfall[v, t] for v in m.Vehicles for t in m.Periods)
        return transport + fixed + late + minutil

    model.obj = Objective(rule=objective_rule, sense=minimize)

    # ------------------------------
    # KISITLAR (1..21)
    # ------------------------------

    # 1) Paket origin'den tam 1 çıkış (ready'den sonra)
    def package_origin_rule(m, p):
        o, r = m.PackageOrigin[p], m.PackageReady[p]
        return sum(m.y[p, v, o, j, t] for v in m.Vehicles for j in m.Cities for t in m.Periods if j != o and t >= r) == 1
    model.package_origin_constraint = Constraint(model.Packages, rule=package_origin_rule)

    # 2) Paket hedefe tam 1 varış
    def package_destination_rule(m, p):
        d = m.PackageDest[p]
        return sum(m.y[p, v, i, d, t] for v in m.Vehicles for i in m.Cities for t in m.Periods if i != d) == 1
    model.package_destination_constraint = Constraint(model.Packages, rule=package_destination_rule)

    # 3) Ana depodan en az bir kez geçiş (origin/dest depo değilse)
    def main_depot_rule(m, p):
        o, d = m.PackageOrigin[p], m.PackageDest[p]
        if o == main_depot or d == main_depot:
            return Constraint.Skip
        through = sum(m.y[p, v, i, main_depot, t] for v in m.Vehicles for i in m.Cities for t in m.Periods if i != main_depot) \
                + sum(m.y[p, v, main_depot, j, t] for v in m.Vehicles for j in m.Cities for t in m.Periods if j != main_depot)
        return through >= 1
    model.main_depot_constraint = Constraint(model.Packages, rule=main_depot_rule)

    # 4) Paket ancak araç gidiyorsa taşınır
    def y_le_x_rule(m, p, v, i, j, t):
        if i == j:
            return Constraint.Skip
        return m.y[p, v, i, j, t] <= m.x[v, i, j, t]
    model.package_vehicle_link = Constraint(model.Packages, model.Vehicles, model.Cities, model.Cities, model.Periods, rule=y_le_x_rule)

    # 5) Kapasite
    def capacity_rule(m, v, i, j, t):
        if i == j:
            return Constraint.Skip
        return sum(m.PackageWeight[p] * m.y[p, v, i, j, t] for p in m.Packages) <= m.VehicleCapacity[v]
    model.capacity_constraint = Constraint(model.Vehicles, model.Cities, model.Cities, model.Periods, rule=capacity_rule)

    # 6) SOFT min. doluluk (sadece ana depodan çıkış)
    def min_utilization_soft_rule(m, v, t):
        departures = sum(m.x[v, main_depot, j, t] for j in m.Cities if j != main_depot)  # 0/1
        loaded = sum(m.PackageWeight[p] * m.y[p, v, main_depot, j, t] for p in m.Packages for j in m.Cities if j != main_depot)
        target = m.MinUtilization[v] * m.VehicleCapacity[v] * departures
        return loaded + m.minutil_shortfall[v, t] >= target
    model.min_utilization_soft = Constraint(model.Vehicles, model.Periods, rule=min_utilization_soft_rule)

    # 7) Paket konumu: her t’de tek şehir
    def pkg_onehot_rule(m, p, t):
        return sum(m.pkg_loc[p, n, t] for n in m.Cities) == 1
    model.pkg_location_onehot = Constraint(model.Packages, model.Periods, rule=pkg_onehot_rule)

    # 8) Ready öncesi origin kilidi ve t=ready’de origin
    def pkg_before_ready_origin_rule(m, p, t):
        o, r = m.PackageOrigin[p], m.PackageReady[p]
        if t < r:
            return m.pkg_loc[p, o, t] == 1
        return Constraint.Skip
    model.pkg_before_ready_origin = Constraint(model.Packages, model.Periods, rule=pkg_before_ready_origin_rule)

    def pkg_before_ready_others_zero_rule(m, p, n, t):
        o, r = m.PackageOrigin[p], m.PackageReady[p]
        if t < r and n != o:
            return m.pkg_loc[p, n, t] == 0
        return Constraint.Skip
    model.pkg_before_ready_others_zero = Constraint(model.Packages, model.Cities, model.Periods, rule=pkg_before_ready_others_zero_rule)

    def pkg_at_ready_origin_rule(m, p):
        o, r = m.PackageOrigin[p], m.PackageReady[p]
        return m.pkg_loc[p, o, r] == 1
    model.pkg_at_ready_origin = Constraint(model.Packages, rule=pkg_at_ready_origin_rule)

    def pkg_at_ready_others_zero_rule(m, p, n):
        o, r = m.PackageOrigin[p], m.PackageReady[p]
        if n != o:
            return m.pkg_loc[p, n, r] == 0
        return Constraint.Skip
    model.pkg_at_ready_others_zero = Constraint(model.Packages, model.Cities, rule=pkg_at_ready_others_zero_rule)

    # 9) Paket konum geçişi (τ=1)
    def pkg_loc_transition_rule(m, p, n, t):
        if t == Tmax:
            return Constraint.Skip
        incoming = sum(m.y[p, v, i, n, t] for v in m.Vehicles for i in m.Cities if i != n)
        outgoing = sum(m.y[p, v, n, j, t] for v in m.Vehicles for j in m.Cities if j != n)
        return m.pkg_loc[p, n, t] + incoming - outgoing == m.pkg_loc[p, n, t + 1]
    model.pkg_location_transition = Constraint(model.Packages, model.Cities, model.Periods, rule=pkg_loc_transition_rule)

    # 10) Çıkış mümkünse o anda orada ol
    def pkg_departure_feasible_rule(m, p, i, t):
        return sum(m.y[p, v, i, j, t] for v in m.Vehicles for j in m.Cities if j != i) <= m.pkg_loc[p, i, t]
    model.pkg_departure_feasible = Constraint(model.Packages, model.Cities, model.Periods, rule=pkg_departure_feasible_rule)

    # 11) Varıştan sonra t+1’de hedef şehirde ol (τ=1)
    def pkg_arrival_feasible_rule(m, p, j, t):
        if t == Tmax:
            return Constraint.Skip
        return sum(m.y[p, v, i, j, t] for v in m.Vehicles for i in m.Cities if i != j) <= m.pkg_loc[p, j, t + 1]
    model.pkg_arrival_feasible = Constraint(model.Packages, model.Cities, model.Periods, rule=pkg_arrival_feasible_rule)

    # 12) Ara şehir akış korunumu
    def flow_conservation_rule(m, p, k):
        o, d = m.PackageOrigin[p], m.PackageDest[p]
        if k == o or k == d:
            return Constraint.Skip
        inflow = sum(m.y[p, v, i, k, t] for v in m.Vehicles for i in m.Cities for t in m.Periods if i != k)
        outflow = sum(m.y[p, v, k, j, t] for v in m.Vehicles for j in m.Cities for t in m.Periods if j != k)
        return inflow == outflow
    model.flow_conservation = Constraint(model.Packages, model.Cities, rule=flow_conservation_rule)

    # 13) Araç kullanım takibi (hareket varsa z=1)
    def vehicle_usage_rule(m, v, t):
        moves = sum(m.x[v, i, j, t] for i in m.Cities for j in m.Cities if i != j)
        return m.z[v, t] >= moves
    model.vehicle_usage = Constraint(model.Vehicles, model.Periods, rule=vehicle_usage_rule)

    # 14) Araç: periyot başına tek hareket
    def vehicle_one_move_rule(m, v, t):
        return sum(m.x[v, i, j, t] for i in m.Cities for j in m.Cities if i != j) <= 1
    model.vehicle_route_out = Constraint(model.Vehicles, model.Periods, rule=vehicle_one_move_rule)

    # 15) Araç başlangıç konumu (t=Tmin’de araç özelinde belirlenen şehirde)
    def vehicle_initial_loc_rule(m, v):
        return m.loc[v, init_loc[v], Tmin] == 1
    model.vehicle_initial_location = Constraint(model.Vehicles, rule=vehicle_initial_loc_rule)

    # 16) Araç: her t’de tek şehir
    def vehicle_loc_onehot_rule(m, v, t):
        return sum(m.loc[v, n, t] for n in m.Cities) == 1
    model.vehicle_location_exists = Constraint(model.Vehicles, model.Periods, rule=vehicle_loc_onehot_rule)

    # 17) Araç konum geçişi (τ=1)
    def vehicle_loc_transition_rule(m, v, n, t):
        if t == Tmax:
            return Constraint.Skip
        incoming = sum(m.x[v, i, n, t] for i in m.Cities if i != n)
        outgoing = sum(m.x[v, n, j, t] for j in m.Cities if j != n)
        return m.loc[v, n, t] + incoming - outgoing == m.loc[v, n, t + 1]
    model.vehicle_location_transition = Constraint(model.Vehicles, model.Cities, model.Periods, rule=vehicle_loc_transition_rule)

    # 18) Araç sadece bulunduğu şehirden ayrılabilir
    def vehicle_move_from_loc_rule(m, v, i, t):
        outgoing = sum(m.x[v, i, j, t] for j in m.Cities if j != i)
        return outgoing <= m.loc[v, i, t]
    model.movement_from_location = Constraint(model.Vehicles, model.Cities, model.Periods, rule=vehicle_move_from_loc_rule)

    # 19) Gecikme tanımı (teslim zamanı - termin ≤ lateness)
    def lateness_rule(m, p):
        d = m.PackageDest[p]
        delivery_t = sum(tt * m.y[p, v, i, d, tt] for v in m.Vehicles for i in m.Cities for tt in m.Periods if i != d)
        deadline = m.PackageReady[p] + m.PackageDeadline[p]
        return m.lateness[p] >= delivery_t - deadline
    model.lateness_calc = Constraint(model.Packages, rule=lateness_rule)

    # 20) Aynı paket aynı i→j segmentini toplamda ≤1
    def package_once_segment_rule(m, p, i, j):
        if i == j:
            return Constraint.Skip
        return sum(m.y[p, v, i, j, t] for v in m.Vehicles for t in m.Periods) <= 1
    model.package_once_per_segment = Constraint(model.Packages, model.Cities, model.Cities, rule=package_once_segment_rule)

    # 21) Hazır olmadan origin'den çıkamaz
    def package_ready_time_rule(m, p, v, i, j, t):
        if i == j or i != m.PackageOrigin[p]:
            return Constraint.Skip
        return m.y[p, v, i, j, t] * t >= m.y[p, v, i, j, t] * m.PackageReady[p]
    model.package_ready_time = Constraint(model.Packages, model.Vehicles, model.Cities, model.Cities, model.Periods, rule=package_ready_time_rule)

    # meta
    meta = {
        "cities": cities,
        "periods_list": periods_list,
        "vehicles": vehicles,
        "packages": packages,
        "distances": distances,
        "vehicle_types": vehicle_types,
        "main_depot": main_depot,
        "MINUTIL_PENALTY": MINUTIL_PENALTY,
        "initial_locations": init_loc
    }
    return model, meta


# ------------------------------
# Sonuçları çıkarma (UI için)
# ------------------------------
def extract_results(model: ConcreteModel, meta: Dict[str, Any]) -> Dict[str, Any]:
    cities = meta["cities"]
    periods = meta["periods_list"]
    vehicles = meta["vehicles"]
    packages = meta["packages"]
    distances = meta["distances"]
    MINUTIL_PENALTY = meta["MINUTIL_PENALTY"]

    results = {}
    total_obj = float(value(model.obj))
    results["objective"] = total_obj

    # Maliyet dağılımı
    transport_cost = 0.0
    for v in vehicles:
        for i in cities:
            for j in cities:
                for t in periods:
                    if i != j and value(model.x[v, i, j, t]) > 0.5:
                        transport_cost += float(value(model.TransportCost[v])) * float(value(model.Distance[i, j]))

    fixed_cost = 0.0
    for v in vehicles:
        for t in periods:
            if value(model.z[v, t]) > 0.5:
                fixed_cost += float(value(model.FixedCost[v]))

    penalty_cost = sum(float(value(model.LatePenalty[p])) * float(value(model.lateness[p])) for p in model.Packages)
    minutil_pen = MINUTIL_PENALTY * sum(float(value(model.minutil_shortfall[v, t])) for v in vehicles for t in periods)

    results["cost_breakdown"] = {
        "transport": transport_cost,
        "fixed": fixed_cost,
        "lateness": penalty_cost,
        "min_util_gap": float(minutil_pen),
    }

    # Araç rotaları
    vehicle_routes = []
    for v in sorted(vehicles):
        entries = []
        for t in periods:
            for i in cities:
                for j in cities:
                    if i != j and value(model.x[v, i, j, t]) > 0.5:
                        moved = []
                        totw = 0.0
                        for p in model.Packages:
                            if value(model.y[p, v, i, j, t]) > 0.5:
                                moved.append(p)
                                totw += float(value(model.PackageWeight[p]))
                        entries.append({
                            "t": t, "from": i, "to": j, "km": float(distances[(i, j)]),
                            "packages": moved, "load_kg": totw,
                            "utilization_pct": (100.0 * totw / float(value(model.VehicleCapacity[v]))) if totw > 0 else 0.0
                        })
        if entries:
            vehicle_routes.append({"vehicle": v, "capacity": float(value(model.VehicleCapacity[v])), "legs": entries})
    results["vehicle_routes"] = vehicle_routes

    # Paketler
    package_summaries = []
    for p in sorted(packages.keys()):
        o = packages[p]["baslangic"]
        d = packages[p]["hedef"]
        r = packages[p]["baslangic_periyot"]
        dl = packages[p]["teslim_suresi"]
        deadline = r + dl

        delivery_time = None
        for t in periods:
            delivered = sum(value(model.y[p, v, i, d, t]) for v in model.Vehicles for i in model.Cities if i != d)
            if delivered > 0.5:
                delivery_time = t
                break

        passed_main = False
        main_depot = meta["main_depot"]
        for v in model.Vehicles:
            for t in periods:
                if (sum(value(model.y[p, v, main_depot, j, t]) for j in cities if j != main_depot) > 0.5 or
                        sum(value(model.y[p, v, i, main_depot, t]) for i in cities if i != main_depot) > 0.5):
                    passed_main = True
                    break
            if passed_main:
                break

        segs = []
        for t in periods:
            for vv in model.Vehicles:
                for i in cities:
                    for j in cities:
                        if i != j and value(model.y[p, vv, i, j, t]) > 0.5:
                            segs.append({"t": t, "from": i, "to": j, "vehicle": vv})

        summary = {
            "id": p,
            "origin": o,
            "dest": d,
            "weight": packages[p]["agirlik"],
            "ready": r,
            "deadline_by": deadline,
            "delivered_at": delivery_time,
            "on_time": (delivery_time is not None and delivery_time <= deadline),
            "passed_main_depot": passed_main,
            "route": sorted(segs, key=lambda s: s["t"]),
            "lateness_hours": max(0, (delivery_time - deadline)) if delivery_time else None,
            "lateness_penalty": float(value(model.LatePenalty[p])) * max(0, (delivery_time - deadline)) if delivery_time else 0.0
        }
        package_summaries.append(summary)

    results["packages"] = package_summaries
    return results


# ------------------------------
# Vision Yardımcıları ve Uç Noktası
# ------------------------------
def _b64_from_filestorage(fs) -> str:
    _ = secure_filename(getattr(fs, "filename", "") or "image")
    data = fs.read()
    return base64.b64encode(data).decode("ascii")


def _build_vision_messages(img_b64: str, cities_ctx: Optional[List[str]]) -> List[Dict[str, Any]]:
    cities_text = f"Geçerli şehirler: {', '.join(cities_ctx)}." if cities_ctx else "Şehirleri düz metinden yorumla."
    sys = (
        "Görüntüden sevkiyat etiketleri, irsaliyeler veya notlardan paket verilerini çıkar. "
        "ÇIKTIYI SADECE JSON olarak ver. Ek açıklama yazma."
    )
    user_text = (
        "İstenen alanlar: id, baslangic, hedef, agirlik, ready, deadline_suresi, ceza. "
        f"{cities_text} Ağırlık kg, para TL, dönem t biriminde. "
        "Okuyamazsan uydurma; boş dizi döndür. "
        'Şema: {"packages":[{"id":"...","baslangic":"...","hedef":"...","agirlik":0,"ready":1,"deadline_suresi":5,"ceza":120}]}'
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": sys}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}
    ]

def _parse_vision_json(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    pkgs = obj.get("packages") or []
    out: List[Dict[str, Any]] = []
    for p in pkgs:
        try:
            out.append({
                "id": str(p["id"]),
                "baslangic": str(p["baslangic"]),
                "hedef": str(p["hedef"]),
                "agirlik": float(p["agirlik"]),
                "ready": int(p["ready"]),
                "deadline_suresi": int(p["deadline_suresi"]),
                "ceza": float(p["ceza"]),
            })
        except Exception:
            continue
    return out


@app.post("/vision/package-extract")
def vision_package_extract():
    """
    Multipart: image (zorunlu), context (opsiyonel JSON: {"cities":[...]})
    Döner: { ok: true, packages: [ {...}, ... ] }
    """
    try:
        if aoai_client is None:
            return jsonify({"ok": False, "error": "Azure OpenAI istemcisi yok."}), 500

        if "image" not in request.files:
            return jsonify({"ok": False, "error": "image dosyası gerekli"}), 400

        # bağlamdan şehirleri al (opsiyonel)
        cities_ctx: Optional[List[str]] = None
        if "context" in request.files:
            try:
                ctx = json.loads(request.files["context"].read().decode("utf-8"))
                if isinstance(ctx, dict) and isinstance(ctx.get("cities"), list):
                    cities_ctx = [str(c) for c in ctx.get("cities")]
            except Exception:
                cities_ctx = None

        img_b64 = _b64_from_filestorage(request.files["image"])
        messages = _build_vision_messages(img_b64, cities_ctx)

        completion = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=400
            response_format={"type": "json_object"}  # JSON zorlaması
        )
        raw = completion.choices[0].message.content or "{}"

        try:
            obj = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(raw[start:end + 1])
            else:
                return jsonify({"ok": False, "error": "Model JSON döndürmedi."}), 502

        packages = _parse_vision_json(obj)
        return jsonify({"ok": True, "packages": packages})

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}", "trace": traceback.format_exc()}), 500


# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def root():
    return send_from_directory(".", "index.html")


@app.route("/dataset", methods=["GET", "PUT", "POST"])
def dataset_endpoint():
    """
    GET  -> dataset.json'u oku ve döndür
    PUT  -> gelen JSON'u doğrula ve dataset.json'a yaz
    POST -> bazı ortamlarda PUT engelli olabilir; POST'u PUT gibi kullan
    """
    try:
        if request.method == "GET":
            if not DATASET_PATH.exists():
                return jsonify({"ok": False, "error": "dataset.json bulunamadı"}), 404
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"ok": True, "dataset": data})

        # PUT veya POST -> kaydet
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Geçersiz JSON: {e}"}), 400

        # Basit şema kontrolü
        required_keys = [
            "cities", "main_depot", "periods",
            "vehicle_types", "vehicle_count",
            "distances", "packages", "minutil_penalty"
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            return jsonify({"ok": False, "error": f"Eksik alanlar: {', '.join(missing)}"}), 400

        # Dosyaya yaz (atomic)
        tmp_path = DATASET_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(DATASET_PATH)

        return jsonify({"ok": True, "message": "dataset.json güncellendi"})

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {e}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Önizleme/forma girilen JSON parametrelerini 'context' olarak alır,
    'messages' listesindeki sohbet geçmişiyle birlikte Azure OpenAI'ye gönderir.
    """
    try:
        if aoai_client is None:
            return jsonify({"ok": False, "error": "Azure OpenAI istemcisi oluşturulamadı (anahtar/endpoint)."}), 500

        payload = request.get_json(force=True) or {}
        user_messages = payload.get("messages", [])
        model_context = payload.get("context", {})

        sys_prompt = f"""
Sen bir lojistik optimizasyon asistanısın. Kullanıcıdan gelen VRP/çok duruşlu taşımacılık
parametrelerini (şehirler, dönemler, ana depo, araç tip/sayıları, mesafeler, paketler,
min. doluluk cezası) kullanarak kısa ve net cevap ver. Her şehir bir depodur ama bir tane ana depo vardır. Alakasız bir şey sorulursa cevap verme. Senin Adın VRP Assist 2.0

Model için kullanılan JSON parametreleri:
{model_context}

Kurallar:
- Sayısal/lojistik sorularda net hesap yap ve kısaca açıkla.
- Tutarsızlık görürsen hangi alanın düzelmesi gerektiğini söyle.
- Gereksiz ayrıntıya girme; anlaşılır ve kısa yanıt üret.
        """.strip()

        completion = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                *user_messages
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer = completion.choices[0].message.content
        return jsonify({"ok": True, "answer": answer})

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}", "trace": traceback.format_exc()}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"ok": False, "error": "Internal Server Error"}), 500


# ==== /solve ====
@app.route("/solve", methods=["POST"])
def solve():
    try:
        # Önce gelen payload'ı dene; yoksa dataset.json'u yükle
        data = request.get_json(silent=True) or {}
        if not data:
            data = load_dataset_from_file()

        model, meta = build_model(data)

        solver_name, solver, is_appsi = pick_solver()
        if solver is None:
            return jsonify({"ok": False, "error": "Uygun MILP çözücüsü bulunamadı."}), 400

        # --- ÇÖZ ---
        if is_appsi:
            results = solver.solve(model)
            term = getattr(results, "termination_condition", None)
        else:
            try:
                results = solver.solve(model, tee=False, load_solutions=True)
            except TypeError:
                results = solver.solve(model, load_solutions=True)
            term = None
            if hasattr(results, "solver") and hasattr(results.solver, "termination_condition"):
                term = results.solver.termination_condition
            else:
                term = getattr(results, "termination_condition", None)

        # --- İnkümbent var mı? ---
        def has_incumbent(m):
            try:
                for _, v in m.x.items():
                    if v.value is not None:
                        return True
            except Exception:
                pass
            return False

        diag = {
            "termination": str(term),
            "solver": solver_name,
            "wallclock_time": getattr(getattr(results, "solver", None), "wallclock_time", None),
            "gap": getattr(getattr(results, "solver", None), "gap", None),
            "status": getattr(getattr(results, "solver", None), "status", None),
        }

        if has_incumbent(model):
            out = extract_results(model, meta)
            return jsonify({"ok": True, "solver": solver_name, "result": out, "diagnostics": diag})

        return jsonify({
            "ok": False,
            "error": f"{MAX_SOLVE_SECONDS} sn içinde uygulanabilir çözüm bulunamadı. Durum: {term}",
            "diagnostics": diag
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}", "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    # Lokal test için:
    app.run(host="0.0.0.0", port=5000, debug=True)

