#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Inferencia por Consola: Tasador Predictivo de Alquileres en Rusia (Moscú)
Modelo Campeón: HistGradientBoosting Regressor (R² = 95.27%, MAE = $84,428 ARS)
"""

import sys
import os
import argparse
import pickle
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), 'model_bundle.pkl')

def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        print(f"Error: No se encuentra el archivo de modelo '{BUNDLE_PATH}'.")
        print("Ejecuta primero: python build_inference_assets.py")
        sys.exit(1)
    with open(BUNDLE_PATH, 'rb') as f:
        return pickle.load(f)

def predict(bundle, district, area_m2, rooms, building_year, building_type, renovation, to_center_km, metro_dist_min, furnished, pets_allowed):
    model = bundle['model']
    feature_cols = bundle['feature_cols']
    district_map = bundle['district_mapping']
    global_mean = bundle['global_mean']
    ohe = bundle['ohe']
    
    # 1. Target encoding del distrito
    district_clean = district.strip()
    district_enc = district_map.get(district_clean, global_mean)
    
    # 2. Features derivadas
    area_per_room = area_m2 / max(rooms, 1)
    accessible_area = area_m2 / (to_center_km + 1.0)
    interaction_area_center = area_m2 * to_center_km
    is_central = 1 if to_center_km <= 5.0 else 0
    building_age = 2026 - building_year
    
    renovation_mapping = {
        'no_renovation': 0, 'sin_remodelar': 0, 'ninguna': 0,
        'cosmetic': 1, 'cosmetica': 1, 'estandar': 1,
        'euro': 2, 'europea': 2,
        'designer': 3, 'diseno': 3, 'lujo': 3
    }
    renov_code = renovation_mapping.get(renovation.lower().strip(), 1)
    
    # 3. One-hot encoding de building_type
    bt_clean = building_type.lower().strip()
    bt_df = pd.DataFrame([{'building_type': bt_clean}])
    bt_encoded = ohe.transform(bt_df)
    bt_cols = ohe.get_feature_names_out(['building_type'])
    
    # 4. Construcción del vector de características
    row_data = {
        'total_area_m2': area_m2,
        'rooms': rooms,
        'building_age': building_age,
        'renovation_code': renov_code,
        'furnished': 1 if furnished else 0,
        'pets_allowed': 1 if pets_allowed else 0,
        'to_center_km_real': to_center_km,
        'metro_distance_min': metro_dist_min,
        'area_per_room': area_per_room,
        'accessible_area': accessible_area,
        'interaction_area_center': interaction_area_center,
        'is_central': is_central,
        'district_encoded': district_enc
    }
    for idx, col in enumerate(bt_cols):
        row_data[col] = bt_encoded[0, idx]
        
    X_pred = pd.DataFrame([row_data])[feature_cols]
    
    # 5. Predicción
    pred_ars = float(model.predict(X_pred)[0])
    pred_rub = pred_ars / 18.0
    mae_ars = 84428.0
    
    return {
        'pred_ars': pred_ars,
        'pred_rub': pred_rub,
        'mae_ars': mae_ars,
        'price_per_m2_ars': pred_ars / area_m2,
        'inputs': {
            'distrito': district_clean,
            'superficie_m2': area_m2,
            'habitaciones': rooms,
            'antiguedad_edificio': f"{building_age} años (Año {building_year})",
            'tipo_edificio': bt_clean,
            'remodelacion': renovation,
            'distancia_centro': f"{to_center_km:.2f} km",
            'distancia_metro': f"{metro_dist_min} min",
            'amoblado': 'Sí' if furnished else 'No',
            'mascotas': 'Sí' if pets_allowed else 'No'
        }
    }

def print_result(res):
    print("\n" + "="*65)
    print("      [*] RESULTADO DE LA TASACION INMOBILIARIA PREDICTIVA")
    print("="*65)
    print("Caracteristicas del Inmueble Ingresado:")
    for k, v in res['inputs'].items():
        print(f"  - {k.replace('_', ' ').capitalize()}: {v}")
    print("-" * 65)
    print(f"  [ARS] ALQUILER ESTIMADO:      $ {res['pred_ars']:,.0f} ARS / mes")
    print(f"  [RUB] EQUIVALENTE EN RUBLOS:  {res['pred_rub']:,.0f} RUB / mes  (tasa: 18 ARS/RUB)")
    print(f"  [m2]  PRECIO POR METRO:       $ {res['price_per_m2_ars']:,.0f} ARS / m2")
    print("-" * 65)
    lower = max(0, res['pred_ars'] - res['mae_ars'])
    upper = res['pred_ars'] + res['mae_ars']
    print(f"  Rango de Tasacion Esperado (+/- 1 MAE de $ {res['mae_ars']:,.0f}):")
    print(f"   $ {lower:,.0f}  <--->  $ {upper:,.0f} ARS / mes")
    print("="*65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Tasador de Alquileres de Moscú (Modelo Gradient Boosting Campeón)")
    parser.add_argument('--district', type=str, help='Nombre del Distrito (ej: Arbat, Presnensky, Tverskoy, Marfino)')
    parser.add_argument('--area', type=float, help='Superficie total en m² (ej: 45.0)')
    parser.add_argument('--rooms', type=int, help='Cantidad de habitaciones (1 a 4)')
    parser.add_argument('--year', type=int, default=2010, help='Año de construcción del edificio (ej: 2015)')
    parser.add_argument('--building_type', type=str, default='monolith', help='Tipo de edificio: brick, monolith, panel, modern_panel, khrushchev, stalin')
    parser.add_argument('--renovation', type=str, default='euro', help='Nivel de reforma: no_renovation, cosmetic, euro, designer')
    parser.add_argument('--to_center_km', type=float, help='Distancia al centro de Moscú en km (ej: 4.5)')
    parser.add_argument('--metro_dist', type=int, default=10, help='Distancia a la estación de metro más cercana en minutos (ej: 8)')
    parser.add_argument('--furnished', action='store_true', help='Incluir si el departamento está amoblado')
    parser.add_argument('--pets', action='store_true', help='Incluir si permite mascotas')
    parser.add_argument('--ejemplo', action='store_true', help='Ejecuta un ejemplo demostrativo predefinido')
    
    args = parser.parse_args()
    bundle = load_bundle()
    
    if args.ejemplo:
        print("\n---> Ejecutando tasación demostrativa (Depto en Arbat, 50 m², reformado)...")
        res = predict(
            bundle=bundle,
            district="Arbat",
            area_m2=50.0,
            rooms=2,
            building_year=2012,
            building_type="monolith",
            renovation="designer",
            to_center_km=2.1,
            metro_dist_min=5,
            furnished=True,
            pets_allowed=False
        )
        print_result(res)
        return

    # Si se pasan los argumentos esenciales por línea de comando
    if args.district and args.area and args.rooms and args.to_center_km is not None:
        res = predict(
            bundle=bundle,
            district=args.district,
            area_m2=args.area,
            rooms=args.rooms,
            building_year=args.year,
            building_type=args.building_type,
            renovation=args.renovation,
            to_center_km=args.to_center_km,
            metro_dist_min=args.metro_dist,
            furnished=args.furnished,
            pets_allowed=args.pets
        )
        print_result(res)
        return

    # Modo interactivo amigable
    print("\n" + "="*65)
    print("   [+] COTIZADOR INTERACTIVO DE ALQUILERES EN MOSCU")
    print("   Modelo: HistGradientBoosting (R2: 95.27%, MAE: $84,428 ARS)")
    print("="*65)
    print("Ingresa los datos del inmueble (presiona Enter para usar los valores por defecto [entre corchetes]):\n")
    
    distrito = input("1. Distrito (ej: Arbat, Presnensky, Tverskoy, Savyolovsky) [Arbat]: ").strip() or "Arbat"
    
    area_in = input("2. Superficie en m² (ej: 45) [50]: ").strip() or "50"
    area_m2 = float(area_in)
    
    rooms_in = input("3. Habitaciones (1, 2, 3 o 4) [2]: ").strip() or "2"
    rooms = int(rooms_in)
    
    year_in = input("4. Año de construcción del edificio (1937 - 2024) [2015]: ").strip() or "2015"
    building_year = int(year_in)
    
    print("\nTipos de edificio disponibles: brick, monolith, panel, modern_panel, khrushchev, stalin")
    btype = input("5. Tipo de construcción [monolith]: ").strip() or "monolith"
    
    print("Nivel de reforma disponible: no_renovation, cosmetic, euro, designer")
    renov = input("6. Remodelación [euro]: ").strip() or "euro"
    
    dist_c_in = input("7. Distancia al Kremlin/Centro en km (ej: 3.5) [2.5]: ").strip() or "2.5"
    to_center_km = float(dist_c_in)
    
    metro_in = input("8. Distancia al metro en minutos (ej: 8) [7]: ").strip() or "7"
    metro_dist_min = int(metro_in)
    
    furn_in = input("9. ¿Está amoblado? (s/n) [s]: ").strip().lower()
    furnished = (furn_in in ['s', 'si', 'y', 'yes', 'true', ''])
    
    pets_in = input("10. ¿Acepta mascotas? (s/n) [n]: ").strip().lower()
    pets_allowed = (pets_in in ['s', 'si', 'y', 'yes', 'true'])
    
    res = predict(
        bundle=bundle,
        district=distrito,
        area_m2=area_m2,
        rooms=rooms,
        building_year=building_year,
        building_type=btype,
        renovation=renov,
        to_center_km=to_center_km,
        metro_dist_min=metro_dist_min,
        furnished=furnished,
        pets_allowed=pets_allowed
    )
    print_result(res)

if __name__ == '__main__':
    main()
