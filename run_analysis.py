#!/usr/bin/env python3
"""
Interactive Analysis Runner for Quant Engine
Allows user to specify parameters at runtime.
"""

import sys
import datetime
from quant.config import TICKERS, BENCHMARK, NUM_PORTFOLIOS, INITIAL_CAPITAL, START_DATE, END_DATE
from quant.main import main

def get_valid_date(prompt):
    """Prompt user for a date in DD/MM/YYYY format and return YYYY-MM-DD."""
    while True:
        date_str = input(prompt).strip()
        if not date_str:
            return None
        
        try:
            # Parse DD/MM/YYYY
            dt = datetime.datetime.strptime(date_str, "%d/%m/%Y")
            # Return YYYY-MM-DD
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            print("❌ Formato inválido. Por favor use DD/MM/YYYY (ej: 31/03/2020)")

def get_assets():
    """Prompt for comma-separated list of tickers."""
    while True:
        input_str = input(f"   Activos (Enter para default: {', '.join(TICKERS)}): ").strip()
        if not input_str:
            return None
        
        tickers = [t.strip().upper() for t in input_str.split(',') if t.strip()]
        if tickers:
            return tickers
        print("❌ Lista vacía. Intente de nuevo.")

def run_interactive():
    print("="*60)
    print("🚀 QUANT ENGINE - CONFIGURACIÓN DE ANÁLISIS")
    print("="*60)
    
    use_defaults = input("\n¿Ejecutar con configuración por defecto del sistema? (S/n): ").lower().strip()
    
    if use_defaults == '' or use_defaults.startswith('s'):
        print("\n✅ Usando configuración por defecto...")
        main()
        return

    print("\n📝 Personalización de Parámetros:")
    print("-" * 40)
    
    # 1. Assets
    custom_tickers = get_assets()
    
    # 2. Benchmark
    custom_benchmark = input(f"   Benchmark (Enter para default: {BENCHMARK}): ").strip().upper()
    if not custom_benchmark:
        custom_benchmark = None
        
    # 3. Start Date
    print(f"   Fecha de Inicio (Enter para default: {START_DATE})")
    custom_start_date = get_valid_date("   Formato DD/MM/YYYY (ej: 31/03/2020): ")
    if not custom_start_date:
        custom_start_date = START_DATE
    
    # 4. Monte Carlo Sims
    while True:
        sims_str = input(f"   Iteraciones Monte Carlo (Enter para default: {NUM_PORTFOLIOS}): ").strip()
        if not sims_str:
            custom_sims = None
            break
        if sims_str.isdigit() and int(sims_str) > 0:
            custom_sims = int(sims_str)
            break
        print("❌ Debe ser un número entero positivo.")
        
    # 5. Initial Capital
    while True:
        cap_str = input(f"   Capital Inicial USD (Enter para default: {INITIAL_CAPITAL}): ").strip()
        if not cap_str:
            custom_capital = None
            break
        try:
            custom_capital = float(cap_str)
            if custom_capital > 0:
                break
        except ValueError:
            pass
        print("❌ Debe ser un valor numérico positivo.")

    # Confirmation
    print("\n📋 Resumen de Configuración:")
    print("-" * 40)
    print(f"• Tickers: {', '.join(custom_tickers if custom_tickers else TICKERS)}")
    print(f"• Benchmark: {custom_benchmark if custom_benchmark else BENCHMARK}")
    print(f"• Fecha Inicio: {custom_start_date if custom_start_date else START_DATE}")
    print(f"• Simulaciones: {custom_sims if custom_sims else NUM_PORTFOLIOS:,}")
    print(f"• Capital: ${custom_capital if custom_capital else INITIAL_CAPITAL:,.2f}")
    
    confirm = input("\n¿Proceder con el análisis? (S/n): ").lower().strip()
    if confirm == '' or confirm.startswith('s'):
        print("\n🚀 Iniciando motor cuantitativo...\n")
        main(
            tickers=custom_tickers,
            benchmark=custom_benchmark,
            start_date=custom_start_date,
            num_simulations=custom_sims,
            initial_capital=custom_capital
        )
    else:
        print("\n❌ Ejecución cancelada.")

if __name__ == "__main__":
    try:
        run_interactive()
    except KeyboardInterrupt:
        print("\n\n👋 Operación interrumpida por el usuario.")
        sys.exit(0)
