import os
import csv
import math

def generate_sample_data():
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', 'gorontalo_listrik_bulanan_2018_2024.csv')
    
    # Base electricity consumption for Gorontalo household sector (in kWh)
    # 2018 base: ~18,500,000 kWh per month with ~4.5% annual growth + seasonal variations
    base_kwh = 18500000.0
    annual_growth = 0.045
    
    months = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ]
    
    # Seasonal weights (Gorontalo weather & holidays: Lebaran / hotter months have higher AC usage)
    seasonal_factors = [
        1.00,  # Jan: normal
        0.96,  # Feb: shorter month
        1.02,  # Mar
        1.04,  # Apr: dry season beginning / Ramadan
        1.07,  # May: Eid / Lebaran peak
        1.03,  # Jun
        1.02,  # Jul
        1.03,  # Aug
        1.05,  # Sep: warmer
        1.08,  # Oct: peak temperature in Gorontalo
        1.06,  # Nov: warm
        1.09   # Dec: holiday season
    ]
    
    rows = []
    month_idx = 0
    for year in range(2018, 2025):
        year_multiplier = (1.0 + annual_growth) ** (year - 2018)
        for m_num in range(1, 13):
            month_name = months[m_num - 1]
            period_str = f"{year}-{m_num:02d}"
            
            # Combine trend, seasonality, and slight deterministic noise
            seasonal = seasonal_factors[m_num - 1]
            pseudo_noise = 1.0 + 0.015 * math.sin(month_idx * 1.7) + 0.01 * math.cos(month_idx * 0.9)
            
            # Covid-19 slight structural shift in 2020: households increased due to WFH
            covid_shift = 1.03 if year == 2020 and m_num >= 4 else (1.02 if year == 2021 else 1.0)
            
            val = round(base_kwh * year_multiplier * seasonal * pseudo_noise * covid_shift)
            
            rows.append({
                "Periode": period_str,
                "Tahun": year,
                "Bulan": month_name,
                "Konsumsi_kWh": val
            })
            month_idx += 1
            
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Periode", "Tahun", "Bulan", "Konsumsi_kWh"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Sample dataset successfully created at: {filepath} with {len(rows)} records.")

if __name__ == '__main__':
    generate_sample_data()
