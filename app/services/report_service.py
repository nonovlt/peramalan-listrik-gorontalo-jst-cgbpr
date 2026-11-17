import os
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from app.config import REPORT_DIR

class ReportService:
    @staticmethod
    def export_csv(predictions: List[Dict[str, Any]], filepath: str) -> str:
        df = pd.DataFrame(predictions)
        df.to_csv(filepath, index=False)
        return filepath

    @staticmethod
    def export_excel(
        historical_summary: Dict[str, Any],
        model_params: Dict[str, Any],
        metrics: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        filepath: str
    ) -> str:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Hasil Peramalan
            df_pred = pd.DataFrame(predictions)
            df_pred.to_excel(writer, sheet_name='Hasil Peramalan', index=False)

            # Sheet 2: Evaluasi & Parameter Model
            meta_data = [
                {"Kategori": "Parameter", "Item": "Jumlah Fitur Lag", "Nilai": model_params.get("lag", "-")},
                {"Kategori": "Parameter", "Item": "Arsitektur Hidden Layer", "Nilai": str(model_params.get("hidden_layers", "-"))},
                {"Kategori": "Parameter", "Item": "Maksimum Epoch", "Nilai": model_params.get("max_epochs", "-")},
                {"Kategori": "Parameter", "Item": "Interval Restart Beale-Powell", "Nilai": model_params.get("restart_interval", "-")},
                {"Kategori": "Parameter", "Item": "Powell Orthogonality Threshold", "Nilai": model_params.get("powell_threshold", "-")},
                {"Kategori": "Metrik Evaluasi", "Item": "MSE (Mean Squared Error)", "Nilai": f"{metrics.get('mse', 0):,.2f}"},
                {"Kategori": "Metrik Evaluasi", "Item": "RMSE (Root Mean Squared Error)", "Nilai": f"{metrics.get('rmse', 0):,.2f}"},
                {"Kategori": "Metrik Evaluasi", "Item": "MAPE (Mean Absolute Percentage Error)", "Nilai": f"{metrics.get('mape', 0):.2f}%"},
            ]
            df_meta = pd.DataFrame(meta_data)
            df_meta.to_excel(writer, sheet_name='Ringkasan Model', index=False)

        return filepath

    @staticmethod
    def export_pdf(
        model_params: Dict[str, Any],
        metrics: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        filepath: str
    ) -> str:
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1e293b"),
            alignment=1, # Center
            spaceAfter=6
        )

        subtitle_style = ParagraphStyle(
            name="SubTitleStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            alignment=1,
            spaceAfter=15
        )

        h2_style = ParagraphStyle(
            name="H2Style",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=6
        )

        normal_style = styles["Normal"]

        elements = []

        # Header Title
        elements.append(Paragraph("LAPORAN HASIL PERAMALAN ENERGI LISTRIK", title_style))
        elements.append(Paragraph("Sektor Rumah Tangga Kota Gorontalo — JST Conjugate Gradient Beale-Powell Restart", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=15))

        # Bagian 1: Konfigurasi Model
        elements.append(Paragraph("1. Konfigurasi Parameter Model JST & CG-BPR", h2_style))
        param_data = [
            ["Parameter", "Nilai Pengaturan", "Keterangan"],
            ["Fitur Lag (Window)", str(model_params.get("lag", 6)), "Jumlah bulan historis sebagai input"],
            ["Hidden Layers", str(model_params.get("hidden_layers", [10])), "Struktur neuron layer tersembunyi"],
            ["Fungsi Aktivasi", "Sigmoid (Hidden) & Linear (Output)", "Standar untuk regresi kontinu"],
            ["Algoritma Optimasi", "CG Beale-Powell Restart (CGB)", "Konjugasi gradien dengan restart ortogonal"],
            ["Powell Threshold", str(model_params.get("powell_threshold", 0.2)), "Batas deteksi kehilangan ortogonalitas"],
            ["Max Epochs", str(model_params.get("max_epochs", 300)), "Batas iterasi pelatihan"],
        ]
        t_param = Table(param_data, colWidths=[150, 150, 220])
        t_param.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        elements.append(t_param)
        elements.append(Spacer(1, 12))

        # Bagian 2: Hasil Evaluasi Akurasi
        elements.append(Paragraph("2. Evaluasi Akurasi Pengujian (Testing Set)", h2_style))
        mape_val = metrics.get('mape', 0)
        eval_data = [
            ["Metrik Evaluasi", "Nilai", "Kategori Akurasi"],
            ["Mean Squared Error (MSE)", f"{metrics.get('mse', 0):,.2f}", "Semakin kecil semakin baik"],
            ["Root Mean Squared Error (RMSE)", f"{metrics.get('rmse', 0):,.2f} kWh", "Estimasi deviasi dalam satuan asli"],
            ["Mean Absolute Percentage Error (MAPE)", f"{mape_val:.2f}%", "Sangat Baik (< 10%)" if mape_val < 10 else "Baik (10-20%)"],
        ]
        t_eval = Table(eval_data, colWidths=[200, 150, 170])
        t_eval.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('TEXTCOLOR', (1, 3), (1, 3), colors.HexColor("#16a34a")), # Green for MAPE
        ]))
        elements.append(t_eval)
        elements.append(Spacer(1, 14))

        # Bagian 3: Tabel Hasil Peramalan
        elements.append(Paragraph("3. Hasil Proyeksi Peramalan Konsumsi Listrik", h2_style))
        table_rows = [["No", "Periode", "Tahun", "Bulan", "Prediksi Konsumsi (kWh)"]]
        for idx, row in enumerate(predictions, 1):
            val_kwh = float(row.get("Konsumsi_Prediksi_kWh", 0))
            table_rows.append([
                str(idx),
                str(row.get("Periode", "-")),
                str(row.get("Tahun", "-")),
                str(row.get("Bulan", "-")),
                f"{val_kwh:,.0f} kWh"
            ])

        t_pred = Table(table_rows, colWidths=[40, 100, 80, 120, 180])
        t_pred.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (4, 1), (4, -1), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        elements.append(t_pred)

        # Footer note
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("* Laporan ini digenerate secara otomatis oleh Sistem Peramalan Konsumsi Energi Listrik Kota Gorontalo.", normal_style))

        doc.build(elements)
        return filepath
