import os
import pandas as pd
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from app.config import REPORT_DIR

class ReportService:
    @staticmethod
    def export_csv(predictions: List[Dict[str, Any]], filepath: str) -> str:
        df = pd.DataFrame(predictions)
        df.to_csv(filepath, index=False)
        return filepath

    @staticmethod
    def export_excel(
        model_params: Dict[str, Any],
        metrics: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]],
        diagnostics: Optional[Dict[str, Any]],
        filepath: str
    ) -> str:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Hasil Peramalan 95% CI
            df_pred = pd.DataFrame(predictions)
            df_pred.to_excel(writer, sheet_name='Hasil Peramalan 95% CI', index=False)

            # Sheet 2: Evaluasi & Parameter Model
            meta_data = [
                {"Kategori": "Parameter", "Item": "Jumlah Fitur Lag (Window)", "Nilai": model_params.get("lag", "-")},
                {"Kategori": "Parameter", "Item": "Arsitektur Hidden Layer", "Nilai": str(model_params.get("hidden_layers", "-"))},
                {"Kategori": "Parameter", "Item": "Fungsi Aktivasi", "Nilai": str(model_params.get("activation", "sigmoid"))},
                {"Kategori": "Parameter", "Item": "Algoritma Optimasi", "Nilai": str(model_params.get("optimizer", "cgb"))},
                {"Kategori": "Parameter", "Item": "Maksimum Epoch", "Nilai": model_params.get("max_epochs", "-")},
                {"Kategori": "Parameter", "Item": "Interval Restart Beale-Powell", "Nilai": model_params.get("restart_interval", "-")},
                {"Kategori": "Parameter", "Item": "Powell Orthogonality Threshold", "Nilai": model_params.get("powell_threshold", "-")},
                {"Kategori": "Metrik Evaluasi", "Item": "MSE (Mean Squared Error)", "Nilai": f"{metrics.get('mse', 0):,.2f}"},
                {"Kategori": "Metrik Evaluasi", "Item": "RMSE (Root Mean Squared Error)", "Nilai": f"{metrics.get('rmse', 0):,.2f}"},
                {"Kategori": "Metrik Evaluasi", "Item": "MAPE (%)", "Nilai": f"{metrics.get('mape', 0):.2f}%"},
            ]
            if diagnostics and "durbin_watson" in diagnostics:
                dw = diagnostics["durbin_watson"]
                meta_data.append({"Kategori": "Diagnostik", "Item": "Statistik Durbin-Watson", "Nilai": f"{dw.get('durbin_watson', '-')} ({dw.get('status', '-')})"})

            df_meta = pd.DataFrame(meta_data)
            df_meta.to_excel(writer, sheet_name='Parameter & Akurasi', index=False)

            # Sheet 3: Benchmark Model (jika ada)
            if benchmark_results and "models" in benchmark_results:
                df_bench = pd.DataFrame(benchmark_results["models"])
                df_bench.to_excel(writer, sheet_name='Komparasi Benchmark', index=False)

        return filepath

    @staticmethod
    def export_pdf(
        model_params: Dict[str, Any],
        metrics: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        benchmark_results: Optional[Dict[str, Any]],
        diagnostics: Optional[Dict[str, Any]],
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
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            alignment=1,
            spaceAfter=4
        )

        subtitle_style = ParagraphStyle(
            name="SubTitleStyle",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#64748b"),
            alignment=1,
            spaceAfter=12
        )

        h2_style = ParagraphStyle(
            name="H2Style",
            parent=styles["Heading2"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=8,
            spaceAfter=5
        )

        normal_style = ParagraphStyle(
            name="NormalSmall",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#334155")
        )

        elements = []

        # Header Title
        elements.append(Paragraph("LAPORAN TEKNIS PERAMALAN ENERGI LISTRIK", title_style))
        elements.append(Paragraph("Sektor Rumah Tangga Kota Gorontalo — Jaringan Saraf Tiruan (JST) CG Beale-Powell Restart", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

        # Bagian 1: Parameter Model
        elements.append(Paragraph("1. Konfigurasi Parameter JST & Algoritma CG-BPR", h2_style))
        param_data = [
            ["Parameter", "Nilai Pengaturan", "Keterangan Ilmiah"],
            ["Fitur Lag (Window)", str(model_params.get("lag", 6)), "Jumlah bulan konsumsi sebelumnya sebagai input"],
            ["Hidden Layers", str(model_params.get("hidden_layers", [10])), "Struktur neuron layer tersembunyi"],
            ["Fungsi Aktivasi", str(model_params.get("activation", "sigmoid")).upper(), "Non-linearitas tersembunyi (Output: Linear)"],
            ["Algoritma Optimasi", "CG Beale-Powell Restart", "Pembaruan arah konjugat dengan reset ortogonalitas"],
            ["Powell Threshold (&eta;)", str(model_params.get("powell_threshold", 0.2)), "Batas kehilangan ortogonalitas |g_k^T g_{k-1}| >= eta*||g_k||^2"],
            ["Maksimum Epoch", str(model_params.get("max_epochs", 300)), "Batas iterasi pelatihan"],
        ]
        t_param = Table(param_data, colWidths=[140, 130, 250])
        t_param.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        elements.append(t_param)
        elements.append(Spacer(1, 8))

        # Bagian 2: Evaluasi Akurasi & Diagnostik
        elements.append(Paragraph("2. Evaluasi Akurasi Pengujian & Diagnostik Residual", h2_style))
        mape_val = metrics.get('mape', 0)
        dw_val = "-"
        dw_status = "-"
        if diagnostics and "durbin_watson" in diagnostics:
            dw_val = str(diagnostics["durbin_watson"].get("durbin_watson", "-"))
            dw_status = str(diagnostics["durbin_watson"].get("interpretation", "-"))

        eval_data = [
            ["Metrik / Uji", "Nilai", "Kategori / Interpretasi"],
            ["Mean Squared Error (MSE)", f"{metrics.get('mse', 0):,.2f}", "Semakin kecil semakin baik"],
            ["Root Mean Squared Error (RMSE)", f"{metrics.get('rmse', 0):,.2f} kWh", "Akurasi deviasi dalam satuan kWh"],
            ["Mean Absolute Percentage Error (MAPE)", f"{mape_val:.2f}%", "Sangat Baik (< 10%)" if mape_val < 10 else "Baik (10-20%)"],
            ["Statistik Durbin-Watson", dw_val, dw_status]
        ]
        t_eval = Table(eval_data, colWidths=[180, 130, 210])
        t_eval.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('TEXTCOLOR', (1, 3), (1, 3), colors.HexColor("#16a34a")),
        ]))
        elements.append(t_eval)
        elements.append(Spacer(1, 8))

        # Bagian 3: Komparasi Benchmark (jika tersedia)
        if benchmark_results and "models" in benchmark_results:
            elements.append(Paragraph("3. Komparasi Benchmark Algoritma Optimasi", h2_style))
            bench_table = [["Model Algoritma", "RMSE (kWh)", "MAPE (%)", "Waktu (s)", "Restart"]]
            for bm in benchmark_results["models"]:
                bench_table.append([
                    bm.get("name", "-"),
                    f"{bm.get('rmse', 0):,.0f}",
                    f"{bm.get('mape', 0):.2f}%",
                    f"{bm.get('elapsed_time_sec', 0):.2f}s",
                    str(bm.get("total_restarts", 0))
                ])
            t_bench = Table(bench_table, colWidths=[200, 100, 80, 70, 70])
            t_bench.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]))
            elements.append(t_bench)
            elements.append(Spacer(1, 8))

        # Bagian 4: Tabel Hasil Peramalan 95% Confidence Interval
        elements.append(Paragraph("4. Tabel Proyeksi Konsumsi Listrik dengan 95% Confidence Interval", h2_style))
        table_rows = [["No", "Periode", "Bulan", "Batas Bawah (95%)", "Prediksi Konsumsi", "Batas Atas (95%)"]]
        for idx, row in enumerate(predictions, 1):
            val_kwh = float(row.get("Konsumsi_Prediksi_kWh", 0))
            low_kwh = float(row.get("Batas_Bawah_95_kWh", val_kwh * 0.95))
            up_kwh = float(row.get("Batas_Atas_95_kWh", val_kwh * 1.05))
            table_rows.append([
                str(idx),
                str(row.get("Periode", "-")),
                f"{row.get('Bulan', '-')} {row.get('Tahun', '')}",
                f"{low_kwh:,.0f} kWh",
                f"{val_kwh:,.0f} kWh",
                f"{up_kwh:,.0f} kWh"
            ])

        t_pred = Table(table_rows, colWidths=[30, 80, 110, 100, 100, 100])
        t_pred.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (3, 1), (-1, -1), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        elements.append(t_pred)

        elements.append(Spacer(1, 14))
        elements.append(Paragraph(
            "* Laporan ini digenerate secara otomatis oleh SIKEL-Gorontalo. Nilai batas bawah dan batas atas dihitung pada tingkat signifikansi alpha = 0.05 (Interval Kepercayaan 95%).",
            normal_style
        ))

        doc.build(elements)
        return filepath
