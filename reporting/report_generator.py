# reporting/report_generator.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from config import Config

class ReportGenerator:
    @staticmethod
    def generate(csv_path):
        df = pd.read_csv(csv_path)
        
        # Ensure output directory exists
        os.makedirs(Config.REPORT_DIR, exist_ok=True)
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(Config.REPORT_DIR, report_filename)
        
        doc = SimpleDocTemplate(report_path, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=24, spaceAfter=20, textColor=colors.HexColor("#1a3a5c"))
        h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=16, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor("#0d7377"))
        h3_style = ParagraphStyle('H3Style', parent=styles['Heading3'], fontSize=12, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#2c3e50"), weight='bold')
        normal_style = styles['Normal']
        
        story = []

        # 1. HEADER & BRANDING
        story.append(Paragraph("ERGO-VISION 🦴", title_style))
        story.append(Paragraph("High-Fidelity Ergonomic Posture Assessment", styles['SubTitle']))
        story.append(Paragraph("-" * 80, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 2. SESSION METADATA
        duration = df['timestamp'].max() - df['timestamp'].min()
        session_date = datetime.now().strftime('%B %d, %Y | %H:%M:%S')
        
        meta_data = [
            [Paragraph(f"<b>Session ID:</b> {os.path.basename(csv_path)}", normal_style), 
             Paragraph(f"<b>Date:</b> {session_date}", normal_style)],
            [Paragraph(f"<b>Duration:</b> {duration:.1f} seconds", normal_style), 
             Paragraph(f"<b>Frame Count:</b> {len(df)}", normal_style)]
        ]
        t_meta = Table(meta_data, colWidths=[3*inch, 3*inch])
        t_meta.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(t_meta)
        story.append(Spacer(1, 25))

        # 3. EXECUTIVE SUMMARY
        story.append(Paragraph("1. Executive Risk Summary", h2_style))
        
        rula_mean = df['RULA_score'].mean()
        reba_mean = df['REBA_score'].mean()
        rula_max = df['RULA_score'].max()
        reba_max = df['REBA_score'].max()

        # Determine overall level
        if rula_max >= 7 or reba_max >= 11:
            risk_text = "CRITICAL ACTION REQUIRED"
            risk_col = colors.red
        elif rula_max >= 5 or reba_max >= 8:
            risk_text = "HIGH RISK - INTERVENTION NEEDED"
            risk_col = colors.orange
        elif rula_max >= 3 or reba_max >= 4:
            risk_text = "MODERATE RISK - MONITOR CLOSELY"
            risk_col = colors.HexColor("#d4a017") # Dark gold
        else:
            risk_text = "ACCEPTABLE - NEGLIGIBLE RISK"
            risk_col = colors.green

        story.append(Paragraph(f"<font color='{risk_col}'><b>OVERALL STATUS: {risk_text}</b></font>", ParagraphStyle('Risk', parent=normal_style, fontSize=14, alignment=1)))
        story.append(Spacer(1, 15))

        exec_data = [
            ['Methodology', 'Average Score', 'Peak Score', 'Action Level'],
            ['RULA (Upper Limb)', f"{rula_mean:.2f}", f"{rula_max}", self._get_rula_action(rula_max)],
            ['REBA (Entire Body)', f"{reba_mean:.2f}", f"{reba_max}", self._get_reba_action(reba_max)]
        ]
        t_exec = Table(exec_data, colWidths=[1.8*inch, 1.4*inch, 1.4*inch, 1.8*inch])
        t_exec.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a3a5c")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
        ]))
        story.append(t_exec)
        story.append(Spacer(1, 20))

        # 4. BILATERAL KINEMATIC ANALYSIS
        story.append(Paragraph("2. Kinematic Breakdown (Bilateral Joint Angles)", h2_style))
        joint_cols = [
            ('Neck', 'neck_deg'), ('Trunk', 'trunk_deg'),
            ('Upper Arm (L)', 'ua_left_deg'), ('Upper Arm (R)', 'ua_right_deg'),
            ('Elbow (L)', 'el_left_deg'), ('Elbow (R)', 'el_right_deg'),
            ('Wrist (L)', 'wr_left_deg'), ('Wrist (R)', 'wr_right_deg')
        ]
        stats_data = [['Joint Component', 'Min (°)', 'Max (°)', 'Mean (°)', 'Strain']]
        
        for name, col in joint_cols:
            if col in df.columns:
                c_min = df[col].min()
                c_max = df[col].max()
                c_mean = df[col].mean()
                c_range = c_max - c_min
                # Simple strain index based on range and max
                strain = "High" if abs(c_max) > 45 or c_range > 60 else "Med" if abs(c_max) > 20 else "Low"
                stats_data.append([name, f"{c_min:.1f}", f"{c_max:.1f}", f"{c_mean:.1f}", strain])
                
        t_stats = Table(stats_data, colWidths=[1.8*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.3*inch])
        t_stats.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d7377")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
        ]))
        story.append(t_stats)
        story.append(Spacer(1, 20))

        # 5. RISK DISTRIBUTION (Visuals)
        story.append(Paragraph("3. Temporal Risk Distribution", h2_style))
        
        os.makedirs(Config.STATIC_DIR, exist_ok=True)
        # Pie Chart: Risk Level Distribution
        risk_counts = df['risk_prediction'].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['#00e5a0', '#ffc94d', '#ff4d6d', '#ff7c3e'])
        plt.title('Time Spent in Risk Zones')
        pie_path = os.path.join(Config.STATIC_DIR, f"temp_pie_{datetime.now().timestamp()}.png")
        plt.savefig(pie_path, transparent=True)
        plt.close()
        
        # Time Series: RULA/REBA
        plt.figure(figsize=(8, 3))
        plt.plot(df['timestamp'], df['RULA_score'], color='#00d4ff', label='RULA', linewidth=1.5)
        plt.plot(df['timestamp'], df['REBA_score'], color='#9b59ff', label='REBA', linewidth=1.5, alpha=0.7)
        plt.fill_between(df['timestamp'], df['RULA_score'], color='#00d4ff', alpha=0.1)
        plt.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='High Risk Threshold')
        plt.xlabel('Time (s)')
        plt.ylabel('Score')
        plt.title('Ergonomic Score Timeline')
        plt.legend(loc='upper right', fontsize='small')
        timeline_path = os.path.join(Config.STATIC_DIR, f"temp_time_{datetime.now().timestamp()}.png")
        plt.savefig(timeline_path)
        plt.close()

        story.append(Image(pie_path, width=3*inch, height=3*inch))
        story.append(Spacer(1, 10))
        story.append(Image(timeline_path, width=6*inch, height=2.2*inch))
        
        story.append(PageBreak())

        # 6. ANOMALIES & CLINICAL OBSERVATIONS
        story.append(Paragraph("4. Anomalies & Clinical Observations", h2_style))
        
        anoms = df[df['anomalies'] != "None"]['anomalies'].unique()
        if len(anoms) > 0:
            story.append(Paragraph("The following posture anomalies were detected during the session:", h3_style))
            anom_list = "<ul>"
            all_anoms = []
            for entry in anoms:
                all_anoms.extend(entry.split("; "))
            unique_anoms = list(set(all_anoms))
            for a in unique_anoms[:10]: # Limit to top 10
                anom_list += f"<li>{a}</li>"
            anom_list += "</ul>"
            story.append(Paragraph(anom_list, normal_style))
        else:
            story.append(Paragraph("No significant posture anomalies detected. Subject maintained safe joint ranges.", normal_style))
        
        story.append(Spacer(1, 20))

        # 7. CORRECTIVE ACTION PLAN
        story.append(Paragraph("5. Corrective Action Plan (CAP)", h2_style))
        
        cap_logic = []
        if rula_max >= 7:
            cap_logic.append("<b>Critical:</b> Redesign workstation immediately to reduce extreme upper limb strain.")
        if reba_max >= 8:
            cap_logic.append("<b>High Priority:</b> Implement material handling aids or adjust task height to reduce full-body load.")
        if df['neck_deg'].max() > 30:
            cap_logic.append("<b>Neck:</b> Elevate monitor/display to eye level. Detected excessive cervical flexion.")
        if df['trunk_deg'].max() > 45:
            cap_logic.append("<b>Trunk:</b> Subject is leaning excessively. Provide lumbar support or bring task closer to body.")
        if df['ua_left_deg'].max() > 60 or df['ua_right_deg'].max() > 60:
            cap_logic.append("<b>Shoulder:</b> Upper arm elevation detected. Lower the keyboard/tooling surface or use armrests.")

        if not cap_logic:
            cap_logic.append("No immediate corrective actions required. Maintain existing ergonomic standards.")

        cap_html = "<ul>" + "".join([f"<li>{item}</li>" for item in cap_logic]) + "</ul>"
        story.append(Paragraph(cap_html, normal_style))
        
        story.append(Spacer(1, 40))
        
        # 8. SIGN-OFF
        story.append(Paragraph("-" * 30, normal_style))
        story.append(Paragraph("Assessed by ErgoVision AI Engine", styles['Italic']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))

        # Build PDF
        doc.build(story)
        
        # Cleanup
        for p in [pie_path, timeline_path]:
            if os.path.exists(p): os.remove(p)
            
        return report_path

    @staticmethod
    def _get_rula_action(score):
        if score >= 7: return "Immediate Change"
        if score >= 5: return "Further Investigate"
        if score >= 3: return "May need change"
        return "Acceptable"

    @staticmethod
    def _get_reba_action(score):
        if score >= 11: return "Very High - Immediate"
        if score >= 8:  return "High Risk"
        if score >= 4:  return "Medium Risk"
        return "Low Risk"