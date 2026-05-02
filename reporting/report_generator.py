# reporting/report_generator.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from config import Config

class ReportGenerator:
    @staticmethod
    def generate(csv_path):
        df = pd.read_csv(csv_path)
        
        # Ensure output directory exists
        os.makedirs(Config.REPORT_DIR, exist_ok=True)
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(Config.REPORT_DIR, report_filename)
        doc = SimpleDocTemplate(report_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # A) HEADER
        story.append(Paragraph("Ergonomic Risk Assessment Report", styles['Title']))
        story.append(Paragraph("CONFIDENTIAL", styles['Italic']))
        story.append(Spacer(1, 12))
        
        duration = df['timestamp'].max() - df['timestamp'].min()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Approximation
        story.append(Paragraph(f"<b>Session Date:</b> {start_time_str}", styles['Normal']))
        story.append(Paragraph(f"<b>Duration:</b> {duration:.1f} seconds", styles['Normal']))
        story.append(Paragraph(f"<b>Total Samples:</b> {len(df)}", styles['Normal']))
        story.append(Spacer(1, 12))

        # B) OVERALL RISK PROFILE
        rula_mean = df['RULA_score'].mean()
        reba_mean = df['REBA_score'].mean()
        if rula_mean > 6 or reba_mean > 8:
            risk_level = "HIGH RISK"
            risk_color = colors.red
        elif rula_mean > 4 or reba_mean > 4:
            risk_level = "MODERATE RISK"
            risk_color = colors.orange
        else:
            risk_level = "LOW RISK"
            risk_color = colors.green
        
        story.append(Paragraph(f"<b>OVERALL RISK PROFILE:</b> <font color='{risk_color}'>{risk_level}</font>", styles['Heading2']))
        story.append(Spacer(1, 12))

        # C) EXECUTIVE SUMMARY TABLE
        story.append(Paragraph("Executive Summary", styles['Heading3']))
        exec_data = [
            ['Metric', 'Mean', 'Peak'],
            ['RULA Score', f"{rula_mean:.1f}", f"{df['RULA_score'].max():.1f}"],
            ['REBA Score', f"{reba_mean:.1f}", f"{df['REBA_score'].max():.1f}"]
        ]
        t1 = Table(exec_data, colWidths=[200, 100, 100])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(t1)
        story.append(Spacer(1, 15))

        # D) JOINT ANGLE STATISTICS
        story.append(Paragraph("Joint Angle Statistics", styles['Heading3']))
        joint_cols = ['neck_deg', 'trunk_deg', 'upper_arm_deg', 'elbow_deg', 'wrist_deg']
        stats_data = [['Joint', 'Min', 'Max', 'Mean', 'Std Dev', '95th %ile']]
        
        for col in joint_cols:
            if col in df.columns:
                c_min = df[col].min()
                c_max = df[col].max()
                c_mean = df[col].mean()
                c_std = df[col].std()
                c_95 = df[col].quantile(0.95)
                name = col.replace('_deg', '').replace('_', ' ').title()
                stats_data.append([name, f"{c_min:.1f}", f"{c_max:.1f}", f"{c_mean:.1f}", f"{c_std:.1f}", f"{c_95:.1f}"])
                
        t2 = Table(stats_data)
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(t2)
        story.append(Spacer(1, 15))

        # F) AI INSIGHTS
        story.append(Paragraph("AI Insights & Critical Joints", styles['Heading3']))
        # Identify the joint with the highest mean absolute deviation from 0
        critical_joint = "None"
        max_dev = 0
        for col in joint_cols:
            if col in df.columns:
                dev = df[col].abs().mean()
                if dev > max_dev:
                    max_dev = dev
                    critical_joint = col.replace('_deg', '').replace('_', ' ').title()
        
        story.append(Paragraph(f"<b>Most Critical Joint:</b> {critical_joint} (Mean absolute deviation: {max_dev:.1f}°)", styles['Normal']))
        anomalies_count = df['anomalies'].apply(lambda x: 0 if x == "None" else 1).sum()
        story.append(Paragraph(f"<b>Total Anomalous Frames Detected:</b> {anomalies_count}", styles['Normal']))
        story.append(Spacer(1, 15))

        # G) VISUAL CHARTS
        story.append(Paragraph("Visual Charts", styles['Heading3']))
        
        # Create temp static dir for plots if missing
        os.makedirs(Config.STATIC_DIR, exist_ok=True)
        
        # Plot 1: Scores
        plt.figure(figsize=(7, 3.5))
        plt.plot(df['timestamp'], df['RULA_score'], label='RULA', color='cyan')
        plt.plot(df['timestamp'], df['REBA_score'], label='REBA', color='purple')
        plt.xlabel('Time (s)')
        plt.ylabel('Risk Score')
        plt.title('RULA & REBA Scores Over Time')
        plt.legend()
        plt.tight_layout()
        plot1_path = os.path.join(Config.STATIC_DIR, f"temp_scores_{datetime.now().timestamp()}.png")
        plt.savefig(plot1_path)
        plt.close()
        
        # Plot 2: Joints
        plt.figure(figsize=(7, 3.5))
        for col in joint_cols:
            if col in df.columns:
                plt.plot(df['timestamp'], df[col], label=col.replace('_deg', ''))
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Joint Angles Over Time')
        plt.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plot2_path = os.path.join(Config.STATIC_DIR, f"temp_angles_{datetime.now().timestamp()}.png")
        plt.savefig(plot2_path)
        plt.close()

        story.append(Image(plot1_path, width=420, height=210))
        story.append(Spacer(1, 10))
        story.append(Image(plot2_path, width=420, height=210))
        story.append(Spacer(1, 15))

        # H) CLINICAL RECOMMENDATIONS
        story.append(Paragraph("Clinical Recommendations", styles['Heading3']))
        rec = "<ul>"
        if risk_level == "HIGH RISK":
            rec += "<li><b>Immediate Action Required:</b> Evaluate workstation setup immediately. The high RULA/REBA scores indicate a substantial risk of musculoskeletal injury.</li>"
            rec += "<li>Introduce frequent micro-breaks and postural variations.</li>"
        elif risk_level == "MODERATE RISK":
            rec += "<li><b>Further Investigation:</b> Consider ergonomic interventions such as adjustable chairs, monitor risers, or anti-fatigue mats.</li>"
        else:
            rec += "<li><b>Maintenance:</b> Current posture appears acceptable. Continue monitoring periodically.</li>"
            
        if "Neck" in critical_joint:
            rec += "<li><b>Neck Strain:</b> Ensure the monitor is at eye level to prevent excessive cervical flexion.</li>"
        elif "Trunk" in critical_joint:
            rec += "<li><b>Trunk Lean:</b> The subject is leaning forward. Adjust chair height or pull the workspace closer.</li>"
        elif "Arm" in critical_joint:
            rec += "<li><b>Arm Elevation:</b> Shoulders are elevated. Lower the work surface or use armrests.</li>"
            
        rec += "</ul>"
        story.append(Paragraph(rec, styles['Normal']))

        # Build PDF
        doc.build(story)
        
        # Cleanup temp plots
        try:
            os.remove(plot1_path)
            os.remove(plot2_path)
        except:
            pass
            
        return report_path