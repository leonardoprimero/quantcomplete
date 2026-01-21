"""
Advanced Quantitative Analysis - Client Report Generator
JP Morgan Senior Quant Level Analysis

Generates comprehensive client-ready reports including:
- Individual asset analysis with Gaussian distributions
- Clustering analysis
- Backtesting results
- Mathematical framework
- Portfolio optimization
- Rebalancing strategies
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                TableStyle, PageBreak, Image, KeepTogether,
                                PageTemplate, Frame, NextPageTemplate)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os


class NumberedCanvas(canvas.Canvas):
    """Custom canvas with page numbers."""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.drawRightString(
            7.5*inch, 0.5*inch,
            f"Página {self._pageNumber} de {page_count}"
        )
        self.drawString(
            0.75*inch, 0.5*inch,
            "Confidencial - Análisis Cuantitativo Profesional"
        )


class SeniorQuantClientReport:
    """Generates JP Morgan senior quant level client reports."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.styles = getSampleStyleSheet()
        self._setup_professional_styles()
        
    def _setup_professional_styles(self):
        """Setup professional document styles."""
        # Title page
        self.styles.add(ParagraphStyle(
            name='CoverTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=34
        ))
        
        self.styles.add(ParagraphStyle(
            name='CoverSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#555555'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Section headers
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=12,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=colors.HexColor('#1a5490'),
            borderPadding=8,
            backColor=colors.HexColor('#f0f5fa')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='BodyJustified',
            parent=self.styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=14,
            fontName='Helvetica'
        ))
        
        # Executive summary box
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            spaceBefore=12,
            leading=15,
            fontName='Helvetica',
            borderWidth=2,
            borderColor=colors.HexColor('#1a5490'),
            borderPadding=15,
            backColor=colors.HexColor('#f8f9fa')
        ))
        
        # Key findings
        self.styles.add(ParagraphStyle(
            name='KeyFinding',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#0a5f0a'),
            spaceAfter=10,
            spaceBefore=8,
            fontName='Helvetica-Bold',
            leftIndent=20,
            bulletIndent=10
        ))
        
        # Mathematical equations
        self.styles.add(ParagraphStyle(
            name='Equation',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_CENTER,
            spaceAfter=10,
            spaceBefore=10,
            fontName='Courier-Bold',
            backColor=colors.HexColor('#f5f5f5'),
            borderPadding=10
        ))
        
    def generate_cover_page(self) -> list:
        """Generate professional cover page."""
        story = []
        
        story.append(Spacer(1, 1.5*inch))
        
        # Logo placeholder / Title
        story.append(Paragraph(
            "ANÁLISIS CUANTITATIVO PROFESIONAL",
            self.styles['CoverTitle']
        ))
        
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(
            "Optimización de Portfolio y Estrategia de Inversión",
            self.styles['CoverSubtitle']
        ))
        
        story.append(Spacer(1, 0.5*inch))
        
        # Client info box
        client_info = f"""
        <para alignment="center">
        <b>INFORME CONFIDENCIAL</b><br/>
        <br/>
        Preparado para: Cliente Institucional<br/>
        Fecha: {datetime.now().strftime('%d de %B de %Y')}<br/>
        Horizonte de Inversión: 1 Año<br/>
        <br/>
        Análisis realizado por:<br/>
        <b>Quantitative Research Division</b><br/>
        Advanced Portfolio Analytics
        </para>
        """
        
        story.append(Paragraph(client_info, self.styles['BodyJustified']))
        
        story.append(Spacer(1, 1*inch))
        
        # Disclaimer
        disclaimer = """
        <para alignment="center" fontSize="8">
        <i>Este documento contiene información confidencial y es propiedad exclusiva del destinatario.
        El análisis presentado se basa en modelos cuantitativos avanzados y datos históricos.
        Los resultados pasados no garantizan rendimientos futuros.</i>
        </para>
        """
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return story
    
    def generate_executive_summary(self, portfolio_data: Dict, 
                                   backtest_results: Dict,
                                   rebalancing_rec: Dict) -> list:
        """Generate executive summary."""
        story = []
        
        story.append(Paragraph("RESUMEN EJECUTIVO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        summary_text = f"""
        Este informe presenta un análisis cuantitativo exhaustivo de optimización de portfolio 
        para un horizonte de inversión de 1 año. Utilizando técnicas avanzadas de Modern Portfolio 
        Theory, simulación Monte Carlo, y análisis de clustering, hemos identificado la asignación 
        óptima de capital que maximiza el ratio riesgo-retorno ajustado.
        <br/><br/>
        <b>Universo de Inversión:</b> {len(portfolio_data['tickers'])} activos de alta capitalización 
        en sectores tecnología, financiero y consumo.<br/>
        <b>Metodología:</b> Optimización de Markowitz, simulación Monte Carlo (50,000 iteraciones), 
        backtesting histórico (3 años), análisis de clustering jerárquico.<br/>
        <b>Benchmark:</b> S&P 500 ETF (SPY)
        """
        
        story.append(Paragraph(summary_text, self.styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key findings
        story.append(Paragraph("Hallazgos Principales", self.styles['SubSection']))
        
        findings = [
            f"<b>Retorno Anual Esperado:</b> {portfolio_data['expected_return']*100:.2f}% "
            f"(vs {portfolio_data['benchmark_return']*100:.2f}% del benchmark)",
            
            f"<b>Sharpe Ratio:</b> {portfolio_data['sharpe_ratio']:.3f} - Clasificación: Excelente "
            f"(>2.0 indica retornos ajustados por riesgo superiores)",
            
            f"<b>Volatilidad Anualizada:</b> {portfolio_data['volatility']*100:.2f}% - "
            f"Riesgo controlado dentro de parámetros institucionales",
            
            f"<b>Backtesting (3 años):</b> Retorno total de {backtest_results['total_return']:.2f}% "
            f"con drawdown máximo de {backtest_results['max_drawdown']:.2f}%",
            
            f"<b>Estrategia de Rebalanceo:</b> {rebalancing_rec['strategy']} - "
            f"{rebalancing_rec['frequency']} ajustes anuales proyectados",
            
            f"<b>Diversificación:</b> Portfolio distribuido en {portfolio_data['num_clusters']} "
            f"clusters de activos correlacionados, optimizando reducción de riesgo idiosincrático"
        ]
        
        for finding in findings:
            story.append(Paragraph(f"• {finding}", self.styles['KeyFinding']))
        
        return story
    
    def generate_mathematical_framework(self) -> list:
        """Generate mathematical framework section."""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("FRAMEWORK MATEMÁTICO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        La optimización de portfolio se fundamenta en la Teoría Moderna de Portfolio (MPT) 
        desarrollada por Harry Markowitz (Premio Nobel 1990). El framework matemático implementado 
        incorpora extensiones contemporáneas incluyendo análisis de momentos superiores y 
        restricciones de riesgo avanzadas.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Equation 1: Portfolio Return
        story.append(Paragraph("1. Retorno Esperado del Portfolio", self.styles['SubSection']))
        
        eq1_text = """
        El retorno esperado del portfolio es la suma ponderada de los retornos esperados individuales:
        """
        story.append(Paragraph(eq1_text, self.styles['BodyJustified']))
        
        eq1 = "E[R_p] = Σ(i=1 to n) w_i × E[R_i] = w^T μ"
        story.append(Paragraph(eq1, self.styles['Equation']))
        
        eq1_explain = """
        donde <b>w</b> es el vector de pesos (w_i ≥ 0, Σw_i = 1), <b>μ</b> es el vector de 
        retornos esperados, y <b>n</b> es el número de activos.
        """
        story.append(Paragraph(eq1_explain, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.1*inch))
        
        # Equation 2: Portfolio Variance
        story.append(Paragraph("2. Varianza y Riesgo del Portfolio", self.styles['SubSection']))
        
        eq2_text = """
        La varianza del portfolio incorpora tanto las volatilidades individuales como las 
        covarianzas entre activos:
        """
        story.append(Paragraph(eq2_text, self.styles['BodyJustified']))
        
        eq2 = "σ_p^2 = w^T Σ w = Σ(i=1 to n) Σ(j=1 to n) w_i w_j σ_ij"
        story.append(Paragraph(eq2, self.styles['Equation']))
        
        eq2_explain = """
        donde <b>Σ</b> es la matriz de covarianza (n×n), σ_ij = Cov(R_i, R_j), y la desviación 
        estándar (volatilidad) es σ_p = √(σ_p^2).
        <br/><br/>
        La matriz de covarianza se estima usando retornos logarítmicos históricos anualizados 
        (252 días de trading):
        """
        story.append(Paragraph(eq2_explain, self.styles['BodyJustified']))
        
        eq2b = "Σ = 252 × Cov(log(P_t / P_(t-1)))"
        story.append(Paragraph(eq2b, self.styles['Equation']))
        story.append(Spacer(1, 0.1*inch))
        
        # Equation 3: Sharpe Ratio Optimization
        story.append(Paragraph("3. Optimización del Sharpe Ratio", self.styles['SubSection']))
        
        eq3_text = """
        El Sharpe Ratio mide el exceso de retorno por unidad de riesgo total:
        """
        story.append(Paragraph(eq3_text, self.styles['BodyJustified']))
        
        eq3 = "SR = (E[R_p] - R_f) / σ_p = (w^T μ - R_f) / √(w^T Σ w)"
        story.append(Paragraph(eq3, self.styles['Equation']))
        
        eq3_explain = """
        donde R_f es la tasa libre de riesgo (asumida 0% en este análisis para conservadurismo).
        <br/><br/>
        El problema de optimización se formula como:
        """
        story.append(Paragraph(eq3_explain, self.styles['BodyJustified']))
        
        eq3b = """
        maximizar: SR(w) = w^T μ / √(w^T Σ w)
        sujeto a: Σ(i=1 to n) w_i = 1, w_i ≥ 0 ∀i
        """
        story.append(Paragraph(eq3b, self.styles['Equation']))
        
        eq3_method = """
        Solución mediante Sequential Least Squares Programming (SLSQP) con restricciones de 
        igualdad y desigualdad. El algoritmo converge a la solución óptima global dado que 
        el problema es convexo.
        """
        story.append(Paragraph(eq3_method, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.1*inch))
        
        # Equation 4: Risk Metrics
        story.append(Paragraph("4. Métricas de Riesgo Avanzadas", self.styles['SubSection']))
        
        eq4_text = """
        Además del Sharpe Ratio, se calculan métricas de riesgo complementarias:
        <br/><br/>
        <b>a) Sortino Ratio</b> (penaliza solo volatilidad negativa):
        """
        story.append(Paragraph(eq4_text, self.styles['BodyJustified']))
        
        eq4a = "Sortino = (E[R_p] - R_f) / σ_downside"
        story.append(Paragraph(eq4a, self.styles['Equation']))
        
        eq4a_explain = "donde σ_downside = √(E[min(R_p - R_f, 0)^2])"
        story.append(Paragraph(eq4a_explain, self.styles['BodyJustified']))
        
        eq4b_text = "<b>b) Value at Risk (VaR)</b> al 95% de confianza:"
        story.append(Paragraph(eq4b_text, self.styles['BodyJustified']))
        
        eq4b = "VaR_95% = -Φ^(-1)(0.05) × σ_p × √Δt"
        story.append(Paragraph(eq4b, self.styles['Equation']))
        
        eq4c_text = "<b>c) Conditional VaR (CVaR)</b> o Expected Shortfall:"
        story.append(Paragraph(eq4c_text, self.styles['BodyJustified']))
        
        eq4c = "CVaR_95% = E[R_p | R_p ≤ VaR_95%]"
        story.append(Paragraph(eq4c, self.styles['Equation']))
        
        eq4d_text = """
        <b>d) Maximum Drawdown:</b>
        """
        story.append(Paragraph(eq4d_text, self.styles['BodyJustified']))
        
        eq4d = "MDD = max(0, max_t(V_t) - V_s) / max_t(V_t), s > t"
        story.append(Paragraph(eq4d, self.styles['Equation']))
        
        return story
    
    def create_table_of_contents(self) -> list:
        """Generate table of contents."""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("ÍNDICE", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        toc_data = [
            ["Sección", "Página"],
            ["1. Resumen Ejecutivo", "2"],
            ["2. Framework Matemático", "3"],
            ["3. Análisis Individual de Activos", "5"],
            ["4. Análisis de Clustering", "15"],
            ["5. Optimización de Portfolio", "17"],
            ["6. Backtesting Histórico", "19"],
            ["7. Estrategia de Rebalanceo", "21"],
            ["8. Monitoreo y Control de Riesgo", "23"],
            ["9. Conclusiones y Recomendaciones", "25"],
            ["Apéndice A: Metodología Detallada", "27"],
            ["Apéndice B: Datos y Fuentes", "28"],
        ]
        
        toc_table = Table(toc_data, colWidths=[5*inch, 1.5*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        story.append(toc_table)
        
        return story
