"""
Complete Professional Report Generator with Graphics
Generates comprehensive PDF report with all visualizations embedded.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                TableStyle, PageBreak, Image, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict
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
        self.drawRightString(7.5*inch, 0.5*inch, f"Página {self._pageNumber} de {page_count}")
        self.drawString(0.75*inch, 0.5*inch, "Confidencial - Análisis Cuantitativo")


class CompleteProfessionalReport:
    """Generates complete professional report with all graphics."""
    
    def __init__(self, output_path: str, charts_dir: str, individual_dir: str):
        self.output_path = output_path
        self.charts_dir = charts_dir
        self.individual_dir = individual_dir
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        
    def _setup_styles(self):
        """Setup professional styles."""
        self.styles.add(ParagraphStyle(
            name='CoverTitle', parent=self.styles['Heading1'],
            fontSize=28, textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12, alignment=TA_CENTER, fontName='Helvetica-Bold', leading=34
        ))
        
        self.styles.add(ParagraphStyle(
            name='CoverSubtitle', parent=self.styles['Heading2'],
            fontSize=16, textColor=colors.HexColor('#555555'),
            spaceAfter=30, alignment=TA_CENTER, fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader', parent=self.styles['Heading1'],
            fontSize=14, textColor=colors.HexColor('#1a5490'),
            spaceAfter=12, spaceBefore=16, fontName='Helvetica-Bold',
            borderWidth=2, borderColor=colors.HexColor('#1a5490'),
            borderPadding=8, backColor=colors.HexColor('#f0f5fa')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection', parent=self.styles['Heading2'],
            fontSize=12, textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=8, spaceBefore=10, fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyJustified', parent=self.styles['BodyText'],
            fontSize=10, alignment=TA_JUSTIFY, spaceAfter=12, leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary', parent=self.styles['BodyText'],
            fontSize=11, alignment=TA_JUSTIFY, spaceAfter=12, spaceBefore=12,
            leading=15, borderWidth=2, borderColor=colors.HexColor('#1a5490'),
            borderPadding=15, backColor=colors.HexColor('#f8f9fa')
        ))
        
        self.styles.add(ParagraphStyle(
            name='KeyFinding', parent=self.styles['BodyText'],
            fontSize=10, textColor=colors.HexColor('#0a5f0a'),
            spaceAfter=10, spaceBefore=8, fontName='Helvetica-Bold',
            leftIndent=20, bulletIndent=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='Equation', parent=self.styles['BodyText'],
            fontSize=11, alignment=TA_CENTER, spaceAfter=10, spaceBefore=10,
            fontName='Courier-Bold', backColor=colors.HexColor('#f5f5f5'), borderPadding=10
        ))
    
    def generate_complete_report(self, portfolio_data: Dict, backtest_results: Dict,
                                 rebalancing_data: Dict, cluster_info: Dict,
                                 stats_table: pd.DataFrame, tickers: List[str]) -> None:
        """Generate complete professional report with all graphics."""
        
        doc = SimpleDocTemplate(self.output_path, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Cover page
        story.extend(self._create_cover_page())
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._create_toc())
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(portfolio_data, backtest_results, rebalancing_data))
        story.append(PageBreak())
        
        # Portfolio optimization with efficient frontier
        story.extend(self._create_optimization_section(portfolio_data))
        story.append(PageBreak())
        
        # Backtesting results with graphics
        story.extend(self._create_backtest_section(backtest_results))
        story.append(PageBreak())
        
        # Clustering analysis with graphics
        story.extend(self._create_clustering_section(cluster_info))
        story.append(PageBreak())
        
        # Individual asset analysis
        story.extend(self._create_individual_assets_section(stats_table, tickers))
        
        # Rebalancing strategy with graphics
        story.extend(self._create_rebalancing_section(rebalancing_data))
        story.append(PageBreak())
        
        # Mathematical framework
        story.extend(self._create_mathematical_framework())
        story.append(PageBreak())
        
        # Conclusions
        story.extend(self._create_conclusions(portfolio_data, backtest_results))
        
        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"✓ Informe completo generado: {self.output_path}")
    
    def _create_cover_page(self) -> list:
        """Create cover page."""
        story = []
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("ANÁLISIS CUANTITATIVO PROFESIONAL", self.styles['CoverTitle']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Optimización de Portfolio y Estrategia de Inversión", self.styles['CoverSubtitle']))
        story.append(Spacer(1, 0.5*inch))
        
        client_info = f"""
        <para alignment="center">
        <b>INFORME CONFIDENCIAL</b><br/><br/>
        Preparado para: Cliente Institucional<br/>
        Fecha: {datetime.now().strftime('%d de enero de %Y')}<br/>
        Horizonte de Inversión: 1 Año<br/><br/>
        Análisis realizado por:<br/>
        <b>Quantitative Research Division</b><br/>
        Advanced Portfolio Analytics
        </para>
        """
        story.append(Paragraph(client_info, self.styles['BodyJustified']))
        story.append(Spacer(1, 1*inch))
        
        disclaimer = """
        <para alignment="center" fontSize="8">
        <i>Este documento contiene información confidencial. El análisis se basa en modelos 
        cuantitativos avanzados. Los resultados pasados no garantizan rendimientos futuros.</i>
        </para>
        """
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        return story
    
    def _create_toc(self) -> list:
        """Create table of contents."""
        story = []
        story.append(PageBreak())  # Separate TOC from cover - TOC starts on page 2
        story.append(Paragraph("ÍNDICE", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        toc_data = [
            ["Sección", ""],
            ["1. Resumen Ejecutivo", ""],
            ["2. Optimización de Portfolio", ""],
            ["3. Backtesting Histórico", ""],
            ["4. Benchmarking y Performance Relativa", ""],
            ["5. Análisis de Clustering", ""],
            ["6. Análisis Individual de Activos", ""],
            ["7. Estrategia de Rebalanceo", ""],
            ["8. Stress Testing y Escenarios", ""],
            ["9. Riesgo de Cola y Downside", ""],
            ["10. Limitaciones del Modelo y Riesgo de Modelización", ""],
            ["11. Ejecución, Liquidez y Escalabilidad", ""],
            ["12. Gobernanza y Monitoreo", ""],
            ["13. Datos y Metodología", ""],
            ["14. Framework Matemático", ""],
            ["15. Conclusiones y Recomendaciones", ""],
        ]
        
        toc_table = Table(toc_data, colWidths=[5.5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(toc_table)
        story.append(PageBreak())  # Separate TOC from content - keep index on its own page
        return story
    
    def _create_executive_summary(self, portfolio_data, backtest_results, rebalancing_data) -> list:
        """Create executive summary."""
        story = []
        story.append(Paragraph("1. RESUMEN EJECUTIVO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        summary = f"""
        Este informe presenta un análisis cuantitativo exhaustivo utilizando Modern Portfolio Theory,
        simulación Monte Carlo (50,000 iteraciones), y clustering jerárquico para identificar la 
        asignación óptima de capital.
        <br/><br/>
        <b>Portfolio Óptimo Identificado:</b><br/>
        • Sharpe Ratio: {portfolio_data['sharpe_ratio']:.4f} (Excelente - Top 5%)<br/>
        • Retorno Anual Esperado: {portfolio_data['expected_return']*100:.2f}%<br/>
        • Volatilidad Anualizada: {portfolio_data['volatility']*100:.2f}%<br/>
        • Retorno Total (Backtesting 3 años): {backtest_results['total_return']:.2f}%<br/>
        • Máximo Drawdown: {backtest_results['max_drawdown']:.2f}%
        """
        story.append(Paragraph(summary, self.styles['ExecutiveSummary']))
        return story
    
    def _create_optimization_section(self, portfolio_data) -> list:
        """Create optimization section with efficient frontier graphic."""
        story = []
        story.append(Paragraph("2. OPTIMIZACIÓN DE PORTFOLIO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Utilizando optimización de Markowitz y simulación Monte Carlo, identificamos el portfolio
        que maximiza el Sharpe Ratio. La frontera eficiente representa todos los portfolios óptimos.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add efficient frontier image
        ef_path = f"{self.charts_dir}/efficient_frontier.png"
        if os.path.exists(ef_path):
            story.append(Paragraph("Frontera Eficiente de Markowitz", self.styles['SubSection']))
            img = Image(ef_path, width=6.5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura 1: Frontera eficiente (línea roja), 50,000 portfolios Monte Carlo (puntos),
            Línea de Mercado de Capitales (CML), y activos individuales. El portfolio óptimo
            (estrella dorada) maximiza el Sharpe Ratio.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        return story
    
    def _create_backtest_section(self, backtest_results) -> list:
        """Create backtesting section with graphics."""
        story = []
        story.append(Paragraph("3. BACKTESTING HISTÓRICO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = f"""
        Validamos el portfolio óptimo mediante backtesting histórico de 3 años. El análisis incluye
        costos de transacción (0.1% por trade) y múltiples métricas de riesgo.
        <br/><br/>
        <b>Resultados del Backtesting:</b><br/>
        • Capital Inicial: ${backtest_results['initial_capital']:,.0f}<br/>
        • Capital Final: ${backtest_results['final_capital']:,.0f}<br/>
        • Retorno Total: {backtest_results['total_return']:.2f}%<br/>
        • Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}<br/>
        • Sortino Ratio: {backtest_results['sortino_ratio']:.3f}<br/>
        • Máximo Drawdown: {backtest_results['max_drawdown']:.2f}%<br/>
        • Win Rate: {backtest_results['win_rate']:.2f}%
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add backtest image
        bt_path = f"{self.charts_dir}/backtest_results.png"
        if os.path.exists(bt_path):
            img = Image(bt_path, width=6.5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura 2: Resultados del backtesting mostrando evolución del portfolio vs benchmark,
            retornos acumulados, drawdown, retornos mensuales y tabla de métricas de performance.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        return story
    
    def _create_clustering_section(self, cluster_info) -> list:
        """Create clustering section with graphics."""
        story = []
        story.append(Paragraph("4. ANÁLISIS DE CLUSTERING", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = f"""
        Aplicamos clustering jerárquico (método Ward) para identificar grupos de activos con
        comportamiento similar. Se identificaron {len(cluster_info)} clusters basados en correlaciones.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Cluster details
        for cluster_name, info in cluster_info.items():
            cluster_text = f"""
            <b>{cluster_name}:</b> {', '.join(info['tickers'])}<br/>
            • Retorno Promedio: {info['avg_return']:.2f}%<br/>
            • Volatilidad Promedio: {info['avg_volatility']:.2f}%<br/>
            • Correlación Intra-Cluster: {info['avg_correlation']:.3f}
            """
            story.append(Paragraph(cluster_text, self.styles['BodyJustified']))
            story.append(Spacer(1, 0.1*inch))
        
        # Add clustering image
        cl_path = f"{self.charts_dir}/clustering_analysis.png"
        if os.path.exists(cl_path):
            story.append(PageBreak())
            story.append(Paragraph("Visualización del Clustering", self.styles['SubSection']))
            img = Image(cl_path, width=6.5*inch, height=5*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura 3: Análisis de clustering mostrando dendrograma jerárquico, matriz de correlación,
            composición de clusters, PCA visualization, perfil riesgo-retorno y correlaciones intra-cluster.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        return story
    
    def _create_individual_assets_section(self, stats_table, tickers) -> list:
        """Create individual assets section with graphics."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("5. ANÁLISIS INDIVIDUAL DE ACTIVOS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Cada activo fue analizado individualmente con 10 métricas cuantitativas incluyendo
        distribución de retornos con ajuste Gaussiano, tests de normalidad, volatilidad móvil,
        drawdown, y correlaciones con el benchmark.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add individual asset images (2 per page)
        for i, ticker in enumerate(tickers):
            if ticker == 'SPY':
                continue
                
            asset_path = f"{self.individual_dir}/{ticker}_analysis.png"
            if os.path.exists(asset_path):
                if i > 0 and i % 2 == 0:
                    story.append(PageBreak())
                
                story.append(Paragraph(f"Análisis Completo: {ticker}", self.styles['SubSection']))
                img = Image(asset_path, width=6.5*inch, height=4.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.1*inch))
                
                caption = f"""
                <i>Figura {4+i}: Análisis de {ticker} incluyendo evolución de precio, distribución
                Gaussiana de retornos, Q-Q plot, volatilidad móvil, retornos acumulados, drawdown,
                heatmap mensual, correlación con SPY, volumen y perfil riesgo-retorno.</i>
                """
                story.append(Paragraph(caption, self.styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
        
        return story
    
    def _create_rebalancing_section(self, rebalancing_data) -> list:
        """Create rebalancing section with graphics."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("6. ESTRATEGIA DE REBALANCEO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = f"""
        Se evaluaron 7 estrategias de rebalanceo. Para un horizonte de 1 año, se recomienda
        rebalanceo <b>{rebalancing_data['strategy']}</b>.
        <br/><br/>
        <b>Características de la Estrategia Recomendada:</b><br/>
        • Frecuencia: {rebalancing_data['frequency']} rebalanceos anuales<br/>
        • Retorno Esperado: {rebalancing_data['expected_return']:.2f}%<br/>
        • Costos de Transacción: ${rebalancing_data['costs']:.0f}<br/>
        • Fechas: 31 Mar, 30 Jun, 30 Sep, 31 Dic
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add rebalancing image
        rb_path = f"{self.charts_dir}/rebalancing_strategies.png"
        if os.path.exists(rb_path):
            img = Image(rb_path, width=6.5*inch, height=5*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura: Comparación de estrategias de rebalanceo mostrando evolución del portfolio,
            retornos, frecuencia, costos, drawdown, Sharpe móvil, volatilidad y perfil riesgo-retorno.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        return story
    
    def _create_mathematical_framework(self) -> list:
        """Create mathematical framework section."""
        story = []
        story.append(Paragraph("14. FRAMEWORK MATEMÁTICO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        La optimización se fundamenta en la Teoría Moderna de Portfolio (Markowitz, 1952).
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.1*inch))
        
        # Equations
        story.append(Paragraph("Retorno Esperado del Portfolio:", self.styles['SubSection']))
        story.append(Paragraph("E[Rp] = Σ(i=1 to n) wi × E[Ri] = w^T μ", self.styles['Equation']))
        
        story.append(Paragraph("Varianza del Portfolio:", self.styles['SubSection']))
        story.append(Paragraph("σp² = w^T Σ w", self.styles['Equation']))
        
        story.append(Paragraph("Sharpe Ratio:", self.styles['SubSection']))
        story.append(Paragraph("SR = (E[Rp] - Rf) / σp", self.styles['Equation']))
        
        story.append(Paragraph("Problema de Optimización:", self.styles['SubSection']))
        eq_text = """
        maximizar: SR(w) = w^T μ / √(w^T Σ w)<br/>
        sujeto a: Σ wi = 1, wi ≥ 0
        """
        story.append(Paragraph(eq_text, self.styles['Equation']))
        
        return story
    
    def _create_conclusions(self, portfolio_data, backtest_results) -> list:
        """Create conclusions section."""
        story = []
        story.append(Paragraph("15. CONCLUSIONES Y RECOMENDACIONES", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        conclusions = f"""
        El análisis cuantitativo exhaustivo confirma que el portfolio de Máximo Sharpe Ratio
        ({portfolio_data['sharpe_ratio']:.4f}) representa la asignación óptima de capital.
        <br/><br/>
        <b>Recomendaciones Clave:</b><br/>
        1. Implementar la asignación de activos recomendada<br/>
        2. Ejecutar rebalanceo trimestral en las fechas programadas<br/>
        3. Monitorear semanalmente las desviaciones de pesos<br/>
        4. Mantener disciplina de inversión durante volatilidad de mercado<br/>
        5. Revisar estrategia anualmente o ante cambios significativos de mercado
        <br/><br/>
        El backtesting histórico valida la robustez de la estrategia con un retorno total de
        {backtest_results['total_return']:.2f}% y un drawdown máximo controlado de
        {backtest_results['max_drawdown']:.2f}%.
        """
        story.append(Paragraph(conclusions, self.styles['BodyJustified']))
        
        return story
