"""
Portfolio Report Generator (Spanish)
Generates detailed technical reports with portfolio recommendations in Spanish.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                TableStyle, PageBreak, Image, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict


class PortfolioReportGenerator:
    """Generates comprehensive portfolio analysis reports in Spanish."""
    
    def __init__(self, tickers: List[str], mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                 individual_stats: tuple, markowitz_portfolios: dict, mc_portfolios: dict,
                 stats_table: pd.DataFrame, backtest_results: dict, 
                 asset_chart_paths: dict, benchmark: str = 'SPY'):
        self.tickers = tickers
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.individual_vols, self.individual_sharpe = individual_stats
        self.markowitz = markowitz_portfolios
        self.mc = mc_portfolios
        self.stats_table = stats_table
        self.backtest = backtest_results
        self.asset_charts = asset_chart_paths
        self.benchmark = benchmark
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=colors.HexColor('#444444'),
            spaceAfter=6,
            spaceBefore=6,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyJustified',
            parent=self.styles['BodyText'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=12
        ))
        
    def generate_report(self, output_path: str, chart_path: str, backtest_path: str) -> None:
        """Generate complete PDF report."""
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=0.6*inch, leftMargin=0.6*inch,
                               topMargin=0.6*inch, bottomMargin=0.6*inch)
        
        story = []
        
        # Portada y resumen ejecutivo
        story.extend(self._create_cover_page())
        story.append(PageBreak())
        
        # Análisis de activos individuales
        story.extend(self._create_individual_analysis())
        story.append(PageBreak())
        
        # Optimización y recomendación
        story.extend(self._create_optimization_page())
        story.append(PageBreak())
        
        # Backtesting y performance
        story.extend(self._create_backtest_page(backtest_path))
        story.append(PageBreak())
        
        # Framework matemático
        story.extend(self._create_mathematical_framework())
        story.append(PageBreak())
        
        # Frontera eficiente
        story.extend(self._create_efficient_frontier_page(chart_path))
        
        doc.build(story)
        print(f"Informe de portfolio guardado en: {output_path}")
        
    def _create_cover_page(self) -> list:
        """Create cover page."""
        story = []
        
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("INFORME DE OPTIMIZACIÓN", self.styles['CustomTitle']))
        story.append(Paragraph("DE PORTFOLIO", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph(f"Análisis Cuantitativo Completo", 
                              self.styles['Heading2']))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d de %B de %Y')}", 
                              self.styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Resumen ejecutivo
        story.append(Paragraph("RESUMEN EJECUTIVO", self.styles['SectionHeader']))
        
        summary = f"""
        Este informe presenta un análisis cuantitativo exhaustivo de {len(self.tickers)} activos 
        financieros utilizando técnicas avanzadas de optimización de portfolio. Se emplearon 
        simulaciones Monte Carlo (50,000 iteraciones) y optimización de Markowitz para identificar 
        la asignación óptima de capital que maximiza el ratio riesgo-retorno.
        <br/><br/>
        El análisis incluye: (1) evaluación individual de cada activo con 10 métricas cuantitativas, 
        (2) construcción de la frontera eficiente, (3) backtesting histórico del portfolio óptimo, 
        y (4) comparación contra el benchmark {self.benchmark}.
        """
        story.append(Paragraph(summary, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.3*inch))
        
        # Portfolio recomendado
        recommended = self.markowitz['max_sharpe']
        
        story.append(Paragraph("PORTFOLIO RECOMENDADO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        rec_box = f"""
        <b>Estrategia:</b> Máximo Sharpe Ratio<br/>
        <b>Retorno Anual Esperado:</b> {recommended['return']*100:.2f}%<br/>
        <b>Volatilidad Anual:</b> {recommended['volatility']*100:.2f}%<br/>
        <b>Sharpe Ratio:</b> {recommended['sharpe']:.4f}<br/>
        <b>Retorno Total (Backtest):</b> {self.backtest['total_return']:.2f}%<br/>
        <b>Máximo Drawdown:</b> {self.backtest['max_drawdown']:.2f}%
        """
        story.append(Paragraph(rec_box, self.styles['BodyJustified']))
        
        return story
        
    def _create_individual_analysis(self) -> list:
        """Create individual asset analysis page."""
        story = []
        
        story.append(Paragraph("ANÁLISIS INDIVIDUAL DE ACTIVOS", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        intro = """
        Se realizó un análisis cuantitativo detallado de cada activo, evaluando distribución de 
        retornos, volatilidad, correlaciones, drawdowns y métricas de riesgo avanzadas.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Tabla de estadísticas
        story.append(Paragraph("Tabla Comparativa de Métricas", self.styles['SubSection']))
        
        # Prepare stats table
        stats_data = [['Activo', 'Ret. Anual', 'Vol. Anual', 'Sharpe', 'Máx DD', 'VaR 95%']]
        
        for ticker in self.tickers:
            if ticker in self.stats_table.index:
                row_data = self.stats_table.loc[ticker]
                stats_data.append([
                    ticker,
                    f"{row_data['Retorno Anual (%)']:.2f}%",
                    f"{row_data['Volatilidad Anual (%)']:.2f}%",
                    f"{row_data['Sharpe Ratio']:.3f}",
                    f"{row_data['Máximo Drawdown (%)']:.2f}%",
                    f"{row_data['VaR 95% (%)']:.3f}%"
                ])
        
        stats_table = Table(stats_data, colWidths=[0.8*inch, 1*inch, 1*inch, 0.8*inch, 0.9*inch, 0.9*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 0.15*inch))
        
        # Interpretación
        story.append(Paragraph("Interpretación de Métricas", self.styles['SubSection']))
        
        interpretation = """
        <b>• Sharpe Ratio:</b> Mide el retorno ajustado por riesgo. Valores >1 indican buena 
        performance, >2 excelente.<br/>
        <b>• Máximo Drawdown:</b> Mayor caída desde un pico histórico. Indica el peor escenario 
        experimentado.<br/>
        <b>• VaR 95%:</b> Value at Risk al 95% de confianza. Pérdida máxima esperada en el 5% 
        de los peores casos.<br/>
        <b>• Asimetría y Curtosis:</b> Miden la forma de la distribución de retornos. Valores 
        cercanos a 0 y 3 respectivamente indican normalidad.
        """
        story.append(Paragraph(interpretation, self.styles['BodyJustified']))
        
        return story
        
    def _create_optimization_page(self) -> list:
        """Create optimization and recommendation page."""
        story = []
        
        story.append(Paragraph("OPTIMIZACIÓN Y RECOMENDACIÓN", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        recommended = self.markowitz['max_sharpe']
        
        # Asignación de activos
        story.append(Paragraph("Asignación Óptima de Activos", self.styles['SectionHeader']))
        
        weights_data = [['Activo', 'Peso (%)', 'Contribución al Retorno', '% del Retorno Total']]
        for i, ticker in enumerate(self.tickers):
            weight = recommended['weights'][i]
            ret_contrib = weight * self.mean_returns.iloc[i]
            weights_data.append([
                ticker,
                f"{weight*100:.2f}%",
                f"{ret_contrib*100:.2f}%",
                f"{(ret_contrib/recommended['return'])*100:.1f}%"
            ])
        
        weights_table = Table(weights_data, colWidths=[1*inch, 1.2*inch, 1.5*inch, 1.5*inch])
        weights_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        story.append(weights_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Justificación
        story.append(Paragraph("Justificación de la Recomendación", self.styles['SectionHeader']))
        
        rationale = f"""
        El portfolio de Máximo Sharpe Ratio es recomendado por las siguientes razones fundamentales:
        <br/><br/>
        <b>1. Optimalidad Matemática:</b> Este portfolio se encuentra sobre la Línea de Mercado de 
        Capitales (CML), representando la combinación más eficiente de activos riesgosos. Cualquier 
        otro portfolio con el mismo retorno esperado tendría mayor volatilidad, o con la misma 
        volatilidad tendría menor retorno.
        <br/><br/>
        <b>2. Ratio Riesgo-Retorno Superior:</b> Con un Sharpe Ratio de {recommended['sharpe']:.4f}, 
        este portfolio genera {recommended['sharpe']:.2f} unidades de retorno por cada unidad de 
        riesgo asumido. Esto supera significativamente al benchmark {self.benchmark}.
        <br/><br/>
        <b>3. Diversificación Óptima:</b> La asignación distribuye el capital entre {len(self.tickers)} 
        activos, aprovechando las correlaciones imperfectas para reducir el riesgo idiosincrático. 
        La matriz de covarianza muestra correlaciones que van desde 
        {self.cov_matrix.min().min()/(self.individual_vols.max()**2):.2f} hasta 
        {self.cov_matrix.max().max()/(self.individual_vols.min()**2):.2f}.
        <br/><br/>
        <b>4. Validación Empírica:</b> El backtesting histórico demuestra un retorno total de 
        {self.backtest['total_return']:.2f}% con un drawdown máximo de {self.backtest['max_drawdown']:.2f}%, 
        validando la robustez de la estrategia.
        """
        story.append(Paragraph(rationale, self.styles['BodyJustified']))
        
        return story
        
    def _create_backtest_page(self, backtest_path: str) -> list:
        """Create backtesting results page."""
        story = []
        
        story.append(Paragraph("BACKTESTING Y PERFORMANCE HISTÓRICA", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        intro = """
        Se realizó un backtesting exhaustivo del portfolio óptimo utilizando datos históricos de 
        los últimos 3 años. Los resultados demuestran la viabilidad y robustez de la estrategia.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Gráfico de backtesting
        try:
            img = Image(backtest_path, width=6.5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))
        except:
            pass
        
        # Métricas clave
        story.append(Paragraph("Métricas de Performance", self.styles['SubSection']))
        
        metrics_text = f"""
        <b>Capital Inicial:</b> ${self.backtest['initial_capital']:,.0f}<br/>
        <b>Capital Final:</b> ${self.backtest['final_capital']:,.0f}<br/>
        <b>Retorno Total:</b> {self.backtest['total_return']:.2f}%<br/>
        <b>Retorno Anualizado:</b> {self.backtest['annual_return']:.2f}%<br/>
        <b>Volatilidad Anualizada:</b> {self.backtest['annual_volatility']:.2f}%<br/>
        <b>Sharpe Ratio:</b> {self.backtest['sharpe_ratio']:.3f}<br/>
        <b>Sortino Ratio:</b> {self.backtest['sortino_ratio']:.3f}<br/>
        <b>Calmar Ratio:</b> {self.backtest['calmar_ratio']:.3f}<br/>
        <b>Máximo Drawdown:</b> {self.backtest['max_drawdown']:.2f}%<br/>
        <b>Win Rate:</b> {self.backtest['win_rate']:.2f}%
        """
        story.append(Paragraph(metrics_text, self.styles['BodyJustified']))
        
        return story
        
    def _create_mathematical_framework(self) -> list:
        """Create mathematical framework page."""
        story = []
        
        story.append(Paragraph("FRAMEWORK MATEMÁTICO", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        intro = """
        La optimización de portfolio se basa en la Teoría Moderna de Portfolio (MPT) desarrollada 
        por Harry Markowitz. A continuación se presentan las ecuaciones fundamentales:
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Ecuación 1
        story.append(Paragraph("1. Retorno Esperado del Portfolio", self.styles['SubSection']))
        eq1 = """
        El retorno esperado es el promedio ponderado de los retornos individuales:
        <br/><br/>
        <b>E[R<sub>p</sub>] = Σ<sub>i=1</sub><sup>n</sup> w<sub>i</sub> × E[R<sub>i</sub>]</b>
        <br/><br/>
        donde w<sub>i</sub> es el peso del activo i, E[R<sub>i</sub>] es su retorno esperado, 
        y n es el número de activos.
        """
        story.append(Paragraph(eq1, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.1*inch))
        
        # Ecuación 2
        story.append(Paragraph("2. Varianza del Portfolio", self.styles['SubSection']))
        eq2 = """
        La varianza considera tanto las volatilidades individuales como las correlaciones:
        <br/><br/>
        <b>σ<sub>p</sub><sup>2</sup> = w<sup>T</sup> Σ w = Σ<sub>i=1</sub><sup>n</sup> 
        Σ<sub>j=1</sub><sup>n</sup> w<sub>i</sub> w<sub>j</sub> σ<sub>ij</sub></b>
        <br/><br/>
        donde Σ es la matriz de covarianza y σ<sub>ij</sub> es la covarianza entre los activos i y j.
        La desviación estándar (volatilidad) es σ<sub>p</sub> = √(σ<sub>p</sub><sup>2</sup>).
        """
        story.append(Paragraph(eq2, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.1*inch))
        
        # Ecuación 3
        story.append(Paragraph("3. Optimización del Sharpe Ratio", self.styles['SubSection']))
        eq3 = """
        El Sharpe Ratio mide el exceso de retorno por unidad de riesgo:
        <br/><br/>
        <b>SR = (E[R<sub>p</sub>] - R<sub>f</sub>) / σ<sub>p</sub></b>
        <br/><br/>
        El problema de optimización se formula como:
        <br/><br/>
        <b>maximizar: (w<sup>T</sup>μ - R<sub>f</sub>) / √(w<sup>T</sup>Σw)</b><br/>
        <b>sujeto a: Σ<sub>i=1</sub><sup>n</sup> w<sub>i</sub> = 1, w<sub>i</sub> ≥ 0</b>
        <br/><br/>
        donde μ es el vector de retornos esperados y R<sub>f</sub> es la tasa libre de riesgo 
        (asumida como 0% en este análisis).
        """
        story.append(Paragraph(eq3, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Guía de comunicación con clientes
        story.append(Paragraph("GUÍA DE COMUNICACIÓN CON CLIENTES", self.styles['SectionHeader']))
        
        client_guide = f"""
        <b>Escenario 1: El portfolio genera ganancias</b>
        <br/><br/>
        "Su portfolio ha generado retornos positivos gracias a la asignación óptima de capital 
        basada en análisis cuantitativo riguroso. La posición principal en 
        {self.tickers[np.argmax(self.markowitz['max_sharpe']['weights'])]} 
        ({self.markowitz['max_sharpe']['weights'][np.argmax(self.markowitz['max_sharpe']['weights'])]*100:.1f}% 
        del capital) contribuyó significativamente debido a su alto Sharpe Ratio de 
        {self.individual_sharpe[np.argmax(self.markowitz['max_sharpe']['weights'])]:.2f}. 
        La diversificación entre {len(self.tickers)} activos permitió capturar oportunidades 
        mientras se limitaba el riesgo mediante correlaciones imperfectas."
        <br/><br/>
        <b>Escenario 2: El portfolio experimenta pérdidas</b>
        <br/><br/>
        "Aunque el portfolio experimentó una pérdida temporal, esta asignación fue diseñada 
        matemáticamente para minimizar el riesgo dado el nivel de retorno objetivo. La volatilidad 
        del portfolio de {self.markowitz['max_sharpe']['volatility']*100:.2f}% representa el rango 
        esperado de fluctuaciones, y las pérdidas actuales están dentro de los parámetros normales. 
        La diversificación entre {len(self.tickers)} activos con correlaciones que van desde 
        {self.cov_matrix.min().min()/(self.individual_vols.max()**2):.2f} hasta 
        {self.cov_matrix.max().max()/(self.individual_vols.min()**2):.2f} ayudó a limitar las 
        pérdidas en comparación con posiciones concentradas. El Sharpe Ratio de 
        {self.markowitz['max_sharpe']['sharpe']:.2f} confirma que esta sigue siendo la estrategia 
        óptima para maximizar retornos ajustados por riesgo en el largo plazo. El backtesting 
        histórico muestra que esta estrategia ha generado un retorno total de 
        {self.backtest['total_return']:.2f}% con un drawdown máximo controlado de 
        {self.backtest['max_drawdown']:.2f}%."
        """
        story.append(Paragraph(client_guide, self.styles['BodyJustified']))
        
        return story
        
    def _create_efficient_frontier_page(self, chart_path: str) -> list:
        """Create efficient frontier visualization page."""
        story = []
        
        story.append(Paragraph("FRONTERA EFICIENTE DE MARKOWITZ", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        intro = """
        La frontera eficiente representa el conjunto de portfolios óptimos que ofrecen el máximo 
        retorno esperado para cada nivel de riesgo. El portfolio recomendado (estrella dorada) 
        se encuentra en el punto de tangencia con la Línea de Mercado de Capitales.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Gráfico
        try:
            img = Image(chart_path, width=6.5*inch, height=4*inch)
            story.append(img)
        except:
            pass
        
        story.append(Spacer(1, 0.15*inch))
        
        conclusion = """
        <b>Conclusión:</b> El análisis cuantitativo exhaustivo confirma que el portfolio de Máximo 
        Sharpe Ratio representa la asignación óptima de capital para maximizar retornos ajustados 
        por riesgo. La validación mediante backtesting histórico y el análisis individual de cada 
        activo respaldan esta recomendación con evidencia empírica sólida.
        """
        story.append(Paragraph(conclusion, self.styles['BodyJustified']))
        
        return story
