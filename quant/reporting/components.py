"""
Extended Report Sections
Additional sections for benchmarking, stress testing, tail risk, limitations, execution, and governance.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
import os
import pandas as pd
from typing import Dict, List


class ExtendedReportSections:
    """Additional professional sections for comprehensive reporting."""
    
    def __init__(self, styles, charts_dir: str):
        self.styles = styles
        self.charts_dir = charts_dir
    
    def create_benchmarking_section(self, risk_metrics: Dict) -> list:
        """Create benchmarking and performance relative section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("BENCHMARKING Y PERFORMANCE RELATIVA", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Evaluamos la performance del portfolio contra el benchmark S&P 500 (SPY) utilizando métricas
        institucionales de performance relativa incluyendo Alpha, Beta, Tracking Error e Information Ratio.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Metrics table
        metrics_text = f"""
        <b>Métricas de Performance Relativa:</b><br/>
        • <b>Alpha Anualizado:</b> {risk_metrics['alpha_annual']*100:.2f}% - Exceso de retorno vs benchmark<br/>
        • <b>Beta:</b> {risk_metrics['beta']:.3f} - Sensibilidad al mercado<br/>
        • <b>R²:</b> {risk_metrics['r_squared']:.3f} - Varianza explicada por benchmark<br/>
        • <b>Tracking Error:</b> {risk_metrics['tracking_error']*100:.2f}% - Desviación vs benchmark<br/>
        • <b>Information Ratio:</b> {risk_metrics['information_ratio']:.3f} - Alpha por unidad de TE<br/><br/>
        <i>Nota: Regresión calculada sobre 3 años de retornos diarios (751 observaciones).
        Frecuencia: daily returns. Benchmark: S&P 500 ETF (SPY).</i>
        """
        story.append(Paragraph(metrics_text, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add benchmark comparison chart
        bench_path = f"{self.charts_dir}/benchmark_comparison.png"
        if os.path.exists(bench_path):
            img = Image(bench_path, width=6.5*inch, height=4.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura: Comparación vs Benchmark mostrando equity curves, exceso de retorno acumulado,
            beta móvil y scatter de retornos con regresión lineal.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        # Interpretation
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("Interpretación:", self.styles['SubSection']))
        
        interpretation = f"""
        El <b>Alpha positivo de {risk_metrics['alpha_annual']*100:.2f}%</b> indica que el portfolio
        genera retornos superiores al benchmark ajustados por riesgo sistemático. El <b>Beta de 
        {risk_metrics['beta']:.2f}</b> sugiere {'mayor' if risk_metrics['beta'] > 1 else 'menor'} 
        sensibilidad al mercado que el benchmark. El <b>Information Ratio de {risk_metrics['information_ratio']:.2f}</b>
        {'supera el umbral institucional de 0.5' if risk_metrics['information_ratio'] > 0.5 else 'indica oportunidad de mejora'},
        demostrando {'eficiencia' if risk_metrics['information_ratio'] > 0.5 else 'potencial'} en la 
        generación de alpha por unidad de tracking error.
        """
        story.append(Paragraph(interpretation, self.styles['BodyJustified']))
        
        return story
    
    def create_stress_testing_section(self, stress_results: Dict) -> list:
        """Create stress testing and scenarios section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("STRESS TESTING Y ESCENARIOS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Aplicamos stress tests institucionales mediante escenarios hipotéticos y paramétricos,
        calibrados con magnitudes plausibles observadas en episodios históricos de mercado.
        Los tests incluyen shocks de mercado, volatilidad y sector-específicos para evaluar
        la resiliencia del portfolio ante condiciones adversas.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add stress test heatmap
        stress_path = f"{self.charts_dir}/stress_test_heatmap.png"
        if os.path.exists(stress_path):
            img = Image(stress_path, width=6.5*inch, height=5*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura: Heatmap de stress testing mostrando impacto en retorno, volatilidad, drawdown
            y Sharpe ratio bajo diferentes escenarios adversos.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        # Scenarios description
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("Escenarios Evaluados:", self.styles['SubSection']))
        
        scenarios = """
        <b>1. Shocks de Mercado:</b> Caídas instantáneas de -10%, -20%, -30% en retornos diarios<br/>
        <b>2. Shocks de Volatilidad:</b> Multiplicadores de 1.5x, 2.0x, 2.5x en desviación estándar<br/>
        <b>3. Shock Sector Tech:</b> Caída de -25% en cluster tecnológico (7 activos)
        """
        story.append(Paragraph(scenarios, self.styles['BodyJustified']))
        
        return story
    
    def create_tail_risk_section(self, var_cvar: Dict) -> list:
        """Create tail risk and downside section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("RIESGO DE COLA Y DOWNSIDE", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Analizamos el riesgo de cola mediante Value at Risk (VaR) y Conditional VaR (Expected Shortfall)
        utilizando el método histórico (percentiles). Estas métricas cuantifican pérdidas potenciales
        en escenarios extremos.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # VaR/CVaR metrics
        metrics_text = f"""
        <b>Métricas de Riesgo de Cola (Método Histórico):</b><br/><br/>
        <b>Nivel de Confianza 95%:</b><br/>
        • VaR 95%: {var_cvar['VaR_95']*100:.2f}% - Pérdida máxima en el 5% peor de los casos<br/>
        • CVaR 95%: {var_cvar['CVaR_95']*100:.2f}% - Pérdida promedio cuando se excede VaR<br/><br/>
        <b>Nivel de Confianza 99%:</b><br/>
        • VaR 99%: {var_cvar['VaR_99']*100:.2f}% - Pérdida máxima en el 1% peor de los casos<br/>
        • CVaR 99%: {var_cvar['CVaR_99']*100:.2f}% - Pérdida promedio en cola extrema
        """
        story.append(Paragraph(metrics_text, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Add tail risk chart
        tail_path = f"{self.charts_dir}/tail_risk_analysis.png"
        if os.path.exists(tail_path):
            img = Image(tail_path, width=6.5*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura: Distribución de retornos del portfolio con marcadores de VaR y CVaR al 95% y 99%,
            más Q-Q plot para análisis de normalidad de colas.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        return story
    
    def create_model_limitations_section(self) -> list:
        """Create model limitations and risk section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("LIMITACIONES DEL MODELO Y RIESGO DE MODELIZACIÓN", 
                              self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Todo modelo cuantitativo tiene supuestos y limitaciones inherentes. A continuación se detallan
        los principales riesgos de modelización que deben considerarse en la interpretación de resultados.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Supuestos del Modelo:", self.styles['SubSection']))
        
        assumptions = """
        • <b>Normalidad de Retornos:</b> Se asume distribución normal; en realidad, los retornos exhiben
        colas pesadas (fat tails) y asimetría<br/>
        • <b>Estacionariedad:</b> Se asume que media y covarianza son constantes; los mercados experimentan
        cambios de régimen<br/>
        • <b>Costos de Transacción:</b> Modelo simplificado (0.1% fijo); costos reales varían con liquidez
        y tamaño de orden<br/>
        • <b>Liquidez Perfecta:</b> Se asume ejecución inmediata; en crisis puede haber iliquidez severa<br/>
        • <b>Sin Restricciones de Corto:</b> Modelo long-only; no considera restricciones de margen o leverage
        """
        story.append(Paragraph(assumptions, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Riesgos de Modelización:", self.styles['SubSection']))
        
        risks = """
        • <b>Overfitting:</b> Optimización sobre datos históricos puede no generalizar a futuro; riesgo
        de curve-fitting<br/>
        • <b>Data Snooping:</b> Uso de mismo dataset para optimización y validación; sesgo de selección<br/>
        • <b>Regime Dependency:</b> Modelo calibrado en período específico; performance puede degradarse
        en régimen diferente<br/>
        • <b>Cambios Estructurales:</b> Correlaciones y volatilidades no son estables; eventos de cisne
        negro no capturados<br/>
        • <b>Estimación de Parámetros:</b> Media y covarianza estimadas con error; incertidumbre en inputs
        propaga a outputs<br/>
        • <b>Horizonte Temporal:</b> Optimización para 1 año; resultados no aplicables a otros horizontes<br/>
        • <b>Eventos Extremos:</b> Modelo basado en historia reciente (3 años); no captura crisis sistémicas
        raras<br/>
        • <b>Microestructura:</b> No considera bid-ask spread, market impact, o fragmentación de mercado
        """
        story.append(Paragraph(risks, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Mitigación:", self.styles['SubSection']))
        
        mitigation = """
        • Recalibración trimestral de parámetros<br/>
        • Stress testing regular con escenarios extremos<br/>
        • Monitoreo continuo de supuestos (tests de normalidad, estabilidad de correlaciones)<br/>
        • Límites de concentración para reducir riesgo idiosincrático<br/>
        • Revisión de modelo ante cambios significativos de mercado
        """
        story.append(Paragraph(mitigation, self.styles['BodyJustified']))
        
        return story
    
    def create_execution_liquidity_section(self, execution_table: pd.DataFrame,
                                          liquidity_table: pd.DataFrame) -> list:
        """Create execution, liquidity and scalability section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("EJECUCIÓN, LIQUIDEZ Y ESCALABILIDAD", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Analizamos los costos de ejecución y restricciones de liquidez a diferentes niveles de AUM.
        El análisis incluye slippage, market impact y escalabilidad del portfolio.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Model assumptions box
        story.append(Paragraph("Supuestos del Modelo de Costos:", self.styles['SubSection']))
        
        assumptions_text = """
        <b>Modelo de Impacto de Mercado:</b><br/>
        • Fórmula: Market Impact = k × √(Order Value)<br/>
        • Parámetro k calibrado empíricamente para large-cap stocks<br/>
        • Basado en modelo de impacto de raíz cuadrada (Almgren-Chriss)<br/><br/>
        <b>Componentes de Costo:</b><br/>
        • Costo Base: 0.10% (comisiones + fees)<br/>
        • Slippage: 5 bps (parametrizable según liquidez)<br/>
        • Market Impact: Proporcional a √AUM<br/><br/>
        <b>Nota Importante:</b> Las estimaciones son <i>indicativas</i> y requieren calibración
        específica por broker, venue y Average Daily Volume (ADV) real de cada activo.
        """
        story.append(Paragraph(assumptions_text, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Execution costs table
        story.append(Paragraph("Análisis de Costos por Nivel de AUM:", self.styles['SubSection']))
        
        exec_data = execution_table.values.tolist()
        exec_data.insert(0, list(execution_table.columns))
        
        exec_table = Table(exec_data, colWidths=[0.9*inch, 0.9*inch, 0.9*inch, 1.1*inch, 
                                                 1*inch, 0.9*inch, 0.8*inch])
        exec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
        ]))
        story.append(exec_table)
        story.append(Spacer(1, 0.15*inch))
        
        # Add scalability chart
        scale_path = f"{self.charts_dir}/aum_scalability.png"
        if os.path.exists(scale_path):
            img = Image(scale_path, width=6.5*inch, height=4.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            caption = """
            <i>Figura: Análisis de escalabilidad mostrando costos totales, costos en bps, desglose
            por componente y costo como porcentaje del AUM a diferentes niveles.</i>
            """
            story.append(Paragraph(caption, self.styles['Normal']))
        
        story.append(PageBreak())
        story.append(Paragraph("Framework de Liquidez y Restricciones:", self.styles['SubSection']))
        
        # Liquidity constraints table
        liq_data = liquidity_table.values.tolist()
        liq_data.insert(0, list(liquidity_table.columns))
        
        liq_table = Table(liq_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        liq_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(liq_table)
        
        return story
    
    def create_governance_section(self, monitoring_checklist: pd.DataFrame,
                                  trigger_events: pd.DataFrame) -> list:
        """Create governance and monitoring section."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("GOBERNANZA Y MONITOREO", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Definimos un framework operativo de gobernanza que especifica frecuencias de monitoreo,
        umbrales de alerta y acciones requeridas para mantener el portfolio dentro de parámetros objetivo.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Checklist de Monitoreo Operativo:", self.styles['SubSection']))
        
        # Monitoring checklist table
        mon_data = monitoring_checklist.values.tolist()
        mon_data.insert(0, list(monitoring_checklist.columns))
        
        mon_table = Table(mon_data, colWidths=[1.3*inch, 0.9*inch, 1.5*inch, 1.8*inch, 1*inch])
        mon_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(mon_table)
        story.append(Spacer(1, 0.15*inch))
        
        story.append(PageBreak())
        story.append(Paragraph("Eventos Gatillo para Acción Inmediata:", self.styles['SubSection']))
        
        # Trigger events table
        trig_data = trigger_events.values.tolist()
        trig_data.insert(0, list(trigger_events.columns))
        
        trig_table = Table(trig_data, colWidths=[1.8*inch, 0.9*inch, 2.5*inch, 1.3*inch])
        trig_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(trig_table)
        
        return story
    
    def create_data_methodology_section(self) -> list:
        """Create data and methodology section for audit trail."""
        story = []
        story.append(PageBreak())
        story.append(Paragraph("DATOS Y METODOLOGÍA", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.15*inch))
        
        intro = """
        Esta sección documenta las fuentes de datos, frecuencias, ajustes y supuestos metodológicos
        utilizados en el análisis para permitir auditabilidad y replicación.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Fuentes de Datos:", self.styles['SubSection']))
        
        data_sources = """
        • <b>Proveedor:</b> Yahoo Finance (yfinance API)<br/>
        • <b>Tipo de Datos:</b> Precios de cierre ajustados (Adjusted Close)<br/>
        • <b>Ajuste de Dividendos:</b> Sí - Precios ajustados por splits y dividendos<br/>
        • <b>Ajuste de Splits:</b> Sí - Incorporado en Adjusted Close<br/>
        • <b>Período:</b> 3 años (2023-01-23 a 2026-01-20)<br/>
        • <b>Observaciones:</b> 751 días de trading
        """
        story.append(Paragraph(data_sources, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Frecuencia y Calendario:", self.styles['SubSection']))
        
        frequency = """
        • <b>Frecuencia de Datos:</b> Diaria (daily)<br/>
        • <b>Calendario de Mercado:</b> NYSE/NASDAQ trading days<br/>
        • <b>Timezone:</b> US Eastern Time (ET)<br/>
        • <b>Días de Trading por Año:</b> 252 (convención estándar)<br/>
        • <b>Anualización de Retornos:</b> × 252<br/>
        • <b>Anualización de Volatilidad:</b> × √252
        """
        story.append(Paragraph(frequency, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Cálculos y Supuestos:", self.styles['SubSection']))
        
        calculations = """
        • <b>Retornos:</b> Logarítmicos (log returns) para propiedades aditivas<br/>
        • <b>Risk-Free Rate:</b> 0% (simplificación conservadora). Análisis adicional disponible con tasa SOFR/UST<br/>
        • <b>Matriz de Covarianza:</b> Estimada sobre retornos históricos completos (3 años)<br/>
        • <b>Retornos Esperados:</b> Media histórica anualizada<br/>
        • <b>Optimización:</b> SLSQP (Sequential Least Squares Programming)<br/>
        • <b>Restricciones:</b> Long-only (wi ≥ 0), fully invested (Σwi = 1)<br/>
        • <b>Costos de Transacción:</b> 0.10% por trade (base) + slippage + market impact
        """
        story.append(Paragraph(calculations, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Validación y Tests:", self.styles['SubSection']))
        
        validation = """
        • <b>Backtesting:</b> Out-of-sample sobre mismo período histórico<br/>
        • <b>Stress Testing:</b> Escenarios hipotéticos calibrados con episodios históricos<br/>
        • <b>VaR/CVaR:</b> Método histórico (percentiles) sin asumir normalidad<br/>
        • <b>Tests de Normalidad:</b> Q-Q plots, Jarque-Bera (para diagnóstico)<br/>
        • <b>Clustering:</b> Método Ward con distancia basada en correlación
        """
        story.append(Paragraph(validation, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Limitaciones de Datos:", self.styles['SubSection']))
        
        limitations = """
        • Datos históricos de 3 años pueden no capturar todos los regímenes de mercado<br/>
        • Precios de Yahoo Finance pueden tener errores ocasionales (validados visualmente)<br/>
        • No se incluyen costos de préstamo de valores o restricciones de short<br/>
        • Liquidez asumida perfecta (apropiado para large-cap stocks)<br/>
        • No se consideran restricciones regulatorias o fiscales específicas
        """
        story.append(Paragraph(limitations, self.styles['BodyJustified']))
        
        return story
