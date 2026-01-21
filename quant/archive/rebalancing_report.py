"""
Rebalancing Report Generator (Spanish)
Generates detailed monitoring and rebalancing strategy reports.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                                TableStyle, PageBreak, Image)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import pandas as pd
from typing import Dict, List
import numpy as np


class RebalancingReportGenerator:
    """Generates comprehensive rebalancing strategy reports in Spanish."""
    
    def __init__(self, comparison_df: pd.DataFrame, strategies_results: Dict,
                 target_weights: np.ndarray, tickers: List[str], 
                 horizon_years: int = 1):
        self.comparison_df = comparison_df
        self.strategies = strategies_results
        self.target_weights = target_weights
        self.tickers = tickers
        self.horizon_years = horizon_years
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
        
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#0a5f0a'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#e8f5e9'),
            borderPadding=10
        ))
    
    def generate_report(self, output_path: str, dashboard_path: str) -> None:
        """Generate complete rebalancing strategy report."""
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=0.6*inch, leftMargin=0.6*inch,
                               topMargin=0.6*inch, bottomMargin=0.6*inch)
        
        story = []
        
        # Page 1: Cover and Executive Summary
        story.extend(self._create_cover_page())
        story.append(PageBreak())
        
        # Page 2: Strategy Comparison
        story.extend(self._create_comparison_page())
        story.append(PageBreak())
        
        # Page 3: Recommended Strategy
        story.extend(self._create_recommendation_page())
        story.append(PageBreak())
        
        # Page 4: Implementation Guide
        story.extend(self._create_implementation_page())
        story.append(PageBreak())
        
        # Page 5: Monitoring Framework
        story.extend(self._create_monitoring_page())
        story.append(PageBreak())
        
        # Page 6: Dashboard and Alerts
        story.extend(self._create_dashboard_page(dashboard_path))
        
        doc.build(story)
        print(f"Informe de rebalanceo guardado en: {output_path}")
    
    def _create_cover_page(self) -> list:
        """Create cover page."""
        story = []
        
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("ESTRATEGIA DE MONITOREO", self.styles['CustomTitle']))
        story.append(Paragraph("Y REBALANCEO DE PORTFOLIO", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph(f"Horizonte de Inversión: {self.horizon_years} Año{'s' if self.horizon_years > 1 else ''}", 
                              self.styles['Heading2']))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d de %B de %Y')}", 
                              self.styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("RESUMEN EJECUTIVO", self.styles['SectionHeader']))
        
        summary = f"""
        Este informe presenta un análisis exhaustivo de estrategias de rebalanceo de portfolio 
        para un horizonte de inversión de {self.horizon_years} año{'s' if self.horizon_years > 1 else ''}. 
        Se evaluaron 7 estrategias diferentes: Buy & Hold (sin rebalanceo), rebalanceo calendario 
        (mensual, trimestral, anual) y rebalanceo por umbral (3%, 5%, 10%).
        <br/><br/>
        El análisis considera: (1) retornos totales, (2) costos de transacción, (3) frecuencia de 
        rebalanceo, (4) control de riesgo, y (5) facilidad de implementación. Se proporciona una 
        recomendación específica basada en el perfil de riesgo y los objetivos del inversor.
        """
        story.append(Paragraph(summary, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key findings
        story.append(Paragraph("HALLAZGOS CLAVE", self.styles['SectionHeader']))
        
        # Find best strategy
        best_strategy = self.comparison_df.loc[self.comparison_df['Retorno Neto (%)'].idxmax()]
        
        findings = f"""
        <b>• Mejor Estrategia por Retorno:</b> {best_strategy['Estrategia']} 
        ({best_strategy['Retorno Neto (%)']:.2f}% retorno neto)
        <br/><br/>
        <b>• Rango de Retornos:</b> {self.comparison_df['Retorno Total (%)'].min():.2f}% a 
        {self.comparison_df['Retorno Total (%)'].max():.2f}%
        <br/><br/>
        <b>• Impacto de Costos:</b> Los costos de transacción pueden reducir el retorno entre 
        {self.comparison_df['Costos Transacción ($)'].min():.0f} y 
        {self.comparison_df['Costos Transacción ($)'].max():.0f} dólares
        <br/><br/>
        <b>• Frecuencia Óptima:</b> El análisis sugiere que para un horizonte de {self.horizon_years} 
        año{'s' if self.horizon_years > 1 else ''}, el rebalanceo {'trimestral o por umbral del 5%' if self.horizon_years == 1 else 'semestral o anual'} 
        ofrece el mejor balance entre control de riesgo y costos.
        """
        story.append(Paragraph(findings, self.styles['BodyJustified']))
        
        return story
    
    def _create_comparison_page(self) -> list:
        """Create strategy comparison page."""
        story = []
        
        story.append(Paragraph("COMPARACIÓN DE ESTRATEGIAS", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Comparison table
        story.append(Paragraph("Tabla Comparativa Completa", self.styles['SectionHeader']))
        
        table_data = [['Estrategia', 'Retorno\nTotal (%)', 'Valor\nFinal ($)', 
                      'Num.\nRebal.', 'Costos\nTrans. ($)', 'Retorno\nNeto (%)']]
        
        for _, row in self.comparison_df.iterrows():
            table_data.append([
                row['Estrategia'],
                f"{row['Retorno Total (%)']:.2f}%",
                f"${row['Valor Final ($)']:,.0f}",
                f"{int(row['Num. Rebalanceos'])}",
                f"${row['Costos Transacción ($)']:.0f}",
                f"{row['Retorno Neto (%)']:.2f}%"
            ])
        
        table = Table(table_data, colWidths=[1.2*inch, 0.9*inch, 0.9*inch, 
                                            0.7*inch, 0.9*inch, 0.9*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
        
        # Strategy descriptions
        story.append(Paragraph("Descripción de Estrategias", self.styles['SectionHeader']))
        
        descriptions = """
        <b>1. Buy & Hold:</b> Compra inicial sin rebalanceo. Los pesos derivan naturalmente con 
        los movimientos del mercado. Mínimos costos pero pérdida de control sobre la asignación.
        <br/><br/>
        <b>2. Rebalanceo Mensual:</b> Ajuste a pesos objetivo cada mes. Máximo control pero 
        mayores costos de transacción. Recomendado solo para portfolios muy grandes.
        <br/><br/>
        <b>3. Rebalanceo Trimestral:</b> Ajuste cada 3 meses. Balance entre control y costos. 
        Adecuado para inversores activos con horizonte de 1 año.
        <br/><br/>
        <b>4. Rebalanceo Anual:</b> Ajuste una vez al año. Bajos costos pero menor control. 
        Apropiado para inversores pasivos con horizonte largo.
        <br/><br/>
        <b>5. Umbral 3%:</b> Rebalanceo cuando cualquier activo se desvía >3% del peso objetivo. 
        Alta frecuencia de ajustes, control preciso.
        <br/><br/>
        <b>6. Umbral 5%:</b> Rebalanceo con desviación >5%. Balance óptimo para la mayoría de 
        inversores. Combina control con costos razonables.
        <br/><br/>
        <b>7. Umbral 10%:</b> Rebalanceo con desviación >10%. Baja frecuencia, permite mayor 
        deriva. Adecuado para portfolios pequeños o mercados de baja volatilidad.
        """
        story.append(Paragraph(descriptions, self.styles['BodyJustified']))
        
        return story
    
    def _create_recommendation_page(self) -> list:
        """Create recommendation page."""
        story = []
        
        story.append(Paragraph("RECOMENDACIÓN ESTRATÉGICA", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Determine best strategy based on horizon
        if self.horizon_years == 1:
            recommended = "Trimestral"
            alternative = "Umbral 5%"
        else:
            recommended = "Anual"
            alternative = "Umbral 10%"
        
        # Get recommended strategy data
        rec_data = self.comparison_df[self.comparison_df['Estrategia'] == recommended].iloc[0]
        alt_data = self.comparison_df[self.comparison_df['Estrategia'] == alternative].iloc[0]
        
        story.append(Paragraph(f"Para Horizonte de {self.horizon_years} Año{'s' if self.horizon_years > 1 else ''}", 
                              self.styles['SectionHeader']))
        
        recommendation = f"""
        <b>ESTRATEGIA RECOMENDADA: {recommended.upper()}</b>
        <br/><br/>
        <b>Retorno Esperado:</b> {rec_data['Retorno Neto (%)']:.2f}%<br/>
        <b>Valor Final Proyectado:</b> ${rec_data['Valor Final ($)']:,.0f}<br/>
        <b>Número de Rebalanceos:</b> {int(rec_data['Num. Rebalanceos'])}<br/>
        <b>Costos de Transacción:</b> ${rec_data['Costos Transacción ($)']:.0f}
        """
        story.append(Paragraph(recommendation, self.styles['Recommendation']))
        story.append(Spacer(1, 0.15*inch))
        
        # Justification
        story.append(Paragraph("Justificación", self.styles['SubSection']))
        
        if self.horizon_years == 1:
            justification = f"""
            Para un horizonte de inversión de 1 año, el rebalanceo <b>trimestral</b> es óptimo porque:
            <br/><br/>
            <b>1. Balance Costo-Beneficio:</b> Con 4 rebalanceos al año, se mantiene control sobre 
            la asignación sin incurrir en costos excesivos. Los costos de transacción representan 
            solo {(rec_data['Costos Transacción ($)'] / rec_data['Valor Final ($)']) * 100:.2f}% 
            del valor final del portfolio.
            <br/><br/>
            <b>2. Control de Riesgo:</b> En un horizonte corto, es crucial mantener la exposición 
            al riesgo dentro de los parámetros objetivo. El rebalanceo trimestral previene desviaciones 
            significativas que podrían comprometer los objetivos.
            <br/><br/>
            <b>3. Captura de Oportunidades:</b> Permite aprovechar reversiones a la media cada 3 meses, 
            vendiendo activos que han subido excesivamente y comprando los que han bajado (disciplina 
            de "comprar barato, vender caro").
            <br/><br/>
            <b>4. Simplicidad Operativa:</b> 4 fechas predefinidas al año facilitan la planificación 
            y ejecución. No requiere monitoreo diario como las estrategias por umbral.
            """
        else:
            justification = f"""
            Para un horizonte de inversión de {self.horizon_years} años, el rebalanceo <b>anual</b> 
            es óptimo porque:
            <br/><br/>
            <b>1. Minimización de Costos:</b> Con solo {int(rec_data['Num. Rebalanceos'])} rebalanceos 
            en {self.horizon_years} años, los costos de transacción se minimizan, permitiendo que más 
            capital trabaje para generar retornos.
            <br/><br/>
            <b>2. Horizonte Largo:</b> En inversiones de largo plazo, las fluctuaciones de corto plazo 
            son menos relevantes. El rebalanceo anual es suficiente para mantener la disciplina sin 
            interferir con tendencias de largo plazo.
            <br/><br/>
            <b>3. Eficiencia Fiscal:</b> Menos transacciones significan menos eventos imponibles, 
            optimizando la carga tributaria (especialmente relevante en cuentas no diferidas).
            <br/><br/>
            <b>4. Simplicidad:</b> Una sola fecha al año (ej: fin de año fiscal) facilita la 
            planificación y reduce la carga operativa.
            """
        
        story.append(Paragraph(justification, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Alternative strategy
        story.append(Paragraph("Estrategia Alternativa", self.styles['SubSection']))
        
        alternative_text = f"""
        <b>Alternativa: {alternative}</b>
        <br/><br/>
        Para inversores que prefieren un enfoque más dinámico, la estrategia de <b>{alternative}</b> 
        ofrece ventajas:
        <br/><br/>
        <b>Retorno Esperado:</b> {alt_data['Retorno Neto (%)']:.2f}%<br/>
        <b>Rebalanceos:</b> {int(alt_data['Num. Rebalanceos'])} (basados en condiciones de mercado)<br/>
        <b>Costos:</b> ${alt_data['Costos Transacción ($)']:.0f}
        <br/><br/>
        Esta estrategia rebalancea solo cuando la desviación excede el umbral, adaptándose 
        automáticamente a la volatilidad del mercado. En períodos tranquilos, habrá menos 
        rebalanceos; en períodos volátiles, más ajustes para mantener el riesgo controlado.
        """
        story.append(Paragraph(alternative_text, self.styles['BodyJustified']))
        
        return story
    
    def _create_implementation_page(self) -> list:
        """Create implementation guide page."""
        story = []
        
        story.append(Paragraph("GUÍA DE IMPLEMENTACIÓN", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Target weights
        story.append(Paragraph("Pesos Objetivo del Portfolio", self.styles['SectionHeader']))
        
        weights_data = [['Activo', 'Peso Objetivo (%)', 'Rango Aceptable (%)']]
        for i, ticker in enumerate(self.tickers):
            target = self.target_weights[i] * 100
            lower = max(0, target - 5)
            upper = min(100, target + 5)
            weights_data.append([
                ticker,
                f"{target:.2f}%",
                f"{lower:.2f}% - {upper:.2f}%"
            ])
        
        weights_table = Table(weights_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
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
        
        # Step-by-step process
        story.append(Paragraph("Proceso de Rebalanceo Paso a Paso", self.styles['SectionHeader']))
        
        process = """
        <b>Paso 1: Calcular Pesos Actuales</b><br/>
        Determinar el valor de mercado de cada posición y calcular el peso actual como porcentaje 
        del portfolio total.
        <br/><br/>
        <b>Paso 2: Identificar Desviaciones</b><br/>
        Comparar pesos actuales con pesos objetivo. Calcular la diferencia absoluta para cada activo.
        <br/><br/>
        <b>Paso 3: Determinar Necesidad de Rebalanceo</b><br/>
        • Estrategia Calendario: Verificar si es la fecha programada<br/>
        • Estrategia Umbral: Verificar si alguna desviación excede el umbral
        <br/><br/>
        <b>Paso 4: Calcular Órdenes de Compra/Venta</b><br/>
        Para cada activo, calcular el monto a comprar (si está subponderado) o vender (si está 
        sobreponderado) para volver a los pesos objetivo.
        <br/><br/>
        <b>Paso 5: Ejecutar Transacciones</b><br/>
        Ejecutar órdenes en el mercado. Considerar:<br/>
        • Vender primero los activos sobreponderados<br/>
        • Usar el efectivo para comprar activos subponderados<br/>
        • Minimizar el número de transacciones para reducir costos
        <br/><br/>
        <b>Paso 6: Documentar y Registrar</b><br/>
        Registrar fecha, pesos antes/después, transacciones ejecutadas y costos incurridos.
        """
        story.append(Paragraph(process, self.styles['BodyJustified']))
        
        return story
    
    def _create_monitoring_page(self) -> list:
        """Create monitoring framework page."""
        story = []
        
        story.append(Paragraph("FRAMEWORK DE MONITOREO", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Métricas Clave a Monitorear", self.styles['SectionHeader']))
        
        metrics = """
        <b>1. Desviación de Pesos (Diaria/Semanal)</b><br/>
        Calcular: |Peso Actual - Peso Objetivo| para cada activo<br/>
        Alerta: Si alguna desviación > 5%
        <br/><br/>
        <b>2. Valor Total del Portfolio (Diaria)</b><br/>
        Monitorear evolución del valor total<br/>
        Alerta: Si drawdown > 10% desde máximo histórico
        <br/><br/>
        <b>3. Volatilidad Realizada (Semanal)</b><br/>
        Calcular volatilidad de retornos diarios (rolling 30 días)<br/>
        Alerta: Si volatilidad > 25% anualizada
        <br/><br/>
        <b>4. Sharpe Ratio Móvil (Mensual)</b><br/>
        Calcular Sharpe ratio rolling de 60 días<br/>
        Alerta: Si Sharpe < 1.0
        <br/><br/>
        <b>5. Correlación con Benchmark (Mensual)</b><br/>
        Monitorear correlación con SPY<br/>
        Alerta: Si correlación cambia >0.2 respecto a histórico
        <br/><br/>
        <b>6. Costos Acumulados (Mensual)</b><br/>
        Sumar costos de transacción del período<br/>
        Alerta: Si costos > 0.5% del valor del portfolio
        """
        story.append(Paragraph(metrics, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Alert system
        story.append(Paragraph("Sistema de Alertas", self.styles['SectionHeader']))
        
        alerts = """
        <b>Nivel 1 - Informativo (Verde):</b><br/>
        • Desviación de peso 3-5%<br/>
        • Volatilidad 18-25%<br/>
        • Acción: Monitorear, no requiere acción inmediata
        <br/><br/>
        <b>Nivel 2 - Precaución (Amarillo):</b><br/>
        • Desviación de peso 5-8%<br/>
        • Drawdown 8-12%<br/>
        • Volatilidad > 25%<br/>
        • Acción: Revisar en 1-2 días, considerar rebalanceo anticipado
        <br/><br/>
        <b>Nivel 3 - Acción Requerida (Rojo):</b><br/>
        • Desviación de peso > 8%<br/>
        • Drawdown > 12%<br/>
        • Sharpe ratio < 0.5<br/>
        • Acción: Rebalancear inmediatamente o revisar estrategia
        """
        story.append(Paragraph(alerts, self.styles['BodyJustified']))
        
        return story
    
    def _create_dashboard_page(self, dashboard_path: str) -> list:
        """Create dashboard visualization page."""
        story = []
        
        story.append(Paragraph("DASHBOARD DE MONITOREO", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        intro = """
        El siguiente dashboard proporciona una visualización completa del desempeño de todas las 
        estrategias de rebalanceo evaluadas. Use este dashboard para monitorear la evolución del 
        portfolio y validar la efectividad de la estrategia implementada.
        """
        story.append(Paragraph(intro, self.styles['BodyJustified']))
        story.append(Spacer(1, 0.15*inch))
        
        # Dashboard image
        try:
            img = Image(dashboard_path, width=7*inch, height=5.5*inch)
            story.append(img)
        except:
            story.append(Paragraph("Dashboard no disponible", self.styles['Normal']))
        
        story.append(Spacer(1, 0.15*inch))
        
        # Interpretation guide
        story.append(Paragraph("Guía de Interpretación", self.styles['SubSection']))
        
        interpretation = """
        <b>Panel Superior:</b> Muestra la evolución del valor del portfolio para cada estrategia. 
        Los puntos marcados indican fechas de rebalanceo.
        <br/><br/>
        <b>Retornos y Frecuencia:</b> Compare retornos totales vs. número de rebalanceos para 
        evaluar eficiencia.
        <br/><br/>
        <b>Drawdown:</b> Identifique cuál estrategia ofrece mejor protección en caídas de mercado.
        <br/><br/>
        <b>Sharpe Móvil:</b> Evalúe la consistencia de retornos ajustados por riesgo a lo largo 
        del tiempo.
        <br/><br/>
        <b>Riesgo-Retorno:</b> El gráfico scatter muestra el trade-off entre volatilidad y retorno 
        para cada estrategia.
        """
        story.append(Paragraph(interpretation, self.styles['BodyJustified']))
        
        return story
