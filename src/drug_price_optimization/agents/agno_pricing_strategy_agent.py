import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
from agno.agent import Agent
from agno.tools import Toolkit
import warnings
warnings.filterwarnings('ignore')

class PricingStrategyToolkit(Toolkit):
    """Toolkit for pricing strategy analysis and recommendation generation"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="pricing_strategy_toolkit",
            tools=[
                self.determine_pricing_strategy,
                self.calculate_recommended_price,
                self.generate_rationale,
                self.assess_confidence,
                self.determine_implementation_timeline,
                self.generate_monitoring_plan,
                self.identify_key_risks,
                self.suggest_risk_mitigation,
                self.generate_portfolio_insights
            ],
            **kwargs
        )
        
        self.strategy_types = {
            'aggressive': {
                'description': 'Price below market average to gain market share',
                'target_position': 'below_median',
                'risk_tolerance': 'high',
                'margin_sacrifice': 0.05  # Willing to sacrifice 5% margin
            },
            'balanced': {
                'description': 'Optimize margin while staying competitive',
                'target_position': 'at_median',
                'risk_tolerance': 'medium',
                'margin_sacrifice': 0.02  # Willing to sacrifice 2% margin
            },
            'conservative': {
                'description': 'Protect revenue with minimal price changes',
                'target_position': 'above_median',
                'risk_tolerance': 'low',
                'margin_sacrifice': 0.0  # No margin sacrifice
            }
        }
    
    def determine_pricing_strategy(self, market_intel: Dict, profitability: Dict) -> str:
        """
        Determine the optimal pricing strategy based on market conditions and profitability
        """
        # Extract key metrics
        competition_level = market_intel['market_intelligence']['competition_level']
        market_volatility = market_intel['market_intelligence']['market_volatility']
        price_position = market_intel['market_intelligence']['price_position_vs_median']
        risk_level = profitability['pricing_optimization']['risk_level']
        expected_margin = profitability['pricing_optimization']['expected_margin']
        regulatory_issues = len(profitability['constraint_analysis'].get('compliance_issues', []))
        
        # Decision matrix scoring
        strategy_scores = {'aggressive': 0, 'balanced': 0, 'conservative': 0}
        
        # Competition factor
        if competition_level == 'high':
            strategy_scores['aggressive'] += 3
            strategy_scores['balanced'] += 1
        elif competition_level == 'medium':
            strategy_scores['balanced'] += 3
            strategy_scores['conservative'] += 1
        else:  # low competition
            strategy_scores['conservative'] += 3
            strategy_scores['balanced'] += 1
        
        # Risk factor
        if risk_level == 'high':
            strategy_scores['conservative'] += 2
            strategy_scores['balanced'] += 1
        elif risk_level == 'medium':
            strategy_scores['balanced'] += 2
        else:  # low risk
            strategy_scores['aggressive'] += 2
        
        # Market position factor
        if price_position > 0.2:  # significantly above median
            strategy_scores['aggressive'] += 2
        elif price_position < -0.2:  # significantly below median
            strategy_scores['conservative'] += 2
        else:  # near median
            strategy_scores['balanced'] += 2
        
        # Margin factor
        if expected_margin > 0.4:  # high margin
            strategy_scores['aggressive'] += 1  # Can afford to be aggressive
        elif expected_margin < 0.2:  # low margin
            strategy_scores['conservative'] += 2
        
        # Regulatory constraints factor
        if regulatory_issues > 0:
            strategy_scores['conservative'] += 2
            strategy_scores['balanced'] += 1
        
        # Return strategy with highest score
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def calculate_recommended_price(self, strategy: str, market_intel: Dict, profitability: Dict) -> float:
        """
        Calculate the recommended price based on the chosen strategy
        """
        optimal_price = profitability['pricing_optimization']['optimal_price']
        feasible_min = profitability['pricing_optimization']['feasible_range'][0]
        feasible_max = profitability['pricing_optimization']['feasible_range'][1]
        market_median = market_intel['market_intelligence']['market_median']
        
        if strategy == 'aggressive':
            # Price at lower end of feasible range or slightly below market median
            target_price = min(feasible_min + (feasible_max - feasible_min) * 0.2, market_median * 0.95)
            return max(target_price, feasible_min)  # Ensure within feasible range
            
        elif strategy == 'balanced':
            # Use optimal price or close to market median
            target_price = min(optimal_price, market_median * 1.02)
            return np.clip(target_price, feasible_min, feasible_max)
            
        else:  # conservative
            # Price at higher end of feasible range
            target_price = feasible_max * 0.95
            return min(target_price, feasible_max)
    
    def generate_rationale(self, strategy: str, market_intel: Dict, profitability: Dict) -> str:
        """
        Generate human-readable rationale for the pricing strategy
        """
        competition = market_intel['market_intelligence']['competition_level']
        risk = profitability['pricing_optimization']['risk_level']
        margin = profitability['pricing_optimization']['expected_margin']
        compliance_issues = profitability['constraint_analysis'].get('compliance_issues', [])
        
        rationale_parts = []
        
        # Base strategy explanation
        rationale_parts.append(f"{strategy.title()} pricing strategy selected")
        
        # Market factors
        if competition == 'high':
            rationale_parts.append("high market competition requires competitive positioning")
        elif competition == 'low':
            rationale_parts.append("limited competition allows for premium pricing")
        
        # Risk factors
        if risk == 'high':
            rationale_parts.append("high risk profile necessitates conservative approach")
        elif risk == 'low':
            rationale_parts.append("low risk enables more aggressive pricing")
        
        # Profitability factors
        if margin > 0.35:
            rationale_parts.append("strong margins provide pricing flexibility")
        elif margin < 0.2:
            rationale_parts.append("tight margins require careful price optimization")
        
        # Regulatory factors
        if compliance_issues:
            rationale_parts.append("regulatory constraints limit pricing options")
        
        return "; ".join(rationale_parts) + "."
    
    def assess_confidence(self, market_intel: Dict, profitability: Dict) -> float:
        """
        Calculate confidence score for the recommendation (0.0 to 1.0)
        """
        confidence_factors = []
        
        # Market data quality
        volatility = market_intel['market_intelligence']['market_volatility']
        if volatility < 0.01:
            confidence_factors.append(0.9)  # Low volatility = high confidence
        elif volatility < 0.05:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Regulatory clarity
        compliance_issues = len(profitability['constraint_analysis'].get('compliance_issues', []))
        if compliance_issues == 0:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Competitive intelligence
        competitor_count = market_intel['market_intelligence']['total_competitors']
        if competitor_count > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)  # Less confidence with no competitor data
        
        return np.mean(confidence_factors)
    
    def determine_implementation_timeline(self, strategy: str, market_intel: Dict, profitability: Dict) -> str:
        """
        Determine when to implement the pricing change
        """
        risk_level = profitability['pricing_optimization']['risk_level']
        compliance_issues = len(profitability['constraint_analysis'].get('compliance_issues', []))
        
        if compliance_issues > 0:
            return "pending_regulatory_review"
        elif risk_level == 'high':
            return "phased_rollout"
        elif strategy == 'aggressive':
            return "immediate"
        else:
            return "next_pricing_cycle"
    
    def generate_monitoring_plan(self, strategy: str, market_intel: Dict) -> Dict:
        """
        Generate monitoring recommendations
        """
        competition_level = market_intel['market_intelligence']['competition_level']
        volatility = market_intel['market_intelligence']['market_volatility']
        
        if competition_level == 'high' or strategy == 'aggressive':
            frequency = "weekly"
            kpis = ["market_share", "competitor_pricing", "volume_impact"]
        elif volatility > 0.05:
            frequency = "bi_weekly"
            kpis = ["price_stability", "margin_impact", "customer_response"]
        else:
            frequency = "monthly"
            kpis = ["profitability", "market_position"]
        
        return {
            "monitoring_frequency": frequency,
            "key_metrics": kpis,
            "review_triggers": ["competitor_price_change", "volume_drop_>10%", "margin_drop_>5%"]
        }
    
    def identify_key_risks(self, market_intel: Dict, profitability: Dict, strategy: str) -> List[str]:
        """
        Identify key risks for the pricing strategy
        """
        risks = []
        
        if strategy == 'aggressive':
            risks.append("Potential margin erosion from aggressive pricing")
            if market_intel['market_intelligence']['competition_level'] == 'high':
                risks.append("Competitor price matching may trigger price war")
        
        if profitability['constraint_analysis'].get('compliance_issues'):
            risks.append("Regulatory compliance issues may delay implementation")
        
        if market_intel['market_intelligence']['market_volatility'] > 0.05:
            risks.append("High market volatility increases price uncertainty")
        
        if profitability['pricing_optimization']['expected_margin'] < 0.2:
            risks.append("Low baseline margins limit pricing flexibility")
        
        return risks if risks else ["No significant risks identified"]
    
    def suggest_risk_mitigation(self, market_intel: Dict, profitability: Dict, strategy: str) -> List[str]:
        """
        Suggest risk mitigation actions
        """
        mitigations = []
        
        if strategy == 'aggressive':
            mitigations.append("Monitor competitor responses closely and be prepared to adjust")
            mitigations.append("Implement gradual price changes rather than immediate large adjustments")
        
        if profitability['constraint_analysis'].get('compliance_issues'):
            mitigations.append("Work with regulatory team to resolve compliance issues before implementation")
        
        if market_intel['market_intelligence']['competition_level'] == 'high':
            mitigations.append("Develop value proposition messaging to justify pricing position")
        
        mitigations.append("Establish clear metrics and thresholds for strategy reversal if needed")
        
        return mitigations
    
    def generate_portfolio_insights(self, recommendations: List[Dict], step1_output: Dict, step2_output: Dict) -> List[str]:
        """
        Generate key insights about the portfolio strategy
        """
        insights = []
        
        # Strategy distribution insights
        strategy_counts = {}
        for rec in recommendations:
            strategy = rec['pricing_strategy']['strategy_type']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        dominant_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0]
        insights.append(f"Dominant strategy is {dominant_strategy} pricing ({strategy_counts[dominant_strategy]} drugs)")
        
        # Market condition insights
        market_sentiment = step1_output['market_overview']['market_trends']['market_sentiment']
        insights.append(f"Overall market sentiment is {market_sentiment}")
        
        # Profitability insights
        total_projected_profit = step2_output['portfolio_summary']['total_projected_annual_profit']
        insights.append(f"Total projected annual profit: ${total_projected_profit:,.2f}")
        
        # Risk insights
        risk_dist = step2_output['portfolio_summary']['risk_distribution']
        high_risk_pct = (risk_dist['high'] / sum(risk_dist.values())) * 100
        if high_risk_pct > 20:
            insights.append(f"High percentage ({high_risk_pct:.1f}%) of high-risk drugs requires careful monitoring")
        
        return insights

# Strategy Recommendation Agent using the Toolkit
class StrategyRecommendationAgent(Agent):
    def __init__(self):
        # Create the toolkit instance
        self.pricing_toolkit = PricingStrategyToolkit()
        
        super().__init__(
            name="StrategyRecommendationAgent",
            model="microsoft/DialoGPT-medium",  # Medium model for improved processing tasks
            description="Combines market intelligence and profitability analysis into actionable pricing strategies",
            tools=[self.pricing_toolkit]  # Pass the toolkit instance
        )
    
    def process_drug_strategy(self, market_intel: Dict, profitability: Dict) -> Dict:
        """
        Process a single drug and generate complete strategy recommendation
        """
        # Determine strategy using toolkit
        strategy = self.pricing_toolkit.determine_pricing_strategy(market_intel, profitability)
        
        # Calculate recommended price using toolkit
        recommended_price = self.pricing_toolkit.calculate_recommended_price(strategy, market_intel, profitability)
        
        # Generate outputs using toolkit methods
        rationale = self.pricing_toolkit.generate_rationale(strategy, market_intel, profitability)
        confidence = self.pricing_toolkit.assess_confidence(market_intel, profitability)
        timeline = self.pricing_toolkit.determine_implementation_timeline(strategy, market_intel, profitability)
        monitoring = self.pricing_toolkit.generate_monitoring_plan(strategy, market_intel)
        
        return {
            "drug_id": market_intel['drug_id'],
            "drug_name": market_intel['drug_name'],
            "ndc": market_intel['ndc'],
            "classification": market_intel['classification'],
            "pricing_strategy": {
                "strategy_type": strategy,
                "recommended_price": round(recommended_price, 4),
                "current_optimal_price": profitability['pricing_optimization']['optimal_price'],
                "price_adjustment": round(recommended_price - profitability['pricing_optimization']['optimal_price'], 4),
                "price_adjustment_pct": round((recommended_price - profitability['pricing_optimization']['optimal_price']) / profitability['pricing_optimization']['optimal_price'] * 100, 2),
                "strategy_rationale": rationale,
                "confidence_score": round(confidence, 3)
            },
            "implementation": {
                "timeline": timeline,
                "priority": "high" if profitability['pricing_optimization']['risk_level'] == 'high' else "medium",
                "monitoring_plan": monitoring
            },
            "financial_impact": {
                "projected_margin": profitability['pricing_optimization']['expected_margin'],
                "margin_impact": round((recommended_price - profitability['pricing_optimization']['estimated_acquisition_cost']) / recommended_price - profitability['pricing_optimization']['expected_margin'], 4),
                "annual_profit_impact": round(profitability['pricing_optimization']['profit_projections']['annual_profit_projection'] * ((recommended_price - profitability['pricing_optimization']['estimated_acquisition_cost']) / recommended_price - profitability['pricing_optimization']['expected_margin']), 2)
            },
            "risk_assessment": {
                "overall_risk": profitability['pricing_optimization']['risk_level'],
                "key_risks": self.pricing_toolkit.identify_key_risks(market_intel, profitability, strategy),
                "mitigation_actions": self.pricing_toolkit.suggest_risk_mitigation(market_intel, profitability, strategy)
            }
        }
    
    def generate_portfolio_summary(self, recommendations: List[Dict], step1_output: Dict, step2_output: Dict) -> Dict:
        """
        Generate high-level portfolio strategy summary
        """
        strategy_distribution = {}
        total_profit_impact = 0
        high_priority_count = 0
        
        for rec in recommendations:
            strategy = rec['pricing_strategy']['strategy_type']
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
            total_profit_impact += rec['financial_impact']['annual_profit_impact']
            if rec['implementation']['priority'] == 'high':
                high_priority_count += 1
        
        return {
            "total_drugs_strategized": len(recommendations),
            "strategy_distribution": strategy_distribution,
            "portfolio_profit_impact": round(total_profit_impact, 2),
            "high_priority_implementations": high_priority_count,
            "average_confidence": round(np.mean([rec['pricing_strategy']['confidence_score'] for rec in recommendations]), 3),
            "key_portfolio_insights": self.pricing_toolkit.generate_portfolio_insights(recommendations, step1_output, step2_output)
        }
    
    def process_portfolio(self, step1_output: Dict, step2_output: Dict) -> Dict:
        """
        Main processing pipeline for portfolio strategy recommendations
        """
        print("🎯 Starting pricing strategy analysis...")
        
        # Create drug lookup from step1 data
        market_intel_lookup = {drug['drug_id']: drug for drug in step1_output['drug_intelligence']}
        
        strategy_recommendations = []
        
        # Process each drug from step2 output
        for drug_profitability in step2_output['pricing_recommendations']:
            drug_id = drug_profitability['drug_id']
            
            if drug_id in market_intel_lookup:
                market_intel = market_intel_lookup[drug_id]
                strategy_recommendation = self.process_drug_strategy(market_intel, drug_profitability)
                strategy_recommendations.append(strategy_recommendation)
        
        print(f"💡 Generated strategies for {len(strategy_recommendations)} drugs")
        
        # Generate portfolio summary using toolkit
        portfolio_summary = self.generate_portfolio_summary(strategy_recommendations, step1_output, step2_output)
        
        print(f"📊 Portfolio analysis complete with {portfolio_summary['total_drugs_strategized']} strategic recommendations")
        
        # Final output structure
        result = {
            "agent": "StrategyRecommendationAgent",
            "timestamp": datetime.now().isoformat(),
            "portfolio_strategy_summary": portfolio_summary,
            "drug_strategies": strategy_recommendations
        }
        
        print("✨ Pricing strategy processing complete!")
        return result