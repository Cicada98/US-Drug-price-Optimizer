import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from agno.agent import Agent
from agno.tools import Toolkit
import json
import warnings
warnings.filterwarnings('ignore')

class ProfitabilityCalculationTool(Toolkit):
    """Tool for calculating profitability and margins"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="profitability_calculation_tool",
            tools = [
                self.calculate_base_margins,
                self.calculate_profit_projections
            ],
            **kwargs
        )
    
    def calculate_base_margins(self, current_price: float, drug_type: str) -> Dict:
        """Calculate basic margin information"""
        
        # Estimate acquisition cost (simplified - typically 60-80% of selling price)
        if drug_type == 'Brand':
            # Brand drugs typically have higher margins
            estimated_acquisition_cost = current_price * 0.65  # 35% margin
            target_margin = 0.40  # Target 40% margin for brands
            minimum_margin = 0.25  # Minimum 25% margin
        else:  # Generic
            # Generic drugs have lower margins due to competition
            estimated_acquisition_cost = current_price * 0.75  # 25% margin
            target_margin = 0.30  # Target 30% margin for generics
            minimum_margin = 0.15  # Minimum 15% margin
        
        # Calculate target and minimum prices
        target_price = estimated_acquisition_cost / (1 - target_margin)
        minimum_price = estimated_acquisition_cost / (1 - minimum_margin)
        
        return {
            'estimated_acquisition_cost': float(estimated_acquisition_cost),
            'target_margin': float(target_margin),
            'minimum_margin': float(minimum_margin),
            'target_price': float(target_price),
            'minimum_price': float(minimum_price),
            'current_margin': float((current_price - estimated_acquisition_cost) / current_price) if current_price > 0 else 0
        }
    
    def calculate_profit_projections(self, price: float, acquisition_cost: float, 
                                   competition_level: str) -> Dict:
        """Calculate profit projections at given price"""
        
        # Estimate monthly volume based on competition level
        volume_estimates = {
            'low': 10000,      # Low competition = higher volume
            'moderate': 7500,   # Moderate competition = medium volume  
            'high': 5000,      # High competition = lower volume
            'very_high': 3000  # Very high competition = very low volume
        }
        
        estimated_volume = volume_estimates.get(competition_level, 5000)
        
        # Calculate profitability metrics
        gross_profit_per_unit = price - acquisition_cost
        monthly_gross_profit = gross_profit_per_unit * estimated_volume
        
        # Apply volume sensitivity (higher prices may reduce volume)
        current_margin = gross_profit_per_unit / price if price > 0 else 0
        if current_margin > 0.5:  # Very high margins may hurt volume
            volume_adjustment = 0.8
        elif current_margin > 0.4:  # High margins
            volume_adjustment = 0.9
        else:
            volume_adjustment = 1.0
        
        adjusted_volume = estimated_volume * volume_adjustment
        adjusted_monthly_profit = gross_profit_per_unit * adjusted_volume
        
        return {
            'gross_profit_per_unit': float(gross_profit_per_unit),
            'estimated_monthly_volume': int(adjusted_volume),
            'monthly_gross_profit': float(adjusted_monthly_profit),
            'annual_profit_projection': float(adjusted_monthly_profit * 12),
            'margin_percentage': float(current_margin * 100)
        }

class RegulatoryConstraintTool(Toolkit):
    """Tool for applying regulatory constraints and limits"""
    
    def __init__(self):
        super().__init__(
            name="regulatory_constraint_tool",
            tools = [
                self.apply_regulatory_limits
            ])
        
        # Simplified regulatory constraints for demo
        self.constraints = {
            'medicaid_ffs_multiplier': 1.5,    # Max 150% of benchmark
            'medicare_part_d_multiplier': 1.3,  # Max 130% of benchmark  
            'price_increase_annual_limit': 0.10, # Max 10% annual increase
            'minimum_safety_margin': 0.05      # 5% buffer from regulatory limits
        }
    
    def apply_regulatory_limits(self, price_data: Dict, market_median: float, 
                               drug_type: str) -> Dict:
        """Apply regulatory constraints to pricing"""
        
        target_price = price_data['target_price']
        minimum_price = price_data['minimum_price']
        
        # Calculate regulatory ceilings
        medicaid_ceiling = market_median * self.constraints['medicaid_ffs_multiplier']
        medicare_ceiling = market_median * self.constraints['medicare_part_d_multiplier']
        
        # Use more restrictive ceiling
        regulatory_ceiling = min(medicaid_ceiling, medicare_ceiling)
        
        # Apply safety margin
        safe_ceiling = regulatory_ceiling * (1 - self.constraints['minimum_safety_margin'])
        
        # Calculate regulatory floor (minimum viable price)
        # Typically based on acquisition cost + minimum margin
        regulatory_floor = price_data['minimum_price']
        
        # Adjust target price if it exceeds limits
        constrained_target_price = min(target_price, safe_ceiling)
        constrained_target_price = max(constrained_target_price, regulatory_floor)
        
        # Check compliance status
        compliance_status = "compliant"
        compliance_issues = []
        
        if target_price > safe_ceiling:
            compliance_issues.append(f"Target price exceeds regulatory ceiling by ${target_price - safe_ceiling:.2f}")
            compliance_status = "requires_adjustment"
        
        if target_price < regulatory_floor:
            compliance_issues.append(f"Target price below minimum viable threshold")
            compliance_status = "requires_adjustment"
        
        return {
            'regulatory_ceiling': float(safe_ceiling),
            'regulatory_floor': float(regulatory_floor),
            'constrained_target_price': float(constrained_target_price),
            'compliance_status': compliance_status,
            'compliance_issues': compliance_issues,
            'regulatory_headroom': float(safe_ceiling - constrained_target_price),
            'medicaid_limit': float(medicaid_ceiling),
            'medicare_limit': float(medicare_ceiling)
        }

class OptimalPricingTool(Toolkit):
    """Tool for determining optimal pricing strategy"""
    
    def __init__(self):
        super().__init__(
            name="optimal_pricing_tool",
            tools = [
                self.calculate_optimal_price
            ])
    
    def calculate_optimal_price(self, profitability_data: Dict, regulatory_data: Dict,
                               market_intelligence: Dict) -> Dict:
        """Calculate the optimal price and feasible range"""
        
        # Get key inputs
        target_price = profitability_data['target_price']
        constrained_price = regulatory_data['constrained_target_price']
        regulatory_ceiling = regulatory_data['regulatory_ceiling']
        regulatory_floor = regulatory_data['regulatory_floor']
        
        # Consider market positioning
        competition_level = market_intelligence['competition_level']
        benchmark_max = market_intelligence['benchmark_max_price']
        benchmark_min = market_intelligence['benchmark_min_price']
        
        # Adjust for competition
        if competition_level == 'very_high':
            # In high competition, price more aggressively
            competitive_adjustment = 0.95
        elif competition_level == 'high':
            competitive_adjustment = 0.97
        elif competition_level == 'moderate':
            competitive_adjustment = 1.0
        else:  # Low competition
            # Can price higher with low competition
            competitive_adjustment = 1.02
        
        # Calculate market-adjusted price
        market_adjusted_price = constrained_price * competitive_adjustment
        
        # Ensure within regulatory bounds
        optimal_price = max(regulatory_floor, min(market_adjusted_price, regulatory_ceiling))
        
        # Calculate feasible range
        feasible_min = max(regulatory_floor, benchmark_min * 0.95)
        feasible_max = min(regulatory_ceiling, benchmark_max * 1.05)
        
        # Ensure optimal price is within feasible range
        optimal_price = max(feasible_min, min(optimal_price, feasible_max))
        
        # Assess risk level
        risk_level = self._assess_pricing_risk(optimal_price, market_intelligence, 
                                             regulatory_data, profitability_data)
        
        return {
            'optimal_price': float(optimal_price),
            'feasible_range': [float(feasible_min), float(feasible_max)],
            'competitive_adjustment_factor': float(competitive_adjustment),
            'risk_level': risk_level,
            'pricing_rationale': self._generate_pricing_rationale(
                optimal_price, competition_level, regulatory_data['compliance_status']
            )
        }
    
    def _assess_pricing_risk(self, optimal_price: float, market_intelligence: Dict,
                            regulatory_data: Dict, profitability_data: Dict) -> str:
        """Assess risk level of pricing decision"""
        
        risk_factors = 0
        
        # Competition risk
        if market_intelligence['competition_level'] in ['high', 'very_high']:
            risk_factors += 1
        
        # Regulatory risk
        if regulatory_data['compliance_status'] != 'compliant':
            risk_factors += 2
        
        # Profitability risk  
        if profitability_data['current_margin'] < profitability_data['minimum_margin']:
            risk_factors += 1
        
        # Market positioning risk
        if market_intelligence['price_quartile'] == 'premium' and market_intelligence['competition_level'] == 'very_high':
            risk_factors += 1
        
        # Assess overall risk
        if risk_factors >= 3:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_pricing_rationale(self, optimal_price: float, competition_level: str,
                                  compliance_status: str) -> str:
        """Generate human-readable rationale for pricing decision"""
        
        rationale_parts = []
        
        # Competition consideration
        if competition_level == 'very_high':
            rationale_parts.append("aggressive pricing due to intense competition")
        elif competition_level == 'low':
            rationale_parts.append("premium pricing opportunity with limited competition")
        else:
            rationale_parts.append("balanced pricing considering moderate competition")
        
        # Regulatory consideration
        if compliance_status == 'compliant':
            rationale_parts.append("full regulatory compliance maintained")
        else:
            rationale_parts.append("pricing adjusted for regulatory constraints")
        
        return "Optimal price balances " + " and ".join(rationale_parts) + "."

class ProfitabilityConstraintAgent(Agent):
    """Profitability & Constraint Agent"""
    
    def __init__(self):
        # Use same model as market intelligence for consistency
        model_name = "microsoft/DialoGPT-medium"
        self.ProfitabilityCalculationTool = ProfitabilityCalculationTool()
        self.RegulatoryConstraintTool = RegulatoryConstraintTool()
        self.OptimalPricingTool = OptimalPricingTool()
        
        super().__init__(
            name="ProfitabilityConstraintAgent", 
            model=model_name,
            description="Calculates optimal pricing balancing profitability targets and regulatory constraints",
            tools=[
                self.ProfitabilityCalculationTool,
                self.RegulatoryConstraintTool,
                self.OptimalPricingTool
            ]
        )
    
    def calculate_optimal_pricing(self, intelligence_output: Dict) -> Dict[str, Any]:
        """Main pipeline for profitability and constraint analysis"""
        
        print("💰 Starting profitability and constraint analysis...")
        
        drug_intelligence = intelligence_output['drug_intelligence']
        market_stats = intelligence_output['market_overview']['market_statistics']
        
        # Get tools
        profitability_tool = self.tools[0]
        regulatory_tool = self.tools[1] 
        pricing_tool = self.tools[2]
        
        pricing_recommendations = []
        
        for drug_data in drug_intelligence:
            drug_name = drug_data['drug_name']
            market_intel = drug_data['market_intelligence']
            classification = drug_data['classification']
            
            # Step 1: Calculate base profitability
            current_price = (market_intel['benchmark_min_price'] + market_intel['benchmark_max_price']) / 2
            profitability_data = profitability_tool.calculate_base_margins(
                current_price, classification['brand_generic']
            )
            
            # Step 2: Apply regulatory constraints
            regulatory_data = regulatory_tool.apply_regulatory_limits(
                profitability_data, market_intel['market_median'], classification['brand_generic']
            )
            
            # Step 3: Calculate profit projections
            profit_projections = profitability_tool.calculate_profit_projections(
                regulatory_data['constrained_target_price'],
                profitability_data['estimated_acquisition_cost'],
                market_intel['competition_level']
            )
            # NaN check
            if np.isnan(profit_projections['annual_profit_projection']):
                profit_projections['annual_profit_projection'] = 0.0
            
            # Step 4: Determine optimal pricing
            optimal_pricing = pricing_tool.calculate_optimal_price(
                profitability_data, regulatory_data, market_intel
            )
            
            # Calculate final expected margin at optimal price
            optimal_price = optimal_pricing['optimal_price']
            acquisition_cost = profitability_data['estimated_acquisition_cost']
            expected_margin = (optimal_price - acquisition_cost) / optimal_price if optimal_price > 0 else 0
            
            # Combine all analysis
            recommendation = {
                "drug_id": drug_data['drug_id'],
                "drug_name": drug_name,
                "ndc": drug_data['ndc'],
                "classification": classification,
                "pricing_optimization": {
                    "optimal_price": optimal_pricing['optimal_price'],
                    "feasible_range": optimal_pricing['feasible_range'],
                    "expected_margin": float(expected_margin),
                    "estimated_acquisition_cost": profitability_data['estimated_acquisition_cost'],
                    "profit_projections": profit_projections,
                    "regulatory_status": regulatory_data['compliance_status'],
                    "risk_level": optimal_pricing['risk_level'],
                    "pricing_rationale": optimal_pricing['pricing_rationale']
                },
                "constraint_analysis": {
                    "regulatory_ceiling": regulatory_data['regulatory_ceiling'],
                    "regulatory_floor": regulatory_data['regulatory_floor'],
                    "regulatory_headroom": regulatory_data['regulatory_headroom'],
                    "compliance_issues": regulatory_data['compliance_issues']
                }
            }
            
            pricing_recommendations.append(recommendation)
        
        print(f"💡 Generated pricing recommendations for {len(pricing_recommendations)} drugs")
        
        # Calculate summary statistics
        total_projected_profit = sum(
            rec['pricing_optimization']['profit_projections']['annual_profit_projection'] 
            for rec in pricing_recommendations
            if not np.isnan(rec['pricing_optimization']['profit_projections']['annual_profit_projection'])
        )
        
        avg_margin = np.mean([
            rec['pricing_optimization']['expected_margin'] 
            for rec in pricing_recommendations
        ])
        
        risk_distribution = {}
        for rec in pricing_recommendations:
            risk = rec['pricing_optimization']['risk_level']
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        # Final output structure
        result = {
            "agent": "ProfitabilityConstraintAgent",
            "timestamp": datetime.now().isoformat(),
            "portfolio_summary": {
                "total_drugs_analyzed": len(pricing_recommendations),
                "total_projected_annual_profit": float(total_projected_profit),
                "average_expected_margin": float(avg_margin),
                "risk_distribution": risk_distribution
            },
            "pricing_recommendations": pricing_recommendations,
            "processing_status": {
                "optimization_success": True,
                "drugs_processed": len(pricing_recommendations),
                "compliance_rate": len([r for r in pricing_recommendations 
                                      if r['pricing_optimization']['regulatory_status'] == 'compliant']) / len(pricing_recommendations)
            }
        }
        
        print("✨ Profitability and constraint analysis complete!")
        return result

# # Usage function
# def run_profitability_constraint_agent(intelligence_output: Dict):
#     """Run the profitability constraint agent on market intelligence output"""
    
#     agent = ProfitabilityConstraintAgent()
#     result = agent.calculate_optimal_pricing(intelligence_output)
    
#     return result

# # Example usage in notebook:
# """
# # Load market intelligence output from Step 1  
# with open('intelligence_output.json', 'r') as f:
#     intelligence_output = json.load(f)

# # Run profitability and constraint analysis
# profitability_result = run_profitability_constraint_agent(intelligence_output)

# # Save structured output for next agent
# with open('profitability_output.json', 'w') as f:
#     json.dump(profitability_result, f, indent=2)

# print(f"Generated pricing for {profitability_result['processing_status']['drugs_processed']} drugs")
# print(f"Average expected margin: {profitability_result['portfolio_summary']['average_expected_margin']:.1%}")
# print(f"Total projected annual profit: ${profitability_result['portfolio_summary']['total_projected_annual_profit']:,.0f}")
# """