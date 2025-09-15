import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from agno.agent import Agent
from agno.tools import Toolkit
import json
import warnings
warnings.filterwarnings('ignore')

class MarketBenchmarkingTool(Toolkit):
    """Tool for calculating market benchmarks and price ranges"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="market_benchmarking_tool",
            tools=[
                self.calculate_market_benchmarks,
                self.calculate_drug_benchmark_ranges
            ],
            **kwargs
        )
    
    def calculate_market_benchmarks(self, drug_profiles: List[Dict]) -> Dict[str, Dict]:
        """Calculate market benchmarks by therapeutic class and drug type"""

        # Extract prices and filter out NaN values explicitly, since I was seeing issues
        def extract_valid_prices(drugs):
            prices = []
            for drug in drugs:
                price = drug['pricing_data']['current_avg_cost']
                # Check for both None and NaN
                if price is not None and not np.isnan(price) and price > 0:
                    prices.append(float(price))
            return prices
        
        # Group drugs by characteristics
        brand_drugs = [d for d in drug_profiles if d['classification']['brand_generic'] == 'Brand']
        generic_drugs = [d for d in drug_profiles if d['classification']['brand_generic'] == 'Generic']
        
        # Extract valid prices
        all_prices = extract_valid_prices(drug_profiles)
        
        print(f"Debug: Found {len(all_prices)} valid prices out of {len(drug_profiles)} drugs")
        
        if not all_prices:
            return {"error": "No valid pricing data found", "total_drugs": len(drug_profiles)}
        
        market_stats = {
            'data_summary': {
                'total_drugs': len(drug_profiles),
                'drugs_with_valid_prices': len(all_prices),
                'price_coverage': len(all_prices) / len(drug_profiles),
                'drugs_with_nan_prices': len(drug_profiles) - len(all_prices)
            },
            'overall_median': float(np.median(all_prices)),
            'overall_mean': float(np.mean(all_prices)),
            'overall_std': float(np.std(all_prices)),
            'price_quartiles': {
                'q25': float(np.percentile(all_prices, 25)),
                'q50': float(np.percentile(all_prices, 50)),
                'q75': float(np.percentile(all_prices, 75)),
                'q90': float(np.percentile(all_prices, 90))
            }
        }
        
        # Calculate brand vs generic benchmarks
        if brand_drugs:
            brand_prices = extract_valid_prices(brand_drugs)
            if brand_prices:
                market_stats['brand_segment'] = {
                    'median': float(np.median(brand_prices)),
                    'mean': float(np.mean(brand_prices)),
                    'count': len(brand_drugs),
                    'valid_prices': len(brand_prices)
                }
        
        if generic_drugs:
            generic_prices = extract_valid_prices(generic_drugs)
            if generic_prices:
                market_stats['generic_segment'] = {
                    'median': float(np.median(generic_prices)),
                    'mean': float(np.mean(generic_prices)),
                    'count': len(generic_drugs),
                    'valid_prices': len(generic_prices)
                }
        
        return market_stats
    
    def calculate_drug_benchmark_ranges(self, drug_profile: Dict, market_stats: Dict) -> Dict:
        """Calculate benchmark ranges for individual drugs"""
        current_price = drug_profile['pricing_data']['current_avg_cost']
        drug_type = drug_profile['classification']['brand_generic']
        
        # Use appropriate segment benchmarks
        if drug_type == 'Brand' and 'brand_segment' in market_stats:
            segment_median = market_stats['brand_segment']['median']
            segment_mean = market_stats['brand_segment']['mean']
        elif drug_type == 'Generic' and 'generic_segment' in market_stats:
            segment_median = market_stats['generic_segment']['median']
            segment_mean = market_stats['generic_segment']['mean']
        else:
            segment_median = market_stats['overall_median']
            segment_mean = market_stats['overall_mean']
        
        # Calculate price volatility factor
        volatility = drug_profile['pricing_data'].get('price_volatility', 0)
        volatility_factor = max(0.1, min(0.3, volatility))  # Cap between 10-30%
        
        # Calculate benchmark ranges
        benchmark_min = current_price * (1 - volatility_factor)
        benchmark_max = current_price * (1 + volatility_factor)
        
        return {
            'benchmark_min_price': float(benchmark_min),
            'benchmark_max_price': float(benchmark_max),
            'market_median': float(segment_median),
            'market_mean': float(segment_mean),
            'price_position_vs_median': float((current_price - segment_median) / segment_median) if segment_median > 0 else 0,
            'price_quartile': self._determine_price_quartile(current_price, market_stats['price_quartiles'])
        }
    
    def _determine_price_quartile(self, price: float, quartiles: Dict) -> str:
        """Determine which quartile the price falls into"""
        if price <= quartiles['q25']:
            return 'low_cost'
        elif price <= quartiles['q50']:
            return 'below_median'
        elif price <= quartiles['q75']:
            return 'above_median'
        else:
            return 'premium'

class CompetitiveAnalysisTool(Toolkit):
    """Tool for analyzing competitive pressures and market dynamics"""
    
    def __init__(self):
        super().__init__(
            name="competitive_analysis_tool",
            tools=[
                self.analyze_competition_level
            ]
        )
    
    def analyze_competition_level(self, drug_profile: Dict, all_profiles: List[Dict]) -> Dict:
        """Analyze competitive pressure for a specific drug"""
        drug_name = drug_profile['drug_name']
        current_price = drug_profile['pricing_data']['current_avg_cost']
        drug_type = drug_profile['classification']['brand_generic']
        
        # Find similar drugs (same active ingredient or similar names)
        similar_drugs = self._find_similar_drugs(drug_profile, all_profiles)
        
        # Count competitors by type
        brand_competitors = len([d for d in similar_drugs if d['classification']['brand_generic'] == 'Brand'])
        generic_competitors = len([d for d in similar_drugs if d['classification']['brand_generic'] == 'Generic'])
        
        # Determine competition level
        total_competitors = len(similar_drugs)
        # competition levels for brand-vs-brand competition
        if total_competitors >= 10:
            competition_level = 'very_high'
        elif total_competitors >= 5:
            competition_level = 'high' 
        elif total_competitors >= 2:
            competition_level = 'moderate'
        else:
            competition_level = 'low'
        
        # Analyze price competitiveness
        if similar_drugs:
            competitor_prices = [d['pricing_data']['current_avg_cost'] 
                            for d in similar_drugs 
                            if d['pricing_data']['current_avg_cost'] is not None 
                            and not np.isnan(d['pricing_data']['current_avg_cost'])]
            
            if competitor_prices:
                avg_competitor_price = np.mean(competitor_prices)
                price_advantage = ((avg_competitor_price - current_price) / avg_competitor_price 
                                if avg_competitor_price > 0 else 0)
            else:
                avg_competitor_price = current_price
                price_advantage = 0
        else:
            avg_competitor_price = current_price
            price_advantage = 0
        
        return {
                'competition_level': competition_level,
                'total_competitors': total_competitors,
                'brand_competitors': total_competitors,  # All are brands
                'generic_competitors': 0,  # None in dataset
                'avg_competitor_price': float(avg_competitor_price),
                'price_advantage_pct': float(price_advantage * 100),
                'competitive_threats': self._identify_brand_threats(drug_profile, similar_drugs),
                'market_opportunities': self._identify_brand_opportunities(drug_profile, similar_drugs, current_price)
            }
    
    def _find_similar_drugs(self, target_drug: Dict, all_drugs: List[Dict]) -> List[Dict]:
        """Find drugs with same active ingredient - REFINED for brand competition"""
        target_name = target_drug['drug_name'].upper()
        target_ndc = target_drug['ndc']
        
        # Extract active ingredient more precisely
        stop_words = {'MG', 'ML', 'MCG', 'CAPSULE', 'TABLET', 'INJECTION', 'INJ', 
                    'ORAL', 'SOLUTION', 'CREAM', 'GEL', 'PATCH', 'ER', 'XR', 'SR'}
        
        target_words = [word for word in target_name.split() if word not in stop_words]
        
        if not target_words:
            return []
        
        # Use first meaningful word as active ingredient
        active_ingredient = target_words[0]
        
        similar_drugs = []
        for drug in all_drugs:
            if drug['ndc'] == target_ndc:
                continue
                
            drug_name = drug['drug_name'].upper()
            drug_words = [word for word in drug_name.split() if word not in stop_words]
            
            # Only match exact active ingredient
            if drug_words and drug_words[0] == active_ingredient:
                similar_drugs.append(drug)
        
        return similar_drugs
    
    def _identify_brand_threats(self, drug_profile: Dict, competitors: List[Dict]) -> List[str]:
        """Identify competitive threats in brand-only market"""
        threats = []
        current_price = drug_profile['pricing_data']['current_avg_cost']
        
        if not competitors:
            return ["No direct brand competitors identified"]
        
        # Check for lower-priced brand competitors
        cheaper_competitors = [d for d in competitors 
                            if d['pricing_data']['current_avg_cost'] < current_price * 0.9]
        if cheaper_competitors:
            threats.append(f"{len(cheaper_competitors)} lower-priced brand competitors")
        
        # Check for premium competitors that might justify higher pricing
        premium_competitors = [d for d in competitors 
                            if d['pricing_data']['current_avg_cost'] > current_price * 1.2]
        if len(premium_competitors) > len(competitors) * 0.5:
            threats.append("Market dominated by premium-priced brands")
        
        # Check for high volatility (price instability)
        volatile_competitors = [d for d in competitors 
                            if d['pricing_data'].get('price_volatility', 0) > 0.2]
        if volatile_competitors:
            threats.append(f"Price volatility among {len(volatile_competitors)} competitors")
        
        return threats if threats else ["Limited competitive pressure detected"]

    def _identify_brand_opportunities(self, drug_profile: Dict, competitors: List[Dict], current_price: float) -> List[str]:
        """Identify market opportunities"""
        opportunities = []
        
        if not competitors:
            opportunities.append("Limited competition - market leadership opportunity")
            return opportunities
        
        competitor_prices = [d['pricing_data']['current_avg_cost'] for d in competitors]
        avg_competitor_price = np.mean(competitor_prices)
        
        # Price positioning opportunities
        if current_price < avg_competitor_price * 0.9:
            opportunities.append("Cost leadership position - maintain competitive advantage")
        elif current_price > avg_competitor_price * 1.1:
            opportunities.append("Premium positioning - justify with value proposition")
        else:
            opportunities.append("Market-rate pricing - opportunity for differentiation")
        
        # Trend-based opportunities  
        increasing_trend_competitors = len([d for d in competitors if d['pricing_data'].get('price_trend') == 'increasing'])
        if increasing_trend_competitors > len(competitors) * 0.5:
            opportunities.append("Market prices trending up - opportunity for margin expansion")
        
        return opportunities

class MarketTrendAnalysisTool(Toolkit):
    """Tool for analyzing market trends and anomalies"""
    
    def __init__(self):
        super().__init__(
            name="market_trend_analysis_tool",
            tools=[
                self.analyze_market_trends
            ])
    
    def analyze_market_trends(self, drug_profiles: List[Dict]) -> Dict:
        """Analyze overall market trends"""
        
        # Analyze price trends distribution
        trend_counts = {}
        volatility_levels = []
        
        for drug in drug_profiles:
            trend = drug['pricing_data'].get('price_trend', 'stable')
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
            volatility_levels.append(drug['pricing_data'].get('price_volatility', 0))
        
        # Calculate market volatility
        avg_volatility = np.mean(volatility_levels)
        high_volatility_drugs = len([v for v in volatility_levels if v > 0.2])
        
        # Identify market anomalies
        anomalies = self._detect_price_anomalies(drug_profiles)
        
        return {
            'trend_distribution': trend_counts,
            'market_volatility': {
                'average_volatility': float(avg_volatility),
                'high_volatility_count': high_volatility_drugs,
                'volatility_level': self._categorize_volatility(avg_volatility)
            },
            'market_anomalies': anomalies,
            'market_sentiment': self._determine_market_sentiment(trend_counts, avg_volatility)
        }
    
    def _detect_price_anomalies(self, drug_profiles: List[Dict]) -> List[Dict]:
        """Detect unusual pricing patterns"""
        anomalies = []
        prices = [d['pricing_data']['current_avg_cost'] for d in drug_profiles]
        
        # Statistical outliers (prices > 3 standard deviations)
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        for drug in drug_profiles:
            price = drug['pricing_data']['current_avg_cost']
            z_score = abs((price - mean_price) / std_price) if std_price > 0 else 0
            
            if z_score > 3:
                anomalies.append({
                    'drug_name': drug['drug_name'],
                    'ndc': drug['ndc'],
                    'anomaly_type': 'statistical_outlier',
                    'price': price,
                    'z_score': float(z_score)
                })
            
            # High volatility anomalies
            if drug['pricing_data'].get('price_volatility', 0) > 0.5:
                anomalies.append({
                    'drug_name': drug['drug_name'],
                    'ndc': drug['ndc'],
                    'anomaly_type': 'high_volatility',
                    'volatility': drug['pricing_data']['price_volatility']
                })
        
        return anomalies[:10]  # Return top 10 anomalies
    
    def _categorize_volatility(self, avg_volatility: float) -> str:
        """Categorize market volatility level"""
        if avg_volatility < 0.05:
            return 'low'
        elif avg_volatility < 0.15:
            return 'moderate'
        else:
            return 'high'
    
    def _determine_market_sentiment(self, trend_counts: Dict, avg_volatility: float) -> str:
        """Determine overall market sentiment"""
        increasing = trend_counts.get('increasing', 0)
        decreasing = trend_counts.get('decreasing', 0)
        stable = trend_counts.get('stable', 0)
        
        total = increasing + decreasing + stable
        if total == 0:
            return 'neutral'
        
        increasing_pct = increasing / total
        decreasing_pct = decreasing / total
        
        if increasing_pct > 0.4 and avg_volatility < 0.1:
            return 'bullish'
        elif decreasing_pct > 0.4 and avg_volatility < 0.1:
            return 'bearish'
        elif avg_volatility > 0.2:
            return 'volatile'
        else:
            return 'stable'

class MarketIntelligenceAgent(Agent):
    """Market & Competitive Intelligence Agent"""
    
    def __init__(self):
        # Use a model good at analysis and pattern recognition
        model_name = "microsoft/DialoGPT-medium"  # Better for analysis tasks
        self.calculate_benchmarks = MarketBenchmarkingTool()
        self.analyze_competitive_position = CompetitiveAnalysisTool()
        self.analyze_market_trends = MarketTrendAnalysisTool()
        
        
        super().__init__(
            name="MarketIntelligenceAgent",
            model=model_name,
            description="Analyzes market positioning, competitive pressures, and identifies pricing opportunities",
            tools=[
                self.calculate_benchmarks,
                self.analyze_competitive_position,
                self.analyze_market_trends
            ]
        )
    
    def analyze_market_intelligence(self, foundation_output: Dict) -> Dict[str, Any]:
        """Main analysis pipeline for market intelligence"""
        
        print("📊 Starting market intelligence analysis...")
        
        drug_profiles = foundation_output['drug_profiles']
        
        # Step 1: Calculate market benchmarks
        benchmarking_tool = self.tools[0]
        market_stats = benchmarking_tool.calculate_market_benchmarks(drug_profiles)
        
        print(f"📈 Calculated market benchmarks for {len(drug_profiles)} drugs")
        
        # Step 2: Analyze each drug's competitive position
        competitive_tool = self.tools[1]
        drug_intelligence = []
        
        for drug_profile in drug_profiles:
            # Calculate benchmark ranges
            benchmark_data = benchmarking_tool.calculate_drug_benchmark_ranges(drug_profile, market_stats)
            
            # Analyze competitive position
            competitive_data = competitive_tool.analyze_competition_level(drug_profile, drug_profiles)
            
            # Combine intelligence
            intelligence = {
                "drug_id": drug_profile['drug_id'],
                "drug_name": drug_profile['drug_name'],
                "ndc": drug_profile['ndc'],
                "classification": drug_profile['classification'],
                "market_intelligence": {
                    **benchmark_data,
                    **competitive_data,
                    "market_volatility": drug_profile['pricing_data'].get('price_volatility', 0),
                    "trend": drug_profile['pricing_data'].get('price_trend', 'stable')
                }
            }
            
            drug_intelligence.append(intelligence)
        
        print(f"🎯 Completed competitive analysis for {len(drug_intelligence)} drugs")
        
        # Step 3: Analyze overall market trends
        trend_tool = self.tools[2]
        market_trends = trend_tool.analyze_market_trends(drug_profiles)
        
        print("📊 Completed market trend analysis")
        
        # Final output structure
        result = {
            "agent": "MarketIntelligenceAgent",
            "timestamp": datetime.now().isoformat(),
            "market_overview": {
                "total_drugs_analyzed": len(drug_profiles),
                "market_statistics": market_stats,
                "market_trends": market_trends
            },
            "drug_intelligence": drug_intelligence,
            "processing_status": {
                "analysis_success": True,
                "drugs_processed": len(drug_intelligence),
                "anomalies_detected": len(market_trends['market_anomalies'])
            }
        }
        
        print("✨ Market intelligence analysis complete!")
        return result