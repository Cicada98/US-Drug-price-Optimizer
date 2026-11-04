import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from agno.agent import Agent
from agno.tools import Toolkit
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import warnings
warnings.filterwarnings('ignore')

class DataFoundationToolkit(Toolkit):
    """Toolkit for cleaning, analyzing and structuring NADAC pharmaceutical data"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="data_foundation_toolkit",
            tools=[
                self.clean_drug_names,
                self.normalize_pricing,
                self.standardize_categories,
                self.compute_time_features,
                self.compute_summary_stats,
                self.create_drug_profiles
            ],
            **kwargs
        )
    
    def clean_drug_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract drug name, dosage, and form from NDC descriptions"""
        df = df.copy()

        # Standardize spacing
        df['NDC Description'] = df['NDC Description'].str.strip()
        df['NDC Description'] = df['NDC Description'].str.replace(r'\s+', ' ', regex=True)

        # Extract dosage (mg, mL, g)
        df['Dosage'] = df['NDC Description'].str.extract(r'(\d+\.?\d*\s?(?:mg|mL|g))', flags=re.IGNORECASE)
        df['Dosage'] = df['Dosage'].fillna('Unknown')

        # Extract form (Tablet, Capsule, Injection, Solution, etc.)
        form_patterns = r'(Tablet|Tab|Capsule|Cap|Injection|Inj|Solution|Sol|Syrup|Cream|Ointment|Zydis)'
        df['Drug_Form'] = df['NDC Description'].str.extract(f'({form_patterns})', flags=re.IGNORECASE)[0]
        df['Drug_Form'] = df['Drug_Form'].str.title().fillna('Unknown')

        # Extract drug name = everything before dosage or form
        df['Drug_Name'] = df['NDC Description'].str.replace(r'(\d+\.?\d*\s?(mg|mL|g))', '', flags=re.IGNORECASE, regex=True)
        df['Drug_Name'] = df['Drug_Name'].str.replace(form_patterns, '', flags=re.IGNORECASE, regex=True)
        df['Drug_Name'] = df['Drug_Name'].str.strip()

        return df
    
    def normalize_pricing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize pricing units and handle missing values"""
        df = df.copy()
        
        # Convert NADAC_Per_Unit to numeric, removing $ signs
        if df['NADAC Per Unit'].dtype == 'object':
            df['NADAC Per Unit'] = df['NADAC Per Unit'].str.replace('$', '', regex=False)
            df['NADAC Per Unit'] = pd.to_numeric(df['NADAC Per Unit'], errors='coerce')
        
        # Handle outliers (prices > $1000 or < $0.001 likely errors)
        df.loc[df['NADAC Per Unit'] > 1000, 'NADAC Per Unit'] = np.nan
        df.loc[df['NADAC Per Unit'] < 0.001, 'NADAC Per Unit'] = np.nan
        
        # Fill missing prices with median by drug group
        df['NADAC Per Unit'] = df.groupby(['Drug_Name', 'Classification for Rate Setting'])['NADAC Per Unit'].transform(
            lambda x: x.fillna(x.median())
        )
        
        return df
    
    def standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical variables for NADAC dataset"""
        df = df.copy()
        
        # Standardize Classification for Rate Setting (Brand vs Generic)
        brand_map = {
            'B': 'Brand',
            'B-ANDA': 'Brand',
            'G': 'Generic',
            'Generic': 'Generic',
            'Brand': 'Brand'
        }
        df['Classification for Rate Setting'] = (
            df['Classification for Rate Setting']
            .map(brand_map)
            .fillna('Unknown')
        )
        
        # Standardize OTC (Yes/No -> OTC/Prescription)
        otc_map = {
            'Y': 'OTC',
            'N': 'Prescription'
        }
        df['OTC'] = df['OTC'].map(otc_map).fillna('Unknown')
        
        # Standardize Pharmacy Type Indicator
        pharmacy_map = {
            'C/I': 'Chain/Independent',
            'Chain': 'Chain',
            'Independent': 'Independent'
        }
        df['Pharmacy Type Indicator'] = (
            df['Pharmacy Type Indicator']
            .map(pharmacy_map)
            .fillna('Unknown')
        )
        
        return df


    def compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based features (moving averages & volatility) for NADAC dataset"""
        df = df.copy()
        
        # Ensure Effective Date is datetime
        df['Effective Date'] = pd.to_datetime(df['Effective Date'])
        
        # Sort by drug and date
        df = df.sort_values(['NDC', 'Effective Date'])
        
        # Compute rolling stats (window = 3 entries, since NADAC updates weekly/monthly)
        df['Price_MA'] = df.groupby('NDC')['NADAC Per Unit'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        df['Price_Std'] = df.groupby('NDC')['NADAC Per Unit'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        # Price volatility (std / mean)
        df['Price_Volatility'] = (df['Price_Std'] / df['Price_MA']).fillna(0)
        
        return df

    
    def compute_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute overall dataset summary statistics for NADAC dataset"""
        summary = {
            # Total number of records
            'total_records': len(df),
            
            # Number of unique drugs
            'unique_drugs': df['NDC'].nunique(),
            
            # Date range of the dataset
            'date_range': {
                'start': df['Effective Date'].min().strftime('%Y-%m-%d'),
                'end': df['Effective Date'].max().strftime('%Y-%m-%d')
            },
            
            # Brand vs Generic split (Classification for Rate Setting)
            'brand_generic_split': df['Classification for Rate Setting'].value_counts().to_dict(),
            
            # Pharmacy type split
            'pharmacy_type_split': df['Pharmacy Type Indicator'].value_counts().to_dict(),
            
            # Price statistics (using NADAC Per Unit)
            'price_stats': {
                'mean': float(df['NADAC Per Unit'].mean()),
                'median': float(df['NADAC Per Unit'].median()),
                'std': float(df['NADAC Per Unit'].std()),
                'min': float(df['NADAC Per Unit'].min()),
                'max': float(df['NADAC Per Unit'].max())
            },
            
            # Data quality
            'data_quality': {
                'completeness': float(1 - df['NADAC Per Unit'].isna().sum() / len(df)),
                'missing_records': int(df['NADAC Per Unit'].isna().sum())
            }
        }
        
        return summary
    
    def _calculate_trend(self, group: pd.DataFrame) -> str:
        """
        Calculate price trend direction for a single drug group in the NADAC dataset.
        
        Returns:
            "increasing"  - if recent prices are significantly higher
            "decreasing"  - if recent prices are significantly lower
            "stable"      - if prices haven’t changed much
            "insufficient_data" - if there are fewer than 2 price points
        """
        if len(group) < 2:
            return "insufficient_data"
        
        # Sort by Effective Date
        group_sorted = group.sort_values('Effective Date')
        
        # Average of last 5 prices (most recent) and first 5 prices (older)
        recent_prices = group_sorted['NADAC Per Unit'].tail(5).mean()
        older_prices = group_sorted['NADAC Per Unit'].head(5).mean()
        
        # Percent change
        change_pct = (recent_prices - older_prices) / older_prices if older_prices > 0 else 0
        
        # Determine trend
        if change_pct > 0.05:
            return "increasing"
        elif change_pct < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def create_drug_profiles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create structured drug profiles from NADAC dataset"""
        drug_profiles = []
        
        for ndc, group in df.groupby('NDC'):
            if group.empty:
                continue
            
            # Latest record for basic info
            latest_record = group.loc[group['Effective Date'].idxmax()]
            
            profile = {
                "drug_id": f"ndc_{ndc}",
                "ndc": ndc,
                "drug_name": latest_record['NDC Description'],
                "classification": {
                    "brand_generic": latest_record['Classification for Rate Setting'],
                    "otc_rx": latest_record['OTC']
                },
                "pricing_data": {
                    "current_avg_cost": float(group['NADAC Per Unit'].mean()),
                    "30_day_moving_avg": float(group['Price_MA'].iloc[-1]) if 'Price_MA' in group.columns else None,
                    "price_volatility": float(group['Price_Volatility'].mean()) if 'Price_Volatility' in group.columns else None,
                    "min_price": float(group['NADAC Per Unit'].min()),
                    "max_price": float(group['NADAC Per Unit'].max()),
                    "price_trend": self._calculate_trend(group)
                },
                "data_quality": {
                    "record_count": len(group),
                    "date_range": {
                        "start": group['Effective Date'].min().strftime('%Y-%m-%d'),
                        "end": group['Effective Date'].max().strftime('%Y-%m-%d')
                    },
                    "completeness": float(1 - group['NADAC Per Unit'].isna().sum() / len(group)),
                    "last_updated": group['Effective Date'].max().strftime('%Y-%m-%d')
                }
            }
            
            drug_profiles.append(profile)
        
        return drug_profiles

# Data Foundation Agent using the Toolkit
class DataFoundationAgent(Agent):
    def __init__(self):
        # Create the toolkit instance
        self.data_toolkit = DataFoundationToolkit()
        
        super().__init__(
            name="DataFoundationAgent",
            model="microsoft/DialoGPT-small",  # Small, efficient model for processing tasks
            description="Cleans, normalizes and structures NADAC pharmaceutical data for analysis",
            tools=[self.data_toolkit]  # Pass the toolkit instance
        )
    
    def process_nadac_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main processing pipeline for NADAC data"""
        
        print("🧹 Starting data cleaning and normalization...")
        
        # Step 1: Clean and normalize data using toolkit methods
        df_clean = self.data_toolkit.clean_drug_names(df)
        df_clean = self.data_toolkit.normalize_pricing(df_clean)
        df_clean = self.data_toolkit.standardize_categories(df_clean)
        
        print(f"✅ Cleaned {len(df_clean)} records")
        
        # Step 2: Compute analysis features
        df_analyzed = self.data_toolkit.compute_time_features(df_clean)
        summary_stats = self.data_toolkit.compute_summary_stats(df_analyzed)
        
        print(f"📊 Computed features for {summary_stats['unique_drugs']} unique drugs")
        
        # Step 3: Structure output
        drug_profiles = self.data_toolkit.create_drug_profiles(df_analyzed)
        
        print(f"🗃️ Created {len(drug_profiles)} structured drug profiles")
        
        # Final output structure
        result = {
            "agent": "DataFoundationAgent",
            "timestamp": datetime.now().isoformat(),
            "summary_statistics": summary_stats,
            "drug_profiles": drug_profiles,
            "processing_status": {
                "total_input_records": len(df),
                "total_output_profiles": len(drug_profiles),
                "data_quality_score": summary_stats['data_quality']['completeness'],
                "processing_success": True
            }
        }
        
        print("✨ Data foundation processing complete!")
        return result