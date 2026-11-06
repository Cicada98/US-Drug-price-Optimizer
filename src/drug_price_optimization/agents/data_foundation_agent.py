import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from agno.agent import Agent
from agno.tools import Toolkit
from drug_price_optimization.data.ndc_parser import NdcParts, add_parsed_columns
import json
import warnings
warnings.filterwarnings('ignore')

PRODUCT_COLUMNS = list(NdcParts.__dataclass_fields__)

BRAND_PRICE = 'NADAC Per Unit'
GENERIC_PRICE = 'Corresponding Generic Drug NADAC Per Unit'
GENERIC_DATE = 'Corresponding Generic Drug Effective Date'


def _to_json_value(value):
    """Convert pandas/numpy scalars (and NA) into plain JSON-friendly Python values."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, 'item') else value

class DataFoundationToolkit(Toolkit):
    """Toolkit for cleaning, analyzing and structuring NADAC pharmaceutical data"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="data_foundation_toolkit",
            tools=[
                self.clean_drug_names,
                self.normalize_pricing,
                self.standardize_categories,
                self.compute_generic_premium,
                self.compute_time_features,
                self.compute_summary_stats,
                self.create_drug_profiles
            ],
            **kwargs
        )
    
    def clean_drug_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split NDC descriptions into drug name, release type, strength and dosage form.

        See ``drug_price_optimization.data.ndc_parser`` for the parsing rules.
        """
        df = df.copy()
        df['NDC Description'] = df['NDC Description'].str.strip().str.replace(r'\s+', ' ', regex=True)
        return add_parsed_columns(df, 'NDC Description')
    
    def normalize_pricing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize brand and generic prices and fill gaps from each NDC's own history"""
        df = df.copy()

        for col in (BRAND_PRICE, GENERIC_PRICE):
            if col not in df.columns:
                continue
            # Convert to numeric, removing $ signs
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace('$', '', regex=False), errors='coerce')

            # Zero / near-zero prices are data errors. There is no upper cap: expensive
            # products are real (e.g. FORTEO pens are ~$1,800 per unit).
            df.loc[df[col] < 0.001, col] = np.nan

            # Fill gaps with the median price of the same NDC (same product, same strength)
            df[col] = df.groupby('NDC')[col].transform(lambda x: x.fillna(x.median()))

        return df
    
    def standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical variables for NADAC dataset"""
        df = df.copy()

        # Keep the raw NADAC class: B-ANDA (authorized generic) is benchmarked separately
        # from plain brands (B), so it must survive the Brand/Generic mapping below.
        df['Rate_Setting_Class'] = df['Classification for Rate Setting'].astype(str).str.strip()

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


    def compute_generic_premium(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare each brand price with its matched generic price on the same row.

        Adds:
            Premium_Per_Unit  - brand minus generic, in $ per pricing unit
            Premium_Ratio     - brand / generic (12.0 = brand costs 12x the generic)
        """
        df = df.copy()
        df['Premium_Per_Unit'] = df[BRAND_PRICE] - df[GENERIC_PRICE]
        df['Premium_Ratio'] = df[BRAND_PRICE] / df[GENERIC_PRICE].where(df[GENERIC_PRICE] > 0)
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
            
            # Raw NADAC class split (B vs B-ANDA), counted per NDC
            'rate_setting_class_split': (
                df.drop_duplicates('NDC')['Rate_Setting_Class'].value_counts().to_dict()
                if 'Rate_Setting_Class' in df.columns else {}
            ),

            # Brand-vs-generic premium on each NDC's latest record
            'generic_premium': self._premium_summary(df),

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
    
    def _premium_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Median premium ratio overall and per rate-setting class (B vs B-ANDA)."""
        if 'Premium_Ratio' not in df.columns:
            return {}
        latest = df.sort_values('Effective Date').groupby('NDC').tail(1)
        by_class = (
            latest.groupby('Rate_Setting_Class')['Premium_Ratio'].median().to_dict()
            if 'Rate_Setting_Class' in latest.columns else {}
        )
        return {
            'median_premium_ratio': _to_json_value(latest['Premium_Ratio'].median()),
            'median_premium_ratio_by_class': {k: float(v) for k, v in by_class.items()},
            'ndcs_with_generic_price': int(latest[GENERIC_PRICE].notna().sum()),
            'ndcs_brand_below_generic': int((latest['Premium_Per_Unit'] < 0).sum()),
        }

    def _calculate_trend(self, group: pd.DataFrame, column: str = BRAND_PRICE) -> str:
        """
        Calculate the trend direction of ``column`` for a single drug group in the NADAC dataset.

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
        recent_prices = group_sorted[column].tail(5).mean()
        older_prices = group_sorted[column].head(5).mean()
        
        # Percent change
        change_pct = (recent_prices - older_prices) / older_prices if older_prices > 0 else 0
        
        # Determine trend
        if change_pct > 0.05:
            return "increasing"
        elif change_pct < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _generic_pricing(self, group: pd.DataFrame, latest: pd.Series) -> Optional[Dict[str, Any]]:
        """Brand-vs-generic comparison for one NDC, using its most recent record."""
        if 'Premium_Ratio' not in group.columns or pd.isna(latest.get(GENERIC_PRICE)) or pd.isna(latest[BRAND_PRICE]):
            return None

        brand, generic = float(latest[BRAND_PRICE]), float(latest[GENERIC_PRICE])
        generic_date = pd.to_datetime(latest.get(GENERIC_DATE), errors='coerce')
        return {
            "brand_nadac_per_unit": brand,
            "generic_nadac_per_unit": generic,
            "generic_effective_date": None if pd.isna(generic_date) else generic_date.strftime('%Y-%m-%d'),
            "premium_per_unit": brand - generic,
            "premium_ratio": _to_json_value(latest['Premium_Ratio']),
            # % the brand price would have to drop to match the generic (negative = already below it)
            "cut_to_generic_parity_pct": (brand - generic) / brand * 100 if brand > 0 else None,
            "brand_below_generic": brand < generic,
            "avg_premium_ratio": _to_json_value(group['Premium_Ratio'].mean()),
            "premium_ratio_trend": self._calculate_trend(group, 'Premium_Ratio'),
            "generic_price_trend": self._calculate_trend(group, GENERIC_PRICE),
        }

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
                "product": {
                    col.lower(): _to_json_value(latest_record[col])
                    for col in PRODUCT_COLUMNS if col in latest_record.index
                },
                "classification": {
                    "brand_generic": latest_record['Classification for Rate Setting'],
                    "rate_setting_class": latest_record.get('Rate_Setting_Class'),  # "B" or "B-ANDA"
                    "otc_rx": latest_record['OTC']
                },
                "generic_pricing": self._generic_pricing(group, latest_record),
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
        df_clean = self.data_toolkit.compute_generic_premium(df_clean)

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