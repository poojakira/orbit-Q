import pandas as pd
from orbit_q import config

class FeatureProcessor:
    @staticmethod
    def process_telemetry(raw_data):
        # Convert dictionary from Firebase to DataFrame
        df = pd.DataFrame(raw_data).T if isinstance(raw_data, dict) else pd.DataFrame(raw_data) #
        df["timestamp"] = pd.to_datetime(df["timestamp"]) #
        df = df.sort_values("timestamp") #

        # Rolling Feature Engineering
        for face in df['face'].unique(): #
            mask = df['face'] == face #
            df.loc[mask, "rolling_mean"] = df.loc[mask, "distance_cm"].rolling(config.ROLLING_WINDOW).mean() #
            df.loc[mask, "rolling_std"] = df.loc[mask, "distance_cm"].rolling(config.ROLLING_WINDOW).std() #
        
        return df.dropna() #