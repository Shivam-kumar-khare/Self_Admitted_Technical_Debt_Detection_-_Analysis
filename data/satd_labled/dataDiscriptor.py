import pandas as pd


df=pd.read_csv("./data/satd_labled/v5.3.0_satd.csv")
print("shape=",df.shape)
print("\n10 sample=",df.sample(10))
print("\nInfo=",df.info())
print("\nDescribe",df.describe())