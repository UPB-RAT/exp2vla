import pyarrow.parquet as pq
pf = pq.ParquetFile("/home/summer_school/summer_ws/Dataset/vla-drone-v0.1/data/chunk-000/file-000.parquet")
print(pf.metadata)