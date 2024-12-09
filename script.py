import pandas

TITANIC: pandas.DataFrame = pandas.read_csv(
    "./titanic.csv",
    dtype_backend="pyarrow",
)
