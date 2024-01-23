import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # load data
    df = pd.read_csv('bo_stream_example.csv')

    print(df.head())
    print(df.info())
    print(df.describe())

    # line plot of the data
    a = sns.lineplot(data=df)
    a.get_legend().remove()
    # scatter plot of the data
    b = sns.scatterplot(data=df)
    b.get_legend().remove()

    # save the plot
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig('bo_stream_example.png')