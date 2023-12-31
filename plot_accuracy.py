import argparse
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(args):
    plt.figure(figsize=(8, 6))

    col_name = args[0]

    for i in range(1, len(args), 2):
        filename = args[i]
        label = args[i + 1]

        # Read the CSV file
        df = pd.read_csv(filename)

        # Extract the specified column data
        column_data = df[col_name]

        # Plot the data with the given label
        plt.plot(column_data, label=label)

    # Add labels and legend
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a line plot from CSV files.")
    parser.add_argument("params", nargs="+", help="""<type> <filename> <label> <filename> <label> ...
                        <type> is the column (train, valid, ood), filename the  name of the csv files with those columns.""")
    args = parser.parse_args().params

    # Check if the number of arguments is valid
    if (len(args) + 1) % 2 != 0:
        print("Invalid number of arguments. Each file should have a corresponding type and label.")
        parser.print_help()
    else:
        plot_data(args)