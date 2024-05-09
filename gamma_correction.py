import numpy as np

def gamma_correction(matrix, gamma):
    max_intensity = np.max(matrix)

    table = (max_intensity * ((np.arange(0, max_intensity + 1) / max_intensity) ** gamma)).astype("uint8")

    new_matrix = np.zeros_like(matrix, dtype=np.uint8)
    for i in range(max_intensity + 1):
        new_matrix[matrix == i] = table[i]

    return new_matrix, table

def display_lookup_table(table):
    print("\nLookup Table:")
    for i in range(len(table)):
        if i % 16 == 0:
            print("\n")
        print(f"{table[i]:<4}", end=" ")

if __name__ == "__main__":
    input_matrix = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])

    gamma_value = float(input("Enter the gamma value: "))
    new_matrix, lookup_table = gamma_correction(input_matrix, gamma_value)
    display_lookup_table(lookup_table)


    print("\nOriginal Matrix:")
    print(input_matrix)

    print("\nCorrected Matrix:")
    print(new_matrix)
