excel_file_path = 'F:\Datasets\CA\children touch dataset\Dataset\id-gender-agegroup.csv'


def read_excel(file_path):
    import csv
    result = []
    with open(file_path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(rows) # skips the header
        for row in rows:
            result.append(row[0].split(','))
    return result


result = read_excel(excel_file_path)
print()
