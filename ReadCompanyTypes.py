company_types = []

with open('CompanyTypes.txt', 'r', encoding='utf-8') as data:
    for line in data.readlines():
        company_types.append(line.split()[0])

# print(company_types)
