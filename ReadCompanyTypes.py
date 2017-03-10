def read_company_types():
    company_types = []

    with open('CompanyTypes.txt', 'r', encoding='utf-8') as data:
        for line in data.readlines():
            company_types.append(line.split()[0])

    return company_types

# print(company_types)
