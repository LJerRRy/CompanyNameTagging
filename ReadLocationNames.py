import re

locations = []
with open('LocationNames.txt', 'r', encoding='utf8') as data:
    for line in data.readlines():
        if re.match(r'\d+\.', line) is not None:
            province = locations.append(line.split('.')[1].split()[0])
        elif re.match(r'\d{4}.+。', line) is not None:
            continue
        else:
            locations += line.split()

# print(len(locations))
