rows = {}

with open('merged_file.csv') as f:
    for line in f:
        entities = line.split(',')
        if len(entities) >= 3:
            id_ = entities[0]
            label = entities[1]
            explanation = ''.join(entities[2:])
            if id_ not in rows:
                rows[id_] = [(label, explanation)]
            else:
                rows[id_].append((label, explanation))

cnts = {}
for l in rows.values():
    if len(l) not in cnts:
        cnts[len(l)] = 0
    cnts[len(l)] += 1

print(f'We have the following distribution of id occurence: {cnts}')

for id_, row in rows.items():
    if len(row) > 1:
        labels, explanations = zip(*row)

        if not all(x == labels[0] for x in labels):
            print(f'For row {id_} there are the following explanations:')
            print(explanations)
            print("Mismatching labels!!!")
            print(labels)
            print()
            print()
