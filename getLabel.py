def get_label(filepath) -> list:
    names = ['apple', 'banana', 'beef', 'blueberries', 'bread', 'butter', 'carrot', 'cheese', 'chicken', 'chicken_breast', 'chocolate', 'corn', 'eggs', 'flour', 'goat_cheese', 'green_beans', 'ground_beef', 'ham', 'heavy_cream', 'lime', 'milk', 'mushrooms', 'onion', 'potato', 'shrimp', 'spinach', 'strawberries', 'sugar', 'sweet_potato', 'tomato']
    labels = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            label_num = int(line.split(" ")[0])
            label = names[label_num]
            labels.append(label)
    return list(set(labels))
