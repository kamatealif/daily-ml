import random 
train:float = [
    [0, 0],
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8]
]

def random_num ():
    return round(random.uniform(0, 10.0), 1);
#moduel formula is going to be for this y = x * w 
if __name__ == "__main__":
    random.seed(69)
    w:float = random_num()
    for i in range(len(train)):
        x = train[i][0]
        y = x * w;
        print(f"Actual: {y}, expected: {train[i][1]}")

