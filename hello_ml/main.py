import random 
train = [
    [0, 0],
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8]
]

def random_num ():
    return round(random.uniform(0, 10.0), 1);

def cost(w:float):
    result: flaot = 0.0;
    for i in range(len(train)):
        x:float = train[i][0]
        y_hat:float = x * w;
        y: float = train[i][1]
        distance: float = y - y_hat
        result = distance * distance
    result /= len(train)

    return result 

#moduel formula is going to be for this y = x * w 
if __name__ == "__main__":
    random.seed(69)
    w:float = random_num()
    esp: float = 1e-3
    learning_rate: float = 1e-3;
    epocs: int = 1000;
    for i in range(epocs):

        distance_cost = (cost(w + esp) - cost(w)) / esp

        w -= distance_cost * learning_rate
        print(cost(w))

    print("weightage: ", w)
