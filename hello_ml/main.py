import random 
train = [
    [0, 0],
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8]
]

def random_num (end: float = 10.0):
    return round(random.uniform(0, end), 1);

def cost(w:float, b: float):
    result: float = 0.0;
    for i in range(len(train)):
        x:float = train[i][0]
        y_hat:float = x * w + b ;
        y: float = train[i][1]
        distance: float = y - y_hat
        result += distance * distance
    result /= len(train)

    return result 
#moduel formula is going to be for this y = x * w 
if __name__ == "__main__":
    w:float = random_num()
    b: float = random_num(end = 5.0)
    esp: float = 1e-3
    learning_rate: float = 1e-3
    epocs: int = 4000;
    for i in range(epocs):

        distance_w = (cost(w + esp,b) - cost(w,b)) / esp
        distance_b = (cost(w, b + esp) - cost(w,b)) / esp

        w -= distance_w * learning_rate
        b -= distance_b * learning_rate
        print(f"cost= {cost(w,b)}, w= {w}, b= {b}")
    print("---"*25)

    print("weightage: ", w)
    print("bias: ", b)
