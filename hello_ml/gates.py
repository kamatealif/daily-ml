import random 

# OR Gate
train = [
 [0, 0, 0],
 [1, 0, 1],
 [0, 1, 1],
 [1, 1, 1]
]

train_count = len(train)

def sigmoid(x:float):
    return 1 / (1 + math.exp(-x))

def cost_func(w1, w2):
    result = 0
    for row in train:
        x1: float = row[0]
        x2: float = row[1]
        y: float = row[2]
        y_hat = sigmoid(x1*w1 + x2*w2)
        d = y - y_hat
        result += d * d 
        
    result /= train_count
    return result 

def random_num (end: float = 1.0):
    return random.uniform(0, end)


if __name__ == "__main__":
    random.seed(69)
    w1:float = random_num()
    w2:float = random_num()
    
    esp: float = 1e-3
    rate: float = 1e-3
    for i in range(2000):
        cost = cost_func(w1, w2)
        print(f"w1 : {w1}, w2 : {w2}, cost = {cost}")
        
        dw1: float = (cost_func(w1 + esp, w2) - cost) / esp
        dw2: float = (cost_func(w1, w2 + esp) - cost) / esp
        
        w1 -= rate * dw1
        w2 -= rate * dw2