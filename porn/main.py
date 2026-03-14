from collections import Counter 
text = "Return a string containing a printable representation of an object. For many types, this function makes an attempt to return a string that would yield an object with the same value when passed to eval(); otherwise, the representation is a string enclosed in angle brackets that contains the name of the type of the object together with additional information often including the name and address of the object. A class can control what this function returns for its instances by defining "
def indexer(text):
    hashmap = Counter()
    for i in range(len(text)-1):
        key = f"{text[i]}{text[i+1]}"
        hashmap[key] += 1
    return hashmap
result = indexer(text)
print(result)
