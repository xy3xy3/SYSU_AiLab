def ReverseKeyValue(dict1):
    dict2 = {}
    for key, value in dict1.items():
        dict2[value] = key
    return dict2
if __name__ == "__main__":
    dict1 = {"a": 1, "b": 2, "c": 3}
    print(ReverseKeyValue(dict1))
    