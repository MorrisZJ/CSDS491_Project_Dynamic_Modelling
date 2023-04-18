
def is_same_cluster(cluster, precise_point, prediction_point):
    for index, element in enumerate(cluster):
        if prediction_point in element and precise_point in element:
            return True
    return False

def is_neighbor(size, i, j):
    row_i, col_i = i // size, i % size
    row_j, col_j = j // size, j % size
    if abs(row_i - row_j) == 0 and abs(col_i - col_j) != 0:
        return True
    elif abs(row_i - row_j) != 0 and abs(col_i - col_j) == 0:
        return True
    elif i == j:
        return True
    else:
        return False

def check_prediction(size, cluster, precise_point, prediction_point):
    if is_same_cluster(cluster, 
                       precise_point, 
                       prediction_point) and is_neighbor(size, 
                                                         precise_point, 
                                                         prediction_point):
        return True
    else:
        return False
    
def test(size, cluster, y_test, prediction):
    counter = 0
    counter_precise = 0
    for index, element in enumerate(y_test):
        predict = prediction[index]
        actual = element['path'][-1]
        if check_prediction(size, cluster, actual, predict):
            counter += 1
        if actual == predict:
            counter_precise += 1
    counter = counter / len(y_test)
    counter_precise = counter_precise / len(y_test)
    print(f"The precise accuracy is {counter_precise}")
    print(f"The MCL accuracy is {counter}")