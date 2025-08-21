import os
import numpy as np
import pandas as pd
import itertools

PATH = os.path.join('data', 'handwriting-dataset')
test = pd.read_csv(os.path.join(PATH, 'written_name_test_v2.csv'))
print(list(test['FILENAME']))
print(test['IDENTITY'])

# res = "hell$ll$looo"

# print((res[0] + ''.join([res[i] for i in range(1, len(res)) if res[i] != res[i - 1]])).replace('$', ''))
# print(''.join(letter for letter, _ in itertools.groupby(res)).replace('$', ''))