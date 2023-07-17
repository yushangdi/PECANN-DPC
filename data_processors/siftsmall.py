import numpy as np

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    print(a[:100])
    n = a[0]
    return a[2:].reshape(n, -1).copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

path = '../data/siftsmall/siftsmall_base.bin'
output_path = '../data/siftsmall.data'
result = fvecs_read(path)

print(result)
with open(output_path, 'w') as f:
    for line in result:
        f.write(' '.join([str(num) for num in line]))
        f.write('\n')
