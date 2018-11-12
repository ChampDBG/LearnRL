## packages
import numpy as np

## define
def func1(r, xs):
    '''
    r for radius
    xs for a list with x
    '''
    val = []
    for x in xs:
        if type(x) == int or float:
            x = np.array([x,1])
        elif type(x) == list:
            x = np.array(x.append(1))
        normX = norm2(x)
        val.append(x*r/normX)
    return val

def func2(xs):
    val = []
    for x in xs:
        if type(x) == int or float:
            x = np.array( [ x, pow(norm2(x),2) ] )
        elif type(x) == list:
            x = np.array(x.append(pow(norm2(x),2)))

def norm2(xs):
    xsPow = []
    if xs == int or float:
        xsPow.append(pow(xs,2))
    
    elif xs == list and len(xs) > 0:
        for x in xs:
            xsPow.append(pow(x,2))
    
    norm = np.sqrt(sum(xsPow))
    return norm

## main
def main(radius, xs):
    ys = func1(radius, xs)

if __name__ == '__main__':
    radius = 3
    xs = [x for x in range(10)]
    ys = np.array(func(radius, xs))
    print(ys)
    if ys.shape[1] == 2:
        import matplotlib.pyplot as plt
        plt.scatter(ys[:,0], ys[:,1])
        plt.plot(ys[:,0], ys[:,1])
        plt.show()