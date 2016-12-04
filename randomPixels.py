from random import randint


def gencoordinates(mx, nx, my, ny):
    seen = set()
    x, y = randint(mx, nx), randint(my, ny)

    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(mx, nx), randint(my, ny)
        while (x, y) in seen:
            x, y = randint(mx, nx), randint(my, ny)


if __name__ == '__main__':
    g = gencoordinates(1, 100, 1, 100)
    print next(g)
