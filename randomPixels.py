from random import randint

def gencoordinates(m, n):
	seen = set()
	x, y = randint(m, n), randint(m, n)

	while True:
		seen.add((x, y))
		yield(x,y)
		x, y = randint(m, n), randint(m, n)
		while (x,y) in seen:
			x, y = randint(m,n), randint(m,n)

g = gencoordinates(1, 100)
print next(g)
