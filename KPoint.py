from random import randint

def cxKPoint(ind1, ind2, k):
	print(f'before crossover \nindiviual 1: {ind1}, indiviual 2 {ind2}')
	size = min(len(ind1), len(ind2))
	if k > size:
		k = size
	cxpoints = []
	while(len(cxpoints)<k):
		num = randint(0, size)
		if num not in cxpoints:
			cxpoints.append(num)
	cxpoints.sort()
	print(f'k crossover indices {cxpoints}')
	for i in range(len(cxpoints)):
		p1 = cxpoints[i]
		ind1[p1::], ind2[p1::] = ind2[p1::], ind1[p1::]
	print(f'after crossover \nindiviual 1: {ind1}, indiviual 2 {ind2}')
	return ind1, ind2

ind1 = [6, 2, 4, 2, 1, 6, 9, 2]
ind2 = [8, 3, 6, 3, 5, 1, 8, 3]
cxKPoint(ind1, ind2, 4)