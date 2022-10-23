import itertools

lst = set(itertools.permutations(["0", "1", "1", "3", "3", "4", "4", "4", "2"], 9))
count = len(lst)
b = []
for a in lst:
    if a[0] == '0' or a[0] == '4' or int(a[8]) % 2 == 0:
        count -= 1
    else:
        b.append(a)
