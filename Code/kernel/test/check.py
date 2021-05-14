
fp = open("test/log.txt", 'r')
content = fp.read()

lp = 0
maxlp = 10

for ch in content:
    if ch == '(':
        lp += 1
    elif ch == ')':
        lp -= 1
    assert 0 <= lp <= maxlp

print("pass test!")

