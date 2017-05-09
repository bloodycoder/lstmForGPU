import sys
print(sys.argv[1])
f = open("./output/output.py","a")
f.write(sys.argv[1])
f.write('\n')
f.close()