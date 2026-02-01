lef = {}
rig = {}
rellef = {}
relrig = {}

triple = open("train.txt", "r")
valid = open("valid.txt", "r")
test = open("test.txt", "r")

# Read all lines from train.txt
train_lines = triple.readlines()
for i in range(len(train_lines)):
	content = train_lines[i]
	h,t,r = content.strip().split()
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

# Read all lines from valid.txt
valid_lines = valid.readlines()
for i in range(len(valid_lines)):
	content = valid_lines[i]
	h,t,r = content.strip().split()
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

# Read all lines from test.txt
test_lines = test.readlines()
for i in range(len(test_lines)):
	content = test_lines[i]
	h,t,r = content.strip().split()
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

test.close()
valid.close()
triple.close()

f = open("type_constrain.txt", "w")
f.write("%d\n"%(len(rellef)))
for i in rellef:
	f.write("%s\t%d"%(i,len(rellef[i])))
	for j in rellef[i]:
		f.write("\t%s"%(j))
	f.write("\n")
	f.write("%s\t%d"%(i,len(relrig[i])))
	for j in relrig[i]:
		f.write("\t%s"%(j))
	f.write("\n")
f.close()

rellef = {}
totlef = {}
relrig = {}
totrig = {}
# lef: (h, r)
# rig: (r, t)
for i in lef:
	if not i[1] in rellef:
		rellef[i[1]] = 0
		totlef[i[1]] = 0
	rellef[i[1]] += len(lef[i])
	totlef[i[1]] += 1.0

for i in rig:
	if not i[0] in relrig:
		relrig[i[0]] = 0
		totrig[i[0]] = 0
	relrig[i[0]] += len(rig[i])
	totrig[i[0]] += 1.0

s11=0
s1n=0
sn1=0
snn=0
f = open("test.txt", "r")
test_lines_for_stats = f.readlines()
f.close()

for i in range(len(test_lines_for_stats)):
	content = test_lines_for_stats[i]
	h,t,r = content.strip().split()
	rign = rellef[r] / totlef[r]
	lefn = relrig[r] / totrig[r]
	if (rign < 1.5 and lefn < 1.5):
		s11+=1
	if (rign >= 1.5 and lefn < 1.5):
		s1n+=1
	if (rign < 1.5 and lefn >= 1.5):
		sn1+=1
	if (rign >= 1.5 and lefn >= 1.5):
		snn+=1


f = open("test.txt", "r")
test_lines_for_output = f.readlines()
f.close()

f11 = open("1-1.txt", "w")
f1n = open("1-n.txt", "w")
fn1 = open("n-1.txt", "w")
fnn = open("n-n.txt", "w")
fall = open("test_all.txt", "w")
tot = len(test_lines_for_output)
fall.write("%d\n"%(tot))
f11.write("%d\n"%(s11))
f1n.write("%d\n"%(s1n))
fn1.write("%d\n"%(sn1))
fnn.write("%d\n"%(snn))
for i in range(tot):
	content = test_lines_for_output[i]
	h,t,r = content.strip().split()
	rign = rellef[r] / totlef[r]
	lefn = relrig[r] / totrig[r]
	if (rign < 1.5 and lefn < 1.5):
		f11.write(content)
		fall.write("0"+"\t"+content)
	if (rign >= 1.5 and lefn < 1.5):
		f1n.write(content)
		fall.write("1"+"\t"+content)
	if (rign < 1.5 and lefn >= 1.5):
		fn1.write(content)
		fall.write("2"+"\t"+content)
	if (rign >= 1.5 and lefn >= 1.5):
		fnn.write(content)
		fall.write("3"+"\t"+content)
fall.close()
f11.close()
f1n.close()
fn1.close()
fnn.close()
