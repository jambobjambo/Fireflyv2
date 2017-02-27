lines = []
with open("./RawData/sampleall.txt") as f:
	content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content]
	for x in content:
		line = x.split('"')
		if len(line) > 1 and len(line) < 4:
			to_add = line[0]
			mid = line[1].split(',')
			new_mid = mid[0] + ':' + mid[1]
			end = line[2]
			new = to_add + new_mid + end
			lines.append(new)
		if len(line) > 4 and len(line) < 6:
			to_add = line[0]
			mid1 = line[1].split(',')
			new_mid1 = mid1[0] + ':' + mid1[1]
			mid2 = line[3].split(',')
			new_mid2 = mid1[0] + ':' + mid1[1]
			end = line[2]
			new = to_add + new_mid1 + line[2] + new_mid2 + end
			lines.append(new)
		else:
			lines.append(x)

file = open("./RawData/training_data.txt","w")
for l in lines:
	l = l.split(",")
	file.write(str(l[0]) + ',' + str(l[1]) + ',' + str(l[2]) + ',' + str(l[3]) + ',' + str(l[4]) + ',' + str(l[5]) + ',' + str(l[6]) + ',' + str(l[7]) + '\r')
