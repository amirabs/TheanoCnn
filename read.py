f = open('temp_value', 'r+')
while(1):
	line=f.readline()
	line2=f.readline()
	if not line:
		break
	line=line.replace("\n","")

	values = [float(i) for i in line.split()]
	value= float(line2)

	print values
	print value
