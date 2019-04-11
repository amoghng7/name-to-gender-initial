import sys, os
from feature_extractor import FeatureExtract
import pandas as pd
import re

if not len(sys.argv) > 1:
	print("Please pass the filename that needs to be processed and the region\nRun the command: python %s <filename> <region>" %(str(sys.argv[0])))
	sys.exit()

if not len(sys.argv) > 2:
	print("Please pass the region\nRun the command: python %s <filename> <region>" %(str(sys.argv[0])))
	sys.exit()

if not os.path.isfile(sys.argv[1]) and str(sys.argv[1])[-4:] != '.csv' :
	print("The filename provided does not exist or is not a valid csv file. Please provide a valid file.")
	sys.exit()

updated_file_name = ''
file_name = str(sys.argv[1])
region = str(sys.argv[2])

if '-' in file_name:
	updated_file_name = 'cleaned-'+file_name
	print(updated_file_name)
else:
	updated_file_name = 'cleaned_'+file_name
	print()

update_file = open(updated_file_name, 'a+')

try:

	data = pd.read_csv(file_name)

	print(data.head(10))
	fe = FeatureExtract()
	# new_data = fe.extract(data)

	temp_data = data.dropna().loc[:,['name','gender']].values
	new_data = pd.DataFrame(columns=['name', 'gender'])
	count = 0
	name, gender = '', ''

	for i in temp_data:
		try:
			name, gender = i[0].strip(), i[1].strip()
			name_list = re.split(r'[.@\s]', name)

			if len(name_list) > 1:
				print("Original Name: %s, Gender: %s\n" %(name, gender))
				# print(name_list)

				for nm in name_list:
					print("%d: '%s'" %(name_list.index(nm)+1, nm), end=' ')
				inp = input('\n: ').split(',')

				exp = False

				while(not exp):
					try:
						if inp[0] == '0':
							print("%s is ignored!\n" %(name))
							break

						elif len(inp) == 1:
							print("%s: %s" %(name_list[int(inp[0])-1],gender))
							new_data.loc[count] = [name_list[int(inp[0])-1], gender]
							count+=1

						elif inp[1].strip().isnumeric():
							for j in range(0,len(inp)):
								print("%s: %s" %(name_list[int(inp[j])-1],gender), end=' ')
								new_data.loc[count] = [name_list[int(inp[j])-1],gender]
								count+=1
							print('\n')

						else:
							for j in range(0,len(inp),2):
								print("%s: %s" %(name_list[int(inp[j])-1],inp[j+1]), end=' ')
								new_data.loc[count] = [name_list[int(inp[j])-1],inp[j+1]]
								count+=1
							print('\n')
						exp = True
					except:
						q = input("Looks like you have accidentally entered a wrong input\nWanna try again(y) or skip(n)? ")
						if q.lower() == 'y':
							exp = False
						else:
							exp = True
			elif len(name_list) == 1:
				try:
					print("Name: %s, Gender: %s\n" %(name, gender))
					new_data.loc[count] = [name, gender]
					count+=1
				except Exception as e:
					print("Exception! "+e)


		except Exception as e:
			try:
				print("Exception for %s" %(name))
			except:
				print('Exception!')

	print(new_data.head(10))

except Exception as e:
	print("Exception "+e)
	print("count: "+str(count))

new_data_list = fe.extract(new_data, region)

for i in new_data_list:
	update_file.write(i)
update_file.close()