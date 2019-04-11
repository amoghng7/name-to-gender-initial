import requests 
from bs4 import BeautifulSoup
import time
import string
import re

file = open("indiannames.csv","a+")
gender_list = {'boy':'M', 'girl':'F'}
alphabets = list(string.ascii_lowercase)
file.write("Name, Gender\n")
errors = []

for gender_url, gender in gender_list.items():
	print("\t\t\tGetting all %ss names\n\n" %(gender_url))
	for alpha in alphabets:
		print("Getting names that starts with %s:" %(alpha.upper()))
		URL = "http://www.indianhindunames.com/indian-hindu-%s-name-%s.htm" %(gender_url, alpha)
		print("Using URL: ", URL)
		r = requests.get(URL)
		soup = BeautifulSoup(r.content, 'html5lib')
		try:
			names_left = soup.find("div", id="bodyleft").find_all("p")
			names_right = soup.find("div", id="bodyright").find_all("p")
			names = names_left + names_right
		except Exception as e:
			errors.append(e)

		# print(names)
		try:
			# 
			for name in names:
				try:
					# 
					for p in name.contents:
						try:
							# 
							contents = str(p).split('\n')
							for content in contents:
								try:
									# 
									final_name = str(content).strip()
									if re.match(r'^[a-zA-Z]', final_name) and not '_' in final_name and final_name.strip() != '<br/>':
										first_name = final_name.replace(',', '').strip().split()[0]
										print("|Name: %s\t\t| Gender: %s\t|" %(first_name, gender))
										file.write("%s,%s\n" %(first_name, gender))
								except Exception as e:
									errors.append(e)
						except Exception as e:
							errors.append(e)
				except Exception as e:
					errors.append(e)
		except Exception as e:
			errors.append(e)
		# print(t)
		print()
		time.sleep(60)

file.close()

print("Errors encountered: ")
for error in errors:
	print(error)