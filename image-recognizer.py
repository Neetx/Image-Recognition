from argparser import argvcontrol

def main():
	args, check= argvcontrol()
	if check:
		if args.image:
			print ("c'e")
		else:
			print ("non c'e")
	else:
		print ("Usage: ")

if __name__ == "__main__":
	main()