import sys, getopt

def parse_arguments(argv):
	model=''
	try:
		opts, args = getopt.getopt(argv, "hc:g:i:n:p:m:", ["image_class=","gpu=","input=","net=","probe=","model="])	
	except getopt.GetoptError:
		print('-i <input image type> -n <network> -p <probe> -m <model>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('-n <network> -p <probe> -m <model>')
			sys.exit()
		elif opt in ("-c", "--image_class"):
			img_class = arg
		elif opt in ("-g", "--gpu"):
			gpu = arg
		elif opt in ("-i", "--input"):
			image_type = arg
		elif opt in ("-n", "--net"):
			net = arg
		elif opt in ("-p", "--probe"):
			probe = arg
		elif opt in ("-m", "--model"):
			model = arg
		else:
			print('Error, wrong optional argument.')	
			print('-n <network> -p <probe> -m <model>')
			sys.exit(1)
	return img_class, gpu, image_type, net, probe, model
				
