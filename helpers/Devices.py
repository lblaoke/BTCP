from xml.etree.cElementTree import parse
import torch

#find device with most free memory
def free_device_id(path):
	assert torch.cuda.is_available(),'Cuda not supported'

	#extract xml tree
	try:
		xml_root = parse(path).getroot()
		device_id = 0
		max_free_memory = 0
		num_device = int(xml_root[3].text)
	except:
		return 0

	assert num_device>0,'No available device found'

	#find the most available one
	try:
		for _id in range(num_device):
			xml_text = xml_root[4+_id][24][2].text
			free_memory = int(xml_text.split()[0])

			if free_memory>max_free_memory:
				device_id = _id
				max_free_memory = free_memory
	except:
		return 0

	assert max_free_memory>1024,'Video memory is %d MB, smaller than 1024 MB' % max_free_memory

	return device_id
