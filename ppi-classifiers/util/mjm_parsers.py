import subprocess, re, gzip

def parse_dictionary_list(filename, header=None, delim='\t', max_keys=None, max_lines=None, gzipped=False):
	'''Uses header row in file as keys for dictionaries'''
	
	if gzipped==True:
		handle = gzip.open(filename)
	else:
		handle = open(filename, 'r')
	
	if header == None:
		header_keys = handle.readline().strip().split(delim)
	else:
		header_keys = header
	
	if max_keys != None:
		header_keys = header_keys[:max_keys]
	
	data = []
	line_num = 0
	for l in handle:
		
		if max_lines != None and line_num == max_lines: break
		
		line = l.replace('\n', '').replace('\r', '').split(delim)
		cur_dict = {}
		for i, key in enumerate(header_keys):
			cur_dict[key] = line[i]
		data.append(cur_dict)
		
		line_num += 1
	
	return data


def unzip_res_range(res_range):
	'''Converts ranges in the form: [2-210] or [3-45,47A,47B,51-67] into lists of strings including all numbers in these ranges in order'''
	res_ranges = res_range.strip()[1:-1].split(',')
	index_list = []
	for r in res_ranges:
		if re.match('.+-.+', r):
			a, b = r.split('-')
			index_list += [str(n) for n in range(int(a), int(b)+1)]
		else:
			index_list.append(r)
	
	if index_list == ['']:
		return []
	else:
		return index_list


def zip_res_range(seq):
	'''Converts lists of residues as string in the form "1,2,2E,3,4,5,6,46,67,68,68A,68C,69,70" to
	zipped ranges in the form "[1-2,2E,3-6,46,67-68,68A,68C,69-70]" @author: haoran'''
	
	if type(seq) == list:
		seq = ','.join(seq)
	
	seqout = []
	tempst = ''
	for h in range(seq.count(',')+1):
		i = seq.split(',')[h]
		if i.isdigit() and h < seq.count(',') and str(int(i)+1) == seq.split(',')[h+1]:
			if tempst == '':
				tempst = i
		else:
			if tempst == '':
				seqout.append(i)
			else:
				seqout.append(tempst+'-'+i)
				tempst = ''
	return '['+','.join(seqout)+']'
	
	
def parse_fasta(file_name):
	
	cur_key = ''
	fasta_dict = {}
	keys_ordered = []
	for line in open(file_name, 'r'):
		if line[0] == '>':
			cur_key = line.strip().replace('>', '').split()[0]#.split('_')[0]
			keys_ordered.append(cur_key)
			fasta_dict[cur_key] = ''
		else:
			fasta_dict[cur_key] += line.strip()
			
	return keys_ordered, fasta_dict


def write_fasta(fasta_dict, file_name):
	'''Write contents of a dictionary mapping names to sequences to a fasta file.'''
	output = open(file_name, 'w')
	for k, v in sorted(fasta_dict.items()):
		output.write('>%s\n%s\n' %(k, v))
	output.close()
	
	
def parse_fastq(file_name):
	
	cur_key = ''
	fastq_dict = {}
	i = -1
	keys_ordered = []
	for line in open(file_name, 'r'):
		i += 1
		if i%4==0:
			cur_key = line.strip().replace('@', '').split()[0]
			fastq_dict[cur_key] = {'seq': '', 'qual': ''}
			keys_ordered.append(cur_key)
		elif i%4==1:
			fastq_dict[cur_key]['seq'] = line.strip()
		elif i%4==3:
			fastq_dict[cur_key]['qual'] = line.strip()
			
	return keys_ordered, fastq_dict
	
	
def parse_sam(sam):
	''' Save reference match with highest match quality to read. '''
	
	sam_dict = {}
	
	for line in open(sam, 'r'):
		l = line.split('\t')
		read_id = l[0].split()[0]
		if read_id in sam_dict and int(sam_dict[read_id]['match_quality']) > int(l[4]): continue
		try:
			sam_dict[read_id] = {'strand': l[1], 'match': l[2], 'fasta_index': l[3], 'match_quality': l[4], 'cigar': l[5]}
		except IndexError:
			continue
		
	return sam_dict



