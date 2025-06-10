from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

# read FASTA query sequence
record = SeqIO.read("example_sequence.fasta", "fasta")
print("Query sequence ID:", record.id)
print("Query sequence length:", len(record.seq))

# submit to NCBI
print("Submitting BLAST search to NCBI...")
handle = NCBIWWW.qblast("blastp", "nr", record.seq)

print("BLAST search submitted. Waiting for results...")
with open("blastp_ncbi.xml", "w") as out:
  out.write(handle.read())
handle.close()
print("Results saved to blastp_ncbi.xml")

'''
# parsing .XML results
print("Parsing BLAST results...")
handle = open("blastp_ncbi.xml")
blast_record = NCBIXML.read(handle)
for alignment in blast_record.alignments:
  print(alignment.hit_def)
  for hsp in alignment.hsps:
    print("****Alignment****")
    print("sequence:", alignment.title)
    print("length:", alignment.length)
    print("e value: {:.3g}".format(hsp.expect))
    print("score:", hsp.score)
    print("identities:", hsp.identities)
    print("gaps:", hsp.gaps)
    print("query start:", hsp.query_start)
    print("query end:", hsp.query_end)
    # print("match start:", hsp.match_start)
    print("match end:", hsp.match_end)
    print("subject start:", hsp.sbjct_start)
    print("subject end:", hsp.sbjct_end)
    print("\n")
handle.close()'''