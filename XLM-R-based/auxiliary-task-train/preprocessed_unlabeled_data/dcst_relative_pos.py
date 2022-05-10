import sys
import re
import os
import codecs
import fileinput
import argparse


def pos_cluster(pos):
        
   # clustering the parts of speech
   if pos[0] == 'V':
       pos = 'VB'
   elif pos == 'NNS':
       pos = 'NN'
   elif pos == 'NNPS':
       pos = 'NNP'
   elif 'JJ' in pos:
       pos = 'JJ'
   elif pos[:2] == 'RB' or pos == 'WRB' or pos == 'RP':
       pos = 'RB'
   elif pos[:3] == 'PRP':
       pos = 'PRP'
   elif pos in ['.', ':', ',', "''", '``']:
       pos = '.'
   elif pos[0] == '-':
       pos = '-RB-'
   elif pos[:2] == 'WP':
       pos = 'WP'
 
   return pos


if __name__=="__main__":

   f_filename = sys.argv[1]
   fnew_filename = sys.argv[2]
   
   f=codecs.open(f_filename, encoding='utf-8', errors='ignore')
   f_new = codecs.open(fnew_filename, "w", "utf-8")

   all_sents = f.read().split("\n\n")

   for sent in all_sents:

       lines = sent.split("\n")

       lines_sep = []
       root_index = 0

       for line in lines:

           cols = line.split("\t")

           if cols[6] == 0:
              root_index = cols[0]

           lines_sep.append(cols)

       for i, line in enumerate(lines):

           distance_to_root = 0
           
           head = lines_sep[i][6]

           j = 0
           while int(head) > 0:
       
               next = lines_sep[j]
               j += 1
               if next[0] == head:
                  distance_to_root += 1
                  head = next[6]
                  j = 0
               
               
           
           lines_sep[i][4] = distance_to_root

           new_line = "\t".join([str(elem) for elem in lines_sep[i]])
           f_new.write(new_line+"\n")

       f_new.write("\n")


       
