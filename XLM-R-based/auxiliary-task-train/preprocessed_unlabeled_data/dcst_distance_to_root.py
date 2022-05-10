import sys
import re
import os
import codecs
import fileinput
import argparse


if __name__=="__main__":

   f_filename = sys.argv[1]
   fnew_filename = sys.argv[2]
   
   f=codecs.open(f_filename, encoding='utf-8', errors='ignore')
   f_new = codecs.open(fnew_filename, "w", "utf-8")

   all_sents = f.read().split("\n\n")

   all_sents = all_sents[:-1]


   cnt = 1
   for sent in all_sents:

       print(cnt)
       print(sent)
       cnt +=1
       linesall = sent.split("\n")

       liness = [line for line in linesall if not line.startswith("#")]
       lines = [line for line in liness if not "-" in line.split("\t")[0]]

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


       
