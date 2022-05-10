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

   unique_cnt_list = []
   cnt = 1
   for sent in all_sents:
       cnt +=1
       linesall = sent.split("\n")

       liness = [line for line in linesall if not line.startswith("#")]
       lines = [line for line in liness if not "-" in line.split("\t")[0]]

       lines_sep = []
       root_ind = 0


       for line in lines:

           cols = line.split("\t")

           if cols[6] == '0':
              root_ind = int(cols[0])

           lines_sep.append(cols)

       for idx, line in enumerate(lines_sep):

           
           cnt_punct = 0
                                   
           if idx > root_ind:

              for i in range(root_ind,idx):
                  if lines_sep[i][3] == "PUNCT":
                     cnt_punct += 1

           else:
              for i in range(idx+1,root_ind):
                  if lines_sep[i][3] == "PUNCT":
                     cnt_punct += 1
               
           
           #print(line, root_ind, idx, cnt_punct)
           if cnt_punct not in unique_cnt_list:
              unique_cnt_list.append(cnt_punct)
           lines_sep[idx][4] = cnt_punct

           new_line = "\t".join([str(elem) for elem in lines_sep[idx]])
           f_new.write(new_line+"\n")

       f_new.write("\n")
   print(unique_cnt_list)


       
