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

   for sent in all_sents:

       linesall = sent.split("\n")

       lines = [line for line in linesall if not line.startswith("#")]

       lines_sep = []

       for line in lines:

           cols = line.split("\t")

           lines_sep.append(cols)

       for i, line in enumerate(lines):

           num_of_child = 0

           index = lines_sep[i][0]

           for other in lines_sep:

              if other[6] == index:

                 num_of_child += 1

           lines_sep[i][2] = num_of_child

           
           new_line = "\t".join([str(elem) for elem in lines_sep[i]])
           f_new.write(new_line+"\n")

       f_new.write("\n")


       
