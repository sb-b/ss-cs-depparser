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

       liness = [line for line in linesall if not line.startswith("#")]
       lines = [line for line in liness if not "-" in line.split("\t")[0]]

       lines_sep = []
       root_index = 0

       for line in lines:

           cols = line.split("\t")
        
           head_id = cols[6]

           langid = "_"

           for line2 in lines:
               
               cols2 = line2.split("\t")

               if cols2[0] == head_id:
                  langid = cols2[8]
                  break

           cols[5] = langid


           f_new.write("\t".join(cols)+"\n")

       f_new.write("\n")


               




       
