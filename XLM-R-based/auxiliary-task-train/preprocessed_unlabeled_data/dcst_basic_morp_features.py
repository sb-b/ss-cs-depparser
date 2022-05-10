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
        
           feats = cols[5].split("|")

           
           new_feats = []
           for feat in feats:
             # if any(feat.startswith(s) for s in ["NumType", "Foreign", "Case", "Degree", "VerbForm", "Mood", "Tense", "Aspect", "Voice", "Person"]):
              if any(feat.startswith(s) for s in ["NumType", "Foreign", "Case","Mood", "VerbForm", "Aspect", "Person"]):
                 new_feats.append(feat)

           print(new_feats)
           if len(new_feats) == 0:
              new_feats.append("_")
           cols[5] = "|".join(new_feats)

           new_line = "\t".join(cols)
           f_new.write(new_line+"\n")

       f_new.write("\n")


       
