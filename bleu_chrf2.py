import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='ar')
import os.path
import sys
import numpy as np

import pandas as pd
import numpy as np

#From Moustafa Tohamy
import re
import string
import itertools



# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:20:38 2024

@author: khered
"""
def saveRes(pred_file,gold_file,preds,b_score,c_score):
    raw_data = {
                'model': [pred_file],
                'di': [gold_file],
                'source': [preds],
                'bleu': [b_score],
                'chrf2': [c_score]
    
                }
    df = pd.DataFrame(raw_data,columns = ['model','di','source','bleu','chrf2'])
    
    evalpath= 'eval_trans.csv'
    from os import path
    isexist = path.exists(evalpath)
    if(isexist):
        evaldata = pd.read_csv (r''+evalpath,encoding='utf-8')
        #evaldata=evaldata.append(df)
        evaldata= pd.concat([evaldata, df], ignore_index=True)
    else: 
        evaldata = df
        
    evaldata.to_csv(evalpath,index=False )
    
def eval_dial2msa(model,gold_file,pred_file,mtrx,scores_di,avg):
    raw_data = {
                'model': [model],
                'g_file': [gold_file],
                'p_file': [pred_file],
                'metrx': [mtrx],
                'Egy': [scores_di[0]],
                'Glf': [scores_di[1]],
                'Mgr': [scores_di[2]],
                'Lev': [scores_di[3]],
                'Avg': [avg]    
                }
    df = pd.DataFrame(raw_data,columns = ['model','g_file','p_file','metrx','Egy','Glf','Mgr','Lev','Avg'])
    print(df.iloc[: , -6:])
    evalpath= 'eval_dial2msa.csv'
    from os import path
    isexist = path.exists(evalpath)
    if(isexist):
        evaldata = pd.read_csv (r''+evalpath,encoding='utf-8')
        #evaldata=evaldata.append(df)
        evaldata= pd.concat([evaldata, df], ignore_index=True)
    else: 
        evaldata = df
        
    evaldata.to_csv(evalpath,index=False )


def print_usage():
    print (
    '''Usage:
    python3 bleu_chrf2_Scorer.py  <gold-file> <pred-file>

    For all the 4 dialects 2000 test each:
        
    python bleu_chrf2_Scorer.py  Dial2MSA_test_eval_4_dialects <pred-file>
        ''')
        
## loads labels from file
def load_labels(filename):
    with open(filename, encoding="utf8") as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]
    return labels

def calBleu(gold_file,pred_file):
    reffile = gold_file
    predfile = pred_file
    
    #f2=gold_file[0:len(gold_file)-4]+'1'+gold_file[len(gold_file)-4:len(gold_file)]
    f2=gold_file[0:len(gold_file)-5]+'2'+gold_file[len(gold_file)-4:len(gold_file)]
    f3=gold_file[0:len(gold_file)-5]+'3'+gold_file[len(gold_file)-4:len(gold_file)]
    

    gold_labels = load_labels(gold_file)
        

    
    predicted_labels = load_labels(pred_file)

    if (len(gold_labels) != len(predicted_labels)):
        print ("both files must have same number of instances.")
        print("gold:"+str(len(gold_labels)))
        print("pred:"+str(len(predicted_labels)))
        exit()

    # Open the test dataset human translation file and detokenize the references
    refs1 = []
    refs2 = []
    refs3 = []
    with open(reffile, encoding="utf8") as test:
        for line in test:
            if '\n' in line:
                line=line.replace('\n', '')
            if (line[0]=='"' and line[len(line)-1]=='"'):
                line=line[1:len(line)-1]
            line = line.strip().split() 
            line = md.detokenize(line) 
            refs1.append(line)
    
    filesNo=1
    
    if os.path.isfile(f2):
        with open(f2, encoding="utf8") as test:
            for line in test:
                if '\n' in line:
                    line=line.replace('\n', '')
                if (line[0]=='"' and line[len(line)-1]=='"'):
                    line=line[1:len(line)-1]
                line = line.strip().split() 
                line = md.detokenize(line) 
                refs2.append(line)
        filesNo=filesNo+1
    
    if os.path.isfile(f3):
        with open(f3, encoding="utf8") as test:
            for line in test:
                if '\n' in line:
                    line=line.replace('\n', '')
                if (line[0]=='"' and line[len(line)-1]=='"'):
                    line=line[1:len(line)-1]
                line = line.strip().split() 
                line = md.detokenize(line) 
                refs3.append(line)
        filesNo=filesNo+1



    # Open the translation file by the NMT model and detokenize the predictions
    preds = []

    with open(predfile, encoding="utf8") as pred:  
        for line in pred: 
            if '\n' in line:
                line=line.replace('\n', '')
            if (line[0]=='"' and line[len(line)-1]=='"'):
                line=line[1:len(line)-1]
            line = line.strip().split() 
            line = md.detokenize(line) 
            preds.append(line)



    preds = [pred.strip() for pred in preds]
    refs1 = [rs.strip() for rs in refs1]
    
    if filesNo ==2:
        refs2 = [rs.strip() for rs in refs2]
        refs = [refs1,refs2]
        
    elif filesNo ==3:
        refs2 = [rs.strip() for rs in refs2]
        refs3 = [rs.strip() for rs in refs3]
        refs = [refs1,refs2,refs3]
    
        
    else:
        refs = [refs1]  # Yes, it is a list of list(s) as required by sacreBLEU

        
    # print("Reference 1st sentence:", refs1[0])
    # if filesNo ==2:
    #     print("Reference 2st sentence:", refs2[0])
    # elif filesNo ==3:
    #     print("Reference 2st sentence:", refs2[0])
    #     print("Reference 3st sentence:", refs3[0])
    # print("Pred sentence:", preds[0])  
    # print("--")    
    
    

    # Calculate and print the BLEU and chrf scores
    chrf_lib = CHRF(word_order=2)    
    bleu_lib = BLEU()
    bleu = bleu_lib.corpus_score(preds, refs)
    chrf = chrf_lib.corpus_score(preds, refs)
    print(pred_file)
    print("MSA refrences: "+ str(filesNo)+" refrences files")
    print("bleu score: ", bleu.score)
    print("chrf++ score: ", chrf.score)
    print("====")
    
    if filesNo ==2:
        if 'file12_' in gold_file or 'file23_' in gold_file:
            saveRes(pred_file,gold_file[0:len(gold_file)-5],len(preds),bleu.score,chrf.score)
        else:
            saveRes(pred_file,'2_filesRefs_'+gold_file[0:len(gold_file)-5],len(preds),bleu.score,chrf.score)
    
            df_ref=pd.DataFrame (refs1, columns = ['ref'])
            df_ref['ref'].to_csv('file1_.txt', sep='\t', index=False,header=False)
            calBleu('file1_.txt',pred_file)
            
            df_ref=pd.DataFrame (refs2, columns = ['ref'])
            df_ref['ref'].to_csv('file2_.txt', sep='\t', index=False,header=False)
            calBleu('file2_.txt',pred_file)
        
    elif filesNo ==3:
        
        if 'file12_' in gold_file or 'file23_' in gold_file:
            saveRes(pred_file,gold_file[0:len(gold_file)-5],len(preds),bleu.score,chrf.score)
        else:
            saveRes(pred_file,'3_filesRefs_'+gold_file[0:len(gold_file)-5],len(preds),bleu.score,chrf.score)

            df_ref1=pd.DataFrame (refs1, columns = ['ref'])
            df_ref1['ref'].to_csv('file1_.txt', sep='\t', index=False,header=False)
            calBleu('file1_.txt',pred_file)
            
            df_ref2=pd.DataFrame (refs2, columns = ['ref'])
            df_ref2['ref'].to_csv('file2_.txt', sep='\t', index=False,header=False)
            calBleu('file2_.txt',pred_file)
            
            df_ref3=pd.DataFrame (refs3, columns = ['ref'])
            df_ref3['ref'].to_csv('file3_.txt', sep='\t', index=False,header=False)
            calBleu('file3_.txt',pred_file)
            

            df_ref1['ref'].to_csv('file12_1.txt', sep='\t', index=False,header=False)
            df_ref2['ref'].to_csv('file12_2.txt', sep='\t', index=False,header=False)
            calBleu('file12_1.txt',pred_file)
            

            df_ref2['ref'].to_csv('file23_1.txt', sep='\t', index=False,header=False)
            df_ref3['ref'].to_csv('file23_2.txt', sep='\t', index=False,header=False)
            calBleu('file23_1.txt',pred_file)
        
    else:
        if 'file1_.txt' == gold_file or 'file2_.txt' == gold_file or 'file3_.txt' == gold_file or 'file12_.txt' == gold_file or 'file23_.txt' == gold_file:
            saveRes(pred_file,gold_file[0:len(gold_file)-4],len(preds),bleu.score,chrf.score)
        else:
            saveRes(pred_file,gold_file,len(preds),bleu.score,chrf.score)
    
    return bleu.score,chrf.score

if __name__ == '__main__':

        
    verbose = 0
    if (len (sys.argv) > 4 or len (sys.argv) <3):
        print_usage()
        exit()
        
    if (len (sys.argv) == 4 and sys.argv[3] != "-verbose"):
        print_usage()
        exit()
        
    if (len (sys.argv) == 4):
        verbose = 1

    
    gold_file = sys.argv[1]
    pred_file = sys.argv[2]
    
    pathD='test/'
    
    path_pred=pred_file[0:len(pred_file)-4]
    
    pred_labels = load_labels(pred_file)
    if len(pred_labels) == 8000 and gold_file == "Dial2MSA_test_eval_4_dialects":
        M_C1=pd.DataFrame (pred_labels, columns = ['source'])
        M_C1['source'][0:2000].to_csv(path_pred+'_egy_pred.txt', sep='\t', index=False,header=False)
        M_C1['source'][2000:4000].to_csv(path_pred+'_mgr_pred.txt', sep='\t', index=False,header=False)
        M_C1['source'][4000:6000].to_csv(path_pred+'_glf_pred.txt', sep='\t', index=False,header=False)
        M_C1['source'][6000:8000].to_csv(path_pred+'_lev_pred.txt', sep='\t', index=False,header=False)
        
        b_egy,c_egy=calBleu(pathD+'egy/gold_msa_egy_ts1.txt',path_pred+'_egy_pred.txt')
        b_mgr,c_mgr=calBleu(pathD+'mgr/gold_msa_mgr_ts1.txt',path_pred+'_mgr_pred.txt')
        b_glf,c_glf=calBleu(pathD+'glf/gold_msa_glf_ts1.txt',path_pred+'_glf_pred.txt')
        b_lev,c_lev=calBleu(pathD+'lev/gold_msa_lev_ts1.txt',path_pred+'_lev_pred.txt')
        
        b_scores=[b_egy,b_glf,b_mgr,b_lev]
        b_avg = sum(b_scores) / len(b_scores)
        eval_dial2msa(path_pred,gold_file,pred_file,'bleu',b_scores,b_avg)
        
        c_scores=[c_egy,c_glf,c_mgr,c_lev]
        c_avg = sum(c_scores) / len(c_scores)
        eval_dial2msa(path_pred,gold_file,pred_file,'chrf++',c_scores,c_avg)
    elif  gold_file == "Dial2MSA_test_eval_4_dialects":
        print("Make sure the pred file has 8000 samples")        
    else:
        calBleu(gold_file,pred_file)
