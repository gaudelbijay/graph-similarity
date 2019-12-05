#-*-coding:utf8 -*-
import pandas as pd
import math
import torch
import networkx as nx
import copy
import os
import numpy as np
 
"""
pandas is used for large scale networks
"""
 
 
 
 
 
class LocalMethods(object):
    """
    some tips:
    In this part, we want to implement some basic similarity mesurement methods in network analysis
    the input is a dataframe of pandas with two columns:
   
    index   source  target
    0       2       3
    1       2       4
    .....
     
    The implemented methods are listed as follows:
    1) common neighbours (CN)
    2) adamic-adar index (AA)
    3) resource allocation (RA)
    4) Resource Allocation Based on Common Neighbor Interactions (RA-CNI)
    5) Preferential Attachment Index (PA)
    6) Jaccard Coefficient (JC)
    7) Salton Index (SA)
    8) The Sørensen Index (SO)
    9) Hub Promoted Index (HPI)
    10) Hub Depressed Index (HDI)
    11) Local Leicht-Holme-Newman Index (LLHN)
   
    """
   
    def __init__(self, df_edge_list):
        self.df_edge_list= df_edge_list
       
        pass
       
    def cal_CN(self):
        """
        this method is implemented for CN
        input: self.edge_list ---->    the edge list of a graph
       
        return : df_common_neighbor_count ----> the CN list
            source_x     source_y   similarity
            1               2           18
            ....
       
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
       
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
        #print df_common_neighbor
        df_common_neighbor_count= df_common_neighbor.groupby(['source_x', 'source_y']).count()
        df_common_neighbor_count=df_common_neighbor_count.reset_index()
        #print df_common_neighbor_count
       
        df_common_neighbor_count.rename(columns={'target':'similarity'}, inplace=True)
        df_common_neighbor_count.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_common_neighbor_count
   
    def cal_AA(self):
        """
        this method is implementation for Adamic-Adar index
        input: self.edge_list ---->    the edge list of a graph
       
        return : df_AA_list ----> the AA list
            source_x     source_y   similarity
            1               2           18
            ....
       
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
       
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'count'}, inplace=True)
       
       
       
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor=pd.merge(df_common_neighbor, df_neighbor_count, left_on=['target'],right_on=['source'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[['source_x','source_y','count']]
        df_common_neighbor['count']=map(lambda x: 1.0/math.log(x), df_common_neighbor['count'])
       
       
       
        df_AA_list=df_common_neighbor.groupby(['source_x', 'source_y']).sum()
        df_AA_list= df_AA_list.reset_index()
       
        df_AA_list.rename(columns={'count':'similarity'},inplace=True)
        #print df_AA_list
        df_AA_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_AA_list
       
    def cal_RA(self):
        """
        this method si implemented for Resource Allocation Based on Common Neighbor Interactions
        input: self.edge_list ---->    the edge list of a graph
       
        return : df_RA_list ----> the RA list
            source_x     source_y   similarity
            1               2           18
            ....
       
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
       
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'count'}, inplace=True)
       
       
       
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor=pd.merge(df_common_neighbor, df_neighbor_count, left_on=['target'],right_on=['source'], how='left')
        df_common_neighbor=df_common_neighbor[['source_x','source_y','count']]
        df_common_neighbor['count']=map(lambda x: 1.0/x, df_common_neighbor['count'])
       
        df_RA_list=df_common_neighbor.groupby(['source_x', 'source_y']).sum()
        df_RA_list= df_RA_list.reset_index()
       
        df_RA_list.rename(columns={'source_x':'source','source_y':'target','count':'similarity'}, inplace=True)
        #print df_RA_list
        #df_RA_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_RA_list
       
    def cal_RA_CNI(self):
   
        """
        this method si implem for resource allocation index
        input: self.edge_list ---->    the edge list of a graph
       
        return : df_RA_CNI_list ----> the RA_CNI list
            source_x     source_y   similarity
            1               2           18
            ....
       
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
       
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'count'}, inplace=True)
       
        df_neighbor_count['count']=map(lambda x: 1.0/x, df_neighbor_count['count'])
       
        df_node_pair_gamma=pd.merge(df_all_nodes_pair, df_neighbor_count, on=['source'], how ='left')
        df_node_pair_gamma=pd.merge(df_node_pair_gamma, df_neighbor_count, left_on=['target'], right_on=['source'], how='left').dropna()
        df_node_pair_gamma['count']=df_node_pair_gamma['count_x']-df_node_pair_gamma['count_y']
        df_node_pair_gamma=df_node_pair_gamma[['source_x','source_y','count']]
        df_node_pair_gamma['count']=map(lambda x: abs(x), df_node_pair_gamma['count'])
       
        #print df_node_pair_gamma
        #print self.df_edge_list.shape
       
        """
        get RA
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor=pd.merge(df_common_neighbor, df_neighbor_count, left_on=['target'],right_on=['source'], how='left')
        df_common_neighbor=df_common_neighbor[['source_x','source_y','count']]
       
       
        df_RA_list=df_common_neighbor.groupby(['source_x', 'source_y']).sum()
        df_RA_list= df_RA_list.reset_index()
       
        df_RA_list.rename(columns={'count':'similarity'}, inplace=True)
       
        """
        get CNI
        """
       
        """
        get common neighbors
        """
       
        df_neighbors=df_all_nodes_pair.groupby(['source'])
       
        df_exist_RA_node_pair=df_RA_list[['source_x', 'source_y']]
       
        df_RA_with_neighbor=pd.merge(df_exist_RA_node_pair, df_all_nodes_pair, left_on=['source_x'], right_on=['source'], how='left')
        df_RA_with_neighbor=df_RA_with_neighbor[['source_x','source_y','target']]
        df_RA_with_neighbor.rename(columns={'source_x':'source_x1','source_y':'source_y1','target':'source_x1_nei'},inplace=True)
        #print df_RA_with_neighbor.head()
        df_RA_with_neighbor=pd.merge(df_RA_with_neighbor, df_all_nodes_pair, left_on=['source_y1'], right_on=['source'], how='left')
        #print df_RA_with_neighbor.head()
        df_RA_with_neighbor=df_RA_with_neighbor[['source_x1','source_y1','source_x1_nei', 'target']]
        df_RA_with_neighbor.rename(columns={'target':'source_y1_nei'},inplace=True)
        #print df_RA_with_neighbor.head(2)
       
        df_RA_with_neighbor_with_CNI=pd.merge(df_RA_with_neighbor,df_node_pair_gamma, left_on=['source_x1_nei', 'source_y1_nei'],right_on=['source_x','source_y'], how='left').dropna()
       
        df_RA_with_neighbor_with_CNI=df_RA_with_neighbor_with_CNI[['source_x1','source_y1','count']]
       
       
        df_RA_CNI=df_RA_with_neighbor_with_CNI.groupby(['source_x1','source_y1']).sum()
        df_RA_CNI=df_RA_CNI.reset_index()
       
       
       
        df_RA_CNI_list=pd.merge(df_RA_list, df_RA_CNI, left_on=['source_x','source_y'], right_on=['source_x1', 'source_y1'], how='left').fillna(0)
       
        df_RA_CNI_list['RACNI']=df_RA_CNI_list['similarity']+df_RA_CNI_list['count']
        df_RA_CNI_list=df_RA_CNI_list[['source_x','source_y','RACNI']]
        df_RA_CNI_list.rename(columns={'RACNI':'similarity'}, inplace=True)
       
        df_RA_CNI_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_RA_CNI_list
       
    def cal_PA(self):
        """
        this method is implemented for preferential attachment index
        input: self.df_edge_list
        output: df_PA_list
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        all_nodes=self.df_edge_list['source'].tolist()
        ttt=self.df_edge_list['target'].tolist()
        all_nodes.extend(ttt)
        print('all nodes number is ',len(set(all_nodes)))
       
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
        df_neighbor_count=df_neighbor_count.reset_index()
        #print df_neighbor_count.head()
        df_neighbor_count.rename(columns={'target':'count1'},inplace=True)
       
        df_PA_list=pd.merge(df_all_nodes_pair, df_neighbor_count, left_on=['source'], right_on=['source'], how='left').fillna(0)
        #print df_PA_list.head()
        #print 'gg PA',df_PA_list.shape
        df_PA_list=pd.merge(df_PA_list, df_neighbor_count, left_on=['target'], right_on=['source'], how='left').fillna(0)
        #print 'gg PA',df_PA_list.shape
        df_PA_list['similarity']=df_PA_list['count1_x']*df_PA_list['count1_y']
        df_PA_list=df_PA_list[['source_x','source_y','similarity']]
       
        #print 'gg PA',df_PA_list.shape
        df_PA_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        #print 'gg PA',df_PA_list.head()
        df_PA_list=df_PA_list.drop_duplicates()
        return df_PA_list
       
       
    def cal_JC(self):
   
        """
        this method is implemented for Jacccard Coefficient
        input: self.df_edge_list
        output: df_JC_list
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'nei_count'}, inplace =True)
       
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor_count= df_common_neighbor.groupby(['source_x', 'source_y']).count()
        df_common_neighbor_count=df_common_neighbor_count.reset_index()
        #print df_common_neighbor_count
       
        df_common_neighbor_count.rename(columns={'target':'CN'}, inplace=True)
       
        df_common_neighbor_with_total_neighbor= pd.merge(df_common_neighbor_count, df_neighbor_count, left_on=['source_x'], right_on=['source'], how='left')
       
        df_common_neighbor_with_total_neighbor=df_common_neighbor_with_total_neighbor[['source_x','source_y','CN','nei_count']]
        df_common_neighbor_with_total_neighbor=pd.merge(df_common_neighbor_with_total_neighbor,df_neighbor_count,left_on=['source_y'],right_on=['source'],how='left')
       
        df_common_neighbor_with_total_neighbor['total_neighbor']=df_common_neighbor_with_total_neighbor['nei_count_x']+df_common_neighbor_with_total_neighbor['nei_count_y']-df_common_neighbor_with_total_neighbor['CN']
       
        df_common_neighbor_with_total_neighbor['similarity']=df_common_neighbor_with_total_neighbor['CN']/df_common_neighbor_with_total_neighbor['total_neighbor']
       
        df_JC_list=df_common_neighbor_with_total_neighbor[['source_x','source_y','similarity']].copy()
       
       
       
        df_JC_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_JC_list
       
       
       
    def cal_SA(self):
   
        """
        this method is implemented for Salton Index
        input: self.df_edge_list
        output: df_SA_list
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'nei_count'}, inplace =True)
       
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor_count= df_common_neighbor.groupby(['source_x', 'source_y']).count()
        df_common_neighbor_count=df_common_neighbor_count.reset_index()
        #print df_common_neighbor_count
       
        df_common_neighbor_count.rename(columns={'target':'CN'}, inplace=True)
       
        df_common_neighbor_with_total_neighbor= pd.merge(df_common_neighbor_count, df_neighbor_count, left_on=['source_x'], right_on=['source'], how='left')
       
        df_common_neighbor_with_total_neighbor=df_common_neighbor_with_total_neighbor[['source_x','source_y','CN','nei_count']]
        df_common_neighbor_with_total_neighbor=pd.merge(df_common_neighbor_with_total_neighbor,df_neighbor_count,left_on=['source_y'],right_on=['source'],how='left')
       
        df_common_neighbor_with_total_neighbor['nei_mul_nei']=df_common_neighbor_with_total_neighbor['nei_count_x']*df_common_neighbor_with_total_neighbor['nei_count_y']
       
        df_common_neighbor_with_total_neighbor['nei_mul_nei']=map(lambda x: pow(x,0.5),df_common_neighbor_with_total_neighbor['nei_mul_nei'])
        df_common_neighbor_with_total_neighbor['similarity']=df_common_neighbor_with_total_neighbor['CN']/df_common_neighbor_with_total_neighbor['nei_mul_nei']
       
        df_SA_list=df_common_neighbor_with_total_neighbor[['source_x','source_y','similarity']].copy()
       
       
       
        df_SA_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_SA_list
       
    def cal_SO(self):
   
        """
        this method is implemented for Sørensen Index
        input: self.df_edge_list
        output: df_SO_list
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'nei_count'}, inplace =True)
       
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor_count= df_common_neighbor.groupby(['source_x', 'source_y']).count()
        df_common_neighbor_count=df_common_neighbor_count.reset_index()
        #print df_common_neighbor_count
       
        df_common_neighbor_count.rename(columns={'target':'CN'}, inplace=True)
       
        df_common_neighbor_with_total_neighbor= pd.merge(df_common_neighbor_count, df_neighbor_count, left_on=['source_x'], right_on=['source'], how='left')
       
        df_common_neighbor_with_total_neighbor=df_common_neighbor_with_total_neighbor[['source_x','source_y','CN','nei_count']]
        df_common_neighbor_with_total_neighbor=pd.merge(df_common_neighbor_with_total_neighbor,df_neighbor_count,left_on=['source_y'],right_on=['source'],how='left')
       
        df_common_neighbor_with_total_neighbor['nei_add_nei']=df_common_neighbor_with_total_neighbor['nei_count_x']+df_common_neighbor_with_total_neighbor['nei_count_y']
       
       
        df_common_neighbor_with_total_neighbor['similarity']=2*df_common_neighbor_with_total_neighbor['CN']/df_common_neighbor_with_total_neighbor['nei_add_nei']
       
        df_SO_list=df_common_neighbor_with_total_neighbor[['source_x','source_y','similarity']].copy()
       
       
       
        df_SO_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_SO_list
       
       
    def cal_HPI(self):
   
        """
        this method is implemented for Hub Promoted Index
        input: self.df_edge_list
        output: df_HPI_list
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'nei_count'}, inplace =True)
           
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor_count= df_common_neighbor.groupby(['source_x', 'source_y']).count()
        df_common_neighbor_count=df_common_neighbor_count.reset_index()
        #print df_common_neighbor_count
       
        df_common_neighbor_count.rename(columns={'target':'CN'}, inplace=True)
       
        df_common_neighbor_with_total_neighbor= pd.merge(df_common_neighbor_count, df_neighbor_count, left_on=['source_x'], right_on=['source'], how='left')
       
        df_common_neighbor_with_total_neighbor=df_common_neighbor_with_total_neighbor[['source_x','source_y','CN','nei_count']]
        df_common_neighbor_with_total_neighbor=pd.merge(df_common_neighbor_with_total_neighbor,df_neighbor_count,left_on=['source_y'],right_on=['source'],how='left')
       
        df_common_neighbor_with_total_neighbor['nei_min']=df_common_neighbor_with_total_neighbor[['nei_count_x','nei_count_y']].min(axis=1)
        #print df_common_neighbor_with_total_neighbor.head()
       
       
        df_common_neighbor_with_total_neighbor['similarity']=df_common_neighbor_with_total_neighbor['CN']/df_common_neighbor_with_total_neighbor['nei_min']
       
        df_HPI_list=df_common_neighbor_with_total_neighbor[['source_x','source_y','similarity']].copy()
       
       
       
        df_HPI_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_HPI_list
       
    def cal_HDI(self):
   
        """
        this method is implemented for Hub Depressed Index
        input: self.df_edge_list
        output: df_HDI_list
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'nei_count'}, inplace =True)
       
       
        """
        get common neighbours
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor_count= df_common_neighbor.groupby(['source_x', 'source_y']).count()
        df_common_neighbor_count=df_common_neighbor_count.reset_index()
        #print df_common_neighbor_count
       
        df_common_neighbor_count.rename(columns={'target':'CN'}, inplace=True)
       
        df_common_neighbor_with_total_neighbor= pd.merge(df_common_neighbor_count, df_neighbor_count, left_on=['source_x'], right_on=['source'], how='left')
       
        df_common_neighbor_with_total_neighbor=df_common_neighbor_with_total_neighbor[['source_x','source_y','CN','nei_count']]
        df_common_neighbor_with_total_neighbor=pd.merge(df_common_neighbor_with_total_neighbor,df_neighbor_count,left_on=['source_y'],right_on=['source'],how='left')
       
        df_common_neighbor_with_total_neighbor['nei_min']=df_common_neighbor_with_total_neighbor[['nei_count_x','nei_count_y']].max(axis=1)
        #print df_common_neighbor_with_total_neighbor.head()
       
       
        df_common_neighbor_with_total_neighbor['similarity']=df_common_neighbor_with_total_neighbor['CN']/df_common_neighbor_with_total_neighbor['nei_min']
       
        df_HDI_list=df_common_neighbor_with_total_neighbor[['source_x','source_y','similarity']].copy()
       
        df_HDI_list.rename(columns={'source_x':'source', 'source_y':'target'}, inplace=True)
       
        #print df_HDI_list.head()
        return df_HDI_list
       
       
    def cal_LLHN(self):
   
        """
        this method is implemented for Local Leicht-Holme-Newman Index
        input: self.df_edge_list
        output: df_LLHN_list
        """
       
        df_edge_list_reverse=pd.DataFrame()
        df_edge_list_reverse['source']=self.df_edge_list['target']
        df_edge_list_reverse['target']=self.df_edge_list['source']
        #print df_edge_list_reverse
       
        df_all_nodes_pair=pd.concat([df_edge_list_reverse, self.df_edge_list])
        df_neighbor_count= df_all_nodes_pair.groupby(['source']).count()
        df_neighbor_count=df_neighbor_count.reset_index()
        df_neighbor_count.rename(columns={'target':'nei_count'}, inplace =True)
       
       
        """
        get common neighbours
        drwxr-xr-x 3
        """
       
        df_common_neighbor=pd.merge(df_all_nodes_pair, df_all_nodes_pair, on=['target'], how='left').dropna()
        df_common_neighbor=df_common_neighbor[df_common_neighbor['source_x']!=df_common_neighbor['source_y']]
       
        df_common_neighbor_count= df_common_neighbor.groupby(['source_x', 'source_y']).count()
        df_common_neighbor_count=df_common_neighbor_count.reset_index()
        #print df_common_neighbor_count
       
        df_common_neighbor_count.rename(columns={'target':'CN'}, inplace=True)
       
        df_common_neighbor_with_total_neighbor= pd.merge(df_common_neighbor_count, df_neighbor_count, left_on=['source_x'], right_on=['source'], how='left')
       
        df_common_neighbor_with_total_neighbor=df_common_neighbor_with_total_neighbor[['source_x','source_y','CN','nei_count']]
        df_common_neighbor_with_total_neighbor=pd.merge(df_common_neighbor_with_total_neighbor,df_neighbor_count,left_on=['source_y'],right_on=['source'],how='left')
       
        df_common_neighbor_with_total_neighbor['nei_mul_nei']=df_common_neighbor_with_total_neighbor['nei_count_x']*df_common_neighbor_with_total_neighbor['nei_count_y']
       
        #df_common_neighbor_with_total_neighbor['nei_mul_nei']=map(lambda x: pow(x,0.5),df_common_neighbor_with_total_neighbor['nei_mul_nei'])
        df_common_neighbor_with_total_neighbor['similarity']=df_common_neighbor_with_total_neighbor['CN']/df_common_neighbor_with_total_neighbor['nei_mul_nei']
       
        df_LLHN_list=df_common_neighbor_with_total_neighbor[['source_x','source_y','similarity']].copy()        
        df_LLHN_list.rename(columns={'source_x':'source','source_y':'target'}, inplace=True)
        return df_LLHN_list
       
    def cal_save_all_similarity(self, data_name):
        similarity_dir='../temp/similarity_directory'
       
        if os.path.exists(similarity_dir):
            pass
        else:
            os.mkdir(similarity_dir)
 
        save_dir=similarity_dir+'/'+data_name
        if os.path.exists(save_dir):
            pass
        else:
            os.mkdir(save_dir)
        """  
        1) common neighbours (CN)
        2) adamic-adar index (AA)
        3) resource allocation (RA)
        4) Resource Allocation Based on Common Neighbor Interactions (RA-CNI)
        5) Preferential Attachment Index (PI)
        6) Jaccard Coefficient
        7) Salton Index (SA)
        8) The Sørensen Index (SO)
        9) Hub Promoted Index (HPI)
        10) Hub Depressed Index (HDI)
        11) Local Leicht-Holme-Newman Index (LLHN)
        """
         
        similarity_name_list=['CN','AA','RA','RA_CNI', 'PA','JC','SA','SO','HPI','HDI','LLHN']
        similarity_function_list=[self.cal_CN, self.cal_AA, self.cal_RA, self.cal_RA_CNI, self.cal_PA, self.cal_JC, \
                self.cal_SA, self.cal_SO, self.cal_HPI, self.cal_HDI,self.cal_LLHN]
       
        total=len(similarity_function_list)
        for i,sim_func in enumerate(similarity_function_list):
            print('similarity calculation',sim_func.__name__)
            df_similarity_list=sim_func()
            save_file_name=save_dir+'/'+data_name+'_'+similarity_name_list[i]
            df_similarity_list.to_csv(save_file_name,index=False, sep=' ')
           
           
class GlobalMethods(object):
    """
     in this part, we implement the global simiilarity methods
     1). Katz Index KI
     2). Rooted Page Rank
 
    """
 
    def __init__(self, df_edge_list):
        self.df_edge_list= df_edge_list
       
        pass
   
    def cal_KI(self):
        df_edge=self.df_edge_list[self.df_edge_list['source']!=self.df_edge_list['target']]
        df_edge_inverse = pd.DataFrame()
        df_edge_inverse['source'] = df_edge['target']
        df_edge_inverse['target'] = df_edge['source']
 
        df_all_node_pair = pd.concat([df_edge, df_edge_inverse])
       
        df_all_node_pair['weight']=[1 for i in range (df_all_node_pair.shape[0])]
        pd_nodes_list = pd.DataFrame()
        pd_nodes_list['source'] = df_all_node_pair['source'].drop_duplicates(keep='first')
 
        pd_nodes_list['indicator'] = [1 for i in range(pd_nodes_list.shape[0])]
 
        pd_node_pairs = pd.merge(pd_nodes_list, pd_nodes_list, on=['indicator'], how='left')
 
        pd_node_pairs = pd_node_pairs.drop(['indicator'], axis=1)
 
        pd_node_pairs.rename(columns={'source_x': 'source', 'source_y': 'target'}, inplace=True)
 
        df_full_edge = pd.merge(pd_node_pairs, df_all_node_pair, on=['source', 'target'], how='left')
 
        df_full_edge = df_full_edge.fillna(0)
 
        df_full_edge = df_full_edge.drop_duplicates(['source', 'target'], keep='first')
 
        df_adj = df_full_edge.pivot('source', 'target', 'weight')
        index = df_adj.index.tolist()
        columns = df_adj.columns.tolist()
 
        adj_A = df_adj.values
        adj_I = np.eye(adj_A.shape[0])
 
        adj_A_tensor=torch.from_numpy(adj_A).type(torch.FloatTensor)
        adj_I_tensor=torch.from_numpy(adj_I).type(torch.FloatTensor)
        adj_A_tensor= adj_A_tensor + adj_I_tensor
        eig_value, eig_vector=torch.eig(adj_A_tensor)
 
        eig_value= torch.norm(eig_value, 2,1).numpy()
        spectral_radius = max(eig_value)
        beta = 0.99 *(1.0/spectral_radius)
 
        S_KI_tensor=torch.inverse(adj_I_tensor -beta* adj_A_tensor)- adj_I_tensor
       
       
        row_min,_=torch.min(S_KI_tensor,1,True)
        row_max,_=torch.max(S_KI_tensor,1,True)
        S_KI_tensor=(S_KI_tensor)/(row_max*1.0)
       
        S_KI = S_KI_tensor.numpy()
        all_ones = np.ones(adj_A.shape)
        mask_ski = all_ones - adj_I
        S_KI = S_KI* mask_ski
        S_KI_sigmoid = S_KI
        df_KI_adj = pd.DataFrame(
        S_KI_sigmoid, index=df_adj.index.tolist(), columns=df_adj.columns.tolist())
        # print df_KI_adj.head()
        df_KI_list = df_KI_adj.stack()
 
        df_KI_list = df_KI_list.reset_index()
        df_KI_list.columns = ['source', 'target', 'similarity']
 
        df_KI_list['similarity']=df_KI_list['similarity'].map(lambda x: x if x>0.001 else 0)
        return df_KI_list
   
 
    def cal_RPR(self,):
 
       
        df_edge=self.df_edge_list[self.df_edge_list['source']!=self.df_edge_list['target']]
        df_edge_inverse = pd.DataFrame()
        df_edge_inverse['source'] = df_edge['target']
        df_edge_inverse['target'] = df_edge['source']
 
        df_all_node_pair = pd.concat([df_edge, df_edge_inverse])
       
        df_all_node_pair['weight']=[1 for i in range (df_all_node_pair.shape[0])]
        pd_nodes_list = pd.DataFrame()
        pd_nodes_list['source'] = df_all_node_pair['source'].drop_duplicates(keep='first')
 
        pd_nodes_list['indicator'] = [1 for i in range(pd_nodes_list.shape[0])]
 
        pd_node_pairs = pd.merge(pd_nodes_list, pd_nodes_list, on=['indicator'], how='left')
 
        pd_node_pairs = pd_node_pairs.drop(['indicator'], axis=1)
 
        pd_node_pairs.rename(columns={'source_x': 'source', 'source_y': 'target'}, inplace=True)
 
        df_full_edge = pd.merge(pd_node_pairs, df_all_node_pair, on=['source', 'target'], how='left')
 
        df_full_edge = df_full_edge.fillna(0)
 
        df_full_edge = df_full_edge.drop_duplicates(['source', 'target'], keep='first')
 
        df_adj = df_full_edge.pivot('source', 'target', 'weight')
        index = df_adj.index.tolist()
        columns = df_adj.columns.tolist()
 
        adj_A = df_adj.values
        adj_I = np.eye(adj_A.shape[0])
       
        #adj_A =adj_A + adj_I
 
        adj_P = adj_A * 1.0 / (adj_A.sum(axis=0))
        alpha = 0.9
        adj_P_tensor=torch.from_numpy(adj_P).type(torch.FloatTensor)
        adj_I_tensor = torch.from_numpy(adj_I).type(torch.FloatTensor)
        S_RPR_tensor=(1- alpha) * torch.inverse(adj_I_tensor - alpha*adj_P_tensor)
       
        row_min,_=torch.min(S_RPR_tensor,1,True)
        row_max,_=torch.max(S_RPR_tensor,1,True)
        S_RPR_tensor=(S_RPR_tensor)/(row_max*1.0)
 
 
        S_RPR= S_RPR_tensor.numpy()
        df_RPR_adj = pd.DataFrame(S_RPR, index=df_adj.index.tolist(), columns=df_adj.columns.tolist())
 
        df_RPR_list = df_RPR_adj.stack()
 
        df_RPR_list = df_RPR_list.reset_index()
        df_RPR_list.columns = ['source', 'target', 'similarity']
 
        df_RPR_list['similarity']=df_RPR_list['similarity'].map(lambda x: x if x>0.3 else 0)
        return df_RPR_list
 
    def cal_save_all_similarity(self, data_name):
        similarity_dir='../temp/similarity_directory'
       
        if os.path.exists(similarity_dir):
            pass
        else:
            os.mkdir(similarity_dir)
 
        save_dir=similarity_dir+'/'+data_name
        if os.path.exists(save_dir):
            pass
        else:
            os.mkdir(save_dir)
        """  
        1) Katz Index(KI)
        2) Rooted PageRank(RPR)
       
        """
         
        similarity_name_list=['KI','RPR']
        similarity_function_list=[self.cal_KI, self.cal_RPR]
       
        total=len(similarity_function_list)
        for i,sim_func in enumerate(similarity_function_list):
            print('similarity calculation',sim_func.__name__)
            df_similarity_list=sim_func()
            save_file_name=save_dir+'/'+data_name+'_'+similarity_name_list[i]
            df_similarity_list.to_csv(save_file_name,index=False, sep=' ')
           
 
 
def test_similarity():
    edge_file_name='../temp/transformed_dataset/citeseer/citeseer.edges'
    df_edge_list= pd.read_csv(edge_file_name, delim_whitespace=True, low_memory=False)
    #test_local=LocalMethods(df_edge_list)
    #t=test_local.cal_PA()
       
    test_global=GlobalMethods(df_edge_list)
    t=test_global.cal_RPR()
    print(t)
def main():
    test_similarity()
if __name__=='__main__':
    main()
