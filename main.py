#Made by Kiptyk Kirill

import os
import time
import difflib
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mitie import *
from sys import maxsize
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
from GoogleNews import GoogleNews
from itertools import permutations
from difflib import SequenceMatcher

    
# extracting all urls of news sources by the "Ремонт доріг Київ" request (1 month period)
def extracting_urls(request): 
    googlenews = GoogleNews(period='1m')
    googlenews.set_lang('ua')
    googlenews.search(request)
    result=googlenews.result()
    
    pd.set_option('display.max_colwidth', None)
    df=pd.DataFrame(result)
    
    for i in range(2,3):
        googlenews.getpage(i)
        result=googlenews.result()
        df=pd.DataFrame(result)

    links=df['link']
    
    with open('urls.txt', 'a') as f: #urls.txt
        links = links.to_string(header=False, index=False)
        f.write(links)
        
    with open('urls.txt', 'r') as f:#urls.txt
        urls=f.readlines()
        urls = [line.replace(' ', '') for line in urls]
        
    with open('urls.txt', 'w') as f:#urls.txt
        f.writelines(urls)
       
    df.to_excel("data/articles.xlsx") #data/articles.xlsx
    
    return(urls)

# Scraping news sources with urls from extracting_urls()
def news_scraper(urls):
    for i, url in enumerate(urls):
        print(urls[i])
        response = Request(urls[i])
        response.add_header('user-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6.1 Safari/605.1.15')
        try:
            client = urlopen(response)
            page = client.read()
            client.close()
            
            soup = BeautifulSoup(page, 'html.parser')
            text=soup.get_text()
            text=text.split()
            text=' '.join(str(e) for e in text)
            
            with open(f'data/news_data.txt','a') as f: #data/news_data.txt
                f.write(text+'\n')
                
        except HTTPError:
            print('HTTP error')
        except URLError:
            print('URL error')
        else:
            print ('ok')

# using pretrained model to find all locations from news text file compiled in news_scraper()
def ner_locs(dat_file_loc,text_file_loc):
    print ("loading NER model...")
    ner = named_entity_extractor(dat_file_loc) 
    
    #print ("\nTags output by this NER model:", ner.get_possible_ner_tags())
    
    # Load a text file and convert it into a list of words.  
    file=load_entire_file(text_file_loc).decode('utf8')
    
    tokens=file.split()
    
    #print ("Tokenized input:", tokens)
    entities = ner.extract_entities(tokens)
    #print ("\nEntities found:", entities)
    #print ("\nNumber of entities detected:", len(entities))

    # entities is a list of tuples, each containing the entity tag and a xrange that indicates which tokens are part of the entity.
    locations=[]
    
    for i,e in enumerate(entities):
        range = e[0]
        tag = e[1]
        accuracy = e[2]
        #print(i)
        
        if tag=='LOC':
            entity_text = " ".join(tokens[i] for i in range)
            if accuracy >= 0:     
                locations.append(entity_text)
    
    return(locations)

# calculating time for every path using fuzzy control
def road_repair_status_check(loc_entities,routes,initial_time):
    # saving all locations that visits salesman into one list 
    valuable_locs=[]
    for route in routes:
        for loc in route:
            if loc not in valuable_locs:
                valuable_locs.append(loc)
    
    # 
    loc_rat=[]
    for loc_ent in loc_entities:
        l={}
        for loc in valuable_locs:
            ratio=difflib.SequenceMatcher(None,loc_ent,loc).ratio()
            l[loc]=ratio
            
        max_r=max(l,key=l.get)
        if l[max_r]>0.5:
            loc_rat.append([max_r,l[max_r]])
        l.clear()
    
    #
    unique_locs=[]
    loc_number_ratio=[]
    for loc in loc_rat:
        if loc[0] not in unique_locs:
            unique_locs.append(loc[0]) 
            loc_number_ratio.append([loc[0]])
     
    #       
    n_mentions=[]
    for i,loc in enumerate(unique_locs):
        s=sum(x.count(loc) for x in loc_rat)
        
        r=[]
        for loc_r in loc_rat:
            if loc_r[0]==loc:
                r.append(loc_r[1])

        loc_number_ratio[i].append(s)
        n_mentions.append(s)
        loc_number_ratio[i].append(sum(r)/s)
    
    #
    def detect_outlier(data):
        # find q1 and q3 values
        q1, q3 = np.percentile(sorted(data), [25, 75])
        # compute IRQ
        iqr = q3 - q1
        
        upper_bound = q3 + (1.5 * iqr)
        #lower_bound = q1 - (1.5 * iqr)
        
        outliers = [x for x in data if x >= upper_bound]
        return outliers
    
    outliers=detect_outlier(n_mentions)
    
    for out in outliers:
        if out in n_mentions:
            n_mentions.remove(out)
    
    for outlier in outliers:
        for ent in loc_number_ratio:
            if ent[1]==outlier:
                ent[1]=max(n_mentions)
            
    #print(loc_number_ratio)  
    
    # New Antecedent/Consequent objects hold variables and membership functions
    mentions= ctrl.Antecedent(np.arange(0, max(n_mentions)+1, 1), 'mentions')
    similarity = ctrl.Antecedent(np.arange(0.5, 1.01, 0.01), 'similarity')
    delay = ctrl.Consequent(np.arange(0, 11, 1), 'delay')
    
    mentions.automf(3)
    similarity.automf(3)
    delay.automf(3)
    
    #mentions.view()
    #similarity.view()
    #delay.view()
    #plt.show()
    
    rule1 = ctrl.Rule(mentions['poor'] & similarity['poor'] , delay['poor'])
    rule2 = ctrl.Rule(mentions['average'] & similarity['poor'], delay['average'])
    rule3 = ctrl.Rule(mentions['good'] & similarity['good'], delay['good'])
    rule4 = ctrl.Rule(mentions['poor'] & similarity['good'], delay['average'])
    rule5 = ctrl.Rule(mentions['good'] & similarity['average'], delay['good'])
    rule6 = ctrl.Rule(mentions['poor'] & similarity['average'], delay['average'])
    
    time_delay_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    
    time_delay_sim = ctrl.ControlSystemSimulation(time_delay_ctrl)
    
    # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
    locs_time_delays={}
    
    for ent in loc_number_ratio:
        time_delay_sim.input['mentions']=ent[1]
        time_delay_sim.input['similarity']=ent[2]
        time_delay_sim.compute()
        locs_time_delays[ent[0]]=int(time_delay_sim.output['delay'])
    
    #print(locs_time_delays)

    
    for i,route in enumerate(routes):
        for loc in route:
            if loc in locs_time_delays.keys():
                initial_time[i]+=locs_time_delays[loc]
        

    print('\nLocations where repair works are in progress:')
    print(list(locs_time_delays.keys()))
    print('\nUpdated travel time for every route:')            
    print(initial_time)
    
    return(initial_time)
                
# reading salesmans input file that consist of all available routes and initial travel time for them
def reading_input_data(salesman_routes_file):

    with open(salesman_routes_file, 'r') as f:
        text=f.read().splitlines()
    
    locations=[]
    estimate_time=[]
    for route in text:
        
        if route=='' or route==' ':
            continue
        
        route1=route.split(',')
        locations.append(route1[:-1])
        estimate_time.append(route.split(';')[1])
    #print(locations)
    n_dest=np.roots([1/2,-1/2,-len(estimate_time)])[0]
    
    if n_dest.is_integer()==False:
        print('Wrong input data')
        return 0
    
    
    time_matrix=[[0 for i in range(int(n_dest))] for i in range(int(n_dest))]
    
    for i in range(len(estimate_time)):
        estimate_time[i]=int(estimate_time[i])
    
    n_dest=int(n_dest)
    
    time_matrix=updating_time_matrix(time_matrix,n_dest,estimate_time)
    
    destinations=[]
    i=n_dest-2
    n=0
    destinations.append(locations[i][0])
    
    while i!=len(locations)-1:
        destinations.append(locations[i+n_dest-2-n][0])
        i+=n_dest-2-n
        n+=1
        
    destinations.append(locations[-1][-1])
    
    return(time_matrix,locations,estimate_time,n_dest,destinations)
   
# creating and updating matrix with weights (travel time) for every route, it is used for travelling salesman problem
def updating_time_matrix(time_matrix,n_dest,estimate_time): 
    i=0
    for x in range(n_dest):
        for y in range(n_dest):
            if x>=y:
                continue
            time_matrix[x][y]=time_matrix[y][x]=estimate_time[i]
            i+=1   
    return time_matrix

# implementation of Travelling Salesman Problem, naive approach
def travellingSalesmanProblem(graph,  V):
    # store all vertex apart from source vertex
    vertex = []
    #starting point
    s=0

    for i in range(V):
        if i != s:
            vertex.append(i)
    
    # store minimum weight Hamiltonian Cycle
    min_pathweight = maxsize
    res_path=()
    all_permutations=permutations(vertex)

    for next_permutation in all_permutations:
        #print(i)
        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = s
        for j in next_permutation:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]
        
        # update minimum
        min_pathweight = min(min_pathweight, current_pathweight)
        if min_pathweight==current_pathweight:
            res_path=next_permutation
    
    return(res_path,min_pathweight)    

# implementation of Travelling Salesman Problem, branch and bound method
def branch_and_bound_TSP(adj,N,final_path):
    maxsize = float('inf')
    def copyToFinal(curr_path):
        final_path[:N + 1] = curr_path[:]
        final_path[N] = curr_path[0]
        
 
    # Function to find the minimum edge cost
    # having an end at the vertex i
    def firstMin(adj, i):
        min = maxsize
        for k in range(N):
            if adj[i][k] < min and i != k:
                min = adj[i][k]
        return min
    
    # function to find the second minimum edge
    # cost having an end at the vertex i
    def secondMin(adj, i):
        first, second = maxsize, maxsize
        for j in range(N):
            if i == j:
                continue
            if adj[i][j] <= first:
                second = first
                first = adj[i][j]
    
            elif(adj[i][j] <= second and
                adj[i][j] != first):
                second = adj[i][j]
                
        return second
    
    # function that takes as arguments:
    # curr_bound -> lower bound of the root node
    # curr_weight-> stores the weight of the path so far
    # level-> current level while moving
    # in the search space tree
    # curr_path[] -> where the solution is being stored
    # which would later be copied to final_path[]
    def TSPRec(adj, curr_bound, curr_weight,
                level, curr_path, visited):
        
        global final_res
        # base case is when we have reached level N
        # which means we have covered all the nodes once
        if level == N:
            
            # check if there is an edge from
            # last vertex in path back to the first vertex
            if adj[curr_path[level - 1]][curr_path[0]] != 0:
                
                # curr_res has the total weight
                # of the solution we got
                curr_res = curr_weight + adj[curr_path[level - 1]]\
                                            [curr_path[0]]
                if curr_res < final_res:
                    copyToFinal(curr_path)
                    final_res = curr_res
            return
    
        # for any other level iterate for all vertices
        # to build the search space tree recursively
        for i in range(N):
            
            # Consider next vertex if it is not same
            # (diagonal entry in adjacency matrix and
            #  not visited already)
            if (adj[curr_path[level-1]][i] != 0 and
                                visited[i] == False):
                temp = curr_bound
                curr_weight += adj[curr_path[level - 1]][i]
    
                # different computation of curr_bound
                # for level 2 from the other levels
                if level == 1:
                    curr_bound -= ((firstMin(adj, curr_path[level - 1]) +
                                    firstMin(adj, i)) / 2)
                else:
                    curr_bound -= ((secondMin(adj, curr_path[level - 1]) +
                                    firstMin(adj, i)) / 2)
    
                # curr_bound + curr_weight is the actual lower bound
                # for the node that we have arrived on.
                # If current lower bound < final_res,
                # we need to explore the node further
                if curr_bound + curr_weight < final_res:
                    curr_path[level] = i
                    visited[i] = True
                    
                    # call TSPRec for the next level
                    TSPRec(adj, curr_bound, curr_weight,
                        level + 1, curr_path, visited)
    
                # Else we have to prune the node by resetting
                # all changes to curr_weight and curr_bound
                curr_weight -= adj[curr_path[level - 1]][i]
                curr_bound = temp
    
                # Also reset the visited array
                visited = [False] * len(visited)
                for j in range(level):
                    if curr_path[j] != -1:
                        visited[curr_path[j]] = True
    
    # This function sets up final_path
    def TSP(adj):
        
        # Calculate initial lower bound for the root node
        # using the formula 1/2 * (sum of first min +
        # second min) for all edges. Also initialize the
        # curr_path and visited array
        curr_bound = 0
        curr_path = [-1] * (N + 1)
        visited = [False] * N
    
        # Compute initial bound
        for i in range(N):
            curr_bound += (firstMin(adj, i) +
                        secondMin(adj, i))
    
        # Rounding off the lower bound to an integer
        curr_bound = math.ceil(curr_bound / 2)
    
        # We start at vertex 1 so the first vertex
        # in curr_path[] is 0
        visited[0] = True
        curr_path[0] = 0
    
        # Call to TSPRec for curr_weight
        # equal to 0 and level 1
        TSPRec(adj, curr_bound, 0, 1, curr_path, visited)

    TSP(adj)
    
    return final_path,final_res
 
# visualizing output graph    
def visualize_graph(path,destinations,ans):
    
    pos=[(172,345),(392,339),(466,178),(220,43),(84,114)]
    
    img = plt.imread("background.png")
    fig,ax = plt.subplots()
    ax.imshow(img,extent=[0,500,0,400])
    
    G = nx.Graph() 
    labels=[]
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * (len(destinations)+1)
    
    for i,s in enumerate(destinations):
        G.add_node(i,pos=pos[i])
        
        if i<len(path)-1:
            G.add_edge(path[i],path[i+1])
        
            
        labels.append(f'{i} - {destinations[i]}')
    
    labels.append(f'Min path - {ans} min')
    
    plt.legend(handles, labels, loc='best', fontsize='small', fancybox=True, framealpha=0.6, handlelength=0, handletextpad=0)

    nx.draw(G,pos,node_size=90,node_color='red',with_labels=True,font_size=8)

    plt.show()
    

# Drive Code
def main():
    # matrix for testing TSP
    #dist = [[0, 10, 15, 20],[10, 0, 35, 25],[15, 35, 0, 30],[20, 25, 30, 0]]
    
    input_file='data/routes.txt' #'data/data_test.txt'
    try:
        estimate_time_matrix,routes,initial_time,number_of_destinations,destinations=reading_input_data(input_file)   
    except FileNotFoundError:
        print('Wrong file or file path')
        return 0
    
    print(f'\nDestinations salesman want to visit:\n{destinations}')
    
    # final_path[] stores the final solution
    # i.e. the // path of the salesman.
    final_path = [None] * (number_of_destinations + 1)

    # visited[] keeps track of the already
    # visited nodes in a particular path
    visited = [False] * number_of_destinations
    
    # Stores the final minimum weight
    # of shortest tour.
    global final_res
    final_res = maxsize
    
    path,ans=branch_and_bound_TSP(estimate_time_matrix, number_of_destinations,final_path)
    
    #print(f"\nPath of the most efficient tour:\n{' -> '.join([destinations[i] for i in path])}")
    #print(f"Estimated cost of the most efficient tour: {ans} min\n")
    
    final_path = [None] * (number_of_destinations + 1)
    final_res = maxsize

    
    print('\n---------------------------------------------------------------------------------')
    
    print('\nInitial travel time for every route (from input file):')
    print(initial_time)
    
    print('\nInitial travel time matrix (from input file):')
    for i in estimate_time_matrix:
        print('\t'.join(map(str,i)))
        
    print('\n---------------------------------------------------------------------------------')
    
    # next two functions are time consuming, so be patient or just skip them this time.
    # news_data.txt file created by this two functions is already in 'data' directory
    
    '''
    urls=extracting_urls('Ремонт доріг київ') #'Ремонт доріг київ'
    news_scraper(urls)
    '''

    # uk_model.dat - NER model from lang-uk github with their training set 
    # workspace/mitie/mitie_ner_model_ver1.dat - NER model trained by the author of this program with training data set taken from lang-uk repository

    loc_entities=ner_locs('workspace/mitie/mitie_ner_model_ver1.dat','data/news_data_13_05.txt') #uk_model.dat #workspace/mitie/mitie_ner_model_ver1.dat #data/news_data.txt
    
    new_time=road_repair_status_check(loc_entities,routes,initial_time)
    new_time_matrix=updating_time_matrix(estimate_time_matrix,number_of_destinations,new_time)

    path,ans=branch_and_bound_TSP(new_time_matrix, number_of_destinations,final_path)
    
    print('\nUpdated travel time matrix:')
    for i in new_time_matrix:
        print('\t'.join(map(str,i)))
        
    print('\n---------------------------------------------------------------------------------')
    
    print(f"\nPath of the most efficient tour:\n{' -> '.join([destinations[i] for i in path])}")
    print(f"Estimated cost of the most efficient tour: {ans} min\n")
    visualize_graph(path,destinations,ans)
       
    
if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')
