
# coding: utf-8

# In[ ]:

from itertools import combinations_with_replacement
import numpy as np


# In[ ]:

def euclidean_similarity(vector1,vector2):
    """Compute the Euclidean similarity (i.e : - Euclidean distance) between 2 vectors
    ----------------
    Parameter types:
    vector1: iterable
    vector2: iterable """
    sim = [(a - b)**2 for a, b in zip(vector1, vector2)]
    sim = -sum(sim)
    return(sim)

def parseLine(line, sep) :
    """ Parse the RDD
    ----------------
    Parameter types:
    line: string
    sep: string (ex: ',' or ';')"""
    parsed_line = line.split(sep)
    parsed_line =  [float(_) for _ in parsed_line]
    return(parsed_line)

def similarity_matrix_per_iterator(iterator, similarity_function):
    """ Compute the similarity matrix
    ----------------
    Parameter types:
    iterator: iterator from mapPartitions
    similarity_function: function (ex: euclidean_similarity)"""
    #Store partition in a list and reindex it
    partition = list(iterator)
    nb_elements = len(partition)
    ind = range(nb_elements)
    #Compute similarities
    similarity_matrix = np.empty([nb_elements, nb_elements]) #More efficient than np.zeros to initialize
    for item1,item2 in combinations_with_replacement(zip(ind, partition),2) :
        sim = similarity_function(item1[1][1:],item2[1][1:]) #compute similarity
        similarity_matrix[item1[0], item2[0]] = sim
        if item1[0] != item2[0]:
            similarity_matrix[item2[0], item1[0]] = sim
    return(nb_elements, ind, partition, similarity_matrix)

def add_preferences_to_similarity_matrix(similarity_matrix, preference_type ="median"):
    """ Update the preference value in the similarity matrix, i.e : updates the value of s(k,k)
    Several methods are available depending on the number of clusters expected by the user :
    -'median' : default method. Takes the median of the similarities as the preference value. 
    This method leads to a medium to large number of clusters
    -'minimum' : Takes the minimum of the similarities as the preference value. This method leads to a small number of clusters.
    -'random' : Takes a random number in the preference range mentioned in the article as the preference value. 
    ----------------
    Parameter types:
    similarity_matrix: symetric numpy array
    preference_type: string (only "median" is available for the moment)"""
    if preference_type == "median":
        preference_value = np.median(similarity_matrix)
    np.fill_diagonal(similarity_matrix, preference_value)

def update_responsibility_and_availability(similarity_matrix, responsibility_matrix, availability_matrix, ind, lambda_damping):
    """ Update the responsibility matrix and the availability matrix 
    ----------------
    Parameter types:
    similarity_matrix: symetric numpy array
    responsibility_matrix: numpy array
    availability_matrix: numpy array
    ind: list or numpy vector (list of the "new" index, i.e : index on the partition and not globally in the dataset)
    lambda_damping: float between 0 and 1"""
    #1. Update Responsibility
    temp_sum_availability_similarity = np.add(availability_matrix, similarity_matrix) #compute a(i,k) + s(i,k) for all i,k
    argmax_resp = np.argmax(temp_sum_availability_similarity, axis = 1 ) #compute argmax { a(i,k') + s(i,k') } on ALL k' when i is fixed
    max_resp =  temp_sum_availability_similarity[ind,argmax_resp][:,None] #NB : [:,None] converts the row of max values into a column
    #Compute the new r(i,k) when k is not equal to argmax { a(i,k') + s(i,k') }
    temp_responsibility_matrix = np.subtract(similarity_matrix,max_resp) 
    #Compute the new r(i,k) when k is equal to  argmax { a(i,k') + s(i,k') } :
    temp_sum_availability_similarity[ind,argmax_resp] = -np.inf #Set the (i,k)th element to -inf when k is an argmax for row i
    max_resp2 = np.max(temp_sum_availability_similarity, axis = 1) #compute new max for row i (the real max value of the row being set to -inf here)
    temp_responsibility_matrix[ind,argmax_resp] = similarity_matrix[ind,argmax_resp] - max_resp2
    #Damping
    responsibility_matrix = (1 - lambda_damping) * temp_responsibility_matrix + lambda_damping*responsibility_matrix
    #Remove temporary variables
    del temp_sum_availability_similarity
    del temp_responsibility_matrix
    
    #2. Update Availability for i != k
    temp_matrix = np.maximum(0, responsibility_matrix)
    temp_matrix = np.sum(temp_matrix, axis = 0) - temp_matrix #Compute el(i,k) = sum(max(0,r(i',k)) - max(0, r(i,k))
    temp_matrix_diag = temp_matrix.diagonal() #Store the updated a(k,k) values
    temp_matrix = np.minimum(0, responsibility_matrix.diagonal()) + temp_matrix
    temp_matrix = np.minimum(0, temp_matrix)

    #3. Update Availability for i ==k
    np.fill_diagonal(temp_matrix, temp_matrix_diag) #set the diagonal values to the updated a(k,k) values computed above
    
    #Damping
    availability_matrix = (1 - lambda_damping) * temp_matrix + lambda_damping*availability_matrix
    return(responsibility_matrix, availability_matrix)

def center_decision(responsibility_matrix, availability_matrix):
    """ Find the center chosen by each individual 
    (using the "new" index, i.e: the index on the partition and not the global index of the dataset)
    ----------------
    Parameter types:
    responsibility_matrix: numpy array
    availability_matrix: numpy array"""
    temp_matrix = responsibility_matrix + availability_matrix
    decision = np.argmax(temp_matrix, axis = 1)
    return(decision)

def affinity_propagation_clustering_per_partition(iterator, similarity_function = euclidean_similarity , preference_type ="median", 
                                                  lambda_damping = 0.5, max_iter = 200, max_unchange_count = 3):
    """
    Terminal conditions : max_iter or unchanged decisions for some number of iterations
    We need to add the terminal condition (2) of the article"""
    #1. Compute similarity matrix
    nb_elements, ind, partition, similarity_matrix = similarity_matrix_per_iterator(iterator, similarity_function)
    #similarity_matrix = convert_sim_to_numpy(nb_elements, similarity_matrix)
    #2. Compute preferences
    add_preferences_to_similarity_matrix(similarity_matrix, preference_type)
    #3. Initialize
    responsibility_matrix = np.zeros([nb_elements, nb_elements]) 
    availability_matrix = np.zeros([nb_elements, nb_elements])
    decision = ind
    unchange_count = 0
    #4. Update responsibility and availability until one of the terminal conditions is met
    for it in xrange(max_iter) :
        responsibility_matrix, availability_matrix = update_responsibility_and_availability(similarity_matrix, responsibility_matrix, 
                                                                                            availability_matrix, ind, lambda_damping)
        temp_decision = center_decision(responsibility_matrix, availability_matrix)
        if np.array_equal(temp_decision, decision) : 
            unchange_count = unchange_count + 1
            if unchange_count >=  max_unchange_count :
                decision = temp_decision
                break
        else: 
            decision = temp_decision
    #5. Aggregate points with the same centers and add the center coordinates (using the initial index)
    decision = sorted(zip(decision, ind), key = lambda x : x[0])
    from itertools import groupby
    output = list()
    for key, group in groupby(decision, lambda x: x[0]):
        output.append((partition[key][0], [partition[j][0] for i,j in group], partition[key][1:], similarity_matrix[0,0]))
    return(output)


# In[ ]:

def getRoots3(aNeigh):
    """ Connected Components Research in a Graph
    ----------------
    Parameter types:
    aNeigh: graph repesented by a dictionnary with key = node and value = list of connected nodes """
    def findRoot(aNode,aRoot):
        while aNode != aRoot[aNode][0]:
            aNode = aRoot[aNode][0]
        return (aNode,aRoot[aNode][1])
    myRoot = {} 
    for myNode in aNeigh.keys():
        myRoot[myNode] = (myNode,0)  
    for myI in aNeigh: 
        for myJ in aNeigh[myI]: 
            (myRoot_myI,myDepthMyI) = findRoot(myI,myRoot) 
            (myRoot_myJ,myDepthMyJ) = findRoot(myJ,myRoot) 
            if myRoot_myI != myRoot_myJ: 
                myMin = myRoot_myI
                myMax = myRoot_myJ 
                if  myDepthMyI > myDepthMyJ: 
                    myMin = myRoot_myJ
                    myMax = myRoot_myI
                myRoot[myMax] = (myMax,max(myRoot[myMin][1]+1,myRoot[myMax][1]))
                myRoot[myMin] = (myRoot[myMax][0],-1) 
    myToRet = {}
    for myI in aNeigh: 
        myToRet[myI] = findRoot(myI,myRoot)[0]
    return myToRet  


# In[ ]:

def affinity_propagation_clustering(name_file, sep = ',', cluster_aggregation_parameter = 0.5, similarity_function = euclidean_similarity,
                                   preference_type ="median", lambda_damping = 0.5, max_iter = 200, max_unchange_count = 3):
    """ Create an RDD from a text file and return the Affinity Propagation Clustering result 
    ----------------
    Parameter types:
    name_file: string (path to text file) """
    rdd = sc.textFile(name_file)
    rdd = rdd.map(lambda line : parseLine(line, sep))
    #1. Affinity Propagation Clustering on each partition
    rdd2 = rdd.mapPartitions(lambda iterator : affinity_propagation_clustering_per_partition(iterator, 
                                                                                             similarity_function = euclidean_similarity , 
                                                                                             preference_type = preference_type, 
                                                                                             lambda_damping = lambda_damping, 
                                                                                             max_iter = max_iter, 
                                                                                             max_unchange_count = max_unchange_count), 
                         preservesPartitioning=True)
    #2. Compute similarities between all centers
    rdd_dist = rdd2.cartesian(rdd2).map(lambda u : ((u[0][0], u[1][0]),euclidean_similarity(u[0][2], u[1][2]),
                                     cluster_aggregation_parameter*((u[0][3] + u[1][3])/2)))
    #3. Filter centers which are close to each other (the threshold depends on the preference values and the cluster_aggregation_parameter)
    edges = rdd_dist.filter(lambda u : u[1] > u[2]).map(lambda u: u[0])
    #4. Format the selected centers as the nodes of a graph where there is an edge is their similarity was higher than the threshold
    edges = edges.map(lambda nameTuple: (nameTuple[0], [ nameTuple[1] ]))     .reduceByKey(lambda a, b: a + b)  # combine lists: ([1,2,3] + [4,5]) becomes [1,2,3,4,5]
    #5. Collect results on the master CPU and find the connected components on the graph (centers that need to be aggregated in one center)
    connected_components = getRoots3(dict(edges.collect()))
    #6. Broadcast the results from the master to the workers
    connected_components_broadcast = sc.broadcast(connected_components)
    #7. Assign the new centers to the Affinity Propagation Clustering on each partition results
    rdd3 = rdd2.map(lambda (center, points, coordinates, preference) : (connected_components_broadcast.value[center], points))
    #8. Aggregate the lists of points that belong to the same cluster (same center = same key)
    result_clustering = rdd3.reduceByKey(lambda a, b: a + b)
    return(result_clustering)

