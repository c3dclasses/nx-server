#------------------------------------------------
# file: nx-functions.py
# desc: defines functions used by the nx-server
#------------------------------------------------

from statistics import median
from statistics import mean
import networkx as nx
import collections
import json
import logging
import requests
import time


####################################
# nx request and response functions

#-----------------------------------------------
# name: nx_do_request()
# desc: performs a request
#-----------------------------------------------
def nx_do_request(request_url, request_payload=None):
    # These two lines enable debugging at httplib level (requests->urllib3->http.client)
    # You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
    # The only thing missing will be the response.body which is not logged.
    try:
        import http.client as http_client
    except ImportError:
        # Python 2
        import httplib as http_client

    http_client.HTTPConnection.debuglevel = 1

    # You must initialize loggi++*ng, otherwise you'll not see debug output.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

    # use empty user agent so server's Mod_Security don't complain 
    headers= {
        "User-Agent":"",
        "Accept": "*/*",
        'Content-Type': 'application/json'
    }
    # try to get the json input from the server
    try:
        if(request_payload is not None):
            print("setting the data -", request_payload)
            return requests.post(request_url, headers=headers, data=json.dumps(request_payload),verify=False)
        else: 
            print("getting the data")
            return requests.request("get", request_url, headers=headers, data=None).json()
    except Exception as e:
        print("ERROR: nx_do_request() - ", e)
        return None
## end nx_do_request()


#-----------------------------------------------
# name: do_request()
# desc: does a request
#-----------------------------------------------
def nx_do_request(request_url, request_payload=None):
    # These two lines enable debugging at httplib level (requests->urllib3->http.client)
    # You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
    # The only thing missing will be the response.body which is not logged.
    try:
        import http.client as http_client
    except ImportError:
        # Python 2
        import httplib as http_client

    http_client.HTTPConnection.debuglevel = 1

    # You must initialize loggi++*ng, otherwise you'll not see debug output.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

    # use empty user agent so server's Mod_Security don't complain 
    headers= {
        "User-Agent":"",
        "Accept": "*/*",
        'Content-Type': 'application/json'
    }
    # try to get the json input from the server
    try:
        if(request_payload is not None):
            print("setting the data -", request_payload)
            return requests.post(request_url, headers=headers, data=json.dumps(request_payload),verify=False)
        else: 
            print("getting the data")
            return requests.request("get", request_url, headers=headers, data=None).json()
    except Exception as e:
        print("ERROR: nx_do_request() - ", e)
        return None
# end nx_do_request()


########################################
# nx get/set/exec call input functions

#--------------------------------------------------------------
# name: nx_get_call_input()
# desc: gets the nx call object containing input to pass to nx 
#---------------------------------------------------------------
def nx_get_call_input():
    return nx_do_request("http://www.kevlewis.com/projects/nx/nx-get-call-input.php")


#--------------------------------------------------------------------------
# name: nx_set_call_output()
# desc: sets the nx call object by updating the call object with new data 
#--------------------------------------------------------------------------
def nx_set_call_output(output_json):
    return nx_do_request("http://www.kevlewis.com/projects/nx/nx-set-call-output.php", output_json)


#---------------------------------------------------------------
# name: nx_exec_call()
# desc: perform the networkx request
#---------------------------------------------------------------
def nx_exec_call(call_input):
    if(call_input is None):
        return None
    
    # check if there is a call object
    if(call_input["input"]["function_name"] == "erdos-renyi"):
        return nx_erdos_renyi_graph(call_input)
   
    if(call_input["input"]["function_name"] == "barabasi-albert"):
        return nx_barabasi_albert_graph(call_input)
    return None
# end nx_exec_call()


##---------------------------------------------------------
## name: nx_update_status()
## desc: updates the status of the call object
##---------------------------------------------------------
def nx_update_status(call_obj, percentage, message):
    status = call_obj["status"]
    status["message"] = message
    status["percentage"] = percentage
    nx_set_call_output(call_obj)


#######################################
# _barabasi albert graphs

##---------------------------------------------------------
## name: nx_barabasi_albert_graph()
## desc: 
##---------------------------------------------------------
def nx_barabasi_albert_graph(call_obj):
    function_params = call_obj["input"]["function_params"]
    function_name = call_obj["input"]["function_name"]  
    if(function_params is None):
        return 
        
    n = int(function_params["n"])
    m = int(function_params["m"])
    t = int(function_params["t"])
    profiletypes = function_params["vprofiletype"]
    bcharts = bool(function_params["bcharts"])

     # run the er algorithm and return the graph
    output={}
    if(bcharts is False):
        nx_update_status(call_obj, "5", f"Network X Functions - starting barabasi_albert_graph on n={n}, m={m}.....")
        G = nx.barabasi_albert_graph(n, m, seed=None) 
        nx_update_status(call_obj, "5", f"Network X Functions - finished barabasi_albert_graph on n={n}, m={m}.....")
        output["nodes"]=[list(G.nodes), None]
        output["edges"]=[list(G.edges), None]
    else:
        nx_update_status(call_obj, "5", f"Network X Functions - Starting ER Trails on n={n}, m={m}, t={t}.....")
        output["charts"] = do_ba_trails(n,m,t,profiletypes,call_obj)
        nx_update_status(call_obj, "90", f"Network X Functions - Completed ER Trails on n={n}, m={m}, t={t}.....")

    if(bcharts):    
        output["charts"]["n"] = n
        output["charts"]["m"] = m
        output["charts"]["t"] = t
        output["charts"]["name"] = function_name
        output["charts"]["bcharts"] = bcharts

    # set the output object
    call_obj["output"] = output

    # set the status object
    call_obj["status"]["percentage"] = 100
    call_obj["status"]["message"] = "done"
    return call_obj
## end nx_barabasi_albert_graph()


def do_ba(n,m):
    return nx.barabasi_albert_graph(n, m, seed=None) 
        

##--------------------------------------------------------------
## name: do_ba_trails()
## desc: returns histograms for t generated BA graphs
##--------------------------------------------------------------
def do_ba_trails(n, m, t, profiletypes, call_obj):
    if(t <= 0):
        return None
    histograms = {}
    for i in range(t):
        nx_update_status(call_obj, "50", f"started trail: {i+1}/{t}")
        G = do_ba(n, m)	
        profile_sequences = compute_graph_profile_sequences(G, profiletypes)
        for profile_name, profile_sequence in profile_sequences.items():
           # if(profile_name == "deg"):
            try:
                histograms[profile_name] = get_profile_histogram(histograms[profile_name], profile_sequence)
            except:
                histograms[profile_name] = get_profile_histogram({}, profile_sequence)
            histograms[profile_name]["_name"] = profile_name
            nx_update_status(call_obj, "50", f"processing trail: {i+1}/{t} - generating histogram for profile: {profile_name}")
            
        nx_update_status(call_obj, "50", f"completed trail: {i+1}/{t}")
       
    return {
        "histograms": histograms,
        "n": n,
        "m": m,
        "t": t
    }
# end do_ba_trails()     


#######################################
# erdos renyi graphs

#--------------------------------------------------------------
# name: nx_erdos_renyi_graph()
# desc: perform the er model
#--------------------------------------------------------------
def nx_erdos_renyi_graph(call_obj):
    # get the input
    if(call_obj["input"]):
        function_name = call_obj["input"]["function_name"]  
        function_params = call_obj["input"]["function_params"]
        if(function_params):
            n = int(function_params["n"])
            p = float(function_params["p"])
            p2 = float(function_params["p2"])
            t = int(function_params["t"])
            pstep = float(function_params["pstep"]) 
            bcharts = bool(function_params["bcharts"])
            bpoverx = bool(function_params["bpoverx"])
            profiletypes = function_params["vprofiletype"]
            maxsequencelength = function_params["maxsequencelength"]
            maxvertexcount = function_params["maxvertexcount"]
        # end if
    # end if

    # run the er algorithm and return the graph
    output={}
    if(bcharts is False):
        G = do_er(n, p)  
        output["nodes"]=[list(G.nodes), None]
        output["edges"]=[list(G.edges), None]
    elif(bpoverx):
        nx_update_status(call_obj, "5", f"Network X Functions - Starting ER Trails over P on n={n}, p1={p}, p2={p2}, pstep={pstep}, t={t}.....")
        output["charts"] = do_er_trails_over_p(n,p,p2,pstep,t,profiletypes,call_obj)
        nx_update_status(call_obj, "90", f"Network X Functions - Completed ER Trails over P on n={n}, p1={p}, p2={p2}, pstep={pstep}, t={t}.....")
    else:
        nx_update_status(call_obj, "5", f"Network X Functions - Starting ER Trails on n={n}, p={p}, t={t}.....")
        output["charts"] = do_er_trails(n,p,t,profiletypes,call_obj)
        nx_update_status(call_obj, "90", f"Network X Functions - Completed ER Trails on n={n}, p={p}, t={t}.....")
        
    if(bpoverx or bcharts):    
        output["charts"]["name"] = function_name
        output["charts"]["n"] = n
        output["charts"]["p"] = p
        output["charts"]["p1"] = p
        output["charts"]["p2"] = p2
        output["charts"]["pstep"] = pstep
        output["charts"]["t"] = t
        output["charts"]["bcharts"] = bcharts
        output["charts"]["bpoverx"] = bpoverx
        output["charts"]["maxvertexcount"] = maxvertexcount
        output["charts"]["maxsequencelength"] = maxsequencelength
        
    # set the output object
    call_obj["output"] = output
    # set the status object
    call_obj["status"]["percentage"] = 100
    call_obj["status"]["message"] = "done"
    
    return call_obj
# end nx_erdos_renyi_graph()

##--------------------------------------------------------------
## name: do_er()
## desc: generates and returns an ER graph
##--------------------------------------------------------------
def do_er(n, p):
    return nx.erdos_renyi_graph(n, p, None, directed=False)
# end do_er()


##--------------------------------------------------------------
## name: do_er_trails()
## desc: returns histograms for t generated ER graphs
##--------------------------------------------------------------
def do_er_trails(n, p, t, profiletypes, call_obj):
    if(t <= 0):
        return None
    histograms = {}
    for i in range(t):
        nx_update_status(call_obj, "50", f"started trail: {i+1}/{t}")
        G = do_er(n, p)	
        profile_sequences = compute_graph_profile_sequences(G, profiletypes)
        for profile_name, profile_sequence in profile_sequences.items():
           # if(profile_name == "deg"):
            try:
                histograms[profile_name] = get_profile_histogram(histograms[profile_name], profile_sequence)
            except:
                histograms[profile_name] = get_profile_histogram({}, profile_sequence)
            histograms[profile_name]["_name"] = profile_name
            nx_update_status(call_obj, "50", f"processing trail: {i+1}/{t} - generating histogram for profile: {profile_name}")
            
        nx_update_status(call_obj, "50", f"completed trail: {i+1}/{t}")
       
    return {
        "histograms": histograms,
        "n": n,
        "p": p,
        "t": t
    }
# end do_er_trails()     


##-------------------------------------------------------------------------------------------
## name: do_er_trails_over_p()
## desc: returns the highest vertexcounts and sequence length over arange of probabilities
##-------------------------------------------------------------------------------------------
def do_er_trails_over_p(n,p1,p2,pstep,t,profiletypes,call_obj):
    p_over_x = []
    highest_vertex_counts = {}
    sequence_lengths = {}
    hists = []
    ##p1 = //int(p1*100)
    ##p2 = int(p2*100)
    ##pstep = int(pstep*100)
    
    print(n, p1, p2, pstep, t)

    p = p1
    while(p<=p2):
        ##for prob in range(p1,p2+1,pstep):
        ##    p = prob/100
        nx_update_status(call_obj, "50", f"started probability: {p}")
        do_er_trails_out = do_er_trails(n,p,t,profiletypes,call_obj)
        histograms = do_er_trails_out["histograms"]
        p_over_x.append(p)
        hists.append(histograms)
        for profile_name, histogram in histograms.items():
            try:
                highest_vertex_counts[profile_name].append(histogram["_max_vc"])
                sequence_lengths[profile_name].append(histogram["_max_len"])
            except:
                highest_vertex_counts[profile_name]=[]
                highest_vertex_counts[profile_name].append(histogram["_max_vc"])
                sequence_lengths[profile_name]=[]
                sequence_lengths[profile_name].append(histogram["_max_len"])
        nx_update_status(call_obj, "50", f"ending probability: {p}")
        
        if(p < 1.0 and (p + pstep) > 1.0):
            p = 1.0
        else:
            p += pstep


    return {
        "_p_over_x": p_over_x,
        "_max_vc": highest_vertex_counts,
        "_max_len": sequence_lengths,
        "_histograms": hists
    } ## end return
# end do_er_trails_over_p

##############################
# other functions

#---------------------------------------------------------------
# name: compute_graph_profile_sequences()
# desc: computes the neighborhood profiles for a graph
#---------------------------------------------------------------
def compute_graph_profile_sequences(G, profiletypes):
    g_profile = {}
    for v in list(G.nodes):
        v_profile = compute_vertex_profiles(v, G)
        for profile_type, bused in profiletypes.items():
            ##for profile_type, value in v_profile.items():
            if profile_type not in g_profile:
                g_profile[profile_type] = []
            #g_profile[profile_type].append(value)
            g_profile[profile_type].append(v_profile[profile_type])
        # end for
    # end for
    return g_profile
# end compute_graph_profile_sequences()


#-----------------------------------------------------------------
# name: compute_vertex_profiles()
# desc: computes the neighborhoodcompute_vertex_profiles profile for given vertex
#-----------------------------------------------------------------
def compute_vertex_profiles(v1, G):
    deg1 = G.degree[v1]
    imin = deg1
    imax = deg1
    emax = 0
    emin = 0 if (deg1 <= 0) else deg1
    esum = 0
    isum = deg1
    ifreq = deg1
    efreq = 0
    degtypes = {}
    degrees = []
    if(deg1 > 0):
        for v2 in list(G.adj[v1]):
            deg2 = G.degree[v2]
            degrees.append(deg2)
            isum += deg2
            esum += deg2
            if(degtypes is None or deg2 not in degtypes):
                degtypes[deg2] = 0
            degtypes[deg2] += 1
            imin = int(min(imin,deg2))
            emin = int(min(emin,deg2))
            imax = int(max(imax,deg2))
            emax = int(max(emax,deg2))
        ## end for
    ## end if

    ## set the profiles
    profiles = {}
    profiles["deg"] = deg1
    profiles["imin"] = imin
    profiles["emin"] = emin
    profiles["imax"] = imax
    profiles["emax"] = emax
    profiles["isum"] = isum
    profiles["esum"] = esum
    iavg = (isum/(deg1+1)) if deg1>0 else 0
    eavg = (esum/deg1) if deg1>0 else 0
    profiles["iavg"] = iavg
    profiles["eavg"] = eavg
    
    # exclusive
    ndegrees = list(degtypes.keys())
    ndegfreq = list(degtypes.values())
    ediv = len(ndegrees)    # diversity of degrees
    emed = median(ndegrees) # median
    ndegfreq.sort(reverse=True)
    
    efreq = freq(degtypes) #ndegfreq[0] if(len(ndegfreq)>0) else 0
    print(efreq)

    if(deg1 not in degtypes):
        degtypes[deg1] = 0

    # inclusive
    degtypes[deg1] += 1
    degrees.append(deg1)
    ndegrees = list(degtypes.keys())
    ndegfreq = list(degtypes.values())
    idiv = len(ndegrees) 
    imed = median(ndegrees)
    ndegfreq.sort(reverse=True)   # sort reverse 
    ifreq =  freq(degtypes) #ndegfreq[0] if(len(ndegfreq)>0) else 0
    
    # set the other profiles
    profiles["ifreq"] = ifreq
    profiles["efreq"] = efreq
    profiles["idiv"] = idiv
    profiles["ediv"] = ediv
    profiles["imed"] = imed
    profiles["emed"] = emed
    profiles["iran"] = imax-imin
    profiles["eran"] = emax-emin
    profiles["hap"] = deg1-eavg
    
    print(profiles)
    return profiles
# end compute_vertex_profiles()


##----------------------------------------------------------------------
## name: get_profile_histogram()
## desc: returns histograms the histogram for a given profile sequence
##----------------------------------------------------------------------
def get_profile_histogram(histogram, profile_sequence):
    if(len(histogram) == 0):
        histogram = {}
        histogram["_max_len"] = 0
        histogram["_max_vc"] = 0
        histogram["_min_len"] = 0
        histogram["_min_vc"] = 0
        histogram["_seq_prf"] = []
        histogram["_min"] = {}
        histogram["_sum"] = {}
        histogram["_avg"] = {}
        histogram["_max"] = {}
        histogram["_err"] = {}
        histogram["low_degree"] = 0
        histogram["high_degree"] = 0

    _min = histogram["_min"]
    _max = histogram["_max"]
    _sum = histogram["_sum"]
    _avg = histogram["_avg"]
    _err = histogram["_err"]
    _seq_prof = histogram["_seq_prf"]

    degreeCount = collections.Counter(profile_sequence)
    _seq_prof.append(dict(degreeCount))
    histogram["_max_len"] = max(histogram["_max_len"], len(degreeCount))     # the maximum length of the vertex in the sequence
    for degree, count in degreeCount.items():
        _min[degree] = min(_min.get(degree, count), count)
        _max[degree] = max(_max.get(degree, count), count)
        _sum[degree] = _sum.get(degree, 0) + count
        _avg[degree] = _sum[degree]
        _err[degree] = (_max.get(degree) - _min.get(degree)) * 0.5
        histogram["_max_vc"] = max(histogram["_max_vc"], _max.get(degree))     # highest vertex in the sequence
        histogram["low_degree"] = min(histogram["low_degree"], degree)
        histogram["high_degree"] = max(histogram["high_degree"], degree)
      
    return histogram
# end get_profile_histogram()

def freq(degfreq):
	freqdeg = []
	maxv = -1
	for deg, freq in degfreq.items():
		if(freq > maxv):
			maxv = freq
			freqdeg = [int(deg)]
		elif(freq == maxv):
			freqdeg.append(int(deg))
	if(maxv == -1):
		return 0
	return mean(freqdeg) if(len(freqdeg) > 1) else freqdeg[0]

#G = nx.gnp_random_graph(100,0.02)
#print(compute_graph_profile_sequences(G))
#print(json.dumps(do_er_trails(5,0.5,2), indent=2))
#print("")
#print("")
#print(json.dumps(do_er_trails_over_p(4, 0, 100, 33, 2), indent=2))
#do_er_trails(5,0.5,2)