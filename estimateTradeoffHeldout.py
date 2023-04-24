# This script estimates the memory-surprisal tradeoff from raw text data.
# It expects the data partitioned into train and held-out partitions, each represented as lists 
# of words. A simple way of preparing this split is by randomly selecting some fraction 
# (e.g. 20%) of sentences for the held-out partition, assigning the rest to the train partition,
# and then concatenating the sentences within a paritition (in random order, if discourse
# structure is to be disregarded) into two lists, serving as the ``train`` and ``dev``
# arguments of the function.
# For ``args``, the defaults should generally produce stable results. They can be tuned
# to minimize held-out surprisal if necessary.
# Feel free to contact Michael Hahn if you have any questions about usage.


from math import log, exp
def calculateMemorySurprisalTradeoff(train, dev, args):
    """ Input: 
        - train (list): training data for fitting an n-gram model
        - dev (list): held-out data for calculating surprisals
        - args (dict) with entries 
           - cutoff: the maximal n to construct an n-gram model (e.g., n=10)
           - gamma (int, e.g. 1): threshold for choosing between interpolation and backoff. If gamma > 1, rare prefixes are not used for interpolation, and trigger backoff instead.
           - alpha (e.g. 1.0): Parameter for absolute discounting in Kneser-Ney
           - delta (e.g. 1.0) parameter describing smoothing of unigram probabilities
        Returns:
          - auc: Area below the memory-surprisal tradeoff curve
          - devSurprisalTable: lists surprisal by n, from n=0 up to cutoff
    """
    itos = list(set(train) | set(dev))
      
    # For the algorithm, we will need to reverse the corpora
    dev = dev[::-1]
    train = train[::-1]

    # idev and itrain are suffix arrays: They contain indices to positions in the text, ordered by the alphabetical position of the suffixes starting at them.
    idev = range(len(dev))
    itrain = range(len(train))
    
    idev = sorted(idev, key=lambda i:dev[i:i+20])
    itrain = sorted(itrain, key=lambda i:train[i:i+20])
    
    # The inverse map of idev/itrain: For each position in the texts, indicates its position in those lists.    
    idevInv = [x[1] for x in sorted(zip(idev, range(len(idev))), key=lambda x:x[0])]
    itrainInv = [x[1] for x in sorted(zip(itrain, range(len(itrain))), key=lambda x:x[0])]
    
    # Basic sanity check
    assert idev[idevInv[5]] == 5
    assert itrain[itrainInv[5]] == 5
    
    
    # Helper function. For a given span length k (>= 0), computes for each position in the heldout set (`dev`), the location of the first and last entries in itrain where the same string of length k starts as after the given position in idev.
    def getStartEnd(k):
       start = [0 for _ in dev]
       end = [len(train)-1 for _ in dev]
       if k == 0:
          return start, end
       # Start is the FIRST train place that is >=
       # End is the FIRST train place that is >
       l = 0
       l2 = 0
       for j in range(len(dev)):
         prefix = tuple(dev[idev[j]:idev[j]+k])
         while l2 < len(train):
            prefix2 = tuple(train[itrain[l2]:itrain[l2]+k])
            if prefix <= prefix2:
                 start[j] = l2
                 break
            l2 += 1
         if l2 == len(train):
            start[j] = l2
         while l < len(train):
            prefix2 = tuple(train[itrain[l]:itrain[l]+k])
            if prefix < prefix2:
                 end[j] = l
                 break
            l += 1
         if l == len(train):
            end[j] = l
         start2, end2 = start[j], end[j]
         assert start2 <= end2
         if start2 > 0 and end2 < len(train):
           assert prefix > tuple(train[itrain[start2-1]:itrain[start2-1]+k])
           assert prefix <= tuple(train[itrain[start2]:itrain[start2]+k])
           assert prefix >= tuple(train[itrain[end2-1]:itrain[end2-1]+k])
           assert prefix < tuple(train[itrain[end2]:itrain[end2]+k])
       return start, end
    
    
    lastProbability = [None for _ in idev]
    newProbability = [None for _ in idev]
    
    devSurprisalTable = []
    for k in range(0,args.cutoff):
#       print(k)
       startK, endK = getStartEnd(k) # Possible speed optimization: There is some redundant computation here, could be reused from the previous iteration. But the algorithm is very fast already.
       startK2, endK2 = getStartEnd(k+1)
       cachedFollowingCounts = {}
       for j in range(len(idev)):
    #      print(dev[idev[j]])
          if dev[idev[j]] in ["PAD", "SOS"]:
             continue
          start2, end2 = startK2[j], endK2[j]
          devPref = tuple(dev[idev[j]:idev[j]+k+1])
          if start2 > 0 and end2 < len(train):
            assert devPref > tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]), (devPref, tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]))
            assert devPref <= tuple(train[itrain[start2]:itrain[start2]+k+1]), (devPref, tuple(train[itrain[start2]:itrain[start2]+k+1]))
            assert devPref >= tuple(train[itrain[end2-1]:itrain[end2-1]+k+1])
            assert devPref < tuple(train[itrain[end2]:itrain[end2]+k+1])
    
          assert start2 <= end2
    
          countNgram = end2-start2
          if k >= 1:
             if idev[j]+1 < len(idevInv):
               prefixIndex = idevInv[idev[j]+1]
               assert dev[idev[prefixIndex]] == dev[idev[j]+1]
       
               prefixStart, prefixEnd = startK[prefixIndex], endK[prefixIndex]
               countPrefix = prefixEnd-prefixStart
               if countPrefix < args.gamma: # there is nothing to interpolate with, just back off
                  assert k > 0
                  newProbability[j] = lastProbability[j]
               else:
                  assert countPrefix >= countNgram, (countPrefix, countNgram)
       
                  following = set()
                  if (prefixStart, prefixEnd) in cachedFollowingCounts:
                      followingCount = cachedFollowingCounts[(prefixStart, prefixEnd)]
                  else:
                    for l in range(prefixStart, prefixEnd):
                      if k < itrain[l]+1:
                         following.add(train[itrain[l]-1])
                         assert devPref[1:] == tuple(train[itrain[l]-1:itrain[l]+k])[1:], (k, itrain[l], l, devPref , tuple(train[itrain[l]-1:itrain[l]+k]))
                    followingCount = len(following)
                    cachedFollowingCounts[(prefixStart, prefixEnd)] = followingCount
                  if followingCount == 0:
                      newProbability[j] = lastProbability[j]
                  else:
                      #assert countNgram > 0 # Sanity check when using the naive estimator only.
                      probability = log(max(countNgram - args.alpha, 0.0) + args.alpha * followingCount * exp(lastProbability[j])) -  log(countPrefix)
                      newProbability[j] = probability
             else:
                newProbability[j] = lastProbability[j]
          elif k == 0:
                  probability = log(countNgram + args.delta) - log(len(train) + args.delta * len(itos))
                  newProbability[j] = probability
       lastProbability = newProbability 
       newProbability = [None for _ in idev]
       assert all([x is None or x <=0 for x in lastProbability])
       try:
           lastProbabilityFiltered = [x for x in lastProbability if x is not None]
           surprisal = - sum([x for x in lastProbabilityFiltered])/len(lastProbabilityFiltered)
       except ValueError:
    #       print >> sys.stderr, "PROBLEM"
     #      print >> sys.stderr, lastProbability
           surprisal = 1000
       devSurprisalTable.append(surprisal)
     #  print("Surprisal", surprisal, len(itos))
    #print(devSurprisalTable)
    for k in range(len(devSurprisalTable)):
        devSurprisalTable[k] = min(devSurprisalTable[:k+1])
    mis = [devSurprisalTable[i] - devSurprisalTable[i+1] for i in range(len(devSurprisalTable)-1)]
    print(mis)
    tmis = [mis[x]*(x+1) for x in range(len(mis))]
    #print(mis)
    #print(tmis)
    auc = 0
    memory = 0
    mi = 0
    for i in range(len(mis)):
       mi += mis[i]
       memory += tmis[i]
       auc += mi * tmis[i]
    #print("MaxMemory", memory)
    assert 7>memory
    auc += mi * (7-memory)
    #print("AUC", auc)
    auc = 7*devSurprisalTable[0] - auc
    print(mis)
    return auc, devSurprisalTable

