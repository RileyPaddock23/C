import itertools
 
def find_subsets(s, n):
    return list(itertools.combinations(s, n))

def powerset(n):
  p = []
  for size in range(len(n)):
    p += [list(x) for x in find_subsets(n,size)]
  
  return p+[n]

def check_disjoint(subset, disj_sub):
  for x in subset:
    if x in disj_sub:
      return False
  else:
    return True


def check_subset(poss_subsets):
  total = 0
  done = False
  for subset in poss_subsets:
    if done: break
    for disj_sub in poss_subsets:
      if sum(subset) == sum(disj_sub) and check_disjoint(subset, disj_sub):
        #   print(subset, disj_sub)
          total += 1
          done = True
          break

  if total >= 1:
    return True
  else:
    return False
s = 44
x = 7
record = [(2,2),(3,4),(4,7),(5,13),(6,24)]
done = False
while False:
    if done: break
    poss_groups = [list(j) for j in find_subsets([i for i in range(1,s+1)],x)]# every possible subset of 10 people ages 1-(s-1)
    # print(poss_groups)
    #note, we don't need to consider duplicate ages because then we just grab 2 people of the same age and have our groups
    total = 0
    for i,poss in enumerate(poss_groups):
        # print(str(round(i/len(poss_groups),3)), end="")
        poss_subsets = powerset(poss)
        poss_subsets.remove([])
        if check_subset(poss_subsets):#if every poss group has a 2 disjoint sets with the same age sum
            total += 1
        else:
            print(s,poss)
            print("above set had no equal disjoint subsets")
            done = True
            break

    if total == len(poss_groups):
        print("every possible group of ("+str(x)+") with ages up to ("+str(s)+") has an equal disjoint subset")
        s+=1

rec = "[12,134]"
split = rec.index(",")
print(split)
m = int(rec[1:split])
e = int(rec[split+1:-1])
print(m,e)
print(f"Recieved {[m,e]}")