from typing import List
from itertools import combinations


nums = [3,0,-2,-1,1,2]
result = []
for ele in nums:
    tmp = nums.copy()
    tmp.remove(ele)
    F = False
    for ele2 in tmp:
        if F:
            break
        else:
            tmp2 = tmp.copy()
            tmp2.remove(ele2)
            for ele3 in tmp2:
                if F:
                    break
                else:
                    if ele3 == -(ele + ele2):
                        h = [ele, ele2, ele3]
                        h.sort()
                        if h not in result:
                            result.append(h)
                            F = True


print(result)

    






