#调用函数 sDLP(g,h,p) 返回 g^x≡h (mod p) 的一个解
#Shanks's Babystep-Giantstep Algorithm
from gmpy2 import invert,iroot
from Crypto.Util.number import getPrime
g=391190709124527428959489662565274039318305952172936859403855079581402770986890308469084735451207885386318986881041563704825943945069343345307381099559075
h=391190709124527428959489662565274039318305952172936859403855079581402770986890308469084735451207885386318986881041563704825943945069343345307381099559075
p=13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084096

class node:
    def _init_(self):
        self.vue=0
        self.num=0
def cmp(a):
      return a.vue
def init_list(first,g,n,p):
      List=[]
      temp=node()
      temp.vue,temp.num=first,0
      List.append(temp)
      for i in range(1,n+1):
            temp=node()
            temp.num = i
            temp.vue = List[i-1].vue * g % p
            List.append(temp)
      List.sort(key=cmp)
      return List
def sDLP(a,b,p):
    ans=p
    n=iroot(p,2)[0]+1
    L1=init_list(1,a,n,p)
    aa=pow(invert(a,p),n,p)
    L2=init_list(b,aa,n,p)
    i = 0
    j = 0
    while True :
        if (i>=n or j>=n): break
        while (L1[i].vue < L2[j].vue and i<n): i += 1
        while (L1[i].vue > L2[j].vue and j<n): j += 1
        if L1[i].vue == L2[j].vue :
            x=L1[i].num+L2[j].num*n
            return int(x)
print(sDLP(g,h,p))