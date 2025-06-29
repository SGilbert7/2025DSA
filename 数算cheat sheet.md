# 数算cheat sheet

## 一、几大主要板块

### 1.链表linked-list

```python
#单向链表
class Listnode:
    def __init__(self,val=0,next=None)：
        self.val=val
        self.next=next
def insert(head,index,val):   #在特定位置插入某值#
    new_node=Listnode(val)
    if index==0:
         new_node.next=head
         return new_node
    cur=head
    pre=None
    cnt=0
    while cur and cnt<index:
         pre=cur
         cur=cur.next
         cnt+=1
    if cnt=index and pre:
         pre.next=new_node
         new_node.next=cur
         return head
    return head
def delete(head,val):   #删去某个特定值所在节点
    dummy=Listnode(0)
    dummy.next=head
    pre,cur=dummy,head
    while cur:
        if cur.val==val:
            pre.next=cur.next
            break
        pre,cur=cur,cur.next
    return dummy.next
#双向链表
class Listnode:
    def __init__(self,val=0,prev=None,next=None):
        self.val=val
        self.prev=prev
        self.next=next
#反转链表
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur=head
        pre=None
        while cur:
            tmp=cur.next
            cur.next=pre
            pre=cur
            cur=tmp
        return pre
```

### 2.树

```python
#二叉树
class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right
def preOrder(node):      #前序遍历
    if node is None:
        return []
    return [node.val]+preOrder(node.left)+preOrder(node.right)
def inOrder(node):       #中序遍历
    if node is None:
        return []
    return inOrder(node.left)+[node.val]+inOrder(node.right)
def postOrder(node):     #后序遍历
    if node is None:
        return []
    return postOrder(node.left)+postOrder(node.right)+[node.val]
def levelOrder(root):    #层序遍历
        p = []
        r = deque([root])
        if root:
            p.append(root.val)
        while r:
            node = r.popleft()
            if node:
                if node.left:
                    r.append(node.left)
                    p.append(node.left.val)
                if node.right:
                    r.append(node.right)
                    p.append(node.right.val)
        return p
#多叉树
class TreeNode:
    def __init__(self,val,child=None):
        self.val=val
        self.child=child if child else []
#二叉树的操作
def left_rotate(node):  #左旋
    if not node or not node.right:
        return node
    x=node.right
    y=x.left
    x.left=node
    node.right=y
    return x
def right_rotate(node):  #右旋
    if not node or not node.left:
        return node
    x=node.left
    y=x.right
    x.right=node
    node.left=y
    return x
#Huffman编码树
import heapq
class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None
    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight
def huffman_encoding(char_freq):
    heap = [Node(freq, char) for char, freq in char_freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]
def external_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.weight
    return (external_path_length(node.left, depth + 1) +
            external_path_length(node.right, depth + 1))
```

### 3.并查集

```python
class DisjointSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def Union(self, x, y):
        root_x= self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        self.parent[root_y] = root_x
```

### 4.栈

```python
##最小栈
class Minstack:
    def __init__(self):
        self.stack=[]
        self.min_stack=[]
    def push(self,x):
        self.stack.append(x)
        if not self.min_stack:
            self.min_stack.append(x)
        else:self.min_stack.append(min(x,self.min_stack[-1]))
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
    def top(self):
        return self.stack[-1]
    def getMin(self):
        return self.min_stack[-1]
##中序转后序表达式
import re
k={'+':1,'-':1,'*':2,'/':2}
n=int(input())
for _ in range(n):
    ss=input()
    s=re.split(r'([+\-*/()])',ss)
    s=list(y for y in s if y!='')
    stack=[]
    p=[]
    for x in s:
        if x not in k and x!='(' and x!=')':
            p.append(x)
        elif x=='(':
            stack.append(x)
        elif x==')':
            t=stack.pop()
            while t!='(':
                p.append(t)
                t=stack.pop()
        else:
            while stack and stack[-1]!='(' and k[x]<=k[stack[-1]]:
                p.append(stack.pop())
            stack.append(x)
    while stack:
        p.append(stack.pop())
    print(' '.join(p))
```



## 二、重要算法

### 1.归并排序

```python
def mid(p):
    if len(p)<=1:
        return p,0
    m=len(p)//2
    left,l=mid(p[:m])
    right,r=mid(p[m:])
    Q,q=cnt(left,right)
    ans=l+r+q
    return Q,ans  #输出排序后的序列和逆序数
def cnt(left,right):
    c=[]
    i,j=0,0
    v=0
    while i < len(left) and j <len(right):
        if left[i]<=right[j]:
            c.append(left[i])
            i+=1
        else:
            c.append(right[j])
            j+=1
            v+=len(left)-i
    c.extend(left[i:])
    c.extend(right[j:])
    return c,v
```

### 2.拓扑排序

```python
from collections import deque
n,m=map(int,input().split())
    graph={x:[] for x in range(1,n+1)}
    degree={x:0 for x in range(1,n+1)}
    for _ in range(m):
        x,y=map(int,input().split())
        degree[y]+=1
        graph[x].append(y)
    r=deque([x for x in degree if degree[x]==0])
    cnt=0
    t_order=[]   ##储存拓扑排序
    while r:
        node=r.popleft()
        cnt+=1
        t_order.append(node)
        for v in graph[node]:
            degree[v]-=1
            if degree[v]==0:
                r.append(v)
    if cnt==len(graph):   ##无环
    else:        ##有环
        
注：若拓扑排序需要按节点顺序排，需要用heapq结构
```

### 3.最小生成树

```python
from heapq import heappop,heappush
def prim(graph,st,n):
    v=set()
    min_heap=[]
    v.add(st)
    for u,w in graph[st]:
        heappush(min_heap,(w,st,u))
    result=[]
    ans=0
    while min_heap and len(result)<n-1:
        w,u,node=heappop(min_heap)
        if node not in v:
            v.add(node)
            result.append((u,node,w))
            ans+=w
            for neighbor,weight in graph[node]:
                if neighbor not in v:
                    heappush(min_heap,(weight,node,neighbor))
     return ans,result
```

### 4.Bellman-Ford

```python
def bellman_ford(graph,start):
    dis={node:float('inf') for node in graph}
    dis[start]=0
    for _ in range(len(graph)-1):
        for u in graph:
            for v,w in graph[u]:
                if dis[u]+w<dis[v]:
                    dis[v]=dis[u]+w
    for u in graph:
        for v,w in graph[u]:
            if dis[u]+w<dis[v]:
                return True  #存在负权环
    return dis
P.S 检测正权环时只需改变判断方向和改变初始值（inf改成0）
```

### 5.Dijkstra

```python
from heapq import heappop,heappush
def dijkstra(graph,start):
    dis={node:float('inf') for node in graph}
    dis[start]=0
    queue=[(0,start)]
    while queue:
        d,node=heappop(queue)
        if d<=dis[node]:
            for u,w in graph[node]:
                nd=d+w
                if nd<dis[u]:
                    dis[u]=nd
                    heappush(queue,(nd,u))
    return dis
```

### 6.KMP算法

```python
#求字符串最大循环节
next_arr[0]=0
for i in range(1,n):
    j=next_arr[i-1]
    while j>0 and s[i]!=s[j]:
        j=next_arr[j-1]
    next_arr[i]=j+(1 if s[i]==s[j] else 0)
L=next_arr[-1]
c=n-L
if C>0 and n%c==0:
    print(s[:c])
else:
    print(s)
```

