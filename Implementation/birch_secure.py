import random

def power(a,b):
    ans = sint(1)
    for _ in range(b):
        ans = ans*a
    return ans

def sqrt(a):
    x0 = sfix(1)
    ans = x0
    ave = sint(2)
    for i in range(10):
        ans = (x0 + (a/x0)) / ave
        x0 = ans
    return ans

class Vector(dict):
    n = sint(1)

    def __init__(self,*args,**kwargs):
        self._squared = None
        self.ls = self
        super(Vector,self).__init__(*args,**kwargs)
    
    def __hash__(self):
        assert self.item,"Cannot hash a value without its item"
        return hash(self.item)
    
    def __add__(self,other):
        output = {}
        for key,val in self.items():
            if key in other:
                output[key] = val + other[key]
            else:
                output[key] = val
        
        for key,val in other.items():
            if key not in self:
                output[key] = val
        
        return Vector(output)
    
    def __sub__(self,other):
        output = {}
        for key,val in self.items():
            if key in other:
                output[key] = val - other[key]
            else:
                output[key] = val
        
        for key,val in other.items():
            if key not in self:
                output[key] = sint(-1) * val
        
        return Vector(output)
    
    def __div__(self,scalar):
        #print "div:",scalar
        output = {}
        #python divisions..
        scalar = sfix(scalar)
        for key,val in self.items():
            output[key] = val / scalar
        return Vector(output)
    
    def __mod__(self,other):
        #"Dot Product"
        result = sfix(0.0)
        for key,val in self.items():
            if key in other:
                result = result + (val * other[key])
        return result
    
    def __pow__(self,p):
        output = {}
        for key,val in self.items():
            #p should not be sint
            output[key] = power(val,p)
        return Vector(output)
    
    def sqrt(self):
        output = {}
        for key,val in self.items():
            output[key] = sqrt(val)
        return Vector(output)
    
    def length(self):
        length = sint(0)
        for val in self.values():
            length = length + power(val,2)
        return sqrt(length)
    
    @property
    def squared(self):
        if not self._squared:
            self._squared = self % self
        return self._squared
    
    def distance(self,other):
        distance = sint(0)
        matching = []
        for key,val in self.items():
            if key in other:
                matching.append(key)
                distance = distance + power((val - other[key]),2)
            else:
                distance = distance + power(val,2)

        for key,val in other.items():
            if key not in self:
                distance = distance + power(val,2)
        
        return sqrt(distance)
        
# SMALL 2D SAMPLE (takes ~ 2 seconds on a Intel Core2 Duo 2.5GHz):
B = sint(10)
T = sint(35)
sample_dimensions = 2
sample_vectors_fillpercentage = 100
num_sample_points = 10000

vectors = []

print_ln('creating vectors.. (%s-dimensional, %s% filled, %s vectors, branch %s, treshold %s)', sample_dimensions, sample_vectors_fillpercentage, num_sample_points, B.reveal(), T.reveal())

for i in range(num_sample_points):
    v = Vector()
    hit = False
    while not hit:
        h = 1
        for x in range(sample_dimensions):
            if random.randint(0,100/sample_vectors_fillpercentage) == 0:
                    hit = True
                    v[sint(x)] = sint(random.randint(0,100))
                    # print_ln('%s',x)
    vectors.append(v)
    if i == 10:
        for a, b in v.items():
            print_ln('%s: %s', a.reveal(), b.reveal())
        v2 = Vector()
        v2 = v.sqrt()
        for a, b in v2.items():
            print_ln('%s: %s', a.reveal(), b.reveal())
print_ln('starting clustering...')

# build the first node - it's our root
root = Node()

print_ln('Root built')