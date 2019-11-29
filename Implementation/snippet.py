class Vector(dict):
    n = 1
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
                output[key] = -val
        
        return Vector(output)
    
    def __div__(self,scalar):
        #print "div:",scalar
        output = {}
        #python divisions..
        scalar = float(scalar)
        for key,val in self.items():
            output[key] = val / scalar
        return Vector(output)
    
    def __mod__(self,other):
        "Dot Product"
        result = 0.0
        for key,val in self.items():
            if key in other:
                result += val * other[key]
        return result
    
    def __pow__(self,p):
        output = {}
        for key,val in self.items():
            output[key] = val ** p
        return Vector(output)
    
    def sqrt(self):
        output = {}
        for key,val in self.items():
            output[key] = sqrt(val)
        return Vector(output)
    
    def length(self):
        length = 0
        for val in self.values():
            length += val**2
        return sqrt(length)
    
    @property
    def squared(self):
        if not self._squared:
            self._squared = self % self
        return self._squared
    
    def distance(self,other):
        distance = 0
        num = 0
        matching = []
        for key,val in self.items():
            num += 1
            if key in other:
                matching.append(key)
                distance += (val - other[key]) ** 2
            else:
                distance += val ** 2

        for key,val in other.items():
            if key not in self:
                num += 1
                distance += val ** 2
        
        return sqrt(distance)