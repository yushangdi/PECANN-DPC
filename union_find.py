


class UnionFind:

    def __init__(self):
        self.parents = {}

    def add(self, a):
        if a not in self.parents:
            self.parents[a] = a
        
    def unite(self, a, b):
        self.add(a)
        self.add(b)
        a = self.find(a)
        b = self.find(b)
        self.parents[a] = b

    def find(self, a):
        if self.parents[a] != a:
            self.parents[a] = self.find(self.parents[a])
            return self.parents[a]
        return a