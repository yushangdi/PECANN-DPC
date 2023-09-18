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
        chain = []

        while self.parents[a] != a:
            chain.append(a)
            a = self.parents[a]

        for b in chain:
            self.parents[b] = a

        return a
