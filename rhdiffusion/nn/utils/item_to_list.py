class Item2List:

    def __init__(self, total_len, pos):
        self.total_len = total_len
        self.pos = pos

    def __call__(self, item):
        self.list = [None] * self.total_len
        self.list[self.pos] = item
        return self.list
