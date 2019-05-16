from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class LTLGridWorldState(GridWorldState):
    ''' Class for Grid World States when LTL task is given'''
    def __init__(self, x, y, q):
        GridWorldState.__init__(self,x,y)
        self.q = q
        self.data.append(q)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.q) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, LTLGridWorldState) and self.x == other.x and self.y == other.y and self.q == other.q
