from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.ltl.LTLGridWorldStateClass import LTLGridWorldState
from simple_rl.ltl.LTLautomataClass import LTLautomata
from simple_rl.mdp.MDPClass import MDP

class LTLGridWorldMDP(GridWorldMDP):
    ''' Class for a Grid World MDP with the given LTL'''

    def __init__(self, ltltask='F a', ap_map={'a':(1,1)}, width=5,
                height=3,
                init_loc=(1,1),
                rand_init=False,
                goal_locs=[(5,3)],
                lava_locs=[()],
                walls=[],
                is_goal_terminal=True,
                gamma=0.99,
                init_state=None,
                slip_prob=0.0,
                step_cost=0.0,
                lava_cost=0.01,
                name="gridworld"):

        GridWorldMDP.__init__(self, width, height, init_loc, rand_init, goal_locs,
                lava_locs, walls, is_goal_terminal, gamma, init_state, slip_prob, step_cost,
                lava_cost, name)

        self.ap_map = ap_map
        self.automata = LTLautomata(ltltask) # construct automata
        self.init_q = self.automata.init_state
        #initialize
        init_state = LTLGridWorldState(self.init_loc[0], self.init_loc[1],self.init_q) if init_state is None or self.rand_init else self.init_state
        MDP.__init__(self, LTLGridWorldMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state,
                     gamma=gamma)
        self.cur_state = init_state

    def _transition_func(self, state, action):
        next_state_xy = super()._transition_func(state, action)
        print('{}, {}'.format(next_state_xy.x, next_state_xy.y))
        # evaluate APs
        evaluated_APs = self._evaluate_APs(next_state_xy, self.ap_map)
#        if (next_state_xy.x == 6) & (next_state_xy.y == 3):
#            evaluated_APs = {'r':True}
#        else:
#            evaluated_APs = {'r':False}

        next_q = self.automata.transition_func(state.q, evaluated_APs)
        next_state = LTLGridWorldState(next_state_xy.x,next_state_xy.y,next_q)

        next_state._is_terminal = self.automata.terminal_func(next_q)
        return next_state

    def _reward_func(self, state, action):

        return self.automata.reward_func(state.q)

    def _evaluate_APs(self, state_env, ap_map):
        evaluated_APs = {}
        for key in ap_map.keys():
            if (state_env.x == ap_map[key][0]) & (state_env.y == ap_map[key][1]):
                evaluated_APs[key] = True
            else:
                evaluated_APs[key] = False

        return evaluated_APs

#    def _evaluate(self, state_env): # evaluate atomic propositions

if __name__=='__main__':
    A = LTLGridWorldMDP()
    print('done')