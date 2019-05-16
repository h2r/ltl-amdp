from sympy import *
import spot
import itertools

class LTLautomata():
    def __init__(self, ltlformula = 'Fr'):
        #self.APs = ap_dict # atomic propositions, dictionary = {'x': 'L1_1', 'y': 'L1_2'}, 'x': level1, room1
        self.formula = ltlformula # ltl formula ex. Always(x)
        self._ltl2automaton() # table
        self.reward={'goal': 100, 'fail': -100, 'others': -1}
        self.subproblem_flag = 0 # if the flag is on, solve the problem to change from stay to goal
        self.subproblem_goal = -1
        self.subproblem_stay = -1

    def _ltl2automaton(self): # automata={state s: transition(s)}, transition(s)={ltl: next state(s')}
        # translate LTL formula to automata
        A = spot.translate(self.formula, 'BA', 'complete')

        self.aut_spot = A
        self.num_sets = A.num_sets()
        self.num_states = A.num_states()
        self.init_state = A.get_init_state_number()

        # read APs
        self.APs = []
        for ii in range(0, len(A.ap())):
            self.APs.append(A.ap()[ii].ap_name())


        # make transition dictionary
        self.trans_dict = {}
        bdict = A.get_dict()
        for s in range(0, A.num_states()):
            dict_sub = {}
            for t in A.out(s):

                dict_sub[spot.bdd_format_formula(bdict,t.cond).replace('!','~')] = t.dst

            self.trans_dict[s] = dict_sub


    def transition_func(self, q, evaluated_APs):
        # evaluate aps
        # ex. evaluated_APs = {'r': True, 'b': False}
        # evaluated_APs: dict --> string

        # define symbols
        for ap in self.APs:
            exec('%s = symbols(\'%s\')' % (ap,ap))

        # return the next state
        for key in self.trans_dict[q].keys():
            if key == '1':
                return self.trans_dict[q][key]
            flag = (eval(key)).subs(evaluated_APs)

            if flag == True: # if ltl of the edge is true,
                return self.trans_dict[q][key]    # transition occurs

        return -1

    def reward_func(self, q):
        if self.subproblem_flag == 0:
            # manually first
            if self.aut_spot.state_is_accepting(q): # get positive reward if it visits an accepting state
                return self.reward['goal']
            elif (~self.aut_spot.state_is_accepting(q)) & ('1' in self.trans_dict[q].keys()): # if the agent cannot reach the accepting state
                if self.trans_dict[q]['1'] == q:
                    return self.reward['fail']
            else:
                return self.reward['others'] # cost of actions
        else:  # subproblem mode
            if q == self.subproblem_goal:
                return self.reward['goal']
            elif q == self.subproblem_stay:
                return self.reward['others']
            else: return self.reward['fail']

    def terminal_func(self, q):
        if self.subproblem_flag == 0:
            if self.aut_spot.state_is_accepting(q): # get positive reward if it visits an accepting state
                return True
            elif (~self.aut_spot.state_is_accepting(q)) & ('1' in self.trans_dict[q].keys()): # if the agent cannot reach the accepting state
                if self.trans_dict[q]['1'] == q:
                    return True
            else:
                return False
        else:
            if q == self.subproblem_goal:
                return True
            else:
                return False

    def get_accepting_states(self):
        states = [s for s in range(0,self.num_states) if self.aut_spot.state_is_accepting(s)]

        return states

    def findpath(self, sq, gq): # sq: start state, gq: goal state
        paths = [[sq]] # save all path
        words = [[]]

        paths_result = []
        words_result = []


        while len(paths)!=0:
            Q_paths = paths
            Q_words = words
            #Q_stays = stays # atomic proposition to stay at the current state
            paths = []
            words = []
            #stays = []

            for ii in range(0,len(Q_paths)):
                s_cur = Q_paths[ii][-1]
                if s_cur != gq:
                    for alphabet in self.trans_dict[s_cur].keys():

                        s_next = self.trans_dict[s_cur][alphabet]
                        if s_next not in Q_paths[ii]: # append a path if s_next is not a visited state
                            word_tmp = Q_words[ii].copy()
                            word_tmp.append(alphabet)
                            words.append(word_tmp)

                            paths_tmp = Q_paths[ii].copy()
                            paths_tmp.append(s_next)
                            paths.append(paths_tmp)
                else:
                    paths_result.append(Q_paths[ii])
                    words_result.append(Q_words[ii])

        return paths_result, words_result

    def findpath_simple(self, sq, gq, ap_maps): # sq: start state, gq: goal state

        edge_level = {}  # assign level of all edges
        for q1 in self.trans_dict.keys():
            for formula in self.trans_dict[q1].keys():
                if formula == '1':
                    edge_level[(q1, self.trans_dict[q1][formula])] = -1
                else:
                    edge_level[(q1, self.trans_dict[q1][formula])] = min([ap_maps[ap][0] for ap in ap_maps.keys() if ap in formula])


        paths = [[sq]] # save all path
        words = [[]]

        paths_result = []
        words_result = []


        while len(paths)!=0:
            Q_paths = paths
            Q_words = words
            #Q_stays = stays # atomic proposition to stay at the current state
            paths = []
            words = []
            #stays = []

            for ii in range(0,len(Q_paths)):
                s_cur = Q_paths[ii][-1]
                if s_cur != gq:
                    for alphabet in self.trans_dict[s_cur].keys():

                        s_next = self.trans_dict[s_cur][alphabet]
                        if s_next not in Q_paths[ii]: # append a path if s_next is not a visited state
                            word_tmp = Q_words[ii].copy()
                            word_tmp.append(alphabet)
                            words.append(word_tmp)

                            paths_tmp = Q_paths[ii].copy()
                            paths_tmp.append(s_next)
                            paths.append(paths_tmp)
                else:
                    paths_result.append(Q_paths[ii])
                    words_result.append(Q_words[ii])

        return paths_result, words_result

    def _simplify_dict(self, AP_tree): # TODO: Complete
        trans_dict_simplified = {}
        for key in self.trans_dict.keys():
            cur_dict = {}
            for key2 in self.trans_dict[key].keys():
                if key2 == '1':
                    cur_dict[key2] = self.trans_dict[key][key2]
                elif not self._check_contradiction(AP_tree, logical_formula=key2):
                    cur_dict[key2] = self.trans_dict[key][key2]

            trans_dict_simplified[key] = cur_dict

        self.trans_dict = trans_dict_simplified


    def _check_contradiction(self, AP_tree, logical_formula = "a & b"):

        # define symbols
        ap_set = [ap for ap in self.APs if ap in logical_formula]
        for ap in ap_set:
            exec('%s = symbols(\'%s\')' % (ap, ap))

        num_ap = len(ap_set)

        # True False Table
        Flag_contradiction = False

        TF_table  = list(itertools.product([True, False], repeat=num_ap))  # true false table
        #del_list = []
        for ii in range(0, len(TF_table)):
            cur_comb = TF_table[ii]
            flag_remove = False

            for jj in range(0, len(cur_comb)):
                ap = ap_set[jj]

                for i_ap in [x for x in range(0, len(cur_comb)) if x != jj]:  # index of ap
                    ap2 = ap_set[i_ap]

                    if cur_comb[jj]:  # True
                        if ap2 in AP_tree[ap]['same'] and cur_comb[i_ap]:
                            #del_list.append(ii)
                            flag_remove = True
                        elif ap2 in AP_tree[ap]['lower_not'] and cur_comb[i_ap]:
                            #del_list.append(ii)
                            flag_remove = True
                        elif ap2 in AP_tree[ap]['higher'] and not cur_comb[i_ap]:
                            #del_list.append(ii)
                            flag_remove = True
                        elif ap2 in AP_tree[ap]['higher_not'] and cur_comb[i_ap]:
                            #del_list.append(ii)
                            flag_remove = True
                    else:
                        if ap2 in AP_tree[ap]['lower'] and cur_comb[i_ap]:
                            #del_list.append(ii)
                            flag_remove = True

                    if flag_remove:
                        break
                if flag_remove:
                    break

            if not flag_remove:   # check true or false
                evaluated_APs = {}
                for jj in range(0, num_ap):
                    evaluated_APs[ap_set[jj]] = cur_comb[jj]

                Flag_contradiction = Flag_contradiction or (eval(logical_formula)).subs(evaluated_APs) # sub



        return (not Flag_contradiction)





if __name__ == '__main__':
    #ap_dict = {'x': 'L1_1', 'y':'L1_2'}
    A = LTLautomata('~b U (G ~r) & F b')
    paths, words = A.findpath(1,2)
    print(A.transition_func(A.init_state,{'r':True}))

