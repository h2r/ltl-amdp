# Python imports.
from __future__ import print_function
from collections import defaultdict

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP
from simple_rl.amdp.AMDPTaskNodesClass import AbstractTask, RootTaskNode

class AMDPAgent(object):
    ''' Generic solver for all abstr_domains that adhere to the AMDP framework (Gopalan et al). '''
    def __init__(self, root_grounded_task, policy_generators, base_mdp):
        '''
        AbstractMDP solver class
        Args:
            root_grounded_task (RootTaskNode)
            policy_generators (list) of type objects (one for each level below the root)
            base_mdp (MDP): Lowest level environment MDP
        '''
        self.root_grounded_task = root_grounded_task
        self.policy_generators = policy_generators
        self.base_mdp = base_mdp
        self.state_stack = []
        self.policy_stack = []
        for i in range(len(policy_generators)):
            self.state_stack.append(State())
            self.policy_stack.append(defaultdict())
        self.max_level = len(self.policy_generators) - 1
        self.action_to_task_map = defaultdict()
        self._construct_action_to_node_map(root_grounded_task)
        self.max_iterate = 100 # YS

    def solve(self):
        base_state = self.base_mdp.init_state
        self.state_stack[0] = base_state

        # Project env MDP init state to all higher levels
        for i in range(1, len(self.policy_generators)):
            pg = self.policy_generators[i]
            base_state = pg.generate_abstract_state(base_state)
            self.state_stack[i] = base_state

        # Start decomposing the highest-level task in hierarchy
        self._decompose(self.root_grounded_task, self.max_level)

    def _decompose(self, grounded_task, level, verbose=False):
        '''
        Ground high level tasks into environment level actions and then execute
        in underlying environment MDP
        Args:
            grounded_task (AbstractTask): TaskNode representing node in task hierarchy
            level (int): what depth we are in our AMDP task hierarchy (base MDP is l0)
            verbose (bool): debug mode
        '''
        print('Decomposing action {} at level {}'.format(grounded_task, level))
        state = self.state_stack[level]

        policy = self.policy_generators[level].generate_policy(state, grounded_task)
        if level > 0:
            num_iterate = 0   # YS
            while not grounded_task.is_terminal(state):
                action = policy[state]
                self.policy_stack[level][state] = action
                self._decompose(self.action_to_task_map[action], level-1)
                state = self.state_stack[level]
                num_iterate = num_iterate + 1   # YS
                if num_iterate > self.max_iterate:   # YS
                    break
        else:
            num_iterate = 0
            while not grounded_task.is_terminal(state):
                action = policy[state]
                self.policy_stack[level][state] = action
                if verbose: print('({}, {})'.format(state, action))
                reward, state = self.base_mdp.execute_agent_action(action)
                self.state_stack[level] = state
                num_iterate = num_iterate + 1  # YS
                if num_iterate > self.max_iterate:  # YS
                    break

        if level < self.max_level:
            projected_state = self.policy_generators[level+1].generate_abstract_state(self.state_stack[level])
            self.state_stack[level+1] = projected_state

    def _construct_action_to_node_map(self, root_node):
        '''
        Use DFS to create a dictionary mapping string actions to AbstractTask nodes
        Args:
            root_node (AbstractTask): TaskNode representing AMDP action
        '''
        if root_node is not None:
            if root_node.action_name not in self.action_to_task_map:
                self.action_to_task_map[root_node.action_name] = root_node
            for child_node in root_node.subtasks:
                self._construct_action_to_node_map(child_node)
