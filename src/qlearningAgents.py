# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
    """

    DEFAULT_Q_VALUE = 0.0

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.__q = {}

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        legal_actions = self.getLegalActions(state)
        is_without_legal_actions = not legal_actions
        if is_without_legal_actions:
            return 0.0

        action_value = self.__get_value_in_q_values(state, action)
        if action_value is not None:
            return action_value

        return self.__create_action_in_q_values(state, action)

    def __update_q_values(self, state, action, newValue):
        if self.__action_already_added(state, action):
            self.__q[state][action] = newValue

    def __create_action_in_q_values(self, state, action):
        actionValues = self.__q.get(state)
        if actionValues is None:
            self.__q[state] = {}
            self.__q[state][action] = self.DEFAULT_Q_VALUE
            return self.DEFAULT_Q_VALUE

        self.__q[state][action] = self.DEFAULT_Q_VALUE
        return self.DEFAULT_Q_VALUE

    def __get_value_in_q_values(self, state, action):
        if self.__action_already_added(state, action):
            return self.__q[state][action]

        return None

    def __action_already_added(self, state, action):
        action_values = self.__q.get(state)
        if action_values is not None and action_values.get(action) is not None:
            return True

        return False

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        best_action = self.computeActionFromQValues(state)

        if best_action is None:
            return 0.0

        return self.getQValue(state, best_action)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        without_legal_actions = not legal_actions
        if without_legal_actions:
            return None

        better_action = None
        better_value = -math.inf

        for action in legal_actions:
            action_value = self.getQValue(state, action)
            if action_value > better_value or (action_value == better_value and util.flipCoin(0.5)):
                better_value = action_value
                better_action = action

        return better_action

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)
        "*** YOUR CODE HERE ***"
        should_take_random_action = util.flipCoin(self.epsilon)
        if should_take_random_action:
            return random.choice(legal_actions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf

        """
        "*** YOUR CODE HERE ***"
        action_value = self.getQValue(state, action)
        next_state_better_value = self.computeValueFromQValues(nextState)
        new_action_value = (1 - self.alpha) * action_value + self.alpha * (
            reward + self.discount * next_state_better_value
        )

        self.__update_q_values(state, action, new_action_value)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        total_value = 0
        features = self.featExtractor.getFeatures(state, action)
        for feature_name, feature_value in features.items():
            total_value += self.weights[feature_name] * feature_value

        return total_value

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        action_value = self.getQValue(state, action)
        better_value_in_next_state = self.computeValueFromQValues(nextState)

        newGeneralValue = reward + self.discount * better_value_in_next_state - action_value

        features = self.featExtractor.getFeatures(state, action)
        for feature_name, feature_value in features.items():
            new_weight_value = self.weights[feature_name] + self.alpha * newGeneralValue * feature_value
            self.__update_weight_value(feature_name, new_weight_value)

    def __update_weight_value(self, feature_name, newValue):
        self.weights[feature_name] = newValue

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
