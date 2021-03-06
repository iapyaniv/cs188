# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        # print(legalMoves[chosenIndex])

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodHeuristic = -1
        ghostHeuristic = 0

        # find min manhattan distance to ghost
        for ghost in newGhostStates:
        	dist = util.manhattanDistance(newPos, ghost.getPosition())
        	if ghost.scaredTimer > 0:
        		if dist == 0:
        			ghostHeuristic += sys.maxint
        		elif dist < ghost.scaredTimer:
        			ghostHeuristic += 1 / float(dist)
        	elif dist < 2:
        		return -sys.maxint - 1
        	
        # eat the last pellet!
        if len(newFood.asList())==0:
        	return sys.maxint

        # find min manhattan distance to food
        for food in newFood.asList():
        	dist = util.manhattanDistance(food, newPos)
        	if dist < foodHeuristic or foodHeuristic == -1:
        		foodHeuristic = 1 / float(dist)

		return successorGameState.getScore() + ghostHeuristic + foodHeuristic


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        result = self.searchStates(gameState, self.depth, 0)
        return result[1]

    def searchStates(self, gameState, depth, agentIndex, action=None):
    	"""
    	Recursive helper function for depth-limited minimax algorithm.
    	Performs DFS postorder traversal to depth defined by SearchAgent. 
    	"""

    	if depth==0 or gameState.isWin() or gameState.isLose():
        	value = float(self.evaluationFunction(gameState))
        	return (value, action)

        if agentIndex==0:
        	bestValue = float("-inf")
        	bestAction = None
        	for action in gameState.getLegalActions(0):
        		successor = gameState.generateSuccessor(0, action)
        		result = self.searchStates(successor, depth, agentIndex+1, action)
        		v = result[0]
        		if v > bestValue:
        			bestValue = v
        			bestAction = action
        	return (bestValue, bestAction)

        elif agentIndex > 0:
        	bestValue = float("inf")
        	bestAction = None
        	for action in gameState.getLegalActions(agentIndex):
        		successor = gameState.generateSuccessor(agentIndex, action)
        		if agentIndex==gameState.getNumAgents()-1:
        			result = self.searchStates(successor, depth-1, 0, action)
        		else:
        			result = self.searchStates(successor, depth, agentIndex+1, action)
        		v = result[0]
        		if v < bestValue:
        			bestValue = v
        			bestAction = action
        	return (bestValue, bestAction)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        result = self.searchStates(gameState, self.depth)
        return result[1]

    def searchStates(self, gameState, depth, agentIndex=0, alpha=float("-inf"), beta=float("inf"), action=None):
        """
          Recursive helper function for minimax with alpha-beta pruning.
          Performs DFS postorder traversal to depth defined by SearchAgent.
        """	

        if depth==0 or gameState.isWin() or gameState.isLose():
			value = float(self.evaluationFunction(gameState))
			return (value, action)

        if agentIndex==0:
        	bestValue = float("-inf")
        	bestAction = None
        	for action in gameState.getLegalActions(0):
        		successor = gameState.generateSuccessor(0, action)
        		result = self.searchStates(successor, depth, agentIndex+1, alpha, beta, action)
        		v = result[0]
        		if v > bestValue:
        			bestValue = v
        			bestAction = action
        		alpha = max(alpha, bestValue)
        		if beta < alpha:
        			break
        	return (bestValue, bestAction)

        elif agentIndex > 0:
        	bestValue = float("inf")
        	bestAction = None
        	for action in gameState.getLegalActions(agentIndex):
        		successor = gameState.generateSuccessor(agentIndex, action)
        		if agentIndex==gameState.getNumAgents()-1:
        			result = self.searchStates(successor, depth-1, 0, alpha, beta, action)
        		else:
        			result = self.searchStates(successor, depth, agentIndex+1, alpha, beta, action)
        		v = result[0]
        		if v < bestValue:
        			bestValue = v
        			bestAction = action
        		beta = min(beta, bestValue)
        		if beta < alpha:
        			break
        	return (bestValue, bestAction)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

