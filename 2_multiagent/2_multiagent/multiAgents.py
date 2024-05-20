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
import math
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = newFood.asList()
        foodDistances = list()
        for food in newFood:
            foodDistances.append(manhattanDistance(newPos,food))
        if foodDistances:
            neareastFoodDistance = min(foodDistances)
        else :
            neareastFoodDistance = 1

        ghostDistances = list()
        scaredGhostDistances = list()
        for ghost in newGhostStates :
            if ghost.scaredTimer == 0 :
                ghostDistances.append(manhattanDistance(newPos, ghost.getPosition()))
            if ghost.scaredTimer > 0 :
                scaredGhostDistances.append(manhattanDistance(newPos,ghost.getPosition()))

        foodScore = -1 * math.log10(neareastFoodDistance)

        ghostPenalty = 0
        scaredGhostScore = 0
        for distance in ghostDistances:
            try :
                ghostPenalty += -1 * math.log10(distance)
                scaredGhostScore += 1 * math.log10(distance)
            except :
                ghostPenalty += -1 * math.log10(distance + 1)
                scaredGhostScore += -1 * math.log10(distance + 1)

        totalScore = successorGameState.getScore() + foodScore - ghostPenalty + scaredGhostScore

        return totalScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(agentIndex, depth, gameState):

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            
            actions = gameState.getLegalActions(agentIndex)
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)
            
            if agentIndex + 1 < gameState.getNumAgents():
                nextAgentIndex = agentIndex + 1
            else:
                nextAgentIndex = 0
            if nextAgentIndex == 0 :
                nextDepth = depth + 1 
            else :
                nextDepth = depth
            
            bestAction = None
            if agentIndex == 0:
                bestScore = -99999
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgentIndex, nextDepth, successor)[0]
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                return (bestScore, bestAction)
            
            else:
                bestScore = 99999
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgentIndex, nextDepth, successor)[0]
                    if score < bestScore:
                        bestScore = score
                        bestAction = action
                return (bestScore, bestAction)
        
        action = minimax(0, 0, gameState)[1]
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            
            if agentIndex + 1 < gameState.getNumAgents():
                nextAgent = agentIndex + 1
            else:
                nextAgent = 0

            nextDepth = depth
            if nextAgent == 0:
                nextDepth += 1

            actions = gameState.getLegalActions(agentIndex)

            if actions == None:
                return (self.evaluationFunction(gameState), None)

            bestAction = None
            if agentIndex == 0:
                value = -99999
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)[0]
                    if score > value:
                        value, bestAction = score, action
                    alpha = max(alpha, value)
                    if alpha > beta:
                        break
            else:
                value = 99999
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)[0]
                    if score < value:
                        value, bestAction = score, action
                    beta = min(beta, value)
                    if beta < alpha:
                        break
            return value, bestAction

        try:
            action = alphaBeta(0, 0, gameState, -99999, 99999)[1]
            if action != None :
                return action 
            else :
                Directions.STOP
        except Exception as e:
            print(f"Error in AlphaBetaAgent.getAction: {str(e)}")
            return Directions.STOP

        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState, depth, agentIndex=0):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return (self.evaluationFunction(gameState), None)
            
            if agentIndex + 1 < gameState.getNumAgents():
                nextAgent = agentIndex + 1
            else:
                nextAgent = 0

            nextDepth = depth
            if nextAgent == 0:
                nextDepth -= 1

            actions = gameState.getLegalActions(agentIndex)

            scores = list()
            if agentIndex == 0:
                for action in actions:
                    expectimaxScore = expectimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextAgent)[0]
                    scores.append((expectimaxScore,action))
                return max(scores)
            
            else:
                for action in actions:
                    expectimaxScore = expectimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextAgent)[0]
                    scores.append(expectimaxScore)
                    totalExpectimaxScore = sum(scores) / len(scores)
                return (totalExpectimaxScore, None)

        action = expectimax(gameState, self.depth)[1]
        return action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    pacmanPosition = currentGameState.getPacmanPosition()

    foodList = currentGameState.getFood().asList()
    foodDistances = list()
    minFoodDistance = 0
    for food in foodList:
        foodDistances.append(manhattanDistance(pacmanPosition, food))
    if foodDistances:
        minFoodDistance = min(foodDistances)
    else :
        1

    ghostStates = currentGameState.getGhostStates()
    ghostPositions = list()
    for ghost in ghostStates :
        ghostPositions.append(ghost.getPosition())

    ghostDistances = list() 
    for ghostPos in ghostPositions :
        ghostDistances.append(manhattanDistance(pacmanPosition, ghostPos))

    if ghostDistances :
        minGhostDistance = min(ghostDistances)
    else :
        99999

    scaredTimes = list()
    for ghost in ghostStates:
        scaredTimes.append(ghost.scaredTimer)
    
    if len(scaredTimes) > 0:
        minScaredTime = min(scaredTimes)
    else :
        1

    capsulesLeft = len(currentGameState.getCapsules())

    if minFoodDistance > 0:
        score -= minFoodDistance

    if minGhostDistance < 1 and minScaredTime < 1:
        score -= 99999

    score -= len(foodList)
    score -= capsulesLeft

    return score

# Abbreviation
better = betterEvaluationFunction
