# factorOperations.py
# -------------------
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


from bayesNet import Factor
import operator as op
import util

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print "Factor failed joinFactorsByVariable typecheck: ", factor
            raise ValueError, ("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict

    NOTES: P(A|B) is the conditional distribution of A given B, and A is 
    conditioned on B. In this project, B is referred to as a "conditioned 
    variable" (since it's being conditioned on), and then A is called 
    "unconditioned." Better terminology might be something like "conditioned 
    on" for B and "not conditioned on" for A. 
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print "Factor failed joinFactors typecheck: ", factor
            raise ValueError, ("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))

    tempConditionedVariables = set()
    inputUnconditionedVariables = set()
    variableDomains = factors[0].variableDomainsDict()

    # construct factor inputs
    for factor in factors:
        for variable in factor.conditionedVariables():
            tempConditionedVariables.add(variable)
        for variable in factor.unconditionedVariables():
            inputUnconditionedVariables.add(variable)

    inputConditionedVariables = {v for v in tempConditionedVariables if v not in inputUnconditionedVariables}

    joinedFactor = Factor(inputUnconditionedVariables, inputConditionedVariables, variableDomains)

    # calculate joint probabilities
    for assignment in joinedFactor.getAllPossibleAssignmentDicts():
        probability = 1
        for factor in factors:
            try:
                probability *= factor.getProbability(assignment)
            except(ValueError):
                continue
        joinedFactor.setProbability(assignment, probability)

    return joinedFactor


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict

        NOTES: Definition of marginalization; if you have distribution P(A,B|C),
        P(A=a|C=c)= Sum over b of P(A=a, B=b|C=c)
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        # construct new factor
        unconditionedVariables = {v for v in factor.unconditionedVariables() if v != eliminationVariable}
        marginalizedFactor = Factor(unconditionedVariables, factor.conditionedVariables(), factor.variableDomainsDict())

        # calculate marginalized probabilities
        eliminatedVariableDomain = factor.variableDomainsDict()[eliminationVariable]
        for assignment in factor.getAllPossibleAssignmentDicts():
            marginalizedProbabilities = []
            for value in eliminatedVariableDomain:
                assignment[eliminationVariable] = value
                marginalizedProbabilities.append(factor.getProbability(assignment))
            marginalizedFactor.setProbability(assignment, sum(marginalizedProbabilities))
        return marginalizedFactor

    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict

    NOTES: Let's say your factor consists of an unconditioned variable W (weather)
    which has a domain {sunny}. Since W always = sunny, you could say that your
    factor is conditioned on the weather being sunny; for this reason, unconditioned
    variables with domain of size 1 are moved to conditioned variables.
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print "Factor failed normalize typecheck: ", factor
            raise ValueError, ("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    # construct new factor
    variableDomains = factor.variableDomainsDict()
    conditionedVariables = factor.conditionedVariables()
    unconditionedVariables = set()

    for variable in factor.unconditionedVariables():
        if len(variableDomains[variable]) == 1:
            conditionedVariables.add(variable)
        else:
            unconditionedVariables.add(variable)

    normalizedFactor = Factor(unconditionedVariables, conditionedVariables, variableDomains)

    # normalize probabilities
    sumProbabilities = 0
    for assignment in factor.getAllPossibleAssignmentDicts():
        sumProbabilities += factor.getProbability(assignment)

    if sumProbabilities == 0:
        return None

    for assignment in factor.getAllPossibleAssignmentDicts():
        normalizedFactor.setProbability(assignment, factor.getProbability(assignment) / sumProbabilities)

    return normalizedFactor