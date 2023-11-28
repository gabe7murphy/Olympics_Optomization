import cvxpy as cp 

# Number of cities 
m = 4   

# Parameters for each city 
# Converted scores out of 10 to a 0-1 scale for normalization
Cost = [0.6, 1.0, 0.7, 0.9]  # Stockholm, Salt Lake City, Bern, Mersailles
Infrastructure = [0.8, 1.0, 0.8, 0.8]
Accessibility = [0.9, 0.9, 0.9, 0.7]
Popularity = [0.8, 0.5, 0.8, 0.8]
Environmental = [1.0, 0.5, 0.9, 0.7]
Safety = [0.8, 0.7, 0.9, 0.7]  # Added safety scores

# Gabe's Weights 
WeightCost, WeightInfrastructure, WeightAccessibility, WeightPopularity, WeightEnvironmental, WeightSafety = [0.5, 0.2, 0.05, 0.19, 0.02, 0.04]

# Decision variables 
X = cp.Variable(m, boolean=True)

# Objective Function 
objective = cp.Maximize(cp.sum(Cost * WeightCost * X + 
                               Infrastructure * WeightInfrastructure * X + 
                               Accessibility * WeightAccessibility * X + 
                               Popularity * WeightPopularity * X + 
                               Environmental * WeightEnvironmental * X + 
                               Safety * WeightSafety * X))

# Gabe's Constraints 
# Converted constraints out of 10 to a 0-1 scale
constraints = [
    # Each city can host only one sport event
    cp.sum(X) == 1,
    # Gabe's specific constraints
    Cost * X >= 0.5,  # Minimum budget score of 5
    Infrastructure * X >= 0.4,  # Minimum infrastructure score of 4
    Accessibility * X >= 0.5,  # Minimum accessibility score of 5
    Popularity * X >= 0.4,  # Minimum popularity score of 4
    Environmental * X >= 0.7,  # Minimum environmental score of 7
    Safety * X >= 0.5  # Minimum safety score of 5
]

# Define and solve the problem 
problem = cp.Problem(objective, constraints)
problem.solve()

# Output results 
print("Status:", problem.status)
print("Optimal value:", problem.value)
print("City selection (X):\n", X.value)
