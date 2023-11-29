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
Safety = [0.8, 0.7, 0.9, 0.7]  

# Gabe's Weights
WeightCost, WeightInfrastructure, WeightAccessibility, WeightPopularity, WeightEnvironmental, WeightSafety = [0.5, 0.2, 0.05, 0.19, 0.02, 0.04]

# Decision variables
X = cp.Variable(m, boolean=True)

# Objective Function
objective = cp.Maximize(
    cp.sum(cp.multiply(Cost, cp.multiply(WeightCost, X))) +
    cp.sum(cp.multiply(Infrastructure, cp.multiply(WeightInfrastructure, X))) +
    cp.sum(cp.multiply(Accessibility, cp.multiply(WeightAccessibility, X))) +
    cp.sum(cp.multiply(Popularity, cp.multiply(WeightPopularity, X))) +
    cp.sum(cp.multiply(Environmental, cp.multiply(WeightEnvironmental, X))) +
    cp.sum(cp.multiply(Safety, cp.multiply(WeightSafety, X)))
)

# Gabe's Constraints
# Converted constraints out of 10 to a 0-1 scale
constraints = [
    # Gabe's specific constraints
    cp.sum(X) == 1,  # Each city can host only one sport event
    cp.sum(cp.multiply(Cost, X)) >= 0.8,  # Minimum budget score of 8
    cp.sum(cp.multiply(Infrastructure, X)) >= 0.4,  # Minimum infrastructure score of 4
    cp.sum(cp.multiply(Accessibility, X)) >= 0.5,  # Minimum accessibility score of 5
    cp.sum(cp.multiply(Popularity, X)) >= 0.4,  # Minimum popularity score of 4
    cp.sum(cp.multiply(Environmental, X)) >= 0.7,  # Minimum environmental score of 7
    cp.sum(cp.multiply(Safety, X)) >= 0.5  # Minimum safety score of 5
]

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Output results
print("Status:", problem.status)
print("Optimal value:", problem.value)
city_names = ["Stockholm", "Salt Lake City", "Bern", "Marseilles"]
selected_city = city_names[X.value.argmax()] if problem.status == 'optimal' else "No optimal city found"
print("Selected city:", selected_city)
