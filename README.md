# Olympics_Optomization
This project addresses the complex process of choosing the Olympic host city by employing linear modeling techniques to optimize where the Olympics should be held. The integration of mathematical algorithms not only offers an optimal solution for a seamless event but also includes relevant constraints to maximize the efficiency of the premier location of the world's premier sporting event. This will be done by using determining a linear program and using a gourbi solver in python to find the solution. 

Decision Variables: 
𝑋𝑖 \newline
𝑋𝑖=1  means the city is chosen to host the Olympics. \newline
𝑋𝑖=0 means the city is not chosen to host the Olympics. \newline
i is {1-4} for the 4 potential cities. \newline

Constraints: 

Budget Constraints: Limit the overall cost of hosting the Olympics. 
Infrastructure Constraints: Ensure that the chosen spot has the necessary infrastructure. 
Accessibility Constraints: Ensure that the chosen spot is easily accessible. 
Popularity Constraints: Ensure that the chosen spot is popular and can attract visitors. 
Environmental Constraints: Ensure that the chosen spot meets environmental standards. 
Safety Constraints: Ensure that the chosen spot is safe for visitors. 
Only can choose one city 
