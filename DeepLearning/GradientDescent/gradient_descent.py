from numpy import *

def compute_error_for_line_given_points(b,m,points):
    #initialize it at 0
    total_error = 0
    #for every data point we have
    for i in range(len(points)):
        #get the x,y value
        x,y = points[i, 0], points[i, 1]
        #get the difference, square it , add it to the total
        total_error += (y-(m*x + b))**2

    #get the average
    return total_error / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    #this is where our gradient starts
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(len(points)):
        x,y = points[i, 0], points[i, 1]
        #direction with respect to b and m
        #computing partial derivatives of our error function
        b_gradient = -(2/N) * (y - ((m_current*x) + b_current))
        m_gradient = (2/N) * x * (y - ((m_current*x) + b_current))


    #update our b and m values using our partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return (new_b, new_m)

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, number_iterations):
    #starting b and m
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(number_iterations):
        # update b and m with the new and more accurate b and m by performing a gradient step
        b,m = step_gradient(b, m, array(points), learning_rate)
    return (b,m)

def run():
    # Step 1 - collect our data
    points = genfromtxt('./Data/data.csv', delimiter=',')

    #Step 2 - define our hyperparameter
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    number_of_iterations = 1000

    #Step 3 - Train our model
    print('Starting gradient descent at b = {0}, m={1}, error={2}'.format(initial_b,initial_m,
                                    compute_error_for_line_given_points(initial_b, initial_m, points)))

    b,m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, number_of_iterations)

    print('Ending gradient descent at b = {0}, m={1}, error={2}'.format(b, m,
                                    compute_error_for_line_given_points(b, m, points)))

if __name__ == "__main__":
    run()