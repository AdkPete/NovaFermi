import numpy as np
import scipy.optimize as opt ##For comparison purposes
import multiprocessing as mp

def test_1(x):
    '''
    simple 1D test function
    '''
    
    return x[0] ** 2

def test_2(x):
    '''
    Trivial 2D test function, global minimum at (0,0)
    '''
    return x[0] ** 2 + x[1] ** 2

def rosenbrock(x):
    '''
    rosenbrock function for testing optimizers
    '''
    a = 1
    b = 100
    return (a - x[0]) ** 2 + b * (x[1]  - x[0] ** 2 ) ** 2
    
def ackley(x):
    '''
    ackley function for testing optimizers
    this particular function has many local minima, causing issues for hillclimbers
    '''
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    x - np.array(x)
    T1 = -a * np.exp(-b * np.sqrt( ( 1. / d ) * np.sum(x ** 2) ) )
    T2 = -np.exp( (1. / d) * np.sum(np.cos(c * x)))
    return T1 + T2 + a + np.exp(1)


def uniform_random_sampler(N , bounds):
    '''
    Function to generate random samples.
    
    Parameters
    ----------
    N : int
        Number of samples to generatet
    Bounds : array
        Each element of the array should be a 
        tuple (or list/array) with length 2, that
        defines the limits of each parameter
    '''
    
    result = []
    
    for b in bounds:
    
        sample = np.random.uniform(b[0] , b[1] , N)
        result.append(sample)
        

    result = np.array(result)
    return np.transpose(result)


def eval_pop(f , population, nproc):
    '''
    simple function to evaluate the fitness of all of the
    members of the given population.
    Parameters
    ----------
    f : function
        evaluation function
    population : array
        Current population
    '''
    
    fitness = []
    
    '''
    for i in population:
        fitness.append(f(i))
    return np.array(fitness)
    '''
    
    with mp.Pool(nproc,maxtasksperchild=1) as P:
        mapres = P.imap(f , population, chunksize =1)
        for i in mapres:
            fitness.append(i)
    return np.array(fitness)

def mutation(creature , bounds , mutation_rate = 0.5):
    '''
    Function designed to handle mutations in the population
    each creature will have a probability of containing a mutation
    governed by  mutation_rate
    '''
    
    param_odds = mutation_rate / len(creature)
    for i in range(len(creature)):
        mutate = np.random.rand()
        if mutate < param_odds:
            creature[i] = np.random.uniform(bounds[i][0], bounds[i][1] , 1)[0]
            
    return creature

def softmax_function(x):
    '''
    Function to normalize an array such that it sums to one.
    This uses a softmax function, to help aid with any floating point
    round off issues
    '''
    x /= np.max(x) ##Here to avoid overflows in e^x
    return np.exp(x) / np.sum(np.exp(x))

def reproduce(f , bounds , population, fitness , N_new , mutation_rate, nproc):
    
    '''
    Function to take in the current population, and
    to create the next generation. Population size
    will be preserved. It is assumed that the
    population is sorted in order of decreasing fitness
    
    Parameters
    ----------
    f : function
        evaluation function
    bounds : array
        parameter boundaries
    population : array
        current population
    N_new : int
        number of new creatures to create
    '''
    
    new_sltns = []
    
    for i in range(N_new):
        nx = []
        
        ii = np.where(abs(fitness) < 1e-200)
        fitness[ii] = 1e-200 ##Prevents any issues with 1 / small number = inf
        
        prob = softmax_function(1.0 / fitness[:N_new])
        
        parents = np.random.choice(np.array(range(N_new)) , 2 , replace = False ,  p = prob)
        
        parent1 = population[parents[0]]
        parent2 = population[parents[1]]
       
        for k in range(len(parent1)):
            p1 = parent1[k]
            p2 = parent2[k]
            
            nparam = np.random.normal((p1 + p2) / 2.0 , abs(p2 - p1) , 1)[0]

            while nparam < bounds[k][0] or nparam > bounds[k][1]:
                ##Generate new sltns until we find one withing the bounds
                nparam = np.random.normal((p1 + p2) / 2.0 , abs(p2 - p1) , 1)[0]

            nx.append(nparam)

        nx = mutation(nx , bounds , mutation_rate)
        new_sltns.append(nx)

    new_sltns = np.array(new_sltns)


    new_fitness = eval_pop(f , new_sltns, nproc = nproc)
    
    ii = np.arange(len(population) - N_new , len(population) , 1 )
    
    
    population[ii] = new_sltns
    fitness[ii] = new_fitness
    #fitness = eval_pop(f , population)
    
    ii = np.argsort(fitness)
    
    population = population[ii]
    fitness =fitness[ii]
    infcount = list(fitness).count(np.inf)
    while infcount > 0:
        fitness = reproduce(f , bounds , population , fitness , infcount, mutation_rate, nproc)
        infcount = list(fitness).count(np.inf)
    
    return population , fitness , len(new_fitness)

def rastrigin(x):
    '''
    Rastrigin function for testing optimizers
    This function has many local minima, causing issues for hillclimbers
    '''
    A = 10
    d = len(x)
    return A * d + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))



def opt_w_random_samples(f , bounds , N):
    
    '''
    Tries to optimize a function by random sampling across
    the domnain. This is highly inefficient of course, but
    serves as a base comparison for better algorithms.
    '''
    
    result = uniform_random_sampler(N , bounds)
    f_vals = []
    x_vals = []
    for x in result:
        f_vals.append(f(x))
        x_vals.append(x)
    
    min_i = f_vals.index(min(f_vals))
    print ("Best f(x):" , f_vals[min_i])
    print ("Best x :" , result[min_i])
    return result[min_i] , f_vals[min_i]

def genetic_algorithm(f , bounds , popsize = 10 , Niter = 100, Gensize = 0.5 , mutation_rate = 0.25, nproc = 1):
    '''
    This is the main function that runs the actual
    genetic algorithm
    '''
    Neval = 0
    new = int(Gensize * popsize)
    ##First step is to initialize a population of solutions
    population = uniform_random_sampler(popsize , bounds)
    
    ##Determine fitness, and sort in order of increasing f eval
    fitness = eval_pop(f , population, nproc)
    ii = np.argsort(fitness)
    fitness = fitness[ii]
    population = population[ii]
    
    ##Iterate Niter times, generating new populations
    for i in range(Niter):
        population , fitness , nfeval = reproduce(f,bounds,population,fitness,new , mutation_rate, nproc=nproc)
        Neval += nfeval
        
    print ("Best f(x) :", fitness[0])
    print ("Best x:" , population[0])
    return population[0] , fitness[0] , Neval
    
if __name__ == "__main__":

    print ("Now we test on a harder function")
    print ("True solution is at (0,0)")
    
    print ("Genetic Algorithm w/ <2000 Function evals")
    xg , fg , Neval= genetic_algorithm(ackley , [(-30,30),(-30,30)] , popsize = 24 , Niter = 100 , mutation_rate = .1)
    print (xg , fg, Neval)
    
    print ("Scipy's Differential Evolution optimizer")
    x0 = np.random.uniform(-30,30,2)
    res = opt.differential_evolution(ackley , [(-30,30),(-30,30)] , maxiter = 50 , popsize = 24 , tol = 1e-6 , init = 'random' , x0 = x0)
    print (res.x , res.fun , res.nfev)
    
    print ("Random sampling")
    xr , fr = opt_w_random_samples(ackley , [(-30,30),(-30,30)] , 600)
    print (xr , fr)
    
    print ("\n\n")
    print ("Now we test on a rosenbrock function")
    print ("Genetic Algorithm w/ <2000 Function evals")
    xg , fg , Neval= genetic_algorithm(rosenbrock , [(-30,30),(-30,30)] , popsize = 150 , Niter = 30 , mutation_rate = .01)
    print (xg , fg, Neval)
    
    print ("Scipy's Differential Evolution optimizer")
    x0 = np.random.uniform(-30,30,2)
    res = opt.differential_evolution(rosenbrock , [(-30,30),(-30,30)] , maxiter = 50 , popsize = 24 , tol = 1e-6 , init = 'random' , x0 = x0)
    print (res.x , res.fun , res.nfev)
    
    print ("Random sampling")
    xr , fr = opt_w_random_samples(rosenbrock , [(-30,30),(-30,30)] , 600)
    print (xr , fr)
    

    print ("\n\n")
    print ("High-dimensionality test")
    
    print ("Genetic Algorithm w/ <2000 Function evals")
    xg , fg , Neval= genetic_algorithm(ackley , [(-30,30),(-30,30),(-30,30),(-30,30),(-30,30)] , popsize = 100 , Niter = 300 , mutation_rate = .1)
    print (fg, Neval)
    
    print ("Scipy's Differential Evolution optimizer")
    x0 = np.random.uniform(-30,30,5)
    res = opt.differential_evolution(ackley , [(-30,30),(-30,30),(-30,30),(-30,30),(-30,30)] , maxiter = 50 , popsize = 50 , tol = 1e-7 , init = 'random' , x0 = x0)
    print (res.fun , res.nfev)
    
    print ("Scipy minimize")
    x0 = np.random.uniform(-30,30,5)
    res = opt.basinhopping(ackley , x0 , niter=100)
    print (res.fun , res.nfev)
    if fg < res.fun:
        print ("Genetic Algorithm found a better solution than Scipy's DE")
        

    print ("\n\n")
    print ("Now we test on a Rastrigin function")
    print ("Genetic Algorithm w/ <2000 Function evals")
    xg , fg , Neval= genetic_algorithm(rastrigin , [(-30,30),(-30,30),(-30,30),(-30,30),(-30,30)] , popsize = 500 , Niter = 1000 , mutation_rate = .1)
    print (fg, Neval)
    print ("Scipy's Differential Evolution optimizer")
    x0 = np.random.uniform(-30,-30,5)
    res = opt.differential_evolution(rastrigin , [(-30,30),(-30,30),(-30,30),(-30,30),(-30,30)] , maxiter = 50 , popsize = 50 , tol = 1e-7 , init = 'random' , x0 = x0)
    print (res.fun , res.nfev)
    if fg < res.fun:
        print ("Genetic Algorithm found a better solution than Scipy's DE")
    
    print ("Scipy minimize")
    x0 = np.random.uniform(-30,30,5)
    res = opt.basinhopping(rastrigin , x0 , niter=100)
    print (res.fun , res.nfev)