import numpy as np
import csv
import cvxpy as cp
import matplotlib.pyplot as plt
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",  #Your API key
)

def create_prompt(): # create prompt
    meta_prompt_start = f'''You are an AI assistant specialized in solving EV charging problems. \
        You have been provided with a predefined function called solve_EV that can solve various EV charging problems.\n\n'''

    code = f'''
    def MPC_Solver(num_of_vehicles, timesteps, initial_states, max_power,
               terminal_states, dept_time, power_capacity, plot_fig):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    max_sum_u = cp.Parameter(name='max_sum_u')
    u_max = cp.Parameter(num_of_vehicles, name='u_max')
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value=initial_states
    max_sum_u.value = power_capacity
    u_max.value=max_power*np.ones((num_of_vehicles, ))

    obj = 0
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for t in range(timesteps):
        constr += [x[:,t+1] == x[:,t] + u[:,t],
                   u[:,t] <= u_max,
                   u[:,t] >= 0,
                   cp.sum(u[:,t]) <= max_sum_u,
                   u[:,t] <= (t*np.ones_like(dept_time)<dept_time)*100.0+0.000001]
        obj += cp.sum(cp.log(u[:,t]))
    #constr+=[u[5,9]<=0.1]
    obj -= cp.norm(x[:, -1]-x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve(solver=cp.ECOS)
    #print(x.value[:,-1])
    #print(u.value)

    if plot_fig==True:
        plt.plot(x.value[0])
        plt.plot(u.value[0])
        plt.show()
    #print("DOne")

    print(x.value, u.value)
    return x.value, u.value \n\n
    '''
    
    meta_prompt_end = f'''When a user asks you to solve an EV charging problem, you will ask the user for the necessary parameters as input with natural language. \
    Then, based on user's answer, you need to generate code to utilize the function. Here is an example:\
num_of_vehicles = 3
timesteps = 10
initial_states = [0, 0, 0]
max_power = 10.0
terminal_states = [70, 50, 100]
dept_time = [8, 6, 10]
power_capacity = 20
plot_fig = True

MPC_Solver(num_of_vehicles, timesteps, initial_states, max_power,
               terminal_states, dept_time, power_capacity, plot_fig)
        '''
    return meta_prompt_start + code + meta_prompt_end

text = create_prompt()
chat_completion = client.chat.completions.create(
messages=[
    {
        "role": "system",
        "content": text,
    },
    {
        "role": "user",
        "content": "Can you help me decide what is the best way to charge my EV.",
    },
    {
        "role": "assistant",
        "content": '''Of course, I'd be happy to help. To provide the best result, I need some specific details from you. Could you please tell me the following:

1. How many vehicles do you have that you need to charge?
2. Over how many timesteps are you planning to charge your vehicles? A timestep could be an hour, for example.
3. What are the initial states (charge levels) of your vehicles? Please list for each vehicle if more than one.
4. What's the maximum power each of your vehicles can consume?
5. What are the final desired states (charge levels) for each of your vehicles?
6. Until what timestep does each of your vehicles not need to start charging? Please list for each vehicle if more than one.
7. What's the total power capacity you want to use for charging all your vehicles?
8. Finally, would you like a plot figure of the charging status?''',
    },
        {
        "role": "user",
        "content": '''1. five, 
        2. 20 hours, 
        3. They all start from zero,
        4. 10,
        5. 100,
        6. [10, 12, 16, 18, 20],
        7. 30,
        8, Yes!
        ''',
    },
],
model="gpt-4",
)
#print(text)
print(chat_completion.choices[0].message.content)
output = chat_completion.choices[0].message.content

'''
Thank you for the information. Based on your input, we can utilize our EV charging solver as follows:

```python
num_of_vehicles = 5
timesteps = 20
initial_states = [0, 0, 0, 0, 0] # starting from zero for all vehicles
max_power = 10.0
terminal_states = [100, 100, 100, 100, 100] # desired state is 100 for all vehicles
dept_time = [10, 12, 16, 18, 20] # staggered start times for charging
power_capacity = 30
plot_fig = True

MPC_Solver(num_of_vehicles, timesteps, initial_states, max_power,
               terminal_states, dept_time, power_capacity, plot_fig)
```

This code will calculate the optimal charging strategy for your five EVs over 20 hours, considering the power constraints and desired charging levels you've specified.\
The function also produces a plot to visualize the plan. Be aware that the function uses a complex mathematical problem to optimise EV charging, so it might take a little time to get the output.
'''