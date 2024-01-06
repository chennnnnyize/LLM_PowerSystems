import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from openai import OpenAI
import time

start_time = time.time()
print(torch.__version__)
print(torch.cuda.is_available())

load_demand = 400 ##(Mw)
units = 5 # the total number of thermal units

# In[] minimum and maximum generating limits of each unit
output_min = [28,90,68,76,19] ## P_i^th,min (MW)
output_max = [206,284,189,266,53] ## P_i^th,max (MW)

# In[] the cost coef a b c of these thermal units
coef_A = [3,4.05,4.05,3.99,3.88] ## a_i^th ($/MW^2)
coef_B = [20,18.07,15.55,19.21,26.18] ## b_i^th ($/MW)
coef_C = [100,98.87,104.26,107.21,95.31] ## c_i^th ($)

device = "cuda"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",  #Your API key
)


def calculate(thermal_output):
    # cost function
    cost = 0
    for i in range(units):
        cost += coef_A[i]*thermal_output[i]*thermal_output[i]+coef_B[i]*thermal_output[i]+coef_C[i]
    
    # Constranits violation
    max_operating_penalty = 0
    min_operating_penalty = 0
    balance_penalty = max(0, load_demand - sum(thermal_output))
    for i in range(units):
        max_operating_penalty += max(0, thermal_output[i]-output_max[i])
        min_operating_penalty += max(0, output_min[i]-thermal_output[i])
    return cost

def satisfy_load_constraints(thermal_output):
    if np.sum(thermal_output)>=400: return True
    else: return False
    
def satisfy_range_constraints(thermal_output):
    satisfies_constraints = all(output_min[i] <= thermal_output[i] <= output_max[i] for i in range(len(thermal_output)))
    return satisfies_constraints

thermal_output = np.array([135, 93, 70, 87, 30])

loss = calculate(thermal_output)

d = {'loss': [loss], 'p1': [thermal_output[0]], 'p2': [thermal_output[1]], 'p3': [thermal_output[2]], 'p4': [thermal_output[3]], 'p5': [thermal_output[4]]}
loss_list = [loss] # collect all losses for plotting at the end
p1_list = [thermal_output[0]]
p2_list = [thermal_output[1]]
p3_list = [thermal_output[2]]
p4_list = [thermal_output[3]]
p5_list = [thermal_output[4]]
df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
df.sort_values(by=['loss'], ascending=False, inplace=True)


def is_number_isdigit(s): # function for parsing str response from LLM
    n1 = s[0].replace('.','',1).replace('-','',1).strip().isdigit()
    n2 = s[1].replace('.','',1).replace('-','',1).strip().isdigit()
    n3 = s[2].replace('.','',1).replace('-','',1).strip().isdigit()
    n4 = s[3].replace('.','',1).replace('-','',1).strip().isdigit()
    n5 = s[4].replace('.','',1).replace('-','',1).strip().isdigit()
    return n1 * n2 * n3 * n4 * n5

def check_last_solutions(loss_list, last_nums): # function that stops optimization when the last 4 values of the loss function < 1
    if len(loss_list) >= last_nums:
        last = loss_list[-last_nums:]
        return all(num < 1 for num in last)

def create_prompt(num_sol): # create prompt
    meta_prompt_start = f'''You need assistance in solving an optimization problem. This problem involves 5 optimization variables, \
     namely p1, p2, p3, p4, and p5. These variables are subject to constraints defined by their minimum and maximum values: p_min=[28, 90, 68, 76, 19] \
     and p_max=[206, 284, 189, 266, 53]. Additionally, the sum of p1, p2, p3, p4, and p5 must be greater than or equal to 400. \
     Your objective is to provide values for p1, p2, p3, p4, and p5 that satisfy the constraints and minimize the optimization objective. \
     Below are some previous solution and their objective value pairs. The pairs are arranged in descending order based on their function values, where lower values are better.\n\n'''

    solutions = ''
    if num_sol > len(df.loss):
        num_sol = len(df.loss)

    for i in range(num_sol):
        solutions += f'''input:\np1={df.p1.iloc[-num_sol + i]:.3f}, p2={df.p2.iloc[-num_sol + i]:.3f}, p3={df.p3.iloc[-num_sol + i]:.3f}, p4={df.p4.iloc[-num_sol + i]:.3f}, p5={df.p5.iloc[-num_sol + i]:.3f} \nvalue:\n{df.loss.iloc[-num_sol + i]:.3f}\n\n''' 
    
    meta_prompt_end = f'''Give me a new (p1, p2, p3, p4, p5) pair that is different from all pairs above, and has a function value lower than
any of the above. Do not give me any explaination, the form of response must stritly follow the example: p1, p2, p3, p4, p5 = 123, 80, 99, 101, 37'''
    return meta_prompt_start + solutions + meta_prompt_end

def create_prompt_2(num_sol): # create prompt
    meta_prompt_start = f'''You need assistance in solving an optimization problem. This problem involves 5 optimization variables, \
     namely p1, p2, p3, p4, and p5. These variables are subject to constraints defined by their minimum and maximum values: p_min=[28, 90, 68, 76, 19] \
     and p_max=[206, 284, 189, 266, 53]. Additionally, the sum of p1, p2, p3, p4, and p5 must be greater than or equal to 400. \
     Your objective is to provide values for p1, p2, p3, p4, and p5 that satisfy the constraints and minimize the optimization objective. \
     Below are some previous solution and their objective value pairs. The pairs are arranged in descending order based on their function values, where lower values are better.\n\n'''

    solutions = ''
    if num_sol > len(df.loss):
        num_sol = len(df.loss)

    for i in range(num_sol):
        solutions += f'''input:\np1={df.p1.iloc[-num_sol + i]:.3f}, p2={df.p2.iloc[-num_sol + i]:.3f}, p3={df.p3.iloc[-num_sol + i]:.3f}, p4={df.p4.iloc[-num_sol + i]:.3f}, p5={df.p5.iloc[-num_sol + i]:.3f} \nvalue:\n{df.loss.iloc[-num_sol + i]:.3f}\n\n''' 
    
    meta_prompt_end = f'''Give me a new (p1, p2, p3, p4, p5) pair that is different from all pairs above, and has a function value lower than
any of the above. Do not give me any explaination, the form of response must stritly follow the example: p1, p2, p3, p4, p5 = 123.11, 80.2, 99.67, 101.52, 37'''
    return meta_prompt_start + solutions + meta_prompt_end

num_solutions = 15 # number of observations to feed into the prompt

for i in range(300):
    #print('start creating prompt...')
    text = create_prompt(num_solutions)
    #print('finish creating prompt...')
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": text,
        }
    ],
    model="gpt-4",
)
    #print(text)
    print(chat_completion.choices[0].message.content)
    output = chat_completion.choices[0].message.content
    #print('finish generating output...')
    response = output.split("p1, p2, p3, p4, p5 =")[1].strip()
    #print('response:',response)
    
    if "\n" in response:
        response = response.split("\n")[0].strip()
        
    if "," in response:
        numbers = response.split(',')
    print('numbers',i,numbers)
    
    if len(numbers)==5:
        if is_number_isdigit(numbers):
            p1, p2, p3, p4, p5 = float(numbers[0].strip()), float(numbers[1].strip()), float(numbers[2].strip()), float(numbers[3].strip()), float(numbers[4].strip())
            thermal_ = np.array([p1, p2, p3, p4, p5])
            if satisfy_load_constraints(thermal_) and satisfy_range_constraints(thermal_):
                print('flag')
                #print('thermal_output',thermal_)
                loss = calculate(thermal_)
                loss_list.append(loss)
                p1_list.append(p1)
                p2_list.append(p2)
                p3_list.append(p3)
                p4_list.append(p4)
                p5_list.append(p5)
                new_row = {'loss': loss, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}
                new_row_df = pd.DataFrame(new_row, index=[0])
                df = pd.concat([df, new_row_df], ignore_index=True)
                df.sort_values(by='loss', ascending=False, inplace=True)

    #print(f'{p1=} {p2=} {p3=} {p4=} {p5=} loss={loss:.3f}')

    if check_last_solutions(loss_list, 3):
        break

iterations = range(1, len(loss_list) + 1)

plt.plot(iterations, loss_list, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost over iterations')
plt.grid(True)
plt.savefig('loss_plot.png')
print(*loss_list, sep='\n')
print(df)

torch.save(np.array(loss_list),'loss.pt')
torch.save(np.array(p1_list),'p1.pt')
torch.save(np.array(p2_list),'p2.pt')
torch.save(np.array(p3_list),'p3.pt')
torch.save(np.array(p4_list),'p4.pt')
torch.save(np.array(p5_list),'p5.pt')
df.to_excel('opfgpt.xlsx', index=False)

end_time = time.time()
execution_time = end_time - start_time

print("Program execution time:", execution_time, "seconds")