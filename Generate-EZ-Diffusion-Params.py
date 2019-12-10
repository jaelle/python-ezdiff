import pandas as pd
import numpy as np
import math

# Edit Configuration:
subject_ids = [771,795,796,800,818,819,820,832,833,844,845,846,861,862,877,893,895,896,897,898,899,908,909,910,911,925,926,954,955]
shift_locations = [-90,-45,0,45,90]
output_file = 'EZ-shifted-stats-12-6-2019.csv'

# TODO: Edit the path to individual reaction time files in the main() function below.

# EZ Diffusion model functions
def logit(p):
    return math.log(p/(1-p))

def ezdiff(subject_id,MRT,VRT,p):

    if p == 0:
        print("Oops, only errors for subject " + subject_id + "!")
    elif p == 0.5:
        print("Oops, chance performance for " + subject_id + "!")
    elif p == 1:
        print("Oops,  only correct responses for " + subject_id + "!")
    
    s = 0.1
    s2 = s*s # scaling parameter squared
    L = logit(p)
    x1 = (L*p*p - L*p + p - 0.5)
    x = L * x1 / VRT
    v = np.sign(p - 0.5) * s*math.pow(x,0.25)
    a = s2*logit(p)/v
    y = -v*a/s2
    MDT = (a/(2*v))*(1-math.exp(y))/(1+math.exp(y))
    Ter = MRT - MDT
    
    return([v,a,Ter])

def main(subject_ids, shift_locations, output_file):
    for subject_id in subject_ids:
        # TODO: Make sure the path below points to individual reaction time files.
        # The files need to have the following trial information, ordered by columns in the following order: [condition, accurate (1 or 0), reaction time in seconds]
        behavioral_all = pd.read_csv('shifted/vigilance_' + str(subject_id) + '_shift.csv',header=None,names=['location','accuracy','RT'])
        mean = behavioral_all['RT'].mean()
        std = behavioral_all['RT'].std()
        
        # remove trials below three standard deviations of the mean
        behavioral = behavioral_all[behavioral_all['RT'] > (mean - 3*std)]
        
        # save subject_id parameters as new row in results
        row = [subject_id]
        
        # calculate EZ parameters for each shift location
        for loc in shift_locations:
            shift_trials = behavioral[behavioral['location'] == loc]
            
            # scale to seconds
            mean = shift_trials['RT'].mean() / 1000
            
            # scale to seconds squared
            var = shift_trials['RT'].var() / 1000000
            
            acc = shift_trials['accuracy'].mean()
      
            [v,a,Ter] = ezdiff(subject_id,mean,var,acc)
        
            row += [mean,var,acc,v,a,Ter]
        
        results += [row]

    # setup column names based on shift locations
    column_names = ['subject_id']

    for loc in shift_locations:
        column_names += ['mean_'+str(loc),
                         'var_'+str(loc),
                         'acc_'+str(loc),
                         'v_'+str(loc),
                         'a_'+str(loc),
                         'Ter_'+str(loc)]

    pd.DataFrame(results).to_csv(output_file,index=False, header=column_names)

main(subject_ids,shift_locations,output_file)
