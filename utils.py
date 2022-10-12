import numpy
import pandas
from pathlib import Path
import matplotlib.pyplot as plt
import json
import argparse

def logg_values_for_plots(train_fitness_over_time,valid_fitness_over_time,valid_accuracy_over_time,folder,name):
    """
    #saves the fitness values (on training set and validation set) and validation set accuracy,  as csv files

    :param train_fitness_over_time:list with pairs (generation_number,trainingset-fitness)
    :param valid_fitness_over_time: list with pairs (generation_number,validationset-fitness)
    :param valid_accuracy_over_time: list with pairs (generation_number,validationset-accuracy)
    :param folder: folder to save files in
    :param name: file to save logs to (will be turned into several different log files that starts with the same path)
    :return: None
    """
    train_fitness_over_time = numpy.array(train_fitness_over_time)
    valid_fitness_over_time = numpy.array(valid_fitness_over_time)
    valid_accuracy_over_time = numpy.array(valid_accuracy_over_time)

    train_fitness = pandas.DataFrame(
        {"generation": train_fitness_over_time[:, 0], "train_fitness": train_fitness_over_time[:, 1]})
    valid_fitness = pandas.DataFrame(
        {"generation": valid_fitness_over_time[:, 0], "valid_fitness": valid_fitness_over_time[:, 1]})
    valid_accuracy = pandas.DataFrame(
        {"generation": valid_accuracy_over_time[:, 0], "valid_accuracy": valid_accuracy_over_time[:, 1]})

    #save the dataframes as csv files
    valid_fitness.to_csv(Path(folder)/Path(name+ "valid_fitness.csv"))
    train_fitness.to_csv(Path(folder)/Path(name+ "train_fitness.csv"))
    valid_accuracy.to_csv(Path(folder)/Path(name+ "valid_accuracy.csv"))

def log_result(pygad_parameters,other_parameters,time,test_accuracy):
    """
    Stores a dictionary with the experiment settings, time used for experiment and final test_accuracy as a .json file

    :param pygad_parameters: the parameters used to train with pygad
    :param other_parameters: the parameters for the experiment not directly realated to pygad (folder to save logs etc)
    :param time: the total time the experiment took
    :param test_accuracy: the test accuracy of the best individual of the final population
    :return: None
    """
    comined_dictionary = dict(list(pygad_parameters.items())+list(other_parameters.items()))
    comined_dictionary["time"]=str(time)
    comined_dictionary["test_accuracy"] = str(test_accuracy)
    with open(Path(other_parameters["Folder"])/Path(other_parameters["Name"]+".json"), "w") as out_file:
        json.dump(comined_dictionary, out_file, indent=6)

def plot(train_fitness_over_time,valid_fitness_over_time,valid_accuracy_over_time,folder,name,test_accuracy):
    
    """
    plot the training and valid fitness, together with the validations set accuracy, time series.

    :param train_fitness_over_time:
    :param valid_fitness_over_time:
    :param valid_accuracy_over_time:
    :param folder:
    :param name:
    :param test_accuracy:
    :return: path to plot file
    """
    #make sure there are no old plots active
    plt.clf()
    

    train_fitness_over_time=numpy.array(train_fitness_over_time)
    valid_fitness_over_time = numpy.array(valid_fitness_over_time)
    valid_accuracy_over_time= numpy.array(valid_accuracy_over_time)

    train_fitness = pandas.DataFrame({"generation":train_fitness_over_time[:,0], "train_fitness": train_fitness_over_time[:, 1]})
    valid_fitness = pandas.DataFrame({"generation":valid_fitness_over_time[:,0],  "valid_fitness":valid_fitness_over_time[:, 1]})
    valid_accuracy = pandas.DataFrame({"generation": valid_accuracy_over_time[:, 0], "valid_accuracy": valid_accuracy_over_time[:, 1]})
    train_fitness=train_fitness.merge(valid_fitness, left_on='generation', right_on='generation')
    train_fitness = train_fitness.merge(valid_accuracy, left_on='generation', right_on='generation')
    fig= train_fitness.plot(x="generation").get_figure()
    fig.suptitle(folder+"/"+name, fontsize=10)
    plotfile_path ="./" + folder + "/" + name + "test_accuracy_"+str(test_accuracy)+".png"
    fig.savefig(plotfile_path)
    return plotfile_path




if __name__ == "__main__":
    """
    plot the result of an experiment
    
    """
    usage_example="example usage: \n "+r"python utils.py --train_fitness .\medium_post_results\first_experiment\first_trytrain_fitness.csv --valid_fitness .\medium_post_results\first_experiment\first_tryvalid_fitness.csv --valid_accuracy .\medium_post_results\first_experiment\first_tryvalid_accuracy.csv"
    # Initialize parser
    parser = argparse.ArgumentParser(
                                    epilog=usage_example,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("--train_fitness", help="a path to csv file",required=True)
    parser.add_argument("--valid_fitness", help="a path to csv file",required=True)
    parser.add_argument("--valid_accuracy", help="a path to csv file",required=True)
    parser.add_argument("--test_accuracy", help="e.g 92.0",required=True)
    parser.add_argument("--name", help="e.g testplot",required=True)
    parser.add_argument("--folder", help="e.g testfolder",required=True)
    args = parser.parse_args()
    
    
    valid_fitness = pandas.read_csv(args.valid_fitness)
    train_fitness = pandas.read_csv(args.train_fitness)
    valid_accuracy = pandas.read_csv(args.valid_accuracy)
    
    
    #remove the index wich we dont want to plott  
    valid_fitness= valid_fitness.drop(columns= ['Unnamed: 0']) 
    train_fitness= train_fitness.drop(columns= ['Unnamed: 0']) 
    valid_accuracy= valid_accuracy.drop(columns= ['Unnamed: 0']) 
    
 
    
    plot(train_fitness_over_time=train_fitness.to_numpy(),valid_fitness_over_time=valid_fitness.to_numpy(),valid_accuracy_over_time=valid_accuracy.to_numpy(),folder=args.folder,name=args.name,test_accuracy=args.test_accuracy)



