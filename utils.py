import numpy
import pandas
from pathlib import Path
import json

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


    train_fitness_over_time=numpy.array(train_fitness_over_time)
    valid_fitness_over_time = numpy.array(valid_fitness_over_time)
    valid_accuracy_over_time= numpy.array(valid_accuracy_over_time)

    train_fitness = pandas.DataFrame({"generation":train_fitness_over_time[:,0], "train_fitness": train_fitness_over_time[:, 1]})
    valid_fitness = pandas.DataFrame({"generation":valid_fitness_over_time[:,0],  "valid_fitness":valid_fitness_over_time[:, 1]})
    valid_accuracy = pandas.DataFrame({"generation": valid_accuracy_over_time[:, 0], "valid_accuracy": valid_accuracy_over_time[:, 1]})
    train_fitness=train_fitness.merge(valid_fitness, left_on='generation', right_on='generation')
    train_fitness = train_fitness.merge(valid_accuracy, left_on='generation', right_on='generation')
    fig= train_fitness.plot(x="generation").get_figure()
    fig.suptitle(folder+"/"+name, fontsize=20)
    plotfile_path ="./" + folder + "/" + name + "test_accuracy_"+str(test_accuracy)+".png"
    fig.savefig(plotfile_path)
    return plotfile_path


