import os
import torch
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import textwrap
import pygad
from torch.autograd import Variable
from torch import optim
from pygad import torchga
import model as model_lib
import dataset as dataset_lib
import utils
from argparse import Namespace
import json
from PIL import Image

#Get more printouts by settign this to True
verbose=False

# Pygad use a lot of callbacks that are tricky to send data to as arguments.
# We therfore need a lot of global variables. We store them all in a single dictionary
global_variables ={"train_fitness_over_time":[],"valid_fitness_over_time":[],"valid_accuracy_over_time":[],"data_inputs":None,"data_outputs":None,"train_loaders_iter":None,"device":None,"loss_func":None,"last_generation_time":None,"other_parameters":{}}

def ga_train(other_parameters,pygad_parameters, model, loaders):
    """
    Training a model with pygad GA based optimiser

    :param pygad_parameters: dictionary with the parameters used to train with pygad
    :param other_parameters: other parameters. e.g folders to save log files to
    :param model: the pytorch ann-model
    :param loaders: the dataset loaders
    :return: the best solution
    """
    # we do not need to compute any gradients since GA based optimizers dont use the gradients for optimisation
    model.eval()
    with torch.no_grad():
        #save all parameters that the different pygad callbacks need acess to in a single  dictionary
        global_variables["other_parameters"]=other_parameters
        global_variables["pygad_parameters"] = pygad_parameters
        global_variables["train_loaders_iter"] = iter(loaders["train"])

        # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
        torch_ga = torchga.TorchGA(model=model,
                                   num_solutions=pygad_parameters["sol_per_pop"])

        # load the first batch , more batches are loaded in the callback function callback_generation()
        (global_variables["data_inputs"], global_variables["data_outputs"]) = next(global_variables["train_loaders_iter"])
        # Transfer to GPU
        (global_variables["data_inputs"], global_variables["data_outputs"]) = global_variables["data_inputs"].to(global_variables["device"]), global_variables["data_outputs"].to(global_variables["device"])

        # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
        # num_parents_mating=Number of solutions to be selected as parents in the mating pool.
        ga_instance = pygad.GA(num_generations=pygad_parameters["num_generations"],
                               num_parents_mating=pygad_parameters["num_parents_mating"],
                               initial_population=torch_ga.population_weights,
                               fitness_func=fitness_func,
                               on_generation=callback_generation,
                               init_range_low=pygad_parameters["init_range_low"],
                               init_range_high=pygad_parameters["init_range_high"],
                               parent_selection_type=pygad_parameters["parent_selection_type"],
                               keep_parents=pygad_parameters["keep_parents"],
                               K_tournament=pygad_parameters["K_tournament"],
                               crossover_type=pygad_parameters["crossover_type"],
                               crossover_probability=pygad_parameters["crossover_probability"],
                               mutation_type=pygad_parameters["mutation_type"],
                               mutation_percent_genes=pygad_parameters["mutation_percent_genes"],
                               mutation_by_replacement=pygad_parameters["mutation_by_replacement"],
                               random_mutation_min_val=pygad_parameters["random_mutation_min_val"],
                               random_mutation_max_val=pygad_parameters["random_mutation_max_val"]
                               )
        ga_instance.run()

        if other_parameters["plot_fitness"]:
            # After the generation is complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
            ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4,save_dir="./"+other_parameters["Folder"]+"/"+other_parameters["Name"])
            plt.close('all')


        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        # Make predictions based on the best solution.
        predictions = pygad.torchga.predict(model=model,
                                            solution=solution,
                                            data=global_variables["data_inputs"])

        abs_error = global_variables["loss_func"](predictions, global_variables["data_outputs"])
        print("Absolute Error : ", abs_error.cpu().detach().numpy())
        return solution

def callback_generation(ga_instance):
    """
    This function is called by pygad as a callback everytime there is a new generation made
    logg validation, training fitness and validationset accuracy
    load next batch of data(if batchsize is smaller than totall dataset size) to use for evaluating the next generation

    :param ga_instance: the genetic algorithm instance that is training
    :return: None
    """

    #Print some information about the training to the screen
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    #best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()
    #best_solution_fitness = ga_instance.best_solutions_fitness[-1]
    best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    print("Fitness    = {fitness}".format(fitness=best_solution_fitness))
    if global_variables["last_generation_time"]:
        print("time for one generation: "+str(time.time()-global_variables["last_generation_time"]))
    global_variables["last_generation_time"] = time.time()

    #logg validation, training fitness and validationset accuracy
    if (ga_instance.generations_completed%10) ==0:
        (data_inputs, data_outputs) = next(iter(global_variables["loaders"]["test"])) #load the data and labels from validationset
        # Transfer to GPU
        (global_variables["data_inputs"], global_variables["data_outputs"]) = data_inputs.to(global_variables["device"]), data_outputs.to(global_variables["device"])
        valid_fitness = fitness_func(solution=best_solution, sol_idx=best_match_idx)
        valid_accuracy=accuracy_func(solution=best_solution)
        print("evalutaing on validation set, fitness:"+str(valid_fitness))
        print("evalutaing on validation set, accuracy:" + str(valid_accuracy))

        global_variables["train_fitness_over_time"].append([ga_instance.generations_completed,best_solution_fitness])
        global_variables["valid_fitness_over_time"].append([ga_instance.generations_completed,valid_fitness])
        global_variables["valid_accuracy_over_time"].append([ga_instance.generations_completed,valid_accuracy])
        #update training loggs
        utils.logg_values_for_plots(global_variables["train_fitness_over_time"], global_variables["valid_fitness_over_time"], global_variables["valid_accuracy_over_time"],folder=global_variables["other_parameters"]["Folder"], name=global_variables["other_parameters"]["Name"])
        #save best model
        model_lib.save_model(solution=best_solution,name=global_variables["other_parameters"]["Name"].rstrip(".csv")+".pt",model=global_variables["model"],folder=global_variables["other_parameters"]["Folder"])


    #change what data should be used for doing the fitness evaluation that drives the evolutionary selection
    print("loading new batch:")
    try:
        (data_inputs, data_outputs) = next(global_variables["train_loaders_iter"])
    except StopIteration:
        global_variables["train_loaders_iter"] = iter(global_variables["loaders"]["train"])
        (data_inputs, data_outputs) = next(global_variables["train_loaders_iter"])

    # Transfer to GPU
    (global_variables["data_inputs"], global_variables["data_outputs"]) = data_inputs.to(global_variables["device"]), data_outputs.to(global_variables["device"])




def test(model,loaders,solution):
    """
    Calculate accuracy on the test set

    :param model: the pytorch model
    :param loaders: dataset loaders
    :param solution: the genome of the individual we want to instantiate the model with weights from
    :return: the average accuracy of the solution evaluated on teh complete test set
    """
    #load the best weights
    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    # Test the model

    model.eval()
    with torch.no_grad():
        nr_of_tested_batches = 0
        sum=0
        for images, labels in loaders['test']:
            (images, labels) = images.to(global_variables["device"]), labels.to(global_variables["device"])
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            sum+=accuracy
            nr_of_tested_batches+=1

    return sum/nr_of_tested_batches


def accuracy_func(solution):
    """
    Calculate the solutions accuracy on the data in the global_variables["data_inputs"]

    :param solution: the genome of the individual for which we want to check its accuracy on MNIST
    :return: the accuracy of the solution on data_inputs
    """

    model_weights_dict = torchga.model_weights_as_dict(model=global_variables["model"],
                                                       weights_vector=solution)
    global_variables["model"].load_state_dict(model_weights_dict)
    predictions = global_variables["model"](global_variables["data_inputs"])
    pred_y = torch.max(predictions, 1)[1].data.squeeze()
    accuracy = (pred_y == global_variables["data_outputs"]).sum().item() / float(global_variables["data_outputs"].size(0))

    if verbose:
        print("accuracy:"+str(accuracy))

    return accuracy


def fitness_func(solution, sol_idx):
    """
    Calculate a fitness value for a solution

    :param solution: the genome of the individual for which we want to check its fitness on MNIST
    :param sol_idx:
    :return:
    """

    model_weights_dict = torchga.model_weights_as_dict(model=global_variables["model"],
                                                       weights_vector=solution)
    global_variables["model"].load_state_dict(model_weights_dict)
    predictions = global_variables["model"](global_variables["data_inputs"])

    solution_fitness = 1.0 / (global_variables["loss_func"](predictions, global_variables["data_outputs"]).cpu().detach().numpy() + 0.00000001)
    if verbose:
        print("solution_fitness:"+str(solution_fitness))

    return solution_fitness

def main(other_parameters,pygad_parameters):
    """
    Optimize a models performance in MNIST with the pygad library

    :param pygad_parameters: dictionary with the parameters used to train with pygad
    :param other_parameters: other parametrs. e.g folders to save log files to
    :return: accuracy of the best model on the MNIST test-set
    """


    time_main_start=time.time()

    # Device configuration
    if (torch.cuda.is_available() and not other_parameters["Use_cpu"]):
        global_variables["device"] = torch.device('cuda')
    else:
        global_variables["device"] = torch.device('cpu')

    print("using device: "+str(global_variables["device"]))

    #MODEL
    global_variables["model"]= model_lib.LeNet5(n_classes=10)
    global_variables["model"] = global_variables["model"].to(global_variables["device"])

    #LOSS FUNCTION
    global_variables["loss_func"]=nn.CrossEntropyLoss()



    #DATASET
    global_variables["loaders"] = dataset_lib.get_loaders(train_batch_size=other_parameters["Batchsize"])


    #START OPTIMIZATION
    print("starting ga training/evolution..")
    solution =ga_train(other_parameters=other_parameters,pygad_parameters=pygad_parameters, model=global_variables["model"], loaders=global_variables["loaders"])
    print("finnished  ga training/evolution..")

    test_accuracy = test(model=global_variables["model"], loaders=global_variables["loaders"], solution=solution)
    plotfile_path = utils.plot(global_variables["train_fitness_over_time"], global_variables["valid_fitness_over_time"], global_variables["valid_accuracy_over_time"], global_variables["other_parameters"]["Folder"], global_variables["other_parameters"]["Name"],test_accuracy)

    time_main_end = time.time()
    total_time = time_main_end-time_main_start
    print("train.py main() took : "+str(total_time))

    #LOG RESULT
    utils.log_result(global_variables["pygad_parameters"],global_variables["other_parameters"], total_time, test_accuracy)
    if other_parameters["plot_fitness"]:
        Image.open(plotfile_path).show()

    return test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training a MNIST digit classifier with GA',formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent('''\
         example usage:
            python train_GA.py --config path/to/configuration1.json path/to/configuration1.json
         '''))

    parser.add_argument('--config',
                        help='path/to/config.json', nargs='+',required=True,default=["./initial_atempt.json"])
    parsed_arguments = parser.parse_args()



    #Do a training for each configfile
    for config_file in parsed_arguments.config:
        print("Config files used for training:")
        print(config_file)
        with open(config_file,"r") as jsonfile:
            loaded_settings = json.load(jsonfile)

        pygad_parameter_names = [ "num_generations","num_parents_mating","sol_per_pop","init_range_low","init_range_high","parent_selection_type","keep_parents","K_tournament","crossover_type","crossover_probability","mutation_type","mutation_percent_genes","mutation_by_replacement","random_mutation_min_val","random_mutation_max_val"]
        # create a dictionary with parameters to pygad
        pygad_parameters = dict([parameter for parameter in loaded_settings.items() if parameter[0] in pygad_parameter_names])
        #create a dictionary with parameters not directly related to pygad
        other_parameters = dict([parameter for parameter in loaded_settings.items() if parameter[0] not in pygad_parameter_names])
        print("pygad_parameters:"+str(pygad_parameters))
        print("other_parameters:" + str(other_parameters))

        print("loaded settings:")
        print("##########################")
        print(loaded_settings)
        print("##########################")
        time.sleep(2)

        #Make a folder for storing results
        os.makedirs(loaded_settings["Folder"], exist_ok=True)
        #Optimize with pygad
        average_accuracy=main(pygad_parameters=pygad_parameters,other_parameters=other_parameters)
        print("average_accuracy :" + str(average_accuracy))





