import numpy as np
import random

def getNextGeneration(accuracies, border_matrix, top_count, mutation_prop=0, distribution=[0.5, 0.25, 0.15, 0.1, 0.0], distance_threshold=0.05, mutation_scale=0.05, cross='recombine'):

    if distance_threshold > 0.0:
            #we want to penalize borders that are too close together
        adapted_accuracies = getAdjustedAccuracies(accuracies, border_matrix, distance_threshold)
    else:
        adapted_accuracies = accuracies.copy()

    ordered_adj_list = adapted_accuracies.argsort()[::-1]
    ordered_adj_matrix = border_matrix[ordered_adj_list]

    ordered_list = accuracies.argsort()[::-1]
    ordered_matrix = border_matrix[ordered_list]

    offspring_count = len(adapted_accuracies) - top_count
    offspring_matrix = getOffspring(ordered_matrix, offspring_count, distribution, cross)

    if mutation_prop > 0.0:    #perform mutation if applicable
        performMutation(offspring_matrix, mutation_prop, mutation_scale)

        #combine top and new
    new_gen_matrix = np.vstack((ordered_adj_matrix[:top_count], offspring_matrix))
    new_gen_matrix = np.sort(new_gen_matrix, axis=1)    #ensure borders are ordered left to right

    return new_gen_matrix


def getOffspring(ordered_matrix, count, distribution, combine):
    offspring_matrix = np.zeros((count, len(ordered_matrix[0,:])))

    cum_distribution = np.cumsum(distribution)
    distr_parts = len(distribution)

    prop_split = np.hstack(([0], [1/distr_parts for _ in range(distr_parts)]))
    cum_prop = np.cumsum(prop_split)
    bin_counts = np.round(cum_prop*count)

    for index in range(count):
        sample1 = getRandomSample(ordered_matrix, cum_distribution, bin_counts)
        sample2 = getRandomSample(ordered_matrix, cum_distribution, bin_counts)

        if combine == 'crossover':
            offspring_matrix[index, :] = crossover(sample1, sample2)
        elif combine == 'recombine':
            offspring_matrix[index, :] = recombine(sample1, sample2)
        elif combine == 'collapse':
            offspring_matrix[index, :] = collapse(sample1, sample2)

    return offspring_matrix


    ##this method simply gives a 50% to each border
def recombine(sample1, sample2):
    child_sample = np.zeros(len(sample1))
    for index in range(len(sample1)):
        if random.random() > 0.5:
            child_sample[index] = sample1[index]
        else:
            child_sample[index] = sample2[index]
    return child_sample


    ##The crossover method cuts borders in
def crossover(sample1, sample2):
    child_sample = []

    sample_length = len(sample1)

    left_items = round(random.uniform(0.25, 0.75) * sample_length)
    right_items = sample_length - left_items

    crossover_point = random.random()

    index1 = 0
    while index1 < sample_length and sample1[index1] < crossover_point:
        index1 += 1

    index2 = sample_length-1
    while index2 >= 0 and sample2[index2] > crossover_point:
        index2 -= 1

    for index in range(index1, index1+right_items):
        child_sample.append(sample1[index%sample_length])

    for index in range(index2-left_items+1, index2+1):
        child_sample.append(sample2[index%sample_length])


    return np.array(child_sample)

def collapse(sample1, sample2, threshold=0.05):
    combined_borders = np.hstack([sample1, sample2])
    combined_borders = np.sort(combined_borders)    #ensure borders are ordered left to right
    combined_borders = filterNearby(combined_borders, len(sample1), threshold) #combine closeby borders
    new_borders = np.empty((len(sample1)))
    counter = 0
    while counter < len(sample1):
        index = random.randint(0, len(combined_borders)-1)
        new_borders[counter] = combined_borders[index]
        combined_borders = np.delete(combined_borders,[index])
        counter += 1
    return new_borders


def distanceRight(border_array):
    border_distances = np.hstack([border_array[1:] - border_array[:-1], (1+border_array[0]) - border_array[-1]])
    min_index = np.argmin(border_distances)
    return (border_distances, min_index)


def filterNearby(combined_borders, samples, threshold):
    counter = 0
    border_distances, min_index = distanceRight(combined_borders)
    while len(combined_borders) > samples and border_distances[min_index] < threshold:
        if min_index == len(combined_borders)-1:
            merged_border = ((combined_borders[min_index] + combined_borders[0]+1)/2) % 1
            if merged_border > combined_borders[min_index-1]:
                combined_borders[min_index] = merged_border
                combined_borders = np.delete(combined_borders,[0])
            else:
                combined_borders[0] = merged_border
                combined_borders = np.delete(combined_borders, [min_index])
        else:
            combined_borders[min_index] = (combined_borders[min_index] + combined_borders[min_index+1])/2
            combined_borders = np.delete(combined_borders,[min_index+1])

        border_distances, min_index = distanceRight(combined_borders)

        if counter == 3: threshold -= 0.01
        counter += 1

    return combined_borders

def getRandomSample(border_matrix, cum_distribution, bin_count):
    random_float = random.random()
    for index, cum_top in enumerate(cum_distribution):
        if random_float <= cum_top:
            lower_bound = bin_count[index]
            upper_bound = bin_count[index+1]
            sample = random.randint(lower_bound, upper_bound)
            return border_matrix[sample, :]


def performMutation(offspring_matrix, mutation_prop, mutation_scale):
    (rows, columns) = offspring_matrix.shape
    for _ in range(int(rows*columns*mutation_prop)):

        index_row = random.randint(0, rows-1)
        index_column = random.randint(0, columns-1)
        offspring_matrix[index_row, index_column] = offspring_matrix[index_row, index_column] + np.random.normal(scale=mutation_scale)

        #we want to make sure values don't go out of bounds
    offspring_matrix = np.mod(offspring_matrix, 1)

    return offspring_matrix


def getAdjustedAccuracies(accuracies, border_matrix, distance_threshold):

    adj_accuracies = np.empty(accuracies.shape)
    for index, accuracy in enumerate(accuracies):
        row = border_matrix[index, :]
        min_dist = 1
        for column_index in range(len(row)):

            if column_index == len(row)-1:  #for the last column we wrap around
                distance = (row[0] + 1) - row[column_index]
            else:
                distance = row[column_index+1] - row[column_index]

            if distance < min_dist:
                min_dist = distance

                #we punish if distance < 0.05
        if min_dist <= distance_threshold/5:   #ensure no division by zero
            adj_accuracies[index] = accuracy - (0.5/0.01)/100
        elif min_dist < distance_threshold:
            adj_accuracies[index] = accuracy - (0.5/min_dist)/100

    return adj_accuracies


def initializeBorderMatrix(model_count, parts, folder):
    try:
        border_matrix = np.load(folder + 'genetic_border_matrix_' + str(parts) + 'parts.npy')
        print('Loading the old results')
    except:
        border_matrix = initializeNewBorderMatrix(model_count, parts)
    return border_matrix


def initializeNewBorderMatrix(model_count, parts):
        border_matrix = np.random.random((model_count, parts))
        border_matrix = np.sort(border_matrix, axis=1)
        return border_matrix
