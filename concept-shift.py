#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
from shutil import copy
from collections import Counter
import matplotlib.pyplot as plt
from optparse import OptionParser

class ConceptShift():
    
    def __init__(self, input_path, k, window_size, min_topics, shift_prob, output_path):
        self.filepaths = self.read_filepaths(input_path)
        self.remaining_topics = list(np.arange(len(self.filepaths)))
        self.time_windows, self.ground_truth = ([] for i in range(2))
        self.num_topics = k
        self.window_size = window_size
        self.min_topics = min_topics
        self.shift_prob = shift_prob
        self.output_path = output_path
        self.window_num = 0
        self.cur_topics = self.generate_initial_topics()
              
    def read_filepaths(self, directory):
        """ Reads dataset directory and stores the filepaths (documents) from each topic """
        folder_paths = [os.path.join(directory, folder) for folder in os.listdir(directory) if not folder.startswith('.')]
        filepaths = [[os.path.join(cur_folder, cur_file) for cur_file in os.listdir(cur_folder)] for cur_folder in folder_paths]
        return filepaths
    
    def generate_initial_topics(self):
        """ Generates the initial set of topics """
        initial_topics = random.sample(self.remaining_topics, self.num_topics)
        self.remaining_topics = [topic for topic in self.remaining_topics if topic not in initial_topics]
        return initial_topics
    
    def generate_window(self):
        """ Generates a time window of filepaths (documents) of a fixed size """
        self.window_num += 1
        cur_window = []
        for i in range(self.window_size):
            if len(self.cur_topics) < self.min_topics: break
            topic_num = np.random.choice(self.cur_topics)
            doc_num = random.randint(0, len(self.filepaths[topic_num])-1)
            cur_window.append(self.filepaths[topic_num][doc_num])
            del self.filepaths[topic_num][doc_num]
            if len(self.filepaths[topic_num]) == 0: self.cur_topics.remove(topic_num) # If a topic runs out of documents
        self.calculate_topic_distribution(cur_window)
        self.time_windows.append(cur_window)
        
    def choose_next_window_topics(self):
        """ Chooses randomly whether to activate a concept shift for the next window or to keep the current selection of topics """
        if random.uniform(0,1) < self.shift_prob: 
            if(random.randint(0,1) == 0): # Removes a topic
                topic_num = np.random.choice(self.cur_topics)
                self.cur_topics.remove(topic_num)
                self.remaining_topics.append(topic_num)      
            else: 
                if len(self.cur_topics) == len(self.filepaths): return # Can't add a topic if already using all available topics
                topic_num = np.random.choice(self.remaining_topics)
                self.remaining_topics.remove(topic_num)
                self.cur_topics.append(topic_num)              
       
    def calculate_topic_distribution(self, window):
        """ Calculates the number of filepaths (documents) for each topic in a time window """
        topic_distribution = Counter([os.path.basename(os.path.dirname(doc)) for doc in window])
        [print("Warning: Window %s, topic \"%s\" has less than 10 documents." % (self.window_num, topic)) for topic in topic_distribution if topic_distribution[topic] < 10]

    def save_dataset(self):
        """ Saves the dataset into folder designated by output_path """
        if os.path.exists(self.output_path):
            print('Directory already exists. EXITING.')
            sys.exit()
        if not os.path.exists(self.output_path): os.mkdir(self.output_path)
        for window_num, cur_window in enumerate(self.time_windows):
            window_dir = (os.path.join(self.output_path, ('window %s' % str(window_num + 1))))
            if not os.path.exists(window_dir): os.mkdir(window_dir)
            for filepath in cur_window:
                topic = os.path.basename(os.path.dirname(filepath))
                topic_dir = os.path.join(os.path.join(window_dir, topic))
                if not os.path.exists(topic_dir): os.mkdir(topic_dir)
                copy(filepath, topic_dir)
            self.ground_truth.append((len(os.listdir(window_dir))))
    
    def plot_dataset(self): 
        """ Plots the distribution of topics over all time windows """
        plt.plot(self.ground_truth, marker='o')
        plt.ylabel('Number of Topics')
        plt.xlabel('Window Number')
        plt.yticks(list(set(self.ground_truth)))
        plt.savefig(os.path.join(self.output_path, 'shift-plot.pdf'))

#-----------------------------------------------------------------------------------------------------                  
def main():
    parser = OptionParser(usage="usage: %prog [options] corpus_folder")
    parser.add_option("--input", action="store", type="string", dest="input_path", help="filepath to folder containing dataset")
    parser.add_option("-k", action="store", type="int", dest="num_topics", help="number of starting topics", default=5)
    parser.add_option("--window_size", action="store", type="int", dest="window_size", help="number of documents in a time window", default=100)
    parser.add_option("--min_topics", action="store", type="int", dest="min_topics", help="minimum topics before ending", default=3)
    parser.add_option("--shift_prob", action="store", type="float", dest="shift_prob", help="probability of a concept shift occuring", default=0.05)
    parser.add_option("--output", action="store", type="string", dest="output_path", help="filepath and folder name where dataset will be stored", default=0.05)
    parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
    (options, args) = parser.parse_args()

    gen = ConceptShift(options.input_path, options.num_topics, options.window_size, options.min_topics, options.shift_prob, options.output_path)
    
    print("Generating dataset...")    
    while(len(gen.cur_topics) >= gen.min_topics):
        gen.generate_window()
        gen.choose_next_window_topics()
    print("Saving dataset...")
    gen.save_dataset()	
    print("Generating plot of dataset...")
    gen.plot_dataset()
    print("Finished")

#-----------------------------------------------------------------------------------------------------     
if __name__ == "__main__":
	main()