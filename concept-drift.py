#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
from shutil import copy
from collections import Counter
from optparse import OptionParser

class ConceptDrift():
    
    def __init__(self, input_path, k, window_size, decrease_windows, increase_windows, min_topics, drift_prob, output_path):
        self.filepaths = self.read_filepaths(input_path)
        self.topic_mappings = self.create_topic_mappings(input_path)
        self.remaining_topics = list(np.arange(len(self.filepaths)))
        self.time_windows, self.drift_list =  ([] for i in range(2))
        self.num_topics = k
        self.window_size = window_size
        self.decrease_windows = decrease_windows
        self.increase_windows = increase_windows
        self.min_topics = min_topics
        self.drift_prob = drift_prob
        self.output_path = output_path
        self.drift = False
        self.cur_topics, self.probabilities = self.generate_initial_topics()
        self.increase_topic = 0
        self.decrease_topic = 0
        self.increase_prob = 0
        self.counter = 0
        self.remove_topics = set()
               
    def read_filepaths(self, directory):
        """ Reads dataset directory and stores the filepaths (documents) from each topic  """
        folder_paths = [os.path.join(directory, folder) for folder in os.listdir(directory) if not folder.startswith('.')]
        filepaths = [[os.path.join(cur_folder, cur_file) for cur_file in os.listdir(cur_folder)] for cur_folder in folder_paths]
        return filepaths
    
    def create_topic_mappings(self, directory):
        """ Creates a mapping of topic names to an integer representaion """
        topics = [folder for folder in os.listdir(directory) if not folder.startswith('.')]
        indices = np.arange(len(self.filepaths))
        return dict(zip(indices, topics))
           
    def generate_initial_topics(self):
        """ Generates the initial set of topics """
        cur_topics = random.sample(self.remaining_topics, self.num_topics)
        self.remaining_topics = [topic for topic in self.remaining_topics if topic not in cur_topics]
        probabilities = list(np.random.dirichlet(np.ones(self.num_topics)))
        return cur_topics, probabilities
    
    def generate_window(self):
        """ Generates a time window of filepaths (documents) of a fixed size"""
        cur_window = []
        for i in range(self.window_size):
            if len(self.cur_topics) < self.min_topics: break
            self.probabilities = [float(i)/sum(self.probabilities) for i in self.probabilities] # Normalises the probabilities if sum is slightly greater than 1.0
            topic_num = np.random.choice(self.cur_topics, p=self.probabilities)
            if len(self.filepaths[topic_num]) == 0 and (self.increase_topic == topic_num or self.decrease_topic == topic_num) and self.drift: # If either the increase_topic or decrease_topic runs out of documents during a concept drift
                self.disable_drift()
                self.drift_list.append(self.drift)
            elif len(self.filepaths[topic_num]) == 0 and not self.drift: # If a topic runs out of documents and not in a concept drift
                self.distribute_topic_probabilities(topic_num)
            elif len(self.filepaths[topic_num]) == 0 and self.drift: # If a topic runs out of documents and in a concept drift
                self.remove_topics.add(topic_num)
                continue
            else:
                doc_num = random.randint(0, len(self.filepaths[topic_num])-1)
                cur_window.append(self.filepaths[topic_num][doc_num])
                del self.filepaths[topic_num][doc_num]
        self.drift_list.append(self.drift)
        self.time_windows.append(cur_window)
        
    def distribute_topic_probabilities(self, topic_num): # Only runs when not in a concept drift
        """ Removes a topic and distributes its probability evenly over the remaining currently selected topics """
        index = self.cur_topics.index(topic_num)
        self.cur_topics.remove(topic_num)  
        prob = self.probabilities[index] / len(self.cur_topics)
        del self.probabilities[index]
        self.probabilities = [x + prob for x in self.probabilities]
              
    def choose_next_window_topics(self):
        """ Chooses randomly whether to activate a concept shift for the next window or keep the current selection of topics """
        if random.uniform(0,1) < self.drift_prob and not self.drift and not len(self.remaining_topics) == 0: 
            self.enable_drift()  
            print('Drift enabled')
        if self.drift: 
            self.modify_probabilities() 
            
    def modify_probabilities(self): # Assumes that the number of increase windows is always bigger than the number of decrease windows 
        """ Controls the increase and decrease probability methods """
        if self.counter == self.increase_windows: #
            self.disable_drift()
            return
        if self.counter == self.decrease_windows: self.remove_decreasing_topic()          
        self.increase_probabilities() 
        if self.counter < self.decrease_windows:  self.decrease_probabilities()
        self.counter += 1 
        
    def drift_distribute_probabilities(self):
        """ Removes any topics that ran out of documents during a drift and distirbutes the probabilities amongst the topics that remain  """
        for topic in self.remove_topics:
            del self.probabilities[self.cur_topics.index(topic)]
            self.cur_topics.remove(topic)
        total = sum(self.probabilities)
        for prob in range(len(self.probabilities)):
            self.probabilities[prob] /= total
        
    def remove_decreasing_topic(self): # Assumes that the number of increase windows is always bigger than the number of decrease windows 
        """ Removes the decreasing topic during a drift """
        if self.decrease_topic in self.remove_topics: self.remove_topics.remove(self.decrease_topic)
        decrease_topic_index =  self.cur_topics.index(self.decrease_topic)
        self.cur_topics.remove(self.decrease_topic)
        del self.probabilities[decrease_topic_index]
        self.remaining_topics.append(self.decrease_topic)
        
    def increase_probabilities(self):
        """ Increases the increase topic probability while decreasing all remaining topics accordingly """
        increase_topic_index = self.cur_topics.index(self.increase_topic)
        remaining_prob = self.increase_prob - self.probabilities[increase_topic_index] # How much is left to work with
        remaining_windows = self.increase_windows - self.counter 
        cur_prob = remaining_prob / remaining_windows
        for i in range(len(self.probabilities)):
            if i == increase_topic_index: self.probabilities[i] += cur_prob       
            else:
                self.probabilities[i] -= cur_prob/(len(self.probabilities)-1)
                if self.probabilities[i] < 0: self.probabilities[i] = 0
            
    def decrease_probabilities(self):
         """ Decreases the decrease topic probability while increasing all remaining topics accordingly """
         if self.counter < self.decrease_windows: 
            decrease_topic_index = self.cur_topics.index(self.decrease_topic)      
            remaining_prob = self.probabilities[decrease_topic_index] # How much is left to work with
            remaining_windows = self.decrease_windows - self.counter 
            cur_prob = remaining_prob / remaining_windows
            for i in range(len(self.probabilities)):
                 if i == decrease_topic_index: self.probabilities[i] -= cur_prob
                 else:  self.probabilities[i] += cur_prob/(len(self.probabilities)-1)
                                          
    def enable_drift(self):
        """ Enables concept drift """
        self.drift = True
        self.counter = 0
        self.decrease_topic = np.random.choice(self.cur_topics)
        self.increase_topic = np.random.choice(self.remaining_topics)
        self.remaining_topics.remove(self.increase_topic)
        self.increase_prob = np.mean(self.probabilities)
        self.probabilities.append(0.0)
        self.cur_topics.append(self.increase_topic)
        self.remove_topics = set()
                       
    def disable_drift(self):
        """ Disables concept drift """
        self.drift = False
        self.drift_distribute_probabilities()
        self.counter = 0
        
    def save_dataset(self):
        """ Saves the dataset into folder designated by output_path """
        if os.path.exists(self.output_path):
            print('Directory already exists. EXITING.')
            sys.exit()
        if not os.path.exists(self.output_path): os.mkdir(self.output_path)
        for window_num, cur_window in enumerate(self.time_windows):
            topic_distribution = Counter([os.path.basename(os.path.dirname(doc)) for doc in cur_window])
            remove_topics = [topic for topic in topic_distribution if topic_distribution[topic] < 10]
            window_dir = (os.path.join(self.output_path, "window-{0:02d}".format(window_num + 1)))
            if not os.path.exists(window_dir): os.mkdir(window_dir)
            for filepath in cur_window:
                topic = os.path.basename(os.path.dirname(filepath))
                topic_dir = os.path.join(os.path.join(window_dir, topic))
                if not os.path.exists(topic_dir) and topic not in remove_topics: os.mkdir(topic_dir)
                if topic not in remove_topics: copy(filepath, topic_dir)
            
    def write_csv(self):
        """ Creates an overview of the drift dataset and saves it in as a csv file """
        topic_distributions = []
        num_topics = []
        num_docs = []
        for window_num, cur_folder in enumerate(os.listdir(self.output_path)):
            num_topics.append(len(os.listdir(os.path.join(self.output_path, cur_folder))))
            window_dir = os.path.join(self.output_path, cur_folder) 
            filepaths = []
            for topic in os.listdir(window_dir):
                topic_dir = os.path.join(window_dir,topic)
                for file in os.listdir(topic_dir):
                    filepaths.append(os.path.join(topic_dir, file))
            topic_dict = Counter([os.path.basename(os.path.dirname(doc)) for doc in filepaths])
            num_docs.append(sum(topic_dict.values()))
            topic_distributions.append(self.calculate_window_distribution(len(self.filepaths), self.topic_mappings, topic_dict))
            
        with open(os.path.join(self.output_path,'drift_overview.csv'), 'w') as f:
             heading = ['Num Topics', 'Num Docs', 'Drift']
             for i in range(1,len(self.filepaths)+1):
                 heading.append('Topic %s ' % i)
             f.write(','.join(heading)+'\n')
             for i in range(len(topic_distributions)):
                f.write(str(num_topics[i]) + ',' + str(num_docs[i]) + ',' + str(self.drift_list[i]) + ',' + ','.join(topic_distributions[i]) + '\n')

    def calculate_window_distribution(self, total_topics, topic_mappings, topic_dict):
        """ Calculates the number of documents in each topic for a time window """
        topic_distributions = []
        for i in range(total_topics):
            if topic_mappings[i] in topic_dict:
                topic_distributions.append("%.3f" % (topic_dict[topic_mappings[i]] / sum(topic_dict.values())))
            else:
                topic_distributions.append('0.0')
        return topic_distributions        

#----------------------------------------------------------------------------------------------------- 
def main():
    parser = OptionParser(usage="usage: %prog [options] corpus_folder")
    parser.add_option("--input", action="store", type="string", dest="input_path", help="filepath to folder containing dataset")
    parser.add_option("-k", action="store", type="int", dest="num_topics", help="number of starting topics", default=5)
    parser.add_option("--window_size", action="store", type="int", dest="window_size", help="number of documents in a window", default=100)
    parser.add_option("--decrease_windows", action="store", type="int", dest="decrease_windows", help="number of windows for a topic to be gradually removed", default=10)
    parser.add_option("--increase_windows", action="store", type="int", dest="increase_windows", help="number of windows for a topic to be gradually introduced", default=15)
    parser.add_option("--min_topics", action="store", type="int", dest="min_topics", help="minimum number of topics before ending", default=3)
    parser.add_option("--drift_prob", action="store", type="float", dest="drift_prob", help="probability of a concept drift occuring", default=0.05)
    parser.add_option("--output", action="store", type="string", dest="output_path", help="filepath and folder name where dataset will be stored", default=None)
    (options, args) = parser.parse_args()

    gen = ConceptDrift(options.input_path, options.num_topics, options.window_size, options.decrease_windows, options.increase_windows, options.min_topics, options.drift_prob, options.output_path)
    
    print("Generating dataset...")  
    while(len(gen.cur_topics) >= gen.min_topics):
        gen.generate_window()
        gen.choose_next_window_topics()
    print("Saving dataset...")
    gen.save_dataset()
    print("Generating overview...")
    gen.write_csv()
    print("Finished.")

#-----------------------------------------------------------------------------------------------------     
if __name__ == "__main__":
	main()