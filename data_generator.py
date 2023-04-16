import numpy as np

class data_generator:

    def __init__(self, 
                 size, 
                 transition_model, 
                 emission_model, 
                 time_duration,
                 train_set_size=100, 
                 test_set_size=100, 
                 initial_range=None, 
                 print_process=False, 
                 numb_sound=3):
        
        self.size = size
        self.transition_model = transition_model
        self.emission_model = emission_model
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.time_duration = time_duration
        self.print_process = print_process
        self.numb_sound = numb_sound
        if initial_range == None:
            self.initial_range = [i for i in range(size * size)]
        else:
            self.initial_range = initial_range
            
    def generate_data(self):
        data_size = self.train_set_size + self.test_set_size
        list, x_train, x_test, y_train, y_test = [], [], [], [], []
        for init_position in self.initial_range:
            for i in range(data_size):
                data_label = self.simulation(tran_mod=self.transition_model, 
                                             emission_model=self.emission_model, 
                                             time_duration=self.time_duration, 
                                             size=self.size, 
                                             numb_sound=self.numb_sound, 
                                             start_position=init_position)
                list.append((data_label))
            if self.print_process:
                print(f"Finished simulation of position {init_position}...")
                
        np.random.shuffle(list)
        train_length = len(list) / data_size * self.train_set_size
        for index, ele in enumerate(list):
            if index < train_length:
                x, y = ele
                x_train.append(x)
                y_train.append(y)
            else:
                x, y = ele
                x_test.append(x)
                y_test.append(y)
        self.dataset = (x_train, y_train), (x_test, y_test)

    def simulation(self, tran_mod, emission_model, time_duration, size, numb_sound, start_position=0):
        for i in range(len(tran_mod)):
            sublist_sum = sum(tran_mod[i])
            tran_mod[i] = [x/sublist_sum for x in tran_mod[i]]
        choice_table = [i for i in range(size * size)]
        emission_tabel = [i for i in range(numb_sound)]
        
        n = time_duration
        start_state = start_position
        path, sound, numb_move, prev_state = [start_state], [], 0, start_state
        data_set, label_set = {}, {}
        while n >= 1:
            curr_state = np.random.choice(choice_table, p=tran_mod[prev_state])
            n -= 1
            path.append(curr_state)
            curr_obs = np.random.choice(emission_tabel, p=emission_model[curr_state])
            sound.append(curr_obs)
            if curr_state != prev_state:
                numb_move += 1
            prev_state = curr_state
        data_set['sound'], data_set['numb_move'], data_set['start_state'] = sound, numb_move, start_state
        label_set['path'] = path
        return data_set, label_set

    def get_data(self):
        return self.dataset