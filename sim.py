import simpy
import pandas as pd

# Load the Excel files
itu_df = pd.read_csv('ITU3.csv')
train_df = pd.read_csv('train3.csv')
truck_df = pd.read_csv('truck4.csv')
T1 = 3 * 60  # Convert hours to minutes
T2 = 24 * 60 # Convert hours to minutes



# Define the ITU class
class ITU:
    def __init__(self, env, id, priority=1):
        self.env = env
        self.id = id
        self.priority = priority
        self.train_arrival_time = None
        self.train_departure_time = None
        self.unloaded = False
        self.stored = False
        self.loaded = False
        self.storage_start_time = None
        self.storage_end_time = None


# Define the Train class
class Train:
    def __init__(self, env, id, itus, arrival_time, departure_time):
        self.env = env
        self.id = id
        self.itus = itus
        self.arrival_time = arrival_time
        self.departure_time = departure_time


# Define the StorageYard class
class StorageYard:
    def __init__(self, env, long_term_capacity=1000, buffer_capacity=200, max_stack=2):
        self.env = env
        self.long_term_capacity = long_term_capacity
        self.buffer_capacity = buffer_capacity
        self.max_stack = max_stack
        self.long_term_storage = {}
        self.buffer_storage = {}
        self.itu_entry_times = {}
        self.total_itus_stored = 0
        self.max_utilization = 0
        self.itu_entry_times = {}


    def store_itu(self, itu, storage_type):
        if storage_type == 'buffer':
            storage = self.buffer_storage
            capacity = self.buffer_capacity
        else:
            storage = self.long_term_storage
            capacity = self.long_term_capacity

        available_slot = next((slot for slot in storage if len(storage[slot]) < self.max_stack), None)
        if available_slot is not None:
            storage[available_slot].append(itu)
        elif len(storage) < capacity:
            new_slot = len(storage) + 1
            storage[new_slot] = [itu]
        else:
            print(f"No {storage_type} space available for ITU {itu.id} at time {self.env.now:.2f}")
            return

        itu.stored = True
        itu.storage_start_time = self.env.now
        self.update_utilization()
        self.itu_entry_times[itu.id] = self.env.now
        print(f"ITU {itu.id} stored in {storage_type} at time {self.env.now}")

    def retrieve_itu(self, itu, storage_type):
        storage = self.buffer_storage if storage_type == 'buffer' else self.long_term_storage
        for slot in storage:
            if itu in storage[slot]:
                storage[slot].remove(itu)
                itu.storage_end_time = self.env.now
        entry_time = self.itu_entry_times.pop(itu.id, None)
        if entry_time is not None:
            dwell_time = self.env.now - entry_time
            self.total_dwell_time += dwell_time
            self.total_itus_stored += 1
            print(f"ITU {itu.id} retrieved from {storage_type} at time {self.env.now}, dwell time: {dwell_time}")
                         

    def update_utilization(self):
        long_term_utilization = sum(len(stack) for stack in self.long_term_storage.values()) / (self.long_term_capacity * self.max_stack)
        buffer_utilization = sum(len(stack) for stack in self.buffer_storage.values()) / (self.buffer_capacity * self.max_stack)
        total_utilization = (long_term_utilization * self.long_term_capacity + buffer_utilization * self.buffer_capacity) / (self.long_term_capacity + self.buffer_capacity)
        if total_utilization > self.max_utilization:
            self.max_utilization = total_utilization

    def get_average_dwell_time(self):
        return self.total_dwell_time / self.total_itus_stored if self.total_itus_stored > 0 else 0

    def get_utilization_rate(self):
        return self.max_utilization


# Define the Truck class
class Truck:
    def __init__(self, env, id, arrival_time, purpose, road_gate):
        self.env = env
        self.id = id
        self.arrival_time = arrival_time
        self.purpose = purpose
        self.road_gate = road_gate
        self.waiting_time = 0
        self.service_start_time = None
        self.env.process(self.run())

    def run(self):
        yield self.env.timeout(self.arrival_time)
        arrive_time = self.env.now
        with self.road_gate.resource.request() as req:
            yield req
            self.service_start_time = self.env.now
            self.waiting_time = self.service_start_time - arrive_time
            self.road_gate.total_waiting_time += self.waiting_time
            self.road_gate.total_trucks += 1
            yield self.env.timeout(3)  # Assuming 3 minutes service time
        print(f'Truck {self.id} waited for {self.waiting_time} minutes')


# Define the RoadGate class
class RoadGate:
    def __init__(self, env, gate_id, truck_data):
        self.env = env
        self.gate_id = gate_id
        self.truck_arrivals = {}
        self.total_waiting_time = 0
        self.total_trucks = 0
        self.truck_data = truck_data
        self.resource = simpy.Resource(env, capacity=1)  # Add this line
        self.process_truck_data()
        self.env.process(self.run())

    def process_truck_data(self):
        print("Processing truck data in RoadGate:")
        for _, truck in self.truck_data.iterrows():
            itu_id = str(int(truck['ITU corrispondente']))  # Ensure it's a string without decimal
            arrival_time = self.hours_to_minutes(truck['Orario di arrivo'])
            print(f"Processing truck for ITU {itu_id}, arrival time: {arrival_time}")
            self.truck_arrivals[itu_id] = arrival_time

    def hours_to_minutes(self, time):
        return int(float(time) * 60)

    def get_truck_arrival_time(self, itu):
        arrival_time = self.truck_arrivals.get(str(itu.id), None)
        print(f"Getting truck arrival time for ITU {itu.id}: {arrival_time}")
        return arrival_time

    def run(self):
        for _, truck in self.truck_data.iterrows():
            itu_id = str(int(truck['ITU corrispondente']))
            arrival_time = self.hours_to_minutes(truck['Orario di arrivo'])
            yield self.env.timeout(arrival_time)
            self.env.process(Truck(self.env, itu_id, arrival_time,
                                   truck['ritiro(0)/consegna(1)/entrambi(2)'], self).run())
        yield self.env.timeout(0)
    def hours_to_minutes(self, time):
        return int(float(time) * 60)

    def get_truck_arrival_time(self, itu):
        arrival_time = self.truck_arrivals.get(itu.id, None)
        print(f"Getting truck arrival time for ITU {itu.id}: {arrival_time}")
        return arrival_time

    def average_truck_waiting_time(self):
        return self.total_waiting_time / self.total_trucks if self.total_trucks > 0 else 0


# Define the CraneOperation class
class CraneOperation:
    def __init__(self, env, num_cranes=3):
        self.env = env
        self.num_cranes = num_cranes
        self.cranes = [simpy.Resource(env, capacity=1) for _ in range(num_cranes)]
        self.total_operations = 0
        self.operating = {crane: False for crane in self.cranes}
        self.last_operation_end_time = 0


    def operate(self, operation_time):
        with self.cranes[0].request() as req:
            yield req
            if self.env.now > self.last_operation_end_time:
                idle_time = self.env.now - self.last_operation_end_time
                self.total_idle_time += idle_time
                print(f"Crane idle for {idle_time} minutes")
            yield self.env.timeout(operation_time)
            self.total_operations += 1
            self.last_operation_end_time = self.env.now

    def get_utilization_rate(self, simulation_time):
        return (self.total_operations * 4) / (simulation_time * self.num_cranes) if simulation_time > 0 else 0


# Define the TrainUnloading class
class TrainUnloading:
    def __init__(self, env, road_gates, storage_yard, crane_operation, T1, T2):
        self.env = env
        self.road_gates = road_gates  # Now it's a list of road gates
        self.storage_yard = storage_yard
        self.crane_operation = crane_operation
        self.T1 = T1
        self.T2 = T2
        self.direct_transshipments = 0

    def determine_path(self, itu):
        truck_times = [gate.get_truck_arrival_time(itu) for gate in self.road_gates]
        truck_time = min(time for time in truck_times if time is not None)
        if truck_time is None:
            print(f"ITU {itu.id}: No truck time, using storage")
            return "storage"
        
        train_time = itu.train_arrival_time
        time_difference = abs(truck_time - train_time)
        print(f"ITU {itu.id}: Truck time: {truck_time}, Train time: {train_time}, Time difference: {time_difference}")
        
        if time_difference <= self.T1:
            print(f"ITU {itu.id}: Direct transshipment (within T1 = {self.T1} minutes)")
            return "direct"
        elif time_difference <= self.T2:
            print(f"ITU {itu.id}: Using buffer (within T2 = {self.T2} minutes)")
            return "buffer"
        else:
            print(f"ITU {itu.id}: Using storage (time difference > T2)")
            return "storage"

    def unload_train(self, train):
        for itu in train.itus:
            if not itu.unloaded:
                path = self.determine_path(itu)
                yield self.env.process(self.execute_unloading(itu, path))
                itu.unloaded = True

    def execute_unloading(self, itu, path):
        if path == "direct":
            yield self.env.process(self.direct_transshipment(itu))
        elif path == "buffer":
            yield self.env.process(self.buffer_storage(itu))
        else:
            yield self.env.process(self.long_term_storage(itu))

    def direct_transshipment(self, itu):
        yield self.env.process(self.crane_operation.operate(4))  # Assume 4 minutes for direct transfer
        self.direct_transshipments += 1
        print(f"ITU {itu.id} directly transshipped at time {self.env.now:.2f}")

    def buffer_storage(self, itu):
        yield self.env.process(self.crane_operation.operate(4))
        self.storage_yard.store_itu(itu, 'buffer')
        print(f"ITU {itu.id} stored in buffer area at time {self.env.now:.2f}")

    def long_term_storage(self, itu):
        yield self.env.process(self.crane_operation.operate(4))
        self.storage_yard.store_itu(itu, 'long_term')
        print(f"ITU {itu.id} stored in long-term storage at time {self.env.now:.2f}")

class TrainLoading:
    def __init__(self, env, road_gates, storage_yard, crane_operation, T1, T2):
        self.env = env
        self.road_gates = road_gates  # Now it's a list of road gates
        self.storage_yard = storage_yard
        self.crane_operation = crane_operation
        self.T1 = T1
        self.T2 = T2
        self.direct_transshipments = 0

    def determine_path(self, itu):
        truck_times = [gate.get_truck_arrival_time(itu) for gate in self.road_gates]
        truck_time = min(time for time in truck_times if time is not None)
        if truck_time is None:
            print(f"ITU {itu.id}: No truck time, using storage")
            return "storage"
        
        train_time = itu.train_departure_time
        time_difference = abs(truck_time - train_time)
        print(f"ITU {itu.id}: Truck time: {truck_time}, Train time: {train_time}, Time difference: {time_difference}")
        
        if time_difference <= self.T1:
            print(f"ITU {itu.id}: Direct transshipment (within T1 = {self.T1} minutes)")
            return "direct"
        elif time_difference <= self.T2:
            print(f"ITU {itu.id}: Using buffer (within T2 = {self.T2} minutes)")
            return "buffer"
        else:
            print(f"ITU {itu.id}: Using storage (time difference > T2)")
            return "storage"

    def load_train(self, train):
        for itu in train.itus:
            if not itu.loaded:
                path = self.determine_path(itu)
                yield self.env.process(self.execute_loading(itu, path))
                itu.loaded = True

    def execute_loading(self, itu, path):
        if path == "direct":
            yield self.env.process(self.direct_transshipment(itu))
        elif path == "buffer":
            yield self.env.process(self.buffer_retrieval(itu))
        else:
            yield self.env.process(self.storage_retrieval(itu))

    def direct_transshipment(self, itu):
        yield self.env.process(self.crane_operation.operate(4))
        self.direct_transshipments += 1
        print(f"ITU {itu.id} directly loaded onto train at time {self.env.now:.2f}")

    def buffer_retrieval(self, itu):
        yield self.env.process(self.crane_operation.operate(4))
        self.storage_yard.retrieve_itu(itu, 'buffer')
        print(f"ITU {itu.id} retrieved from buffer area at time {self.env.now:.2f}")

    def storage_retrieval(self, itu):
        yield self.env.process(self.crane_operation.operate(4))
        self.storage_yard.retrieve_itu(itu, 'long_term')
        print(f"ITU {itu.id} retrieved from long-term storage at time {self.env.now:.2f}")

# Define the MainSimulation class
def load_data():
    # Load data from CSV files
    itu_df = pd.read_csv('ITU3.csv')
    train_df = pd.read_csv('train3.csv')
    truck_df = pd.read_csv('truck4.csv')
    print("Sample of loaded truck data:")
    print(truck_df.head())
    return itu_df, train_df, truck_df
class MainSimulation:
    T1 = 6 * 60  # minutes, time window for direct transshipment
    T2 = 24 * 60  # minutes, time window for buffer area vs long-term storage

    def __init__(self, env, itu_df, train_df, truck_df, num_road_gates=1, num_cranes=1):
        self.env = env
        self.storage_yard = StorageYard(env, long_term_capacity=1000, buffer_capacity=200, max_stack=2)
        self.crane_operation = CraneOperation(env, num_cranes=num_cranes)
        self.itu_data = itu_df
        self.truck_data = truck_df
        self.train_data = train_df
        self.num_road_gates = num_road_gates
        self.direct_transshipments = 0
        self.total_itus_handled = 0

        # Initialize road gates first
        self.road_gates = [RoadGate(env, i+1, truck_df) for i in range(num_road_gates)]

        # Now initialize train schedule
        self.trains = self.initialize_train_schedule(train_df)

        # Start the simulation process
        self.env.process(self.run())

    def hours_to_minutes(self, time):
        return int(float(time) * 60)

    def initialize_train_schedule(self, train_df):
        trains = []
        for day in range(7):  # Repeat for 7 days (one week)
            day_offset = day * 1440  # 1440 minutes per day
            for _, train in train_df.iterrows():
                train_id = train['Unnamed: 0']
                is_arrival = 'Arrivo' in train_id
                
                if is_arrival:
                    train_arrival_time = self.hours_to_minutes(train['Orario di arrivo']) + day_offset
                    train_departure_time = None
                else:
                    train_arrival_time = None
                    train_departure_time = self.hours_to_minutes(train['Orario di partenza']) + day_offset

                # Collect ITU IDs from relevant columns
                itu_ids = train.iloc[6:].dropna().astype(int).tolist()

                # Create ITU objects
                itus = [ITU(self.env, str(itu_id)) for itu_id in itu_ids]

                # Set train arrival and departure times for each ITU
                for itu in itus:
                    itu.train_arrival_time = train_arrival_time
                    itu.train_departure_time = train_departure_time
                    if str(itu.id) not in self.road_gates[0].truck_arrivals:
                        print(f"Warning: No truck arrival time found for ITU {itu.id}")

                train_obj = Train(self.env, train_id, itus, train_arrival_time, train_departure_time)
                trains.append(train_obj)
                self.env.process(self.unload_and_load_train(train_obj))
        return trains

    def unload_and_load_train(self, train):
        if train.arrival_time is not None:
            train_unloading = TrainUnloading(self.env, self.road_gates, self.storage_yard, self.crane_operation, self.T1, self.T2)
            yield self.env.process(train_unloading.unload_train(train))
        
        if train.departure_time is not None:
            yield self.env.timeout(30)  # Delay between unloading and loading
            train_loading = TrainLoading(self.env, self.road_gates, self.storage_yard, self.crane_operation, self.T1, self.T2)
            yield self.env.process(train_loading.load_train(train))
        
        self.total_itus_handled += len(train.itus)
        if train.arrival_time is not None:
            self.direct_transshipments += train_unloading.direct_transshipments
        if train.departure_time is not None:
            self.direct_transshipments += train_loading.direct_transshipments


    def run(self):
        for gate in self.road_gates:
            yield self.env.process(gate.run())

    def calculate_kpis(self, print_results=True):
        total_itus = self.total_itus_handled
        percentage_direct = (self.direct_transshipments / total_itus) * 100 if total_itus > 0 else 0

        kpis = {
            "Average Truck Waiting Time": sum(gate.average_truck_waiting_time() for gate in self.road_gates) / len(self.road_gates),
            "Storage Yard Utilization Rate": self.storage_yard.get_utilization_rate() * 100,
            "Total Crane Operations": self.crane_operation.total_operations,
            "Crane Utilization Rate": self.crane_operation.get_utilization_rate(self.env.now) * 100,
            "Throughput (Total ITUs handled)": self.total_itus_handled,
            "Percentage of Direct Transshipments": percentage_direct
        }

        if print_results:
            print("\n--- KPI Report ---")
            for key, value in kpis.items():
                print(f"{key}: {value:.2f}")
        return kpis
    

def main():
    road_gate_scenarios = [1, 2, 3]  # Number of road gates to simulate
    crane_scenarios = [1, 2, 3, 4]  # Number of cranes to simulate
    
    results = {}
    
    for num_gates in road_gate_scenarios:
        for num_cranes in crane_scenarios:
            print(f"\nRunning simulation with {num_gates} road gate(s) and {num_cranes} crane(s)")
            env = simpy.Environment()
            itu_df, train_df, truck_df = load_data()
            main_simulation = MainSimulation(env, itu_df, train_df, truck_df, 
                                             num_road_gates=num_gates, 
                                             num_cranes=num_cranes)
            env.run(until=10080)  # Run the simulation for one week (10,080 minutes)
            kpis = main_simulation.calculate_kpis()
            
            # Store results
            scenario_key = (num_gates, num_cranes)
            results[scenario_key] = kpis
    
    # Print or analyze results
    for (gates, cranes), kpis in results.items():
        print(f"\nResults for {gates} road gate(s) and {cranes} crane(s):")
        for kpi, value in kpis.items():
            print(f"{kpi}: {value}")

if __name__ == "__main__":
    main()
