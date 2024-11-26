import salabim as sim
import pandas as pd

# Load the data from the Excel file
file_path_correct = r'C:\Users\RayanegostaR\Desktop\New folder (8)\truck2.xlsx'
truck_data_excel = pd.read_excel(file_path_correct)

# Extract the relevant data from the Excel file
truck_data_excel['arrival_time_minutes'] = truck_data_excel['Orario di arrivo'].apply(lambda x: int(x) * 60 + int((x - int(x)) * 100))

# Map the truck purposes from the Excel file
purpose_mapping = {0: 'pickup', 1: 'delivery', 2: 'both'}
truck_data_excel['purpose'] = truck_data_excel['ritiro(0)/consegna(1)/entrambi(2)'].map(purpose_mapping)

# Limit to the first 100 trucks
truck_data_excel = truck_data_excel.head(100)

# Simulation environment setup
sim.yieldless(False)
env = sim.Environment(trace=False)
env.animate(True)  # Turn on animation
env.animate_debug(False)  # Turn off debug mode for animation

# Constants
CHECKIN_TIME = 4  # Fixed check-in time in minutes as per the document
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CENTER_Y = SCREEN_HEIGHT // 2
GATE_X = SCREEN_WIDTH - 50
GATE_Y = CENTER_Y  # Set the gate's Y position
TRUCK_HEIGHT = 20
TRUCK_WIDTH = 40  # Width of the truck
TRUCK_SPEED = 2  # Speed at which trucks move (pixels per simulation step)
GATE_WIDTH = 20  # Width of the gate
GATE_HEIGHT = 250  # Height of the gate

# RoadGate Component
class RoadGate(sim.Component):
    def __init__(self, truck_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.truck_data = truck_data
        self.checkin_server = sim.Resource(capacity=1, name="Check-in Server")
        
        # Animation for the road gate
        self.gate_animation = sim.AnimateRectangle(
            spec=(0, 0, GATE_WIDTH, GATE_HEIGHT),
            fillcolor='gray',
            x=GATE_X,
            y=GATE_Y - GATE_HEIGHT // 2  # Center the gate vertically
        )
        
        # Label for the road gate
        self.gate_label = sim.AnimateText(
            text='Road Gate',
            x=GATE_X + GATE_WIDTH + 10,
            y=GATE_Y - GATE_HEIGHT // 2 - 20,
            textcolor='black',
            fontsize=15
        )
        
        # Legend for truck colors
        self.legend_delivery = sim.AnimateText(
            text='Delivery: Blue',
            x=10,
            y=SCREEN_HEIGHT - 60,
            textcolor='blue',
            fontsize=12
        )
        
        self.legend_pickup = sim.AnimateText(
            text='Pickup: Green',
            x=10,
            y=SCREEN_HEIGHT - 40,
            textcolor='green',
            fontsize=12
        )
        
        self.legend_both = sim.AnimateText(
            text='Both: Orange',
            x=10,
            y=SCREEN_HEIGHT - 20,
            textcolor='orange',
            fontsize=12
        )

    def process(self):
        for i, row in self.truck_data.iterrows():
            print(f"Creating Truck {i + 1} at time {row['arrival_time_minutes']} minutes")
            Truck(self.checkin_server, CHECKIN_TIME, row['arrival_time_minutes'], row['purpose'], row['ITU corrispondente'])
            yield self.hold(1)  # Yield for 1 minute between truck generations to prevent overlapping

# Truck Component
class Truck(sim.Component):
    def __init__(self, checkin_server, checkin_time, arrival_time, purpose, itu_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkin_server = checkin_server
        self.checkin_time = checkin_time
        self.arrival_time = arrival_time
        self.purpose = purpose
        self.itu_code = itu_code
        self.x = 0  # Starting position on the x-axis
        self.y = CENTER_Y - TRUCK_HEIGHT // 2  # Vertical center position
        
        # Truck animation
        self.truck_animation = sim.AnimateRectangle(
            spec=(0, 0, TRUCK_WIDTH, TRUCK_HEIGHT),
            fillcolor='blue' if self.purpose == 'delivery' else 'green' if self.purpose == 'pickup' else 'orange',
            x=lambda: self.x,  # Lambda function to update x position
            y=self.y  # Static y position
        )
        
        # Label for the ITU code on the truck
        self.itu_label = sim.AnimateText(
            text=lambda: str(self.itu_code),
            x=lambda: self.x + 5,  # Position the label inside the truck
            y=lambda: self.y + 5,  # Center vertically within the truck
            textcolor='white',
            fontsize=10
        )
        
    def process(self):
        yield self.hold(self.arrival_time)  # Wait until the truck's arrival time
        
        # Move the truck towards the gate
        while self.x < GATE_X - TRUCK_WIDTH:
            self.x += TRUCK_SPEED  # Increment position based on a fixed speed
            yield self.hold(0.1)  # Hold for a short time to create a smooth movement
        
        yield self.request(self.checkin_server)  # Request the check-in server
        yield self.hold(self.checkin_time)  # Simulate the check-in process
        self.release()  # Release the check-in server

# Instantiate the RoadGate with truck data
road_gate = RoadGate(truck_data_excel)

# Run the simulation for one day (1440 minutes)
env.run(1440)
