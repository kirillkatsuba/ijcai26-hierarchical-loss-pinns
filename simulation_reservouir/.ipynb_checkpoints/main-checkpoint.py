from oil_reservoir_synthesizer import OilSimulator

simulator = OilSimulator()

# Build a model with one well and block
simulator.addWell("wellName", seed=997)
simulator.addBlock("5,5,5", seed=31)

# Run simulation
num_steps = 10
fopr_values = []  # oil production rate for each time step
for time_steps in range(num_steps):
    simulator.step(scale=1.0 / num_steps)
    fopr_values.append(simulator.fopr())