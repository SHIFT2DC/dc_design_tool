import numpy as np

# Define the network
buses = [
    {'id': 1, 'type': 'slack', 'V': 1.0},
    {'id': 2, 'type': 'load', 'P': 0.5, 'V': 1.0},
    {'id': 3, 'type': 'converter', 'reference_bus': 2, 'ratio': 2.0, 'V': 1.0}
]

lines = [
    {'from': 1, 'to': 2, 'R': 0.1},
    {'from': 2, 'to': 3, 'R': 0.15}
]

# Identify unknown buses
unknown_buses = [bus for bus in buses if bus['type'] == 'load']
unknown_ids = [bus['id'] for bus in unknown_buses]
num_unknowns = len(unknown_buses)

# Map bus id to index
bus_index = {bus['id']: i for i, bus in enumerate(unknown_buses)}

# Function to get voltage expression
def get_voltage_expr(bus_id, buses, bus_voltages):
    bus = next(b for b in buses if b['id'] == bus_id)
    if bus['type'] == 'slack':
        return bus['V']
    elif bus['type'] == 'load':
        return bus_voltages[bus_index[bus['id']]]
    elif bus['type'] == 'converter':
        ref_bus = next(b for b in buses if b['id'] == bus['reference_bus'])
        return bus['ratio'] * get_voltage_expr(ref_bus['id'], buses, bus_voltages)

# Function to get neighbors of a bus
def get_neighbors(bus_id, lines):
    neighbors = []
    for line in lines:
        if line['from'] == bus_id:
            neighbors.append(line['to'])
        elif line['to'] == bus_id:
            neighbors.append(line['from'])
    return neighbors

# Newton-Raphson iteration
tolerance = 1e-6
max_iterations = 100
V_initial = {bus['id']: bus['V'] for bus in unknown_buses}  # Initial guess

for iteration in range(max_iterations):
    # Build power mismatch vector
    F = np.zeros(num_unknowns)
    for i, bus in enumerate(unknown_buses):
        mismatch = 0
        for line in lines:
            if line['from'] == bus['id']:
                to_bus = next(b for b in buses if b['id'] == line['to'])
                V_to = get_voltage_expr(to_bus['id'], buses, V_initial)
                P = (bus['V'] - V_to) / line['R']
                mismatch += P
            elif line['to'] == bus['id']:
                from_bus = next(b for b in buses if b['id'] == line['from'])
                V_from = get_voltage_expr(from_bus['id'], buses, V_initial)
                P = (V_from - bus['V']) / line['R']
                mismatch += P
        if bus['type'] == 'load':
            mismatch -= bus['P']
        F[i] = mismatch

    # Build Jacobian matrix
    J = np.zeros((num_unknowns, num_unknowns))
    for i, bus in enumerate(unknown_buses):
        for neighbor in get_neighbors(bus['id'], lines):
            if neighbor in unknown_ids:
                j = bus_index[neighbor]
                line = next(l for l in lines if l['from'] == bus['id'] and l['to'] == neighbor or l['from'] == neighbor and l['to'] == bus['id'])
                J[i, j] += -1 / line['R']
        J[i, i] = -sum(1 / line['R'] for line in lines if line['from'] == bus['id'] or line['to'] == bus['id'])

    # Solve for voltage corrections
    delta_V = np.linalg.solve(J, -F)

    # Update voltages
    for i, bus in enumerate(unknown_buses):
        V_initial[bus['id']] += delta_V[i]

    # Check for convergence
    if np.max(np.abs(delta_V)) < tolerance:
        print("Converged in", iteration, "iterations.")
        break
else:
    print("Did not converge in", max_iterations, "iterations.")

# Assign solved voltages to unknown buses
for bus in unknown_buses:
    bus['V'] = V_initial[bus['id']]

# Calculate final voltages for converter buses
for bus in buses:
    if bus['type'] == 'converter':
        ref_bus = next(b for b in buses if b['id'] == bus['reference_bus'])
        bus['V'] = bus['ratio'] * ref_bus['V']

# Calculate power flows in lines
power_flows = {}
for line in lines:
    from_bus = next(b for b in buses if b['id'] == line['from'])
    to_bus = next(b for b in buses if b['id'] == line['to'])
    P = (from_bus['V'] - to_bus['V']) / line['R']
    power_flows[(line['from'], line['to'])] = P

# Output results
print("Bus Voltages:")
for bus in buses:
    print(f"Bus {bus['id']}: V = {bus['V']:.4f} p.u.")

print("\nLine Powers:")
for line, P in power_flows.items():
    print(f"Line {line[0]}-{line[1]}: P = {P:.4f} p.u.")