import matplotlib.pyplot as plt

def plot_voltage(net):
    """
    Plots the voltage profiles of the buses in the given pandapower network.

    Parameters:
    net (pandapowerNet): The pandapower network object.
    """
    # Print a message indicating the start of the plotting process
    print(">Plot voltage profiles")
    
    # Adjust bus indices to start from 1 for better readability
    bus_indices = net.res_bus.index + 1
    
    # Extract the voltage magnitudes in per unit (pu) from the network results
    bus_voltages = net.res_bus.vm_pu
    
    # Enable interactive mode for plotting
    plt.ion()
    
    # Create a new figure with specified size
    plt.figure(figsize=(10, 6))
    
    # Plot the voltage values with markers and lines
    plt.plot(range(len(bus_voltages.values)), bus_voltages.values, marker='o', linestyle='-', color='b')
    
    # Set the title and labels for the plot
    plt.title("Voltage Profiles")
    plt.xlabel("Bus Index")
    plt.ylabel("Voltage (pu)")
    
    # Enable grid for better visualization
    plt.grid(True)
    
    # Calculate the spread of voltage values to set y-axis limits dynamically
    spread = max(bus_voltages) - min(bus_voltages)
    plt.ylim(min(bus_voltages) - 0.1 * spread, max(bus_voltages) + 0.1 * spread)
    
    # Set x-axis ticks and labels to match bus indices
    plt.xticks(ticks=range(len(bus_voltages.values)), labels=bus_indices)
    
    # Display the plot
    plt.show()
