import pandapower.networks as pn
import pandapower as pp
import numpy as np
import json

# Configuration
NUM_SEEDS = 10
NUM_SCENARIOS = 360
LOAD_SCALE_MIN = 0.5
LOAD_SCALE_MAX = 1.5

# Load network
net = pn.ieee_european_lv_asymmetric()

# Save scenario data
def generate_dataset():
    data = []
    for seed in range(NUM_SEEDS):
        np.random.seed(seed)
        for i in range(NUM_SCENARIOS):
            # Generate random load scaling (apply to each load)
            scaling = np.random.uniform(LOAD_SCALE_MIN, LOAD_SCALE_MAX, len(net.asymmetric_load))
            net.asymmetric_load.scaling = scaling
            
            try:
                # Run three-phase power flow
                pp.runpp_3ph(net, verbose=False)
                
                # Check for convergence
                if not net.converged:
                    continue
                
                # Extract results (use correct column names)
                scenario = {
                    "seed": seed,
                    "scenario_id": i,
                    "scaling": scaling.tolist(),
                    "vm_a": net.res_bus_3ph.vm_a_pu.tolist(),
                    "vm_b": net.res_bus_3ph.vm_b_pu.tolist(),
                    "vm_c": net.res_bus_3ph.vm_c_pu.tolist(),
                }
                data.append(scenario)
            except Exception as e:
                continue
    
    # Save to JSON
    with open("dataset.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    generate_dataset()
