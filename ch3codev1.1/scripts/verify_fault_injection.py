
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add path to find sim module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sim.eci_full import simulate_ecifull

def run_and_plot():
    outdir = os.path.join(os.path.dirname(__file__), "verification_plots")
    os.makedirs(outdir, exist_ok=True)
    
    # Define Scenarios (matching make_dataset_cache.py)
    scenarios = [
        ("Nominal",            {}),
        ("Thrust Drop 15%",    {"thrust_drop": 0.15}), 
        ("TVC Rate Limit",     {"tvc_rate_lim_deg_s": 0.3}),
        ("TVC Stuck",          {"tvc_stick": (30.0, 20.0)}), # t=30, dt=20
        ("Sensor Bias",        {"sensor_bias": (1.5, 0.0, 0.0)}),
        ("Event Delay",        {"event_delay": {"S3_ign": 5.0}}),
    ]
    
    results = {}
    
    print("Running Simulations...")
    for name, params in scenarios:
        print(f"  - {name}...")
        # Use fixed seed for reproducibility
        # Increased t_end to 320.0 to cover S3 ignition
        # Reduced noise_std to 0.1 to make sensor bias more observable
        sim_res = simulate_ecifull(t_end=320.0, dt=0.05, seed=42, noise_std=0.1, **params)
        
        # Calculate Altitude from r (if not provided, but we assume h is consistent with r)
        # Actually simulate_ecifull wrapper returns what core returns.
        # Core returns 'h' in hist? Let's check.
        # Core returns 'h' in the dict. But wrapper might not have passed it?
        # Wrapper passes: t, r, v, a_true, a_meas, mass, thrust, q_dyn, n_load, pitch, fpa
        # It does NOT pass 'h' explicitly in my last edit. 
        # But we can compute h from r.
        r = sim_res["r"]
        h = np.linalg.norm(r, axis=1) - 6378137.0
        sim_res["h"] = h
        
        # Calculate Velocity Norm
        v = sim_res["v"]
        vel = np.linalg.norm(v, axis=1)
        sim_res["vel"] = vel
        
        results[name] = sim_res

    print("Generating Plots...")
    
    # List of variables to plot
    # (Key in sim_res, Label, Y-Label)
    plot_vars = [
        ("h", "Altitude", "Altitude (m)"),
        ("vel", "Velocity", "Velocity (m/s)"),
        ("q_dyn", "Dynamic Pressure", "Q (Pa)"),
        ("n_load", "Normal Load", "Load (g)"),
        ("mass", "Mass", "Mass (kg)"),
        ("thrust", "Thrust", "Thrust (N)"),
        ("pitch_cmd_deg", "Pitch Command", "Pitch (deg)"),
        ("fpa_deg", "Flight Path Angle", "Gamma (deg)"),
    ]
    
    # 1. Standard Variables
    for key, title, ylabel in plot_vars:
        plt.figure(figsize=(10, 6))
        for name, res in results.items():
            # Handle missing keys gracefully if any
            if key in res:
                # Downsample for plotting speed/clarity if needed, but 0.05s is fine
                plt.plot(res["t"], res[key], label=name, linewidth=1.5 if name=="Nominal" else 1.0)
        
        plt.title(f"{title} Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        safe_fname = title.lower().replace(" ", "_")
        plt.savefig(os.path.join(outdir, f"compare_{safe_fname}.png"))
        plt.close()
        
    # 2. Measured Acceleration (X and Z)
    # X Axis
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res["t"], res["a_meas"][:, 0], label=name, linewidth=1.5 if name=="Nominal" else 1.0)
    plt.title("Measured Acceleration X Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("a_meas_x (m/s^2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outdir, "compare_a_meas_x.png"))
    plt.close()
    
    # Z Axis
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res["t"], res["a_meas"][:, 2], label=name, linewidth=1.5 if name=="Nominal" else 1.0)
    plt.title("Measured Acceleration Z Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("a_meas_z (m/s^2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outdir, "compare_a_meas_z.png"))
    plt.close()

    print(f"All plots saved to: {outdir}")

if __name__ == "__main__":
    run_and_plot()
