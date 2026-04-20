import subprocess
import os, sys

# List of problems
problems = [
    'burgers_pde',
    'darcy_pde_2d',
    'elliptic_pde',
    'seismic_res5',
    'seismic_res7',
    'seismic_res10',
    'seismic_res14',
    'inverse_scattering', # warning: larger dataset
    'Calderon' # warning: larger dataset
]

for pb in problems:
    print(f"\n=== Running benchmark for {pb} ===", flush=True)
    
    # Build the command with unbuffered flag
    cmd = [sys.executable, "-u", "train_pb.py", pb]
    
    # Ensure child also runs unbuffered
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    
    # Run the command and wait for it to finish
    subprocess.run(cmd, check=True, env=env)