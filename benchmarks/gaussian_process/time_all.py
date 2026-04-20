import subprocess

valid_names = [
    'burgers_pde', 'darcy_pde_2d', 'elliptic_pde',
    'seismic_res5', 'seismic_res7', 'seismic_res10', 'seismic_res14',
    "inverse_scattering", "Calderon"
]

for name in valid_names:
    cmd = ["python", "-u", "prediction.py", name, "--time"]
    print(f"\nRunning: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)