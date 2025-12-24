from pathlib import Path
import json
data_root = Path(__file__).parent.parent / "data" / "raw"


all_trajectories = []
# now iterate all json files
for json_file in data_root.glob("*.json"):
    file_data = json.load(json_file.open())
    all_trajectories.extend(file_data)


print(f"Loaded a total of {len(all_trajectories)} trajectories from all JSON files.")

trajectory_lengths = [len(traj[0]) for traj in all_trajectories]
print(f"Trajectory lengths range from {min(trajectory_lengths)} to {max(trajectory_lengths)} with mean {sum(trajectory_lengths) / len(trajectory_lengths):.2f}.")    

# remove all trajectories with len < 300
filtered_trajectories = [traj for traj in all_trajectories if len(traj[0]) >= 300]
print(f"Filtered out {len(all_trajectories) - len(filtered_trajectories)} trajectories with length less than 300.")
print(f"Remaining trajectories: {len(filtered_trajectories)}")
trajectory_lengths = [len(traj[0]) for traj in filtered_trajectories]
print(f"Filtered trajectory lengths range from {min(trajectory_lengths)} to {max(trajectory_lengths)} with mean {sum(trajectory_lengths) / len(trajectory_lengths):.2f}.")    

# store at data/clean
clean_data_root = data_root.parent / "clean"
clean_data_root.mkdir(exist_ok=True)

clean_file = clean_data_root / "filtered_trajectories.json"
with clean_file.open("w") as f:
    json.dump(filtered_trajectories, f)