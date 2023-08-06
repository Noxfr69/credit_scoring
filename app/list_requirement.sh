#!/bin/bash

# Copy the base requirements to the combined file
cp requirements.txt combined_requirements.txt

# Loop through the directories containing requirements.txt
for dir in mlruns/1/*/artifacts/model; do
  if [ -f "$dir/requirements.txt" ]; then
    # Concatenate the requirements.txt file to the combined file
    cat "$dir/requirements.txt" >> combined_requirements.txt
  fi
done

# Run the Python script to deduplicate the requirements
python << END
from collections import defaultdict

# Dictionary to hold the latest version of each package
latest_versions = defaultdict(str)

# Read the combined file
with open('combined_requirements.txt', 'r') as file:
    for line in file:
        line = line.strip()
        # Split the package name from the version
        if '==' in line:
            package, version = line.split('==', 1)
            # Keep only the latest version of each package
            latest_versions[package] = version

# Write the deduplicated requirements back to a file
with open('combined_requirements.txt', 'w') as file:
    for package, version in latest_versions.items():
        file.write(f'{package}=={version}\n')
END

