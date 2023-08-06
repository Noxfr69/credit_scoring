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
