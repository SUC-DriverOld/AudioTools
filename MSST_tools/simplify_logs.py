# Define the patterns to search for
patterns = [
    "Training loss:",
    "Instr SDR aspiration:",
    "Instr SDR other:",
    "SDR Avg:",
    "Train epoch:"
]

# Open the log file and output file
input_file = "nohup.out"
output_file = "log.txt"

# Read the file and filter lines
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if any(pattern in line for pattern in patterns):
            outfile.write(line)

print(f"Filtered log saved to {output_file}")
