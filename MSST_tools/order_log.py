import re

def extract_info(log_file, output_file):
    patterns = [
        r"Train epoch: (\d+) Learning rate: ([\d.eE+-]+)",
        r"Training loss: ([\d.]+)",
        r"Instr dry sdr: ([\d.]+) \(Std: ([\d.]+)\)",
        r"Instr dry l1_freq: ([\d.]+) \(Std: ([\d.]+)\)",
        r"Instr dry si_sdr: ([\d.]+) \(Std: ([\d.]+)\)",
        r"Instr other sdr: ([\d.]+) \(Std: ([\d.]+)\)",
        r"Instr other l1_freq: ([\d.]+) \(Std: ([\d.]+)\)",
        r"Instr other si_sdr: ([\d.]+) \(Std: ([\d.]+)\)",
        r"Metric avg sdr\s+: ([\d.]+)",
        r"Metric avg l1_freq\s+: ([\d.]+)",
        r"Metric avg si_sdr\s+: ([\d.]+)"
    ]

    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = f.read()

    extracted_paragraphs = []
    current_paragraph = []
    for line in log_data.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                current_paragraph.append(line)
                break

    if current_paragraph:
        extracted_paragraphs.append("\n".join(current_paragraph))

    with open(output_file, 'a') as out:
        for paragraph in extracted_paragraphs:
            out.write(paragraph + "\n")

log_file = 'train.log'
output_file = 'extracted_info.txt'
extract_info(log_file, output_file)
