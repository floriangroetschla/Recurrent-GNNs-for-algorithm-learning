import os
import json
import pandas as pd
import sys

output_file = open(sys.argv[2] + '.json', 'w')
file_with_error = 0
files_processed = 0
output = []

for filename in os.listdir(sys.argv[1]):
   with open(os.path.join(sys.argv[1], filename), 'r') as f:
       lines = f.readlines()
       if len(lines) == 0:
           print('Encountered empty file: ' + filename)
       elif lines[-1] != 'Finished\n':
           print('Error in file ' + filename)
           file_with_error += 1
       else:
           config = json.loads(lines[0].replace("AttrDict(", "").replace(")", "").replace("'", '"').replace("True", "true").replace("False", "false"))
           for line in lines[1:-1]:
               if not line.startswith("Path"):
                data = json.loads(line)
                output.append({**config, **data})
       files_processed += 1
output_file.write(json.dumps(output))
with open(sys.argv[2] + '.json') as input_file:
    df = pd.read_json(input_file)
    df.to_csv(sys.argv[2] + '.csv', encoding='utf-8', index=False)
print('Processed ' + str(files_processed) + ' files')
print('Error percentage: ' + str(float(file_with_error) / files_processed))

