import os

def extract_rs_files(directory):
    with open("1.txt", "a", encoding="utf-8") as output_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") or file.endswith(".ipynb"):
                    file_path = os.path.join(root, file)
                    output_file.write(file_path + "\n")
                    output_file.write("```\n")
                    with open(file_path, "r", encoding="utf-8") as rs_file:
                        output_file.write(rs_file.read())
                    output_file.write("\n```\n")
def extra_file(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open("1.txt", "a", encoding="utf-8") as output_file:
        output_file.write(file + "\n")
        output_file.write("```\n")
        for line in lines:
            output_file.write(line)
        output_file.write("\n```\n")
    return

if os.path.exists("1.txt"):
    os.remove("1.txt")
extract_rs_files("Code")
# extract_rs_files("notebook")