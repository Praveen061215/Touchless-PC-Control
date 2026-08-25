import os
import time

repo_dir = r"c:\Users\prave\Visual Studio Code\Touchless PC Control"
os.chdir(repo_dir)

def make_commit(msg):
    os.system("git add .")
    os.system(f'git commit -m "{msg}"')
    time.sleep(1)

# Commit 1
with open("README.md", "a", encoding="utf-8") as f:
    f.write("\n\n## Contact\nCreated by Praveen. Feel free to reach out for any questions.")
make_commit("docs: add contact information to README")

# Commit 2
with open("requirements.txt", "r") as f:
    content = f.read()
with open("requirements.txt", "w") as f:
    f.write("# Project Dependencies\n" + content)
make_commit("chore: add header to requirements.txt")

# Commit 3
with open(".gitattributes", "w", encoding="utf-8") as f:
    f.write("*.task filter=lfs diff=lfs merge=lfs -text\n")
make_commit("chore: add .gitattributes for LFS tracking")

# Commit 4
with open("VirtualMouse.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("import cv2", "import cv2\nimport logging")
with open("VirtualMouse.py", "w", encoding="utf-8") as f:
    f.write(content)
make_commit("refactor: import logging module in VirtualMouse")

# Commit 5
with open("VirtualMouse.py", "a", encoding="utf-8") as f:
    f.write("\n# End of script\n")
make_commit("style: add EOF comment to VirtualMouse")

# Commit 6
with open("HandTrackingModule.py", "a", encoding="utf-8") as f:
    f.write("\n# Module version 1.0.0\n")
make_commit("docs: add version tag to HandTrackingModule")

# Commit 7
with open("CHANGELOG.md", "w", encoding="utf-8") as f:
    f.write("# Changelog\n\n## [1.0.0] - Initial Release\n- Added virtual mouse functionality using MediaPipe and PyAutoGUI\n- Supported 13 unique hand gestures\n")
make_commit("docs: create initial CHANGELOG.md")

# Commit 8
with open("CONTRIBUTING.md", "w", encoding="utf-8") as f:
    f.write("# Contributing\nPull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.\n")
make_commit("docs: add CONTRIBUTING.md guidelines")

# Commit 9
with open("VirtualMouse.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("import logging", "import logging  # Built-in python logging")
with open("VirtualMouse.py", "w", encoding="utf-8") as f:
    f.write(content)
make_commit("style: add inline comment for logging import")

# Commit 10
with open("README.md", "a", encoding="utf-8") as f:
    f.write("\n\n## License\nThis project is licensed under the MIT License.")
make_commit("docs: add license section to README")

print("Pushing to remote...")
os.system("git push")
print("Done!")
