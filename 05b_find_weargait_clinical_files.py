r"""
05b_find_weargait_clinical_files.py

Search the main WearGait-PD Synapse repository for clinical/demographic
files WITHOUT recursively downloading the entire project.

Requires:
    synapseclient

Uses your existing local Synapse login/profile.

Main WearGait-PD project:
    syn52540892

Outputs a text list of likely clinical/demographic files and their Synapse IDs.
"""

from pathlib import Path
import synapseclient

ROOT_SYN = "syn52540892"

KEYWORDS = (
    "clinical",
    "demographic",
    "updrs",
    "medication",
    "participant",
    "subject info",
    "hoehn",
    "yahr",
)

def walk_children(syn, parent_id, breadcrumb=""):
    for child in syn.getChildren(parent_id):
        child_id = child["id"]
        name = child.get("name", "")
        child_type = child.get("type", "")
        here = f"{breadcrumb}/{name}" if breadcrumb else name

        yield {
            "id": child_id,
            "name": name,
            "type": child_type,
            "path": here,
        }

        # Recurse into folders/projects only.
        if child_type in {"org.sagebionetworks.repo.model.Folder",
                          "org.sagebionetworks.repo.model.Project"}:
            yield from walk_children(syn, child_id, here)

def main():
    syn = synapseclient.Synapse()
    syn.login(silent=True)

    print(f"Searching WearGait-PD repository {ROOT_SYN}...\n")

    hits = []

    for item in walk_children(syn, ROOT_SYN):
        text = f"{item['name']} {item['path']}".lower()

        if any(keyword in text for keyword in KEYWORDS):
            hits.append(item)

    if not hits:
        print("No keyword matches were found.")
        print("If your account can view the repository, open the Files tab and")
        print("look for folders/files containing 'Demographic' or 'Clinical'.")
        return

    print("Likely clinical/demographic items:\n")
    for item in hits:
        print(f"{item['id']} | {item['type']}")
        print(f"  {item['path']}")
        print()

    print("For any CSV file you want to download, use:")
    print('synapse get <SYNAPSE_ID> --downloadLocation "$HOME\\Desktop\\WearGait_PD_Longitudinal\\Clinical_Metadata"')

if __name__ == "__main__":
    main()
