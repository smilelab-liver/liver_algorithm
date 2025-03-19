import uuid

def generate_unique_id():
    return str(uuid.uuid4())

def get_bridge_template():
    return {
        "uid": generate_unique_id(),
        "category": None,
        "vein1": None,
        "vein2": None,
        "thickness": None,
        "length": None,
        "area": None,
    }

def get_fibrosis_template():
    return {
        "uid": generate_unique_id(),
        "name" : None,
        "category": None,
        "area": None,
        "has_bridge": False,
    }

def get_result_template():
    return {
        "fibrosis": [],
        "bridge": []
    }

def check_zone2_exist(result):
    for i, fibrosis in enumerate(result["fibrosis"]):
        if fibrosis["category"] == "zone2":
            return i  
    return -1
def check_fibrosis_exist(result, query_name):
    for i, fibrosis in enumerate(result["fibrosis"]):
        if fibrosis["name"] == query_name:
            return i
    return -1 
