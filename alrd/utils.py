from datetime import datetime
def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")