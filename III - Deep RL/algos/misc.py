def format_time(seconds):
    seconds = int(seconds)

    if seconds == -1:
        return "___"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02}h{minutes:02}min"