import numpy as np
import  datetime

def get_model_save_name():
    """
    Return a save name for a file, depending on the instant the function is called
    """
    
    now = datetime.datetime.now()
    l = [now.month, now.day, now.hour, now.minute]
    l = [str(el) for el in l]
    save_date_name = '-'.join(l[:2]) + '_' + ':'.join(l[2:])
    save_date_name = '0' + save_date_name
    
    return save_date_name