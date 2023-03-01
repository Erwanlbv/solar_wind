import datetime


def get_model_save_name():
    """
    Return a save name for a file, depending on the instant the function
    is called
    """

    now = datetime.datetime.now()
    time_l = [now.month, now.day, now.hour, now.minute]
    time_l = [str(el) for el in time_l]
    save_date_name = '-'.join(time_l[:2]) + '_' + ':'.join(time_l[2:])
    save_date_name = '0' + save_date_name

    return save_date_name
