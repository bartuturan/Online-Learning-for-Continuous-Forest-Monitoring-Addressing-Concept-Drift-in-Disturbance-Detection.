

def select_for_skipconnect(enc_value, time_index=-1):
    #TODO: average or weighted sum over time?
    return enc_value[:, time_index] 


def nfilter_layer(start_filters, n_layer):
    return min(512, start_filters * (2 ** n_layer))
