'''This help library defines celestial class and associted method(s) that will store and process information of
celsestial objects'''

import numpy as np

class Celestial:
    """Generic conteiner to store object information"""

    def __init__(self, name, data, object_type=None, post_processing_data={}):
        """
        Create instance of class
        :param name: <str>
            name of celestial object obtained from filename
        :param data <nested dict>
            keys: represent value type (V2, CP, uv, etc) and values in inner dictionary with
            keys: equal to measurements type and values are actual measurements for the parameter

        :param object_type: <str>
            type of celestial object obtained from filename; default - None
        :param post_processing_data <nested dict>
            keys: represent value type (V2, CP, uv, etc) and values in inner dictionary with
            keys: equal to measurements type and values are actual measurements for the parameter after data processing
        """
        self.name = name
        self.data = data
        self.object_type = object_type
        self.post_processing_data = post_processing_data

    def update_data(self, new_data):
        """"Combines measurements taken for the same object in different days to single dataset
        :param: new_data <nested dict>
            data that needs to be appended to data already associated with object
            keys: represent value type (V2, CP, uv, etc) and values in inner dictionary with
            keys: equal to measurements type and values are actual measurements for the parameter
        :returns None
        """
        for outer_key, inner_dict in new_data.items():
            for inner_key, inner_value in inner_dict.items():
                self.data[outer_key][inner_key] = np.append(self.data[outer_key][inner_key],
                                                            new_data[outer_key][inner_key])
        return None
