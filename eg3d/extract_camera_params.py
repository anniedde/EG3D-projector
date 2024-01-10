import json
import numpy
import os
import dnnlib
import torch
import run_projector

for year in ['2007', '2011', '2015', '2017', '2023']:

    data_loc = f'/playpen-nas-ssd/awang/data/Taylor/Taylor_Swift_{year}/test/preprocessed/'
    dataset_json_loc = data_loc + 'dataset.json'

    with open(dataset_json_loc) as f:
        dataset_json = json.load(f)

    labels = dataset_json['labels']

    for entry in labels:
        img_file_name = entry[0]
        params = entry[1]
        out_loc = data_loc + img_file_name[:-3] + 'npy'

        np_array = numpy.array(params)
        numpy.save(out_loc, params)