def retrieve_indices(tensor, indices):
    '''
    :param tensor:
    :param indices:
    :return: elements of tensor matching the indices
    '''
    flattened_tensor = tensor.flatten(start_dim=2)
    flattened_indices = indices.flatten(start_dim=2)
    return flattened_tensor.gather(dim=2, index=flattened_indices).view_as(indices)