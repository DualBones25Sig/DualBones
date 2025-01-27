import torch.utils.data
from torch.nn.utils.rnn import pad_sequence


def from_data_list(batch):
    data_num = len(batch[0])
    d = []

    for i in range(data_num):
            field_data = [b[i] for b in batch]
            if isinstance(field_data[0], torch.Tensor) and any(len(x) != len(field_data[0]) for x in field_data):
                padded_field = pad_sequence(field_data, batch_first=True, padding_value=0)
                d.append(padded_field)
            else:
                d.append(torch.stack(field_data))

    return tuple(d)



class ModelDatasetLoader(torch.utils.data.DataLoader):

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=None,
                 **kwargs):
        if follow_batch is None:
            follow_batch = []

        def collate(batch):
            elem = batch[0]
            if isinstance(elem, tuple):
                return from_data_list(batch)
            raise TypeError('DataLoader found invalid type: {}'.format(
                type(elem)))

        super(ModelDatasetLoader, self).__init__(dataset,
                                                batch_size,
                                                shuffle,
                                                collate_fn=lambda batch: collate(
                                                    batch),
                                                **kwargs)
