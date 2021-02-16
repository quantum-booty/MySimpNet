from typing import NamedTuple


class Slim(NamedTuple):
    # 300k parameters
    dropout: bool = False

    EPSILON: float = 1e-6
    BN_MOMENTUM: float = 0.99

    INPUT_SHAPE: tuple = (28, 28, 1)

    CONV1_NUM_FILTERS: int = 30
    CONV1_FILTER_SIZE: int = 3

    CONV2_NUM_FILTERS: int = 40
    CONV2_FILTER_SIZE: int = 3

    CONV3_NUM_FILTERS: int = 40
    CONV3_FILTER_SIZE: int = 3

    CONV4_NUM_FILTERS: int = 40
    CONV4_FILTER_SIZE: int = 3

    CONV5_NUM_FILTERS: int = 50
    CONV5_FILTER_SIZE: int = 3

    MAXPOOL1_SIZE: int = 2

    CONV6_NUM_FILTERS: int = 50
    CONV6_FILTER_SIZE: int = 3

    CONV7_NUM_FILTERS: int = 50
    CONV7_FILTER_SIZE: int = 3

    CONV8_NUM_FILTERS: int = 50
    CONV8_FILTER_SIZE: int = 3

    CONV9_NUM_FILTERS: int = 50
    CONV9_FILTER_SIZE: int = 3

    CONV10_NUM_FILTERS: int = 58
    CONV10_FILTER_SIZE: int = 3

    MAXPOOL2_SIZE: int = 2

    CONV11_NUM_FILTERS: int = 58
    CONV11_FILTER_SIZE: int = 3

    CONV12_NUM_FILTERS: int = 70
    CONV12_FILTER_SIZE: int = 3

    CONV13_NUM_FILTERS: int = 90
    CONV13_FILTER_SIZE: int = 3

    OUTPUT_SIZE: int = 10


class Full(NamedTuple):
    # 5 million parameters
    dropout: bool = True

    EPSILON: float = 1e-6
    BN_MOMENTUM: float = 0.95

    INPUT_SHAPE: tuple = (28, 28, 1)

    CONV1_NUM_FILTERS: int = 66
    CONV1_FILTER_SIZE: int = 3

    CONV2_NUM_FILTERS: int = 64
    CONV2_FILTER_SIZE: int = 3

    CONV3_NUM_FILTERS: int = 64
    CONV3_FILTER_SIZE: int = 3

    CONV4_NUM_FILTERS: int = 64
    CONV4_FILTER_SIZE: int = 3

    CONV5_NUM_FILTERS: int = 96
    CONV5_FILTER_SIZE: int = 3

    MAXPOOL1_SIZE: int = 2

    CONV6_NUM_FILTERS: int = 96
    CONV6_FILTER_SIZE: int = 3

    CONV7_NUM_FILTERS: int = 96
    CONV7_FILTER_SIZE: int = 3

    CONV8_NUM_FILTERS: int = 96
    CONV8_FILTER_SIZE: int = 3

    CONV9_NUM_FILTERS: int = 96
    CONV9_FILTER_SIZE: int = 3

    CONV10_NUM_FILTERS: int = 144
    CONV10_FILTER_SIZE: int = 3

    MAXPOOL2_SIZE: int = 2

    CONV11_NUM_FILTERS: int = 144
    CONV11_FILTER_SIZE: int = 3

    CONV12_NUM_FILTERS: int = 178
    CONV12_FILTER_SIZE: int = 3

    CONV13_NUM_FILTERS: int = 216
    CONV13_FILTER_SIZE: int = 3

    OUTPUT_SIZE: int = 10
