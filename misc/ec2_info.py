from rllab import config

all_subnet_info = {
    'ex2.data': {
        "us-west-1a": dict(
            SubnetID="subnet-b0b06ae8", Groups=["sg-53b76234"]),
        "us-west-1b": dict(
            SubnetID="subnet-0b21b46f", Groups=["sg-53b76234"]),
        "us-west-1c": dict(
            SubnetID="subnet-7f904927", Groups=["sg-01b87766"]),
        "us-west-2a": dict(
            SubnetID="subnet-a8a48cdf", Groups=["sg-d9559ca1"]),
        "us-west-2c": dict(
            SubnetID="subnet-6217573b", Groups=["sg-d9559ca1"]),
        "us-west-2b": dict(
            SubnetID="subnet-57253232", Groups=["sg-d9559ca1"]),
    },
}
subnet_info = all_subnet_info[config.BUCKET]
