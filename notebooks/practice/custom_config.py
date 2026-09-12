import copy
class MyConfig:

    class ConfigProperty:

        def __init__(self, name):
            self.name = name

        def __set__(self, obj, value):
            raise AttributeError(f"Read only property '{self.name}' in {type(obj).__name__} object")

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            if self.name not in obj._configs:
                raise ValueError(f"Property '{self.name}' unavailable in {objtype.__name__} object")
            return obj._configs[self.name]

    configs = ["runoff", "area", "elevation"]
    for config in configs:
        probobj = ConfigProperty(config)
        locals()[config] = probobj
    



    def __init__(self, configs):
        """
        Initialize configs with deepcopy of configs dictionary
        """
        self._configs = copy.deepcopy(configs)


if __name__ == "__main__":

    testcases = [{"runoff":89, "area":8549.0, "elevation":5}, {"runoff":89, "elevation":5}, {"runoff":89, "area":8549.0, "elevation":5, "gage":83900} ]
    # for case in testcases:
    #     cfg = MyConfig(case)
    #     print(f"runoff = {cfg.runoff} area = {cfg.area} elev = {cfg.elevation}")
    cfg = MyConfig(testcases[-1])
    print(f"runoff = {cfg.runoff} area = {cfg.area} elev = {cfg.elevation}")

    cfg.area = 9