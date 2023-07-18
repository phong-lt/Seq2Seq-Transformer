class Registry(dict):
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        if module is not None:
            self._register_generic(module_name, module)
            return

        def register_fn(fn):
            self._register_generic(module_name, fn)
            return fn
            
        return register_fn
    
    def _register_generic(self, module_name, module):
        assert module_name not in self, "Module {} is already registered".format(module_name)

        self[module_name] = module
    
    def get(self, name = ""):
        if name not in self.keys():
            raise KeyError(
                "No object named '{}' found in registry!".format(name)
            )
        
        return self[name]

