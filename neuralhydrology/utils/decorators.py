import inspect

def ignoreextra(func):
    """
    Takes a function with arguments and keyword arguments and calls them but ignores unexpected
    keyword arguments passed by the caller instead of raising errors.
    It is useful in situations where we can write a clean code providing same set of parameters to
    different models (as keyword arguments) even if they expect different subsets of them without 
    either unnecessarily long unused parameter list for all models or a boilerplate to filter required
    parameters for each model 
    """
    sig = inspect.signature(func)
    accepted_params = set(sig.parameters.keys())
    def returned_func(*args, **kargs):
        kargs = {k:v for k in kargs if k in accepted_params}
        return func(*args, **kargs)
    return returned_func
        

