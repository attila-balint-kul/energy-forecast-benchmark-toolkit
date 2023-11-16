class Dataset:
    def __init_subclass__(cls, **kwargs):
        """This throws a deprecation warning on subclassing."""
        msg = f"""
        {cls.__name__} is deprecated.
        Import it as 'from enfobench.dataset import Dataset' instead.
        """
        raise DeprecationWarning(msg)

    def __init__(self, *args, **kwargs):  # noqa
        """This throws a deprecation warning on initialization."""
        msg = f"""
        {self.__class__.__name__} is deprecated.
        Import it as 'from enfobench.dataset import Dataset' instead.
        """
        raise DeprecationWarning(msg)
