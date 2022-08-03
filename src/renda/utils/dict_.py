# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


def transform_dicts(
    *dicts: Dict,
    transforms: Sequence[Callable] = [],
) -> Union[Dict, Tuple[Dict]]:
    """
    Transform all leaf objects of one or more dictionaries.

    Args:
        *dicts (Dict): Dictionaries to transform.
        transforms (Sequence[Callable], optional): Transforms to apply to the
        leaf objects of each dictionary passed. Defaults to [].

    Returns:
        Union[Dict, Tuple[Dict]]: If only one dictionary is passed, the
        transformed dictionary is returned. If multiple dictionaries are
        passed, a tuple of the transformed dictionaries is returned.

    Note:
        If you wish to obtain (deep) copies, you need to include the
        appropriate transforms. E.g., in the example below, the NumPy arrays
        stored in ``a`` and tensors stored in ``b`` still share the same
        memory.

    Example:

        >>> a = {
        >>>     "a_leaf": np.ones([3, 3]),
        >>>     "a_node": {
        >>>         "another_leaf": np.ones([3, 3]),
        >>>         "yet_another_leaf": np.ones([3, 3]),
        >>>     },
        >>> }

        # Get a version of 'a' where the leaf objects are PyTorch tensors
        >>> transforms = [torch.from_numpy]
        >>> b = transform_dicts(a, transforms=transforms)

        # Replace leaf objects with None (does not alter 'a' and 'b')
        >>> transforms = [lambda _: None]
        >>> c, d = transform_dicts(a, b, transforms=transforms)

        # Get a deep copy of 'b'
        >>> transforms = [torch.detach, torch.clone]
        >>> e = transform_dicts(b, transforms=transforms)
    """
    dicts_ = []
    for dict_ in dicts:
        dict_ = _transform_dict(dict_, transforms=transforms)
        dicts_.append(dict_)

    if len(dicts_) == 1:
        return dicts_[0]
    else:
        return tuple(dicts_)


def _transform_dict(
    dict_: Dict,
    transforms: Sequence[Callable] = [],
    node: Optional[Dict] = None,
) -> Dict:
    """
    Transform all leaf objects of a single dictionary.
    """
    if node is None:
        node = {}

    for k, v in dict_.items():
        if isinstance(v, dict):
            node[k] = {}
            _transform_dict(v, transforms, node[k])
        else:
            for t in transforms:
                v = t(v)
            node[k] = v

    return node


class OrderedDefaultDict(OrderedDict):
    """
    Python's missing OrderedDefaultDict.

    Args:
        type_ (TYPE): Type that a newly addressed node defaults to.

    Returns:
        None.

    Example::

        >>> a = OrderedDefaultDict(list)

        # Lists are created on the fly
        >>> a["new_key"].append(0)
        >>> assert len(a["another_new_key"]) == 0
    """

    def __init__(self, type_: Any) -> None:
        self.type_ = type_

    def __missing__(self, key: Any) -> Any:
        super().__setitem__(key, self.type_())
        return super().__getitem__(key)
