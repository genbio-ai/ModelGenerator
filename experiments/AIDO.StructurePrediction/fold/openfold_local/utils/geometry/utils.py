
"""Utils for geometry library."""

import dataclasses


def get_field_names(cls):
    fields = dataclasses.fields(cls)
    field_names = [f.name for f in fields]
    return field_names
