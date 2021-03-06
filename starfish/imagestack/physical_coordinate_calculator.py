from typing import Optional, Tuple, Union


from starfish.types import (
    Number,
)


def _calculate_physical_pixel_size(coord_min: Number, coord_max: Number, num_pixels: int) -> Number:
    """Calculate the size of a pixel in physical space"""
    return (coord_max - coord_min) / num_pixels


def _pixel_offset_to_physical_coordinate(
        physical_pixel_size: Number,
        pixel_offset: Optional[int],
        coordinates_at_pixel_offset_0: Number,
        dimension_size: int,
) -> Number:
    """Calculate the physical pixel value at the given index"""
    if pixel_offset:
        # Check for negative index
        if pixel_offset < 0:
            pixel_offset = pixel_offset + dimension_size
        return (physical_pixel_size * pixel_offset) + coordinates_at_pixel_offset_0
    return coordinates_at_pixel_offset_0


def recalculate_physical_coordinate_range(
        coord_min: float,
        coord_max: float,
        dimension_size: int,
        indexer: Union[int, slice],
) -> Tuple[Number, Number]:
    """Given the dimension size and pixel coordinate indexes calculate the corresponding
    coordinates in physical space

    Parameters
    ----------
    coord_min: float
        the minimum physical coordinate value

    coord_max: float
        the maximum physical coordinate value

    dimension_size: int
        The size (number of pixels) to use to calculate physical_pixel_size

    indexer: (int/slice)
        The pixel index or range to calculate.

    Returns
    -------
    The new min and max physical coordinate values of the given dimension
    """
    physical_pixel_size = _calculate_physical_pixel_size(coord_min, coord_max, dimension_size)
    min_pixel_index = indexer if isinstance(indexer, int) else indexer.start
    max_pixel_index = indexer if isinstance(indexer, int) else indexer.stop
    # Add one to max pixel index to get end of pixel
    max_pixel_index = max_pixel_index + 1 if max_pixel_index else dimension_size
    new_min = _pixel_offset_to_physical_coordinate(
        physical_pixel_size, min_pixel_index, coord_min, dimension_size)
    new_max = _pixel_offset_to_physical_coordinate(
        physical_pixel_size, max_pixel_index, coord_min, dimension_size)
    return new_min, new_max


def get_physical_coordinates_of_z_plane(zrange: Tuple[float, float]):
    """Calculate the midpoint of the given zrange."""
    physical_z = (zrange[1] - zrange[0]) / 2 + zrange[0]
    return physical_z
