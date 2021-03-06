from typing import Tuple

import numpy as np
from slicedimage import Tile, TileSet

import starfish.data
from starfish.experiment.experiment import Experiment, FieldOfView
from starfish.types import Axes, Coordinates
from starfish.util.synthesize import SyntheticData


def round_to_x(r: int) -> Tuple[float, float]:
    return (r + 1) * 1000, (r + 1) * 100


def round_to_y(r: int) -> Tuple[float, float]:
    return (r + 1) * 10, (r + 1) * 0.1


def round_to_z(r: int) -> Tuple[float, float]:
    return (r + 1) * 0.01, (r + 1) * 0.001


NUM_ROUND = 5
NUM_CH = 2
NUM_Z = 1
HEIGHT = 10
WIDTH = 10


def get_aligned_tileset():
    alignedTileset = TileSet(
        [Axes.X, Axes.Y, Axes.CH, Axes.ZPLANE, Axes.ROUND],
        {Axes.CH: NUM_CH, Axes.ROUND: NUM_ROUND, Axes.ZPLANE: NUM_Z},
        {Axes.Y: HEIGHT, Axes.X: WIDTH})

    for r in range(NUM_ROUND):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
                tile = Tile(
                    {
                        Coordinates.X: 1,
                        Coordinates.Y: 4,
                        Coordinates.Z: 3,
                    },
                    {
                        Axes.ROUND: r,
                        Axes.CH: ch,
                        Axes.ZPLANE: z,
                    }
                )
                tile.numpy_array = np.zeros((100, 100))
                alignedTileset.add_tile(tile)
    return alignedTileset


def get_un_aligned_tileset():
    unAlignedTileset = TileSet(
        [Axes.X, Axes.Y, Axes.CH, Axes.ZPLANE, Axes.ROUND],
        {Axes.CH: NUM_CH, Axes.ROUND: NUM_ROUND, Axes.ZPLANE: NUM_Z},
        {Axes.Y: HEIGHT, Axes.X: WIDTH})

    for r in range(NUM_ROUND):
        for ch in range(NUM_CH):
            for z in range(NUM_Z):
                tile = Tile(
                    {
                        # The round_to methods generate coordinates
                        # based on the r value, therefore the coords vary
                        # throughout the tileset
                        Coordinates.X: round_to_x(r),
                        Coordinates.Y: round_to_y(r),
                        Coordinates.Z: round_to_z(r),
                    },
                    {
                        Axes.ROUND: r,
                        Axes.CH: ch,
                        Axes.ZPLANE: z,
                    }
                )
                tile.numpy_array = np.zeros((HEIGHT, WIDTH))
                unAlignedTileset.add_tile(tile)
    return unAlignedTileset


def test_fov_order():
    data = SyntheticData()
    codebook = data.codebook()
    tilesets = {"primary": get_aligned_tileset()}
    fovs = [FieldOfView("stack2", tilesets),
            FieldOfView("stack1", tilesets)]
    extras = {"synthetic": True}
    experiment = Experiment(fovs, codebook, extras)
    assert "stack1" == experiment.fov().name
    assert ["stack1", "stack2"] == [x.name for x in experiment.fovs()]


def test_crop_experiment():
    exp = starfish.data.ISS(use_test_data=True)
    x_slice = slice(10, 30)
    y_slice = slice(40, 70)
    image = exp['fov_001'].get_image('primary', x_slice=x_slice, y_slice=y_slice)
    assert image.shape['x'] == 20
    assert image.shape['y'] == 30


def test_fov_aligned_tileset():
    tilesets = {'primary': get_aligned_tileset(), 'nuclei': get_aligned_tileset()}
    fov = FieldOfView("aligned", tilesets)
    # Assert only one coordinate group for each aligned stack
    assert len(fov.aligned_coordinate_groups['primary']) == 1
    assert len(fov.aligned_coordinate_groups['nuclei']) == 1


def test_fov_un_aligned_tileset():
    tilesets = {'primary': get_un_aligned_tileset(), 'nuclei': get_un_aligned_tileset()}
    fov = FieldOfView("unaligned", tilesets)
    # Assert that the number of coordinate groups == NUM_ROUNDS
    assert len(fov.aligned_coordinate_groups['primary']) == NUM_ROUND
    assert len(fov.aligned_coordinate_groups['nuclei']) == NUM_ROUND


def test_fov_partially_aligned_tileset():
    # Create combo of aligned and unaligned tiles
    partially_aligned_tileset = get_aligned_tileset()
    for tile in get_un_aligned_tileset().tiles():
        partially_aligned_tileset.add_tile(tile)
    tileset_dict = {'primary': partially_aligned_tileset}
    fov = FieldOfView("paritally aligned", tileset_dict)
    # Assert the number of tile groups is num rounds + 1 aligned group
    assert len(fov.aligned_coordinate_groups['primary']) == NUM_ROUND + 1
