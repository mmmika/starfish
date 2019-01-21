#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file experiments with spot finding approaches, attempting to produce an integrated model for
identifying spots and/or pixels.

In doing so, it attempts to categorize the data characteristics that could cause a user to leverage
a particular spot finder. So far, these characteristics include:

1. How big are your spots? Are they diffraction-limited, e.g. single-pixel? do they vary in size?
# TODO move this down, but larger spots can use DoG instead of LoG, invariant spots can leverage
# smaller gaussian blurring windows, both of these will tune performance.
# fundamentally, the pixel-spot detector is the limit of "find small spots" when laplacians are
# unnecessary
2. Are your spots perfectly aligned, or do they require alignment?
3. What kind of filtering do you want to run on your spots?

Globally, this explores what users can do, given prior information about their spot characteristics.
"""
import os

import napari_gui
import numpy as np
import starfish.data
import starfish.display
from starfish.image import Filter, Registration, Segmentation
from starfish.spots import SpotFinder, PixelSpotDecoder, TargetAssignment
from starfish.types import Axes, Features
%gui qt5

###################################################################################################
# Pilot on ISS data
#
# ISS data is relatively sparse and has high signal-to-noise for detected spots. It makes it an
# easy substrate to test the approaches.

experiment = starfish.data.ISS()
image = experiment['fov_001']['primary']

###################################################################################################
# Register the data, if it isn't.

registration = Registration.FourierShiftRegistration(
    upsampling=1000,
    reference_stack=experiment['fov_001']['dots']
)
registered = registration.run(image, in_place=False)

# registration produces some bizarre edge effects where the edges get wrapped to maintain the
# image's. original shape This is bad because it interacts with the scaling later, so we need to
# crop these out. This is accomplished by cropping by the size of the shift
cropped = registered.sel({Axes.X: (25, -25), Axes.Y: (10, -10)})
starfish.display.stack(cropped)

###################################################################################################
# remove background from the data and equalize the channels. We're going to test using a weighted
# structuring element in addition to the one built into starfish

from skimage.morphology import ball, disk, opening, white_tophat  # noqa
from functools import partial  # noqa

selem_radii = (7, 10, 13)

def create_weighted_disk(selem_radius):
    s = ball(selem_radius)
    h = int((s.shape[1] + 1) / 2)  # Take only the upper half of the ball
    s = s[:h, :, :].sum(axis=0)  # Flatten the 3D ball to a weighted 2D disc
    weighted_disk = (255 * (s - s.min())) / (s.max() - s.min())  # Rescale weights into 0-255
    return weighted_disk

selems = [create_weighted_disk(r) for r in selem_radii]

# look at the background constructed from these methods

backgrounds = []
for s in selems:
    opening = partial(opening, selem=s)

    backgrounds.append(cropped.apply(
        opening,
        group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False, n_processes=8
    ))

# viewers = [starfish.display.stack(i) for i in backgrounds]
# starfish.display.stack(image)

# it looks like the smallest radius (7) is doing the best job, use that.

tophat = partial(white_tophat, selem=selems[0])

background_subtracted = cropped.apply(
    tophat,
    group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False, n_processes=8
)

# starfish.display.stack(background_subtracted)
# starfish.display.stack(image)

###################################################################################################
# remove slices in z that are not in focus
# this is important to ensure that scaling doesn't fabricate signal in out-of-focus planes. For this
# experiment, which is z-projected, this step can be omitted.

###################################################################################################
# normalize the channels so that they have approximately equal intensity

# two popular approaches are histogram normalization, to stretch the dynamic range, and quantile
# normalization, to equalize two or more images

from copy import deepcopy  # noqa

def quantile_normalize(xarray):
    stacked = xarray.stack(pixels=(Axes.X.value, Axes.Y.value, Axes.ZPLANE.value))
    inds = stacked.groupby(Axes.CH.value).apply(np.argsort)
    pos = inds.groupby(Axes.CH.value).apply(np.argsort)

    sorted_pixels = deepcopy(stacked)
    for v in sorted_pixels.coords[Axes.CH.value]:
        sorted_pixels[v, :] = sorted_pixels[v, inds[v].values].values

    rank = sorted_pixels.mean(Axes.CH.value)

    output = deepcopy(stacked)
    for v in output.coords[Axes.CH.value]:
        output[v] = rank[pos[v].values].values

    return output.unstack("pixels")

quantile_normalized = background_subtracted.xarray.groupby(
    Axes.ROUND.value
).apply(quantile_normalize)
quantile_normalized = starfish.ImageStack.from_numpy_array(quantile_normalized.values)

# starfish.display.stack(quantile_normalized)

# this demonstrates something similar to the above, using skimage functions.

# TODO this is coming in skimage 0.15.0, but isn't on pypi yet.
# from skimage.transform import match_histograms  # noqa

# # note that we should also match over Z, but that this data is projected
# match_r1c1 = partial(
#     match_histograms,
#     reference=background_subtracted.xarray.sel({Axes.CH.value: 0, Axes.ROUND.value: 1})
# )

# histogram_normalized = background_subtracted.apply(match_r1c1, group_by={Axes.ROUND, Axes.CH})

###################################################################################################
# call spots

# we have a bunch of different spot finders, and ways to visualize their outputs. test them all and
# see what happens!

from starfish.spots import SpotFinder  # noqa
from starfish.spots import PixelSpotDecoder  # noqa

SpotFinder.TrackpyLocalMaxPeakFinder
SpotFinder.LocalMaxPeakFinder
SpotFinder.BlobDetector
PixelSpotDecoder.PixelSpotDecoder

# key parameter to understand: intensity histogram across channels. This will be used to pick out
# blobs, and only when the detected blobs decode across channels will they be retained.

# start with 1/1000 pixels as spots and move up from there as necessary
# for blob_dog # TODO enter correct value.
# TODO this threshold might not correspond to local maxima directly -- it's related to the
# "scale space" -- what does this mean?
threshold = np.percentile(np.ravel(quantile_normalized.xarray.values), 99.4)

# start with the BlobDetector; difference of hessians only works in 2d, so we're skipping that.
# TODO think about removing DoH from the BlobDetector
min_sigma = 1  # TODO this appears to pick up _some_ 1-px signal
max_sigma = 4  # TODO can we translate these into pixels to be more intuitive
num_sigma = 9  # how many is enough? #TODO dial it down until things break
bd_dog = SpotFinder.BlobDetector(
    min_sigma, max_sigma, num_sigma, threshold=threshold, detector_method='blob_dog',
    is_volume=False
)
dog_intensities = bd_dog.run(quantile_normalized, reference_image_from_max_projection=True)
starfish.display.stack(quantile_normalized, dog_intensities, mask_intensities=threshold - 1e-5)

# LocalMaxPeakFinder is the same thing as blob_log if run after an inverted LoG filter...
# implies we could find the threshold in the same way.

###################################################################################################
# decode the images

decoded_dog_intensities = experiment.codebook.decode_per_round_max(dog_intensities)

# what fraction of spots decode?
fail_decoding = np.sum(decoded_dog_intensities['target'] == 'nan')
obs = (decoded_dog_intensities.shape[0] - fail_decoding) / decoded_dog_intensities.shape[0]

# what fraction of spots would match by chance?
# each spot has 4 rounds each with 4 options.
exp = 31 / (4 ** 4)

# odds ratio ~ 7.27
obs / exp

###################################################################################################
# segment the cells

seg = Segmentation.Watershed(
    nuclei_threshold=.16,
    input_threshold=.22,
    min_distance=57,
)
label_image = seg.run(image, experiment['fov_001']['nuclei'])

# assign spots to cells
ta = TargetAssignment.Label()
assigned = ta.run(label_image, decoded_dog_intensities)

# this sucks. Let's try Ilastik. Dump a max projection of the nuclei image and dots images.
# import skimage.io
# to_ilastik = np.squeeze(np.maximum(
#     experiment['fov_001']['nuclei'].xarray.values,
#     experiment['fov_001']['dots'].xarray.values
# ))
# skimage.io.imsave(os.path.expanduser('~/Downloads/nuclei.tiff'), to_ilastik)
# # lame, turns out this is just hard to segment :(

###################################################################################################
# Create a count matrix based on this bad segmentation.
counts = assigned.to_expression_matrix()


###################################################################################################
# APPENDIX, WORK IN PROGRESS AFTER THIS POINT.

# blob_dog can have trouble picking up small spots, try the blob_log instead
threshold = np.percentile(np.ravel(quantile_normalized.xarray.values), 98.0)
min_sigma = 0.7
max_sigma = 3  # if set too high, this can merge adjacent spots; translate to pixel radius or w/e?
num_sigma = 15  # how many is enough? #TODO dial it down until things break

# pilot on a single slice, don't bother measuring across yet.
from skimage.feature import blob_log  # noqa
plane = quantile_normalized.xarray.sel(
    {Axes.CH.value: 0, Axes.ROUND.value: 0, Axes.ZPLANE.value: 0}
)
test = blob_lob(
    plane.values,
    min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold
)

bd_log = SpotFinder.BlobDetector(
    min_sigma, max_sigma, num_sigma, threshold=threshold, detector_method='blob_log',
    is_volume=False
)
log_intensities = bd_log.run(quantile_normalized)
starfish.display.stack(quantile_normalized, log_intensities, mask_intensities=threshold - 1e-5)

###################################################################################################
# can we build tooling to match spots across rounds? apply blob_dog with the correct parameters

# alternative: find spots in each round/channel, then align them using mutual nearest neighbors
# we know the right parameters now, so apply this across rounds/channels.
spot_finder = partial(
    blob_dog, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold
)

# this is VERY fast in 2-d
spot_results = quantile_normalized.transform(spot_finder)

# create tidy dataframe
import pandas as pd  # noqa

data = []
for arr, axes in spot_results:
    arr = pd.DataFrame(arr, columns=['y', 'x', 'r'])
    arr[Axes.ROUND.value] = np.full(arr.shape[0], axes[Axes.ROUND])
    arr[Axes.CH.value] = np.full(arr.shape[0], axes[Axes.CH])
    data.append(arr)
pd_spot_results = pd.concat(data, axis=0)

# get all spots from round 1
# how many spots have NN within the 2.5px threshold?
round_1 = pd_spot_results.loc[pd_spot_results[Axes.ROUND.value].values == 0, :]

# do a simple matching now
# we have already filtered by size and intensity by calling spots
# nearest neighbors filter now filters by distance/jitter

# group the outputs by round
# find the k closest spots for
# for each spot called in round 1, match to the nearest k spots in other rounds
from scipy.spatial import cKDTree  # noqa

rounds = set(pd_spot_results[Axes.ROUND.value])
overlaps = []
for r in rounds:
    round_ = pd_spot_results.loc[pd_spot_results[Axes.ROUND.value].values == r, :]
    # check how many spots have self-overlapping values _within round_
    tree = cKDTree(round_[[Axes.X.value, Axes.Y.value]].values)
    results = tree.query_ball_tree(tree, 2.5, p=2)

    # _which_ results have an overlapping point? hypothesis: these may exist across rounds
    has_overlap = [True if len(r) > 1 else False for r in results]
    is_overlapping = (
        pd_spot_results.loc[pd_spot_results[Axes.ROUND.value].values == r].loc[has_overlap]
    )
    overlaps.append(is_overlapping.sort_values(by=[Axes.X.value, Axes.Y.value]))

# write up a visualization routine to visualize the spots in question
# can abuse show_spots
coords = pd.concat(overlaps, axis=0)
coords['z'] = np.full(coords.shape[0], 0)
starfish.display.stack(quantile_normalized, markers=coords[['x', 'y', 'r', 'c', 'z']])

# def find_mutual_nn(data1, data2, k1, k2, n_jobs):
k_index_1 = cKDTree(data1).query(x=data2, k=k1, n_jobs=n_jobs)[1]
k_index_2 = cKDTree(data2).query(x=data1, k=k2, n_jobs=n_jobs)[1]
mutual_1 = []
mutual_2 = []
for index_2 in range(data2.shape[0]):
    for index_1 in k_index_1[index_2]:
        if index_2 in k_index_2[index_1]:
            mutual_1.append(index_1)
            mutual_2.append(index_2)

# ok, I've looked at spots across rounds; they are definitely spots, but they produce some tricky
# cases, the spots are definitely real, they're even slightly shifted. Since the codebook is sparse
# it makes sense to build traces - at least one will be wrong, and hopefully a later hamming
# filter will remove one of them.

# since only 4% of spots do this, it doesn't seem bad to build the traces across rounds.
# this is codebook dependent, I think.

# pick an anchor channel, find nearby spots in each subsequent round.

anchor_round = 0
search_radius = 3
# def anchored_nn_search(anchor_round, search_radius):

threshold = np.percentile(np.ravel(quantile_normalized.xarray.values), 98)
spot_finder = partial(
    blob_dog, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold
)

# this is VERY fast in 2-d
spot_results = quantile_normalized.transform(
    spot_finder, group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}
)

from collections import defaultdict  # noqa

round_data = defaultdict(list)
for arr, axes in spot_results:
    arr = pd.DataFrame(arr, columns=['y', 'x', 'r'])
    arr[Axes.CH.value] = np.full(arr.shape[0], axes[Axes.CH])
    round_data[axes[Axes.ROUND.value]].append(arr)

# this method is nice because we can assess round imbalance!
# we could go back and increase the number of spots to search for in the other rounds...
# and now I understand why they're doing this the way they are...
# goal could be "fraction of spots that decode"
round_dataframes = {k: pd.concat(v, axis=0) for k, v in round_data.items()}

# this will need to be generalized to volumetric data by adding Axes.Z.value
traces = []
template = cKDTree(round_dataframes[anchor_round][[Axes.X.value, Axes.Y.value]])
for r in sorted(set(round_dataframes.keys()) - {anchor_round, }):
    query = cKDTree(round_dataframes[r][[Axes.X.value, Axes.Y.value]])
    traces.append(tree.query_ball_tree(query, search_radius, p=2))

# build a hamming tree from the codebook (is there a way to generalize?)

def measure_all(template_dataframe, image, spot_list):
    template_dataframe = template_dataframe.copy()
    # only rounds 2 and above will have a spot list. consider factoring this out.
    if spot_list:
        spot_indices = np.fromiter((r for r in indices if len(r) == 1), dtype=np.int)
        template_dataframe = template_dataframe.loc[spot_indices]
    # TODO generalize for Z
    intensities = image[template_dataframe[Axes.Y], template_dataframe[Axes.X]]
    template_dataframe['intensity'] = intensities
    return template_dataframe


# Put the traces together; resolve multiplets as they come up.
template_data = round_dataframes[anchor_round]
query_spot_indices = zip(*traces)
num_correct = 0
for anchor, indices in zip(template_data.index, query_spot_indices):
    if any(not r for r in indices):  # if any round didn't find a spot, continue
        continue
    else:
        # resolve multiplets; just toss for now.
        if any(len(r) > 1 for r in indices):
            continue

        # TODO ideally here, build some kind of starfish object (SpotAttributes?)
        # instead, build an IntensityTable directly to sidestep a refactor.

        # TODO interesting CB question (make issue)
        # have _correct_ measurements, how to deal with approximate x, y?

        # how to validate that things are working property in this mode?
        # plot all spots from first channel that _fail_ decoding, visually verify that there is no
        # spot in other rounds
        num_correct += 1

