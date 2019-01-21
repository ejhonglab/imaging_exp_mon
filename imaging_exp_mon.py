#!/usr/bin/env python3

"""
Analyzes the in-progress output of ThorImage and ThorSync to check for any of a
few reasons to stop the experiment.
"""

from os import listdir
from os.path import join, split, isdir
import time
import xml.etree.ElementTree as etree

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import h5py
import matplotlib.pyplot as plt


# The number of seconds the scope is aqcuiring for each odor presentation.
# Only support fixed length here.
presentation_seconds = 60.0
# TODO check this
seconds_before_odor = 5.0


def read_presentation_frames(imaging_file, n_frames, xy, z=None, c=None,
    presentation_num=None):
    """Returns (t,x,y) indexed timeseries.

    xy (array-like): number of pixels along X and Y dimensions
    z (int): number of pixels along Z dimension
    c (int): number of channels
    presentation_num (int): non-negative integer or None. If None, will load
        frames from last presentation.
    """
    # TODO err if presentation_num would run us off the end

    # (16-bit)
    bytes_per_pixel = 2
    
    if z is not None or c is not None:
        raise NotImplementedError

    x, y = xy
    # TODO off by one?
    offset = (-1) * (bytes_per_pixel * x * y * n_frames)

    # 0 = beginning, 1 = current position, 2 = end
    # Goal is to load the frames from the LAST presentation.
    if presentation_num is None:
        from_what = 2
    else:
        offset = (-1) * presentation_num * offset
        from_what = 0

    # From ThorImage manual: "unsigned, 16-bit, with little-endian byte-order"
    dtype = np.dtype('<u2')

    with open(imaging_file, 'rb') as f:
        # TODO TODO actually, how to specify end of where fromfile should load
        # from if that isn't the end of the file?
        # (for offline analysis w/ hardcoded presentation_num)
        # (or just load all frames in that case?)
        f.seek(offset, from_what)
        # TODO maybe check we actually get enough bytes for n_frames?
        data = np.fromfile(f, dtype=dtype)

    data = np.reshape(data, (n_frames, x, y))
    return data


def read_presentation_syncdata():
    """
    """
    # TODO TODO am i just fucked if thorsync isn't using (explicitly?)
    # hdf5 SWMR mode?
    # ask them to implement it that way? get new source and compile myself?

    # TODO 
    syncdata_file = 'SyncData001.hdf5'

    f = h5py.File(syncdata_file, 'r', swmr=True)

    # TODO TODO how (possible?) to only have recent chunks in memory when
    # reading in SWMR mode? how to configure in h5py?
    

    raise NotImplementedError


def xml_root(xml_path):
    """
    """
    return etree.parse(xml_path).getroot()


def get_thorimage_dims(xmlroot):
    """
    """
    lsm_attribs = xmlroot.find('LSM').attrib
    x = int(lsm_attribs['pixelX'])
    y = int(lsm_attribs['pixelY'])
    xy = (x,y)

    # TODO make this None unless z-stepping seems to be enabled
    # + check this variable actually indicates output steps
    #int(xml.find('ZStage').attrib['steps'])
    z = None
    c = None

    return xy, z, c


def get_thorimage_fps(xmlroot):
    """
    """
    lsm_attribs = xmlroot.find('LSM').attrib
    raw_fps = float(lsm_attribs['frameRate'])
    # TODO what does averageMode = 1 mean? always like that?
    # 
    n_averaged_frames = int(lsm_attribs['averageNum'])
    saved_fps = raw_fps / n_averaged_frames
    return saved_fps


def mean_response(baseline, frames):
    """
    """
    response_seconds = 2.0
    n_response_frames = int(np.round(response_seconds * fps))

    # TODO check this doesn't overlap by a frame w/ baseline
    response_frames = frames[:(frames_before_odor + n_response_frames),:,:]

    # TODO equivalent to taking mean in response period first and diffing?
    response = (response_frames - baseline) / baseline
    mean_df_over_f = np.mean(response)

    return mean_df_over_f


def load_thorimage_metadata(directory):
    """
    """
    # TODO does xml get written immediately?
    xml_path = join(directory, 'Experiment.xml')
    xml = xml_root(xml_path)

    fps = get_thorimage_fps(xml)
    xy, z, c = get_thorimage_dims(xml)
    imaging_file = join(directory, 'Image_0001_0001.raw')

    return fps, xy, z, c, imaging_file


def analyze_thorimage_offline(directory):
    """
    """
    # TODO try to share more code w/ monitor_... !

    fps, xy, z, c, imaging_file = load_thorimage_metadata(directory)
    n_presentation_frames = int(np.round(fps * presentation_seconds))

    first_baseline = None
    # Responses
    mean_responses = []
    # Baseline drift
    mean_bn_over_b0 = []

    #mean_movement = []

    # TODO guess presentation num from other params / say if raw
    # doesnt have an integer number of frames assuming all hardcoded params
    # (presentation seconds especially)
    n_presentations = 3

    for n in range(n_presentations):
        frames = read_presentation_frames(imaging_file, n_presentation_frames,
                                          xy, presentation_num=n)

        # Compute baseline
        frames_before_odor = int(np.round(seconds_before_odor * fps))
        baseline = np.mean(frames[:frames_before_odor,:,:], axis=0)

        if first_baseline is None:
            first_baseline = baseline

        baseline_drift = np.mean(baseline / first_baseline)
        mean_bn_over_b0.append(baseline_drift)

        # Check response
        mean_df_over_f = mean_response(baseline,
                                       frames[frames_before_odor:,:,:])

        mean_responses.append(mean_df_over_f)

        print('Mean (baseline / first baseline): {}'.format(baseline_drift))

        print('Mean dF/F in {} seconds after odor: {}'.format(response_seconds,
            mean_df_over_f))


# TODO TODO include global knowledge of odor order, so that all monitor
# functions can save odor specific statistics over time, so that outliers can be
# detected
def monitor_thorimage(directory):
    """
    """
    # TODO TODO how to know when to terminate this program?
    # intercept Thor IPC? wait for a file to finish writing?
    # does Experiment.xml actually have experiment status updated to stopped at
    # end (what about crash cases though...)?
    print('monitor_thorimage on', directory)

    fps, xy, z, c, imaging_file = load_thorimage_metadata(directory)

    first_baseline = None
    # Responses
    mean_responses = []
    # Baseline drift
    mean_bn_over_b0 = []

    #mean_movement = []


    # TODO TODO and how to know when to iterate?
    # watch raw file for writes (is it written all at end of block?)

    n_presentation_frames = int(np.round(fps * presentation_seconds))

    ############ Iterate the rest of the fn ###########
    frames = read_presentation_frames(imaging_file, n_presentation_frames, xy)

    # Compute baseline
    frames_before_odor = int(np.round(seconds_before_odor * fps))
    baseline = np.mean(frames[:frames_before_odor,:,:], axis=0)

    if first_baseline is None:
        first_baseline = baseline

    # TODO maybe compute this wrt just the previous presentation?
    baseline_drift = np.mean(baseline / first_baseline)
    mean_bn_over_b0.append(baseline_drift)

    # Check response
    mean_df_over_f = mean_response(baseline, frames[frames_before_odor:,:,:])
    mean_responses.append(mean_df_over_f)

    # TODO find some appropriate threshold for quality here
    # (doable?)

    # TODO evaluate return to baseline?
    # TODO evaluate movement

    print('Mean (baseline / first baseline): {}'.format(baseline_drift))

    print('Mean dF/F in {} seconds after odor: {}'.format(response_seconds,
        mean_df_over_f))
    import ipdb; ipdb.set_trace()



def monitor_thorsync(directory):
    """
    """
    # TODO evaluate PID responses / drift

    print('monitor_thorsync on', directory)


def is_thorsync_dir(d):
    """Returns whether directory has all ThorSync output files.
    """
    # TODO warn if has SyncData in name but fails this?
    if not isdir(d):
        return False
    
    files = {f for f in listdir(d)}

    have_settings = False
    have_h5 = False
    for f in files:
        # checking for substring
        if 'ThorRealTimeDataSettings.xml' in f:
            have_settings = True
        if '.h5':
            have_h5 = True

    return have_h5 and have_settings


def is_thorimage_dir(d):
    """Returns whether directory has all ThorImage output files.
    """
    if not isdir(d):
        return False
    
    files = {f for f in listdir(d)}

    have_xml = False
    have_processed_tiff = False
    have_extracted_metadata = False
    for f in files:
        if 'Experiment.xml' in f:
            have_xml = True
        elif f.startswith('Image_') and f.endswith('.raw'):
            # TODO TODO let this be missing initially, for case where there is
            # only that .tmp file or whatever, or movie is still in memory
            # check other files and wait?
            have_movie = True
            # TODO use regex to check like Image_0001_0001.raw
        '''
        elif f == split(d)[-1] + '_ChanA.tif':
            have_processed_tiff = True
        elif f == 'ChanA_extracted_metadata.txt':
            have_extracted_metadata = True
        '''
    # TODO any of preview, roimask.raw, roi xaml?

    #if have_xml and have_processed_tiff:
    #if have_xml and have_movie:
    if have_xml:
        '''
        if not have_extracted_metadata:
            warnings.warn('there does not seem to be extracted metadata in' + d)
        '''
        return True
    else:
        return False


def monitor_thor_dir(directory):
    """Dispatches appropriate function for ThorImage and Sync directories.

    Does nothing if directory is neither.
    """
    #time.sleep(5)

    if is_thorimage_dir(directory):
        monitor_thorimage(directory)

    elif is_thorsync_dir(directory):
        monitor_thorsync(directory)

    else:
        print(directory, 'was neither thorsync nor thorimage')
        return


class ThorDirectoryHandler(FileSystemEventHandler):
    # TODO should this be @staticmethod or not? two examples have it...
    # should be in docs though...
    #@staticmethod
    def on_created(self, event):
        print('handling on_created event')
        if not event.is_directory:
            print('was not a directory')
            return

        monitor_thor_dir(event.src_path)
        # TODO TODO need to destroy this handler when above returns
        # what is lifetime of a handler normally?
        # just the event?

    # TODO TODO so can this class use other functions in this file?
    # restrictions? probably won't share globals across threads...

    # TODO check here or in subdir for case where directory is deleted
    # (b/c of a false start, for instance)


#def fn_on_new_subdirs(directory, fn):
def handle_directory(directory, handler_class):
    """
    """
    observer = Observer()
    observer.schedule(handler_class(), directory)
    # TODO why is schedule distinct from start?
    print('starting observer on', directory)
    observer.start()

    # TODO TODO check that this does not put undue load on computer when it is
    # actually involved in acquisition
    # why does it seem to pause when any file is created? should just return,
    # right? fixable? a problem?
    try:
        while True:
            #print('sleeping...')
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()

    # TODO does this return anything? use that to coordinate things?
    print('joining observer')
    observer.join()
    print('done')


def main():
    # TODO maybe option to just look for thorsync / image folders that look
    # current under here, and monitor that experiment, as opposed to checking
    # for new day directories under here?
    '''
    thor_output_dirs = 'D:/'

    for d in thor_output_dirs:
        register_fn_on_new_subdirs(d, fn)
    '''
    #thor_output_dir = 'D:/'
    #experiment_group_directory = join(thor_output_dir, '2019-01-18')
    experiment_group_directory = 'test_dir'
    # TODO how do i want to match up thorimage and thorsync, generally?
    handle_directory(experiment_group_directory, ThorDirectoryHandler)
    
    
if __name__ == '__main__':
    # TODO delete
    analyze_thorimage_offline('test_dir/_001')
    #monitor_thorimage('test_dir/_001')
    import sys; sys.exit()
    #

    # TODO make directory to watch configurable w/ config file
    main()

