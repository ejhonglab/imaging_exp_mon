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


def read_presentation_frames(xy, z=None, c=None):
    """
    xy (array-like): number of pixels along X and Y dimensions
    z (int): number of pixels along Z dimension
    c (int): number of channels
    """
    # TODO prob need to know how long each presentation is beyond thorimage
    # info...
    raise NotImplementedError


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
    # TODO reason not to getroot?
    return etree.parse(xml_path).getroot()


def monitor_thorimage(directory):
    """
    """
    # TODO get xy, z, c from xml
    # (it gets written to all immediately, right?)
    xml_path = join(directory, 'Experiment.xml')
    xml = xml_root()

    # TODO evaluate calcium responses
    # TODO evaluate baseline drift
    # TODO evaluate movement
    print('monitor_thorimage on', directory)


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
    if have_xml and have_movie:
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
    # TODO make directory to watch configurable w/ config file
    main()

