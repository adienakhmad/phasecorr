from pathlib import Path
from phasecorr.pac import PhaseAutocorr
import obspy


class BatchProcessPAC(object):

    def __init__(self, root_folder):
        self.root_path = Path(root_folder)

    def get_iterator(self, file_pattern='*'):
        paths = self.root_path.glob(file_pattern)
        files = (path for path in paths if path.is_file())

        for file in files:
            st = obspy.read(str(file))
            pac = PhaseAutocorr(st)
            yield file, pac
