from __future__ import print_function

import esutil as eu

class NullTesterBase(dict):
    def __init__(self, run):
        """
        parameters
        -----------
        run: string
            is a ngmixer run identifier
        selection: string
            A string representing the selection
        """
        self['run'] = run

    def do_sums(self, *args, **kw):
        """
        run over the files and perform sums
        """
        pass


class PSFShapeTester(NullTesterBase):
    """
    measure the mean shape vs. psf shape
    """

    def do_sums(self, data, sums=None):
        gpsf=data['mcal_gpsf']
        pass
