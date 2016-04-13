from __future__ import print_function
import numpy
import fitsio

def MetacalAveragerBase(dict):
    def __init__(self, step=0.01, chunksize=1000000):
        self['chunksize'] = chunksize
        self['step'] = step

    def process_flist(self, flist):
        """
        run through a set of files, doing the sums for
        averages
        """
        chunksize=self.chunksize
        sums=None
        ntot=0

        for f in flist:
            with fitsio.FITS(f) as fits:
                hdu=fits[1]

                nrows=hdu.get_nrows()
                nchunks = nrows//chunksize

                if (nrows % chunksize) > 0:
                    nchunks += 1

                beg=0
                for i in xrange(nchunks):
                    print("    chunk %d/%d" % (i+1,nchunks))

                    end=beg+chunksize

                    data = hdu[beg:end]
                    ntot += data.size

                    sums=self.do_sums(data, sums=sums)

                    beg = beg + chunksize

        return sums

    def get_shears(self, sums):
        """
        average the shear in each bin
        """

        nbin=sums.size

        means=self.average_sums(sums)

        # corrected for selections
        sh=get_shear_struct(nbin)

        # not corrected for selections
        sh_nocorr=get_shear_struct(nbin)

        R = means['R']
        Rsel = means['Rsel']
        Rpsf = means['Rpsf']
        Rpsf_sel = means['Rpsf_sel']

        for i in xrange(nbin):

            gmean = means['g'][i]
            gsq   = means['gsq'][i]

            # wsum is a count when we are not doing weights
            # need to do get the sums right for weights
            gvar = gsq - gmean**2
            num  = means['wsum']
            gerr = numpy.sqrt(gvar/num)

            c        = (Rpsf + Rpsf_sel)*gpsf[i]
            c_nocorr = Rpsf*gpsf[i]

            shear        = (gmean-c)/(R+Rsel)
            shear_nocorr = (gmean-c_nocorr)/R

            shear_err        = gerr/(R+Rsel)
            shear_nocorr_err = gerr/R

            sh['shear'][i] = shear
            sh['shear_err'][i] = shear_err

            sh_nocorr['shear'][i] = shear_nocorr
            sh_nocorr['shear_err'][i] = shear_nocorr_err

        means['shear'] = sh
        means['shear_nocorr'] = sh_nocorr

        return means

    def do_sums(self, data, sums=None):
        """
        for this base class, no selections, just do
        the overall means
        """

        if sums is None:
            sums=self._get_sums_struct(1)

        # for weights, need to do gsq correctly
        sums['wsum'][0] += data.size
        sums['g'][0]    += data['mcal_g'].sum(axis=0)
        sums['gsq'][0]  += (data['mcal_g']**2).sum(axis=0)
        sums['gpsf'][0] += data['mcal_gpsf'].sum(axis=0)

        for type in ngmix.metacal.METACAL_TYPES_SUB:
            mcalname='mcal_g_%s' % type
            sumname='g_%s' % type

            sums[sumname][0] += data[mcalname].sum(axis=0)

        return sums

    def average_sums(self, sums):
        """
        divide by sum of weights and get g for each field

        Also average the responses over all data
        """

        # g averaged in each field
        g    = sums['g'].copy()
        gsq  = sums['gsq'].copy()
        gpsf = sums['gpsf'].copy()

        winv = 1.0/sums['wsum']
        g[:,0]    *= winv
        g[:,1]    *= winv
        gsq[:,0]  *= winv
        gsq[:,1]  *= winv
        gpsf[:,0] *= winv
        gpsf[:,1] *= winv

        # responses averaged over all fields
        R = zeros(2)
        Rpsf = zeros(2)
        Rsel = zeros(2)
        Rsel_psf = zeros(2)

        factor = 1.0/(2.0*self['step'])

        wsum=sums['wsum'].sum()

        g1p = sums['g_1p'][:,0].sum()/wsum
        g1m = sums['g_1m'][:,0].sum()/wsum
        g2p = sums['g_2p'][:,1].sum()/wsum
        g2m = sums['g_2m'][:,1].sum()/wsum

        g1p_psf = sums['g_1p_psf'][:,0].sum()/wsum
        g1m_psf = sums['g_1m_psf'][:,0].sum()/wsum
        g2p_psf = sums['g_2p_psf'][:,1].sum()/wsum
        g2m_psf = sums['g_2m_psf'][:,1].sum()/wsum

        R[0] = (g1p - g1m)*factor
        R[1] = (g2p - g2m)*factor
        Rpsf[0] = (g1p_psf - g1m_psf)*factor
        Rpsf[1] = (g2p_psf - g2m_psf)*factor

        print("R:",R)
        print("Rpsf:",Rpsf)

        # selection terms
        if self.select is not None:
            s_g1p = sums['s_g_1p'][:,0].sum()/sums['s_wsum_1p'].sum()
            s_g1m = sums['s_g_1m'][:,0].sum()/sums['s_wsum_1m'].sum()
            s_g2p = sums['s_g_2p'][:,1].sum()/sums['s_wsum_2p'].sum()
            s_g2m = sums['s_g_2m'][:,1].sum()/sums['s_wsum_2m'].sum()

            s_g1p_psf = sums['s_g_1p_psf'][:,0].sum()/sums['s_wsum_1p_psf'].sum()
            s_g1m_psf = sums['s_g_1m_psf'][:,0].sum()/sums['s_wsum_1m_psf'].sum()
            s_g2p_psf = sums['s_g_2p_psf'][:,1].sum()/sums['s_wsum_2p_psf'].sum()
            s_g2m_psf = sums['s_g_2m_psf'][:,1].sum()/sums['s_wsum_2m_psf'].sum()

            Rsel[0] = (s_g1p - s_g1m)*factor
            Rsel[1] = (s_g2p - s_g2m)*factor
            Rsel_psf[0] = (s_g1p_psf - s_g1m_psf)*factor
            Rsel_psf[1] = (s_g2p_psf - s_g2m_psf)*factor

            print("Rsel:",Rsel)
            print("Rpsf_sel:",Rsel_psf)

        res=dict(
            sums=sums, # original sum structure
            g=g,       # number of bins size
            gsq=gsq,   # number of bins size
            gpsf=gpsf, # number of bins size
            R=R,       # the following averaged over bins
            Rpsf=Rpsf,
            Rsel=Rsel,
            Rsel_psf=Rsel_psf,
        )
        return res


    def do_selection(self, data, field, type):
        """
        parameters
        ----------
        data: numpy array with fields
            Should have fields at the minimum, perhaps more for selections
                'mcal_g','mcal_gpsf',
                'mcal_g_1p','mcal_g_1m',
                'mcal_g_2p','mcal_g_2m',
                'mcal_g_1p_psf','mcal_g_1m_psf',
                'mcal_g_2p_psf','mcal_g_2m_psf',
        """
        raise NotImplementedError("implement selections in a subclass")

    def _get_sums_struct(self, n):
        """
        get the structure to hold sums over the metacal parameters
        """
        dt=self._get_sums_dt()
        return numpy.zeros(n, dtype=dt)

    def _get_sums_dt(self):
        """
        dtype for the structure to hold sums over the metacal parameters
        """
        dt=[
            ('wsum','f8'),
            ('g','f8',2),
            ('gsq','f8',2), # for variances
            ('gpsf','f8',2),

            ('g_1p','f8',2),
            ('g_1m','f8',2),
            ('g_2p','f8',2),
            ('g_2m','f8',2),
            ('g_1p_psf','f8',2),
            ('g_1m_psf','f8',2),
            ('g_2p_psf','f8',2),
            ('g_2m_psf','f8',2),

            # selection terms
            ('s_wsum_1p','f8'),
            ('s_wsum_1m','f8'),
            ('s_wsum_2p','f8'),
            ('s_wsum_2m','f8'),
            ('s_g_1p','f8',2),
            ('s_g_1m','f8',2),
            ('s_g_2p','f8',2),
            ('s_g_2m','f8',2),

            ('s_wsum_1p_psf','f8'),
            ('s_wsum_1m_psf','f8'),
            ('s_wsum_2p_psf','f8'),
            ('s_wsum_2m_psf','f8'),
            ('s_g_1p_psf','f8',2),
            ('s_g_1m_psf','f8',2),
            ('s_g_2p_psf','f8',2),
            ('s_g_2m_psf','f8',2),
        ]
        return dt





def get_shear_struct(n):
    dt=[('shear','f8',2),
        ('shear_err','f8',2)]

    means = numpy.zeros(n, dtype=dt)
    return means


