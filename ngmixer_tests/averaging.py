from __future__ import print_function
import os
from glob import glob
import numpy
import fitsio
import ngmix

import esutil as eu
from esutil.numpy_util import between

def get_run_flist(run):
    dir=os.environ['NGMIXER_OUTPUT_DIR']
    dir=os.path.join(dir, run,'output')
    pattern=os.path.join(dir, '*.fits')
    print("pattern:",pattern)
    flist=glob(pattern)
    print("found",len(flist),"files")
    return flist

class AveragerBase(dict):
    def __init__(self, step=0.01, chunksize=1000000, matchcat=None):

        self.matchcat=matchcat

        self['chunksize'] = chunksize
        self['step'] = step

        # no selections in this base class
        self['do_select']=False

    def process_run(self, run):
        """
        run through all the collated files for the specified run
        """

        flist=get_run_flist(run)
        return self.process_flist(flist)


    def process_flist(self, flist):
        """
        run through a set of files, doing the sums for
        averages
        """
        chunksize=self['chunksize']
        sums=None
        nf=len(flist)

        for i,f in enumerate(flist):

            print("processing %d/%d: %s" % (i+1,nf,f))
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

                    sums=self.do_sums(data, sums=sums)

                    beg = beg + chunksize

        self.means /= sums['wsum']

        means=self.get_shears(sums)
        means['means'] = self.means
        return means

    def do_sums(self, data, sums=None):
        """
        do all the sums, no binning for base class
        """

        if sums is None:
            sums=self._get_sums_struct(1)

        w=self.do_selection(data)
        print("    kept: %d/%d" % (w.size, data.size))

        # for weights, need to do gsq correctly
        sums['wsum'][0] += w.size
        sums['g'][0]    += data['mcal_g'][w].sum(axis=0)
        sums['gsq'][0]  += (data['mcal_g'][w]**2).sum(axis=0)
        sums['gpsf_sq'][0]  += (data['mcal_gpsf'][w]**2).sum(axis=0)
        sums['gpsf'][0] += data['mcal_gpsf'][w].sum(axis=0)

        for type in ngmix.metacal.METACAL_TYPES_SUB:
            mcalname='mcal_g_%s' % type
            sumname='g_%s' % type

            sums[sumname][0] += data[mcalname][w].sum(axis=0)

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

        for i in xrange(nbin):

            num       = sums['wsum'][i]
            gmean     = means['g'][i]
            gpsf_mean = means['gpsf'][i]

            gsq       = means['gsq'][i]
            gpsf_sq   = means['gpsf_sq'][i]

            R         = means['R'][i]
            Rsel      = means['Rsel'][i]
            Rpsf      = means['Rpsf'][i]
            Rpsf_sel  = means['Rpsf_sel'][i]

            # wsum is a count when we are not doing weights
            # need to do get the sums right for weights
            gvar      = gsq - gmean**2
            gpsf_var  = gpsf_sq - gpsf_mean**2


            print("gpsf:",gpsf_mean)
            c        = (Rpsf + Rpsf_sel)*gpsf_mean
            c_nocorr = Rpsf*gpsf_mean

            vartot        = gvar + gpsf_var*(Rpsf + Rpsf_sel)**2
            vartot_nocorr = gvar + gpsf_var*Rpsf**2

            gerr = numpy.sqrt(vartot/num)
            gerr_nocorr = numpy.sqrt(vartot_nocorr/num)

            shear        = (gmean-c)/(R+Rsel)
            shear_nocorr = (gmean-c_nocorr)/R

            shear_err        = gerr/(R+Rsel)
            shear_nocorr_err = gerr_nocorr/R
            '''
            shear        = (gmean-c)
            shear_nocorr = (gmean-c_nocorr)

            shear_err        = gerr
            shear_nocorr_err = gerr_nocorr
            '''


            sh['shear'][i] = shear
            sh['shear_err'][i] = shear_err

            sh_nocorr['shear'][i] = shear_nocorr
            sh_nocorr['shear_err'][i] = shear_nocorr_err

        means['shear'] = sh
        means['shear_nocorr'] = sh_nocorr

        return means


    def average_sums(self, sums):
        """
        divide by sum of weights and get g for each field

        Also average the responses over all data
        """

        g    = sums['g'].copy()
        gsq  = sums['gsq'].copy()
        gpsf_sq  = sums['gpsf_sq'].copy()
        gpsf = sums['gpsf'].copy()

        winv = 1.0/sums['wsum']
        g[:,0]    *= winv
        g[:,1]    *= winv
        gsq[:,0]  *= winv
        gsq[:,1]  *= winv
        gpsf_sq[:,0]  *= winv
        gpsf_sq[:,1]  *= winv
        gpsf[:,0] *= winv
        gpsf[:,1] *= winv

        # responses averaged over all fields
        R = 0*g
        Rpsf = 0*g
        Rsel = 0*g
        Rpsf_sel = 0*g

        factor = 1.0/(2.0*self['step'])

        g1p = sums['g_1p'][:,0]*winv
        g1m = sums['g_1m'][:,0]*winv
        g2p = sums['g_2p'][:,1]*winv
        g2m = sums['g_2m'][:,1]*winv

        g1p_psf = sums['g_1p_psf'][:,0]*winv
        g1m_psf = sums['g_1m_psf'][:,0]*winv
        g2p_psf = sums['g_2p_psf'][:,1]*winv
        g2m_psf = sums['g_2m_psf'][:,1]*winv

        R[:,0] = (g1p - g1m)*factor
        R[:,1] = (g2p - g2m)*factor
        Rpsf[:,0] = (g1p_psf - g1m_psf)*factor
        Rpsf[:,1] = (g2p_psf - g2m_psf)*factor

        print("R:",R)
        print("Rpsf:",Rpsf)

        # selection terms
        if self['do_select']:
            s_g1p = sums['s_g_1p'][:,0]/sums['s_wsum_1p']
            s_g1m = sums['s_g_1m'][:,0]/sums['s_wsum_1m']
            s_g2p = sums['s_g_2p'][:,1]/sums['s_wsum_2p']
            s_g2m = sums['s_g_2m'][:,1]/sums['s_wsum_2m']

            s_g1p_psf = sums['s_g_1p_psf'][:,0]/sums['s_wsum_1p_psf']
            s_g1m_psf = sums['s_g_1m_psf'][:,0]/sums['s_wsum_1m_psf']
            s_g2p_psf = sums['s_g_2p_psf'][:,1]/sums['s_wsum_2p_psf']
            s_g2m_psf = sums['s_g_2m_psf'][:,1]/sums['s_wsum_2m_psf']

            Rsel[:,0] = (s_g1p - s_g1m)*factor
            Rsel[:,1] = (s_g2p - s_g2m)*factor
            Rpsf_sel[:,0] = (s_g1p_psf - s_g1m_psf)*factor
            Rpsf_sel[:,1] = (s_g2p_psf - s_g2m_psf)*factor

            print("Rsel:",Rsel)
            print("Rpsf_sel:",Rpsf_sel)

        res=dict(
            sums=sums, # original sum structure
            g=g,       # number of bins size
            gsq=gsq,   # number of bins size
            gpsf=gpsf, # number of bins size
            gpsf_sq=gpsf_sq,   # number of bins size
            R=R,       # the following averaged over bins
            Rpsf=Rpsf,
            Rsel=Rsel,
            Rpsf_sel=Rpsf_sel,
        )
        return res


    def do_selection(self, data, **kw):
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

        # s2n check can be removed in new run
        w, = numpy.where( (data['flags'] == 0) & (data['mcal_s2n_r'] != -9999.0))

        return w

    def get_type_string(self, mcal_type):
        """
        mcal_name is for noshear
        mcal_name_{mcal_type} for others
        """
        if mcal_type=='noshear':
            tstr=''
        else:
            tstr='_%s' % mcal_type
        return tstr

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
            ('gpsf_sq','f8',2),

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


class FieldAverager(AveragerBase):
    """
    cuts on one field

    selections is any arbitrary selections, not a simple
    binning
    """
    def __init__(self, field_base, selections, **kw):
        """
        field_base: string
            name of base field
        selections: string or list
            e.g. 'x > 10' or ['x > 10','x > 15']
        matchcat: array
            Catalog with 'coadd_objects_id' to match in sanity cuts
        """
        super(FieldAverager,self).__init__(**kw)


        self['field_base'] = field_base
        self['do_select'] = True
        self['use_logpars']=kw.get('use_logpars',False)

        # single selection
        if not isinstance(selections,(list,tuple)):
            selections = [selections]

        self['selections'] = selections
        self['nbin'] = len(selections)
        self.means = numpy.zeros(self['nbin'])

    def do_sanity_cuts(self, data):
        """
        the s2n cuts are for the bug not propagating flags
        """
        logic = (
              (data['flags'] == 0)
            #& (data['mask_frac'] < 0.1)
            #& (data['mcal_s2n_r'] != val)
            #& (data['mcal_s2n_r_1p'] != val)
            #& (data['mcal_s2n_r_1m'] != val)
            #& (data['mcal_s2n_r_2p'] != val)
            #& (data['mcal_s2n_r_2m'] != val)
            #& (data['mcal_s2n_r_1p_psf'] != val)
            #& (data['mcal_s2n_r_1m_psf'] != val)
            #& (data['mcal_s2n_r_2p_psf'] != val)
            #& (data['mcal_s2n_r_2m_psf'] != val)
            # need to save T_r
            #& (data['mcal_pars'][:,4] > data['mcal_Tpsf'])
            #& (data['mcal_pars'][:,4] > 0.5*data['mcal_Tpsf'])
            #& (data['gauss_logsb'][:,1] < 4)

            #& (numpy.sqrt(data['mcal_pars_cov'][:,2,2]) < 0.3)
            #& (numpy.sqrt(data['mcal_pars_cov'][:,3,3]) < 0.3)
        )

        w,=numpy.where(logic)
        print("    keeping %d/%d from flag sanity cuts" % (w.size,data.size))
        if self.matchcat is not None:
            m,mcat = eu.numpy_util.match(
                data['id'],
                self.matchcat['coadd_objects_id'],
            )
            match_logic = numpy.zeros(data.size, dtype=bool)
            match_logic[m] = True

            w,=numpy.where(match_logic)
            print("    keeping %d/%d from match sanity cuts" % (w.size,data.size))

            logic = logic & match_logic
            w,=numpy.where(logic)
            print("        keeping %d/%d from both cuts" % (w.size,data.size))

        return logic

    def get_selection_args(self, data, mcal_type):
        """
        get the default set
        """
        tstr=self.get_type_string(mcal_type)

        gpsf = data['mcal_gpsf'][:,self['element']]
        Tpsf = data['mcal_Tpsf']

        T = data['mcal_T_r%s' % tstr]

        Tvar = data['mcal_T_err%s' % tstr]
        Terr=numpy.zeros(Tvar.size) + 1.e9

        w,=numpy.where(Tvar > 0.0)
        if w.size > 0:
            Terr[w] = numpy.sqrt(Tvar[w])

        Ts2n = T/Terr

        s2n_field = 'mcal_s2n_r%s' % tstr
        s2n = data[s2n_field]


        return gpsf, s2n, Ts2n, T, Terr, Tpsf

    def do_selection(self, data, mcal_type, binnum):
        """
        cut on flags and s/n
        """
        raise NotImplementedError("implement in concrete class")

    def do_sums(self, data, sums=None):
        """
        do all the sums, no binning for base class
        """

        logic0 = self.do_sanity_cuts(data)

        if sums is None:
            sums=self._get_sums_struct(self['nbin'])

        # first select on the noshear measurement,
        # sum up the estimator g and gpsf,
        # then sum the sheared parameters for R

        for binnum in xrange(self['nbin']):

            cut_logic, fvalues = self.do_selection(data, 'noshear', binnum)
            w, = numpy.where(logic0 & cut_logic)
            print("    kept: %d/%d" % (w.size, data.size))

            # TODO: for weights, need to do gsq correctly
            sums['wsum'][binnum]     +=  w.size
            sums['g'][binnum]        +=  data['mcal_g'][w].sum(axis=0)
            sums['gsq'][binnum]      += (data['mcal_g'][w]**2).sum(axis=0)
            sums['gpsf'][binnum]     +=  data['mcal_gpsf'][w].sum(axis=0)
            sums['gpsf_sq'][binnum]  += (data['mcal_gpsf'][w]**2).sum(axis=0)

            self.means[binnum] += fvalues[w].sum(axis=0)

            for type in ngmix.metacal.METACAL_TYPES_SUB:
                mcalname='mcal_g_%s' % type
                sumname='g_%s' % type

                sums[sumname][binnum] += data[mcalname][w].sum(axis=0)

            # now the selection terms: select on sheared measurements
            # but add up the unsheared estimator g
            for type in ngmix.metacal.METACAL_TYPES_SUB:

                wsumname = 's_wsum_%s' % type
                sumname = 's_g_%s' % type

                cut_logic, fvalues = self.do_selection(data, type, binnum)
                w, = numpy.where(logic0 & cut_logic)

                sums[wsumname][binnum] += w.size
                sums[sumname][binnum]  += data['mcal_g'][w].sum(axis=0)

        return sums

class S2NAverager(FieldAverager):
    """
    averaging only over some s2n selection
    """
    def __init__(self, selections, **kw):
        super(S2NAverager,self).__init__('mcal_s2n_r', selections, **kw)


    def do_selection(self, data, mcal_type, binnum):
        """
        cut on flags and s/n
        """
        gpsf, s2n, Ts2n, T, Terr, Tpsf = self.get_selection_args(data, mcal_type)

        selection=self['selections'][binnum]
        cut_logic = eval(selection)

        return cut_logic, s2n

class S2NTS2NAverager(FieldAverager):
    """
    The following variables are made available for selection

    s2n, Ts2n, T, Tpsf
    """

    def do_selection(self, data, mcal_type, binnum):
        """
        cut on flags and s/n
        """

        gpsf, s2n, Ts2n, T, Terr, Tpsf =self.get_selection_args(data, mcal_type)

        selection=self['selections'][binnum]

        cut_logic = eval(selection)

        return cut_logic, Ts2n


class FieldBinner(FieldAverager):
    def __init__(self,
                 field_base,
                 xmin,
                 xmax,
                 nbin,
                 extra=None, **kw):
        """
        extra probably not useful except in a base clas where
        we have more variables available
        """
        sel=self.make_selections(field_base, xmin, xmax, nbin, extra=extra)

        super(FieldBinner,self).__init__(
            field_base,
            sel,
            **kw
        )

    def make_selections(self, field_base, xmin, xmax, nbin, extra=None):
        selections=[]

        binsize=(xmax-xmin)/float(nbin)

        for i in xrange(nbin):
            ixmin = xmin + i*binsize
            ixmax = ixmin + binsize

            sel='between(%s, %g, %g)' % (field_base,ixmin,ixmax)
            if extra is not None:
                sel += ' & (%s)' % extra

            print(sel)
            selections += [sel]
        return selections

class S2NBinner(FieldBinner):
    """
    Bin by S/N

    following variables are made available for selection
        s2n, Ts2n, T, Tpsf
    """
    def __init__(self,
                 xmin,
                 xmax,
                 nbin,
                 other_selection,
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        field_base='s2n'

        super(S2NBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            extra=other_selection,
            **kw
        )

    def do_selection(self, data, mcal_type, binnum):
        """
        cut on flags and s/n
        """

        gpsf, s2n, Ts2n, T, Terr, Tpsf = \
                self.get_selection_args(data, mcal_type)

        selection=self['selections'][binnum]

        cut_logic = eval(selection)

        return cut_logic, s2n

class LogS2NBinner(S2NBinner):
    """
    Bin by S/N

    following variables are made available for selection
        logs2n, Ts2n, T, Tpsf
    """
    def __init__(self,
                 xmin,
                 xmax,
                 nbin,
                 other_selection,
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        field_base='logs2n'

        # note calling super of super
        super(S2NBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            extra=other_selection,
            **kw
        )

    def get_selection_args(self, data, mcal_type):
        """
        convert s2nto log s2n
        """
        gpsf, s2n, Ts2n, T, Terr, Tpsf=super(LogS2NBinner,self).get_selection_args(data, mcal_type)
        logs2n=numpy.log10( s2n.clip(min=0.001))
        return logs2n, gpsf, s2n, Ts2n, T, Terr, Tpsf

    def do_selection(self, data, mcal_type, binnum):
        """
        cut on flags and s/n
        """

        logs2n, gpsf, s2n, Ts2n, T, Terr, Tpsf = \
                self.get_selection_args(data, mcal_type)

        selection=self['selections'][binnum]

        cut_logic = eval(selection)

        return cut_logic, logs2n

    def doplot(self, x, d, **kw):
        """
        plot the results

        parameters
        ----------
        x: x values
            Should be self.means
        d: dict
            result of running something like process_run or process_flist
        **kw:
            extra plotting keywords
        """
        from pyxtools import plot

        sh=d['shear']['shear']
        sherr=d['shear']['shear_err']

        kw['xlog']=True

        g=plot(10.0**x, sh[:,0], dy=sherr[:,0], color='blue', **kw)
        plot(10.0**x, sh[:,1], dy=sherr[:,1], color='red', g=g, **kw)

        return g


class PSFShapeBinner(FieldBinner):
    """
    Bin by psf shape

    following variables are made available for selection
        gpsf, s2n, Ts2n, T, Tpsf
    """
    def __init__(self,
                 xmin,
                 xmax,
                 nbin,
                 element, # 0 or 1
                 other_selection,
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        field_base='gpsf'

        self['element'] = element

        super(PSFShapeBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            extra=other_selection,
            **kw
        )

    def do_selection(self, data, mcal_type, binnum):
        """
        cut on flags and s/n
        """

        gpsf, s2n, Ts2n, T, Terr, Tpsf = \
                self.get_selection_args(data, mcal_type)

        selection=self['selections'][binnum]

        cut_logic = eval(selection)

        return cut_logic, gpsf

    def doplot(self, d, file=None, **kw):
        """
        plot the results

        parameters
        ----------
        d: dict
            result of running something like process_run or process_flist
        **kw:
            extra plotting keywords
        """
        from pyxtools import plot

        sh=d['shear']['shear']
        sherr=d['shear']['shear_err']

        plt=plot(
            d['means'], sh[:,0], dy=sherr[:,0],
            xlabel=r'$g_{psf}$',
            ylabel=r'$g$',
            color='blue',
            **kw)
        plot(
            d['means'], sh[:,1], dy=sherr[:,1],
            color='red', plt=plt, file=file,
            **kw)

        return plt


class TratioBinner(S2NBinner):
    """
    Bin by S/N

    following variables are made available for selection
        s2n, Ts2n, T, Tpsf, Tratio
    """
    def __init__(self,
                 xmin,
                 xmax,
                 nbin,
                 other_selection,
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        field_base='Tratio'

        # note calling super of super
        super(S2NBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            extra=other_selection,
            **kw
        )

    def get_selection_args(self, data, mcal_type):
        """
        convert s2nto log s2n
        """
        gpsf, s2n, Ts2n, T, Terr, Tpsf=super(TratioBinner,self).get_selection_args(data, mcal_type)

        Tratio=T/Tpsf
        return gpsf, s2n, Ts2n, T, Terr, Tpsf, Tratio

    def do_selection(self, data, mcal_type, binnum):
        """
        cut on flags and s/n
        """

        gpsf, s2n, Ts2n, T, Terr, Tpsf, Tratio = \
                self.get_selection_args(data, mcal_type)

        selection=self['selections'][binnum]

        cut_logic = eval(selection)

        return cut_logic, Tratio


def get_shear_struct(n):
    dt=[('shear','f8',2),
        ('shear_err','f8',2)]

    means = numpy.zeros(n, dtype=dt)
    return means


