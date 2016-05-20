from __future__ import print_function
import os
from glob import glob
import numpy
from numpy import newaxis, sqrt, zeros
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

class AveragerBase(object):
    def __init__(self, step=0.01, chunksize=1000000, matchcat=None, weight_type=None):

        self.matchcat=matchcat

        self.chunksize = chunksize
        self.step = step

        self.element=None

        self.weight_type=weight_type

    def process_run(self, run):
        """
        run through all the collated files for the specified run
        """

        flist=get_run_flist(run)
        return self.process_flist(flist)

    def process_run_cache(self, run):
        """
        use the cached file for speed
        """
        fname=get_run_cache_file(run)

        if not os.path.exists(fname):
            cache_run(run)

        return self.process_flist([fname])

    def process_flist(self, flist):
        """
        run through a set of files, doing the sums for
        averages
        """
        chunksize=self.chunksize
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
        raise NotImplementedError("implement in a base class")

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

            wsum      = sums['wsum'][i]
            gmean     = means['g'][i]
            gpsf_mean = means['gpsf'][i]


            R         = means['R'][i]
            Rsel      = means['Rsel'][i]
            Rpsf      = means['Rpsf'][i]
            Rpsf_sel  = means['Rpsf_sel'][i]
            print("Rpsf:",Rpsf)
            print("Rpsf_sel:",Rpsf_sel)

            # terms for errors on weighted mean
            err2sum = (       sums['gsq'][i] 
                        - 2.0*sums['gwsq'][i]*gmean
                        +     sums['wsqsum'][i]*gmean**2 )

            gerr = sqrt(err2sum)/wsum

            c        = (Rpsf + Rpsf_sel)*gpsf_mean
            c_nocorr = Rpsf*gpsf_mean

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


    def average_sums(self, sums):
        """
        divide by sum of weights and get g for each field

        Also average the responses over all data
        """

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
        R = 0*g
        Rpsf = 0*g
        Rsel = 0*g
        Rpsf_sel = 0*g

        factor = 1.0/(2.0*self.step)

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
            R=R,       # the following averaged over bins
            Rpsf=Rpsf,
            Rsel=Rsel,
            Rpsf_sel=Rpsf_sel,
        )
        return res

    def do_sanity_cuts(self, data):
        """
        the s2n cuts are for the bug not propagating flags
        """
        logic = (data['flags'] == 0)

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


    def do_selection(self, data, mcal_type):
        """
        cut on flags and s/n
        """

        s2n, Ts2n, T, Terr, Tpsf, Tratio = \
                self.get_selection_args(data, mcal_type)

        cut_logic = eval(self.selection)
        return cut_logic

    def get_selection_args(self, data, mcal_type):
        """
        get the default set
        """
        tstr=self.get_type_string(mcal_type)

        # this is the one for selections, not for Rpsf*gpsf
        Tpsf = data['mcal_Tpsf']

        T = data['mcal_T_r%s' % tstr]

        Terr= data['mcal_T_err%s' % tstr]

        Ts2n = T/Terr

        s2n_field = 'mcal_s2n_r%s' % tstr
        s2n = data[s2n_field]

        Tratio=T/Tpsf

        return s2n, Ts2n, T, Terr, Tpsf, Tratio

    def _get_weights(self, data, mcal_type):
        wtype=self.weight_type
        if wtype is None:
            weights=numpy.ones(data.size)
        elif wtype=='err':
            # use the shape error
            SN2=0.21**2

            tstr=self.get_type_string(mcal_type)
            gcov = data['mcal_g_cov%s' % tstr]
            weights = 1.0/(2*SN2 + gcov[:,0,0] + gcov[:,1,1])
        else:
            raise ValueError("bad weight type: %s" % wtype)

        wa = weights[:,newaxis]
        return weights, wa

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
            ('num','i8'),
            ('wsum','f8'),
            ('g','f8',2),
            ('gpsf','f8',2),

            # terms for errors
            ('wsqsum','f8'),
            ('gsq','f8',2),
            ('gwsq','f8',2),

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



class FieldBinner(AveragerBase):
    def __init__(self,
                 field_base,
                 xmin,
                 xmax,
                 nbin,
                 selection=None, **kw):

        self.field_base=field_base
        self.xmin=xmin
        self.xmax=xmax
        self.nbin=nbin
        self.selection=selection

        self.means=zeros(nbin)

        super(FieldBinner,self).__init__(**kw)

    def do_sums(self, data, sums=None):
        """
        do all the sums, no binning for base class
        """

        if sums is None:
            sums=self._get_sums_struct(self.nbin)

        # ideally this selection should not depend on any sheared parameters
        sanity_logic = self.do_sanity_cuts(data)
        sane_data = data[sanity_logic]

        # first, selecting and binning by unsheared data
        if self.selection is not None:
            logic = self.do_selection(sane_data, 'noshear')
            w,=numpy.where(logic)
            print("    finally kept %d/%d after extra cuts" % (w.size, data.size))
            selected_data=sane_data[w]
        else:
            selected_data=sane_data

        h, rev, fvalues = self.bin_data_by_type(selected_data, 'noshear')
        assert h.size==self.nbin,"histogram size %d wrong" % h.size

        for binnum in xrange(h.size):
            if rev[binnum] != rev[binnum+1]:

                w = rev[ rev[binnum]:rev[binnum+1] ]

                bdata = selected_data[w]

                # weights based on non-sheared quantities
                wts,wa=self._get_weights(bdata,'noshear')

                sums['num'][binnum]  += w.size
                sums['wsum'][binnum] += wts.sum()
                sums['g'][binnum]    += (wa*bdata['mcal_g']).sum(axis=0)
                sums['gpsf'][binnum] += (wa*bdata['mcal_gpsf']).sum(axis=0)
                self.means[binnum]   += (wts*fvalues[w]).sum(axis=0)

                # terms for errors
                sums['wsqsum'][binnum] += (wts**2).sum()
                sums['gsq'][binnum]    += (wa**2 * bdata['mcal_g']**2).sum(axis=0)
                sums['gwsq'][binnum]   += (wa**2 * bdata['mcal_g']).sum(axis=0)

                # now the response terms, also based on selections/weights from
                # unsheared data
                for type in ngmix.metacal.METACAL_TYPES:
                    if type=='noshear':
                        continue

                    tstr=self.get_type_string(type)

                    mcalname='mcal_g%s' % tstr
                    sumname='g%s' % tstr

                    if mcalname in bdata.dtype.names:
                        sums[sumname][binnum] += (wa*bdata[mcalname]).sum(axis=0)


        # now, selecting and binning by sheared data
        for type in ngmix.metacal.METACAL_TYPES:
            if type=='noshear':
                continue

            if self.selection is not None:
                logic = self.do_selection(sane_data, type)
                selected_data=sane_data[logic]
            else:
                selected_data=sane_data

            h, rev, fvalues = self.bin_data_by_type(selected_data, type)
            assert h.size==self.nbin,"histogram size %d wrong" % h.size

            for binnum in xrange(h.size):
                if rev[binnum] != rev[binnum+1]:

                    w = rev[ rev[binnum]:rev[binnum+1] ]

                    bdata = selected_data[w]

                    wts,wa=self._get_weights(bdata,type)


                    # mean of unsheared g after selection/weighting based on
                    # sheared parameters
                    wsumname = 's_wsum_%s' % type
                    sumname = 's_g_%s' % type
                    sums[wsumname][binnum] += wts.sum()
                    sums[sumname][binnum]  += (wa*bdata['mcal_g']).sum(axis=0)

        return sums

    def bin_data(self, data, field, element=None):

        if element is not None:
            fdata=data[field][:,self.element]
        else:
            fdata=data[field]

        h,rev = eu.stat.histogram(
            fdata,
            min=self.xmin,
            max=self.xmax,
            nbin=self.nbin,
            rev=True,
        )
        return h, rev, fdata


    def bin_data_by_type(self, data, mcal_type):

        tstr=self.get_type_string(mcal_type)
        field='%s%s' % (self.field_base, tstr)
        if field not in data.dtype.names:
            field=self.field_base
            print("using base name:",field)

        return self.bin_data(data, field, element=self.element)

    def _extract_xvals(self, d):
        return d['means']

    def doplot(self, d, xlabel=None, xlog=False,
               ymin=-0.0049, ymax=0.0049,
               xmin=None,xmax=None,
               **kw):
        """
        plot the results

        parameters
        ----------
        d: dict
            result of running something like process_run or process_flist
        **kw:
            extra plotting keywords
        """
        import pyxtools
        from pyx import graph, deco
        from pyx.graph import axis

        if xlabel is None:
            xlabel='x'

        ydensity=kw.get('ydensity',1.5)
        xdensity=kw.get('ydensity',1.5)

        red=pyxtools.colors('red')
        blue=pyxtools.colors('blue')

        xvals=self._extract_xvals(d)
        sh=d['shear']['shear']
        sherr=d['shear']['shear_err']


        if xlog:
            xcls=axis.log
        else:
            xcls=axis.lin

        xaxis=xcls(
            title=xlabel,
            density=xdensity,
            min=xmin,
            max=xmax,
        )
        yaxis=axis.lin(
            title=r"$\langle g \rangle$",
            density=ydensity,
            min=ymin,
            max=ymax,
        )

        key=graph.key.key(pos='tr')
        g = graph.graphxy(
            width=8,
            key=key,
            x=xaxis,
            y=yaxis,
        )

        c=graph.data.function(
            "y(x)=0",
            title=None,
            min=xvals[0], max=xvals[-1],
        )

        g.plot(c)

        g1values=graph.data.values(
            x=list(xvals),
            y=list(sh[:,0]),
            dy=list(sherr[:,0]),
            title=r'$g_1$',
        )
        g2values=graph.data.values(
            x=list(xvals),
            y=list(sh[:,1]),
            dy=list(sherr[:,1]),
            title=r'$g_2$',
        )

        symbol1=graph.style.symbol(
            symbol=graph.style.symbol.circle,
            symbolattrs=[blue,deco.filled([blue])],
            size=0.1,
        )
        symbol2=graph.style.symbol(
            symbol=graph.style.symbol.triangle,
            symbolattrs=[red,deco.filled([red])],
            size=0.1,
        )

        g.plot(g1values,[symbol1, graph.style.errorbar(errorbarattrs=[blue])])
        g.plot(g2values,[symbol2, graph.style.errorbar(errorbarattrs=[red])])

        if 'file' in kw:
            pyxtools.write(g, kw['file'], dpi=200)

        return g



class LogS2NBinner(FieldBinner):
    """
    Bin by log10( S/N )
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
        field_base='mcal_s2n_r'

        super(LogS2NBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            selection=other_selection,
            **kw
        )

    def bin_data(self, data, field, **kw):

        s2n=data[field]

        logs2n = numpy.log10( s2n.clip(min=0.001) )

        h,rev = eu.stat.histogram(
            logs2n,
            min=numpy.log10(self.xmin),
            max=numpy.log10(self.xmax),
            nbin=self.nbin,
            rev=True,
        )
        return h, rev, s2n

    def doplot(self, d, **kw):
        """
        plot the results

        parameters
        ----------
        d: dict
            result of running something like process_run or process_flist
        **kw:
            extra plotting keywords
        """

        kw['xlabel']=r'$S/N$'
        kw['xlog']=True
        super(LogS2NBinner,self).doplot(d, **kw)

class PSFShapeBinner(FieldBinner):
    """
    Bin by psf shape. We use the psf shape without metacal,
    since we often symmetrize the metacal psf
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

        self.element = element

        super(PSFShapeBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            selection=other_selection,
            **kw
        )

    def doplot(self, d, **kw):
        """
        plot the results

        parameters
        ----------
        d: dict
            result of running something like process_run or process_flist
        **kw:
            extra plotting keywords
        """

        kw['xlabel']=r'$g^{psf}_{%s}$' % (1+self.element,)
        super(PSFShapeBinner,self).doplot(d, **kw)

class TratioBinner(FieldBinner):
    """
    Bin by T/Tpsf
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

        super(TratioBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            selection=other_selection,
            **kw
        )


    def bin_data_by_type(self, data, mcal_type):

        tstr=self.get_type_string(mcal_type)
        Tfield='mcal_T_r%s' % tstr
        Tpsf_field='mcal_Tpsf'

        Tratio = data[Tfield]/data[Tpsf_field]

        h,rev = eu.stat.histogram(
            Tratio,
            min=self.xmin,
            max=self.xmax,
            nbin=self.nbin,
            rev=True,
        )
        return h, rev, Tratio

    def doplot(self, d, **kw):
        """
        plot the results

        parameters
        ----------
        d: dict
            result of running something like process_run or process_flist
        **kw:
            extra plotting keywords
        """

        kw['xlabel']=r'$T/T^{psf}$'
        super(TratioBinner,self).doplot(d, **kw)



def get_shear_struct(n):
    dt=[('shear','f8',2),
        ('shear_err','f8',2)]

    means = numpy.zeros(n, dtype=dt)
    return means

def get_run_cache_file(run):
    fname='$TMPDIR/%s-cache.fits' % run
    fname=os.path.expandvars(fname)
    return fname

def cache_run(run):
    fname=get_run_cache_file(run)
    flist=get_run_flist(run)

    cache_flist(flist, fname)

def cache_flist(flist, filename):
    columns= (
        'id',
        'nimage_tot',
        'flags',
        'box_size',
        'nimage_use',
        'mask_frac',
        'psfrec_T',
        'psfrec_g',
        'mcal_flags',
        'mcal_g_1p',
        'mcal_g_cov_1p',
        'mcal_pars_1p',
        'mcal_T_1p',
        'mcal_T_err_1p',
        'mcal_T_r_1p',
        'mcal_s2n_r_1p',
        'mcal_g_1m',
        'mcal_g_cov_1m',
        'mcal_pars_1m',
        'mcal_T_1m',
        'mcal_T_err_1m',
        'mcal_T_r_1m',
        'mcal_s2n_r_1m',
        'mcal_g_2p',
        'mcal_g_cov_2p',
        'mcal_pars_2p',
        'mcal_T_2p',
        'mcal_T_err_2p',
        'mcal_T_r_2p',
        'mcal_s2n_r_2p',
        'mcal_g_2m',
        'mcal_g_cov_2m',
        'mcal_pars_2m',
        'mcal_T_2m',
        'mcal_T_err_2m',
        'mcal_T_r_2m',
        'mcal_s2n_r_2m',
        'mcal_g_1p_psf',
        'mcal_g_cov_1p_psf',
        'mcal_pars_1p_psf',
        'mcal_T_1p_psf',
        'mcal_T_err_1p_psf',
        'mcal_T_r_1p_psf',
        'mcal_s2n_r_1p_psf',
        'mcal_g_1m_psf',
        'mcal_g_cov_1m_psf',
        'mcal_pars_1m_psf',
        'mcal_T_1m_psf',
        'mcal_T_err_1m_psf',
        'mcal_T_r_1m_psf',
        'mcal_s2n_r_1m_psf',
        'mcal_g_2p_psf',
        'mcal_g_cov_2p_psf',
        'mcal_pars_2p_psf',
        'mcal_T_2p_psf',
        'mcal_T_err_2p_psf',
        'mcal_T_r_2p_psf',
        'mcal_s2n_r_2p_psf',
        'mcal_g_2m_psf',
        'mcal_g_cov_2m_psf',
        'mcal_pars_2m_psf',
        'mcal_T_2m_psf',
        'mcal_T_err_2m_psf',
        'mcal_T_r_2m_psf',
        'mcal_s2n_r_2m_psf',
        'mcal_g',
        'mcal_g_cov',
        'mcal_pars',
        'mcal_pars_cov',
        'mcal_gpsf',
        'mcal_Tpsf',
        'mcal_T',
        'mcal_T_err',
        'mcal_T_r',
        'mcal_s2n_r',
    )

    nf=len(flist)
    print("cacheing to:",filename)
    with fitsio.FITS(filename,'rw',clobber=True) as fits:
        for i,f in enumerate(flist):
            print("%d/%d %s" % (i+1,nf,f))
            data=fitsio.read(f, columns=columns)

            if i==0:
                fits.write(data)
            else:
                fits[-1].append(data)
