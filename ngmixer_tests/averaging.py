"""

>>> numpy.percentile(pg[:,0], [5.0, 95.0])
array([-0.01466833,  0.02719123])

>>> numpy.percentile(pg[:,1], [5.0, 95.0])
array([-0.01281529,  0.02913015])

"""
from __future__ import print_function
import os
from glob import glob
import numpy
from numpy import newaxis, sqrt, zeros, abs, log10
from pprint import pprint
import fitsio
import ngmix
import time

import esutil as eu
from esutil.numpy_util import between

def get_run_flist(run):
    dir=os.environ['NGMIXER_OUTPUT_DIR']
    dir=os.path.join(dir, run,'output')
    pattern=os.path.join(dir, '*.fits')
    print("pattern:",pattern)
    flist=glob(pattern)
    flist.sort()

    print("found",len(flist),"files")
    return flist

def write_result(filename, res):
    print("writing:",filename)
    with fitsio.FITS(filename,'rw',clobber=True) as fits:
        fits.write(res['means'], extname='means')
        fits.write(res['shear'], extname='shear')
        fits.write(res['shear_nocorr'], extname='shear_nocorr')

def read_result(filename):
    print("writing:",filename)

    res={}
    with fitsio.FITS(filename) as fits:
        if 'means' in fits:
            res['means'] = fits['means'].read()
            res['shear'] = fits['shear'].read()
            res['shear_nocorr'] = fits['shear_nocorr'].read()
        else:
            res['means'] = fits[0].read()
            res['shear'] = fits[1].read()
            res['shear_nocorr'] = fits[2].read()

    return res

class AveragerBase(object):
    def __init__(self, step=0.01, chunksize=1000000, matchcat=None, weight_type=None):

        self.matchcat=matchcat

        self.chunksize = chunksize
        self.step = step

        self.element=None

        self.weight_type=weight_type

    def process_run(self, run, start=None, end=None, ntest=None):
        """
        run through all the collated files for the specified run
        """

        flist=get_run_flist(run)

        if start is not None:
            flist=flist[start:end]

        return self.process_flist(flist, ntest=ntest)

    def process_run_cache(self, run, ntest=None):
        """
        use the cached file for speed
        """
        fname=get_run_cache_file(run)

        if not os.path.exists(fname):
            cache_run(run)

        return self.process_flist([fname], ntest=ntest)

    def process_flist(self, flist, ntest=None):
        """
        run through a set of files, doing the sums for
        averages
        """
        tm0=time.time()

        chunksize=self.chunksize
        sums=None
        nf=len(flist)

        n=0
        for i,f in enumerate(flist):

            print("processing %d/%d: %s" % (i+1,nf,f))
            with fitsio.FITS(f) as fits:
                hdu=fits[1]

                nrows=hdu.get_nrows()
                nchunks = nrows//chunksize

                if (nrows % chunksize) > 0:
                    nchunks += 1

                self._set_do_Rpsf(hdu[0:10])

                beg=0
                for i in xrange(nchunks):
                    print("    chunk %d/%d" % (i+1,nchunks))

                    end=beg+chunksize

                    data = hdu[beg:end]

                    sums=self.do_sums(data, sums=sums)

                    beg = beg + chunksize
                    n += data.size

                    if ntest is not None and n >= ntest:
                        break

        # self.means is really a sum
        sums['mean_sum'] = self.means

        self.means_sums = self.means.copy()
        self.means /= sums['mean_sum']/sums['wsum']

        #sums['means_sums'] = self.means_sums
        #sums['means'] = self.means_sums

        means=self.get_shears(sums)
        #means['means_sums'] = sums['means_sums']
        #means['means'] = sums['means_sums']/sums['wsum']

        eu.misc.ptime(time.time()-tm0)
        return means

    def _set_do_Rpsf(self, data):
        names=data.dtype.names
        if 'mcal_g_1m_psf' in names:
            self.do_Rpsf =True
        else:
            self.do_Rpsf =False


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

            c          = (Rpsf + Rpsf_sel)*gpsf_mean
            shear      = (gmean-c)/(R+Rsel)
            shear_err  = gerr/(R+Rsel)

            shear_nocorr = gmean.copy()
            shear_nocorr_err = gerr.copy()

            sh['shear'][i] = shear
            sh['shear_err'][i] = shear_err

            sh_nocorr['shear'][i] = shear_nocorr
            sh_nocorr['shear_err'][i] = shear_nocorr_err

        means['shear'] = sh
        means['shear_nocorr'] = sh_nocorr

        means['means'] = sums['mean_sum']/sums['wsum']

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

        R[:,0] = (g1p - g1m)*factor
        R[:,1] = (g2p - g2m)*factor

        if self.do_Rpsf:
            g1p_psf = sums['g_1p_psf'][:,0]*winv
            g1m_psf = sums['g_1m_psf'][:,0]*winv
            g2p_psf = sums['g_2p_psf'][:,1]*winv
            g2m_psf = sums['g_2m_psf'][:,1]*winv

            Rpsf[:,0] = (g1p_psf - g1m_psf)*factor
            Rpsf[:,1] = (g2p_psf - g2m_psf)*factor

            s_g1p_psf = sums['s_g_1p_psf'][:,0]/sums['s_wsum_1p_psf']
            s_g1m_psf = sums['s_g_1m_psf'][:,0]/sums['s_wsum_1m_psf']
            s_g2p_psf = sums['s_g_2p_psf'][:,1]/sums['s_wsum_2p_psf']
            s_g2m_psf = sums['s_g_2m_psf'][:,1]/sums['s_wsum_2m_psf']

            Rpsf_sel[:,0] = (s_g1p_psf - s_g1m_psf)*factor
            Rpsf_sel[:,1] = (s_g2p_psf - s_g2m_psf)*factor

        print("R:",R)
        print("Rpsf:",Rpsf)

        # selection terms
        s_g1p = sums['s_g_1p'][:,0]/sums['s_wsum_1p']
        s_g1m = sums['s_g_1m'][:,0]/sums['s_wsum_1m']
        s_g2p = sums['s_g_2p'][:,1]/sums['s_wsum_2p']
        s_g2m = sums['s_g_2m'][:,1]/sums['s_wsum_2m']

        Rsel[:,0] = (s_g1p - s_g1m)*factor
        Rsel[:,1] = (s_g2p - s_g2m)*factor

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
        #logic = (data['flags'] == 0)


        if len(data['nimage_use'].shape) > 1:
            nimages=data['nimage_use'].sum(axis=1)
        else:
            nimages=data['nimage_use']
        logic = (data['flags'] == 0)
        #logic = (data['flags'] == 0) & (nimages >= 6)
        #logic = (data['flags'] == 0) & (data['box_size']==32)
        #logic = (data['flags'] == 0) & (data['box_size']==48)
        #logic = (data['flags'] == 0) & (data['box_size'] <= 48)

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

        s2n, Ts2n, T, Terr, Tpsf, gpsf, Tratio = \
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
        gpsf = data['mcal_gpsf']

        T = data['mcal_T_r%s' % tstr]

        Terr= data['mcal_T_err%s' % tstr]

        Ts2n = T/Terr

        s2n_field = 'mcal_s2n_r%s' % tstr
        s2n = data[s2n_field]

        Tratio=T/Tpsf

        return s2n, Ts2n, T, Terr, Tpsf, gpsf, Tratio

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

            ('mean_sum','f8'),
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

        names=data.dtype.names

        if 'mcal_g_1m_psf' in names:
            self.do_Rpsf =True
        else:
            self.do_Rpsf =False

        if sums is None:
            sums=self._get_sums_struct(self.nbin)

        # ideally this selection should not depend on any sheared parameters
        sanity_logic = self.do_sanity_cuts(data)
        sane_data = data[sanity_logic]

        if sane_data.size == 0:
            print("    none passed, skipping")
            return sums

        # first, selecting and binning by unsheared data
        if self.selection is not None:
            logic = self.do_selection(sane_data, 'noshear')
            w,=numpy.where(logic)
            print("    finally kept %d/%d after "
                  "extra cuts" % (w.size, data.size))
            selected_data=sane_data[w]
        else:
            selected_data=sane_data

        if selected_data.size == 0:
            print("    none passed, skipping")
            return sums

        h, rev, fvalues = self.bin_data_by_type(selected_data, 'noshear')
        if h is None:
            return sums

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
                sums['gsq'][binnum]    += \
                        (wa**2 * bdata['mcal_g']**2).sum(axis=0)
                sums['gwsq'][binnum]   += \
                        (wa**2 * bdata['mcal_g']).sum(axis=0)

                # now the response terms, also based on selections/weights
                # from unsheared data

                for type in ngmix.metacal.METACAL_TYPES:
                    if type=='noshear':
                        continue

                    tstr=self.get_type_string(type)

                    mcalname='mcal_g%s' % tstr
                    if mcalname not in names:
                        #print("skipping:",type)
                        continue

                    sumname='g%s' % tstr

                    if mcalname in bdata.dtype.names:
                        sums[sumname][binnum] += \
                                (wa*bdata[mcalname]).sum(axis=0)


        # now, selecting and binning by sheared data
        for type in ngmix.metacal.METACAL_TYPES:
            if type=='noshear':
                continue

            # just make sure we have this field
            tstr=self.get_type_string(type)
            mcalname='mcal_g%s' % tstr
            if mcalname not in names:
                #print("skipping:",type)
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

        try:
            h,rev = eu.stat.histogram(
                fdata,
                min=self.xmin,
                max=self.xmax,
                nbin=self.nbin,
                rev=True,
            )
        except ValueError:
            return None,None,None

        return h, rev, fdata


    def bin_data_by_type(self, data, mcal_type):

        tstr=self.get_type_string(mcal_type)
        field='%s%s' % (self.field_base, tstr)
        if field not in data.dtype.names:
            field=self.field_base
            #print("using base name:",field)

        return self.bin_data(data, field, element=self.element)

    def _extract_xvals(self, d):
        return d['means']

    def doplot(self, d, xlabel=None, xlog=False,
               ymin=-0.0029, ymax=0.0029,
               xmin=None,xmax=None,
               nocorr=False,
               fitlines=False,
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
        from pyx import graph, deco, style
        from pyx.graph import axis

        if xlabel is None:
            xlabel='x'

        xdensity=kw.get('xdensity',1.5)
        ydensity=kw.get('ydensity',1.5)

        red=pyxtools.colors('red')
        blue=pyxtools.colors('blue')

        xvals=self._extract_xvals(d)

        if nocorr:
            sh=d['shear_nocorr']['shear']
            sherr=d['shear_nocorr']['shear_err']
        else:
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

        key=graph.key.key(pos='bl')
        g = graph.graphxy(
            width=8,
            ratio=1.2,
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
            symbolattrs=[red,deco.filled([red])],
            size=0.1,
        )
        symbol2=graph.style.symbol(
            symbol=graph.style.symbol.triangle,
            symbolattrs=[blue,deco.filled([blue])],
            size=0.1,
        )

        g.plot(g1values,[symbol1, graph.style.errorbar(errorbarattrs=[red])])
        g.plot(g2values,[symbol2, graph.style.errorbar(errorbarattrs=[blue])])

        use_errors=True
        if fitlines and not use_errors:
            res1=fitline(xvals,sh[:,0])
            res2=fitline(xvals,sh[:,1])


            fmt='(%(slope).3g +/- %(slope_err).3g) x + (%(offset).3g +/- %(offset_err).3g)'
            print("line1: "+fmt % res1)
            print("line2: "+fmt % res2)

            if self.element is not None:
                tit1=r'$g_1=%.2g ~g^{psf}_%d + %.2g$' % (res1['slope'],self.element+1,res1['offset'])
                tit2=r'$g_2=%.2g ~g^{psf}_%d + %.2g$' % (res2['slope'],self.element+1,res2['offset'])
            else:
                tit1=r'$g_1=%.2g ~x + %.2g$' % (res1['slope'],res1['offset'])
                tit2=r'$g_2=%.2g ~x + %.2g$' % (res1['slope'],res1['offset'])

            c1=graph.data.function(
                "y(x)=%g*x + %g" % tuple( (res1['slope'], res1['offset'])),
                title=tit1,
                min=xvals[0], max=xvals[-1],
            )
            c2=graph.data.function(
                "y(x)=%g*x + %g" % tuple( (res2['slope'], res2['offset'])),
                title=tit2,
                min=xvals[0], max=xvals[-1],
            )

            styles1=[
                graph.style.line([red, style.linestyle.solid]),
            ]
            styles2=[
                graph.style.line([blue, style.linestyle.dashed]),
            ]

            g.plot(c1,styles1)
            g.plot(c2,styles2)

        if fitlines and use_errors:
            # old code
            import fitting
            l1=fitting.fit_line(xvals,sh[:,0],yerr=sherr[:,0])
            l2=fitting.fit_line(xvals,sh[:,1],yerr=sherr[:,1])
            res1=l1.get_result()
            res2=l2.get_result()

            pars1,err1=res1['pars'],res1['perr']
            pars2,err2=res2['pars'],res2['perr']

            fmt='(%.3g +/- %.3g) x + (%.3g +/- %.3g)'
            print("line1: "+fmt % (pars1[0],err1[0],pars1[1],err1[1]))
            print("line2: "+fmt % (pars2[0],err2[0],pars2[1],err2[1]))

            if self.element is not None:
                tit1=r'$g_1=%.2g ~g^{psf}_%d + %.2g$' % (pars1[0],self.element+1,pars1[1])
                tit2=r'$g_2=%.2g ~g^{psf}_%d + %.2g$' % (pars2[0],self.element+1,pars2[1])
            else:
                tit1=r'$g_1=%.2g ~x + %.2g$' % (pars1[0],pars1[1])
                tit2=r'$g_2=%.2g ~x + %.2g$' % (pars2[0],pars2[1])
            c1=graph.data.function(
                "y(x)=%g*x + %g" % tuple(pars1),
                title=tit1,
                min=xvals[0], max=xvals[-1],
            )
            c2=graph.data.function(
                "y(x)=%g*x + %g" % tuple(pars2),
                title=tit2,
                min=xvals[0], max=xvals[-1],
            )

            styles1=[
                graph.style.line([red, style.linestyle.solid]),
            ]
            styles2=[
                graph.style.line([blue, style.linestyle.dashed]),
            ]

            g.plot(c1,styles1)
            g.plot(c2,styles2)


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
                 field_base='mcal_gpsf',
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        #field_base='psfrec_g'
        #field_base='mcal_gpsf'


        super(PSFShapeBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            selection=other_selection,
            **kw
        )

        self.element = element

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
        kw['fitlines']=True
        super(PSFShapeBinner,self).doplot(d, **kw)


class PositionBinner(FieldBinner):
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
                 field_base='mcal_pars',
                 **kw):

        super(PositionBinner,self).__init__(
            field_base,
            xmin,
            xmax,
            nbin,
            selection=other_selection,
            **kw
        )

        self.element = element

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

        kw['xlabel']=r'$cen_{%s}$' % (1+self.element,)
        kw['fitlines']=False
        super(PositionBinner,self).doplot(d, **kw)




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


class PSFTBinner(FieldBinner):
    """
    Bin by psf shape. We use the psf shape without metacal,
    since we often symmetrize the metacal psf
    """
    def __init__(self,
                 xmin,
                 xmax,
                 nbin,
                 other_selection,
                 field_base='mcal_Tpsf',
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        #field_base='psfrec_g'
        #field_base='mcal_gpsf'


        super(PSFTBinner,self).__init__(
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

        kw['xlabel']=r'$T^{psf}$'
        kw['fitlines']=False
        super(PSFTBinner,self).doplot(d, **kw)

class TBinner(FieldBinner):
    """
    Bin by psf shape. We use the psf shape without metacal,
    since we often symmetrize the metacal psf
    """
    def __init__(self,
                 xmin,
                 xmax,
                 nbin,
                 other_selection,
                 field_base='mcal_T_r',
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        #field_base='psfrec_g'
        #field_base='mcal_gpsf'


        super(TBinner,self).__init__(
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

        kw['xlabel']=r'$T$'
        kw['fitlines']=False
        super(TBinner,self).doplot(d, **kw)

class MaskFracBinner(FieldBinner):
    """
    Bin by psf shape. We use the psf shape without metacal,
    since we often symmetrize the metacal psf
    """
    def __init__(self,
                 xmin,
                 xmax,
                 nbin,
                 other_selection,
                 field_base='mask_frac',
                 **kw):

        # note it is abbreviated; ok since in do_selection
        # we used the abbreviated for. Also s2n rather than
        # full
        #field_base='psfrec_g'
        #field_base='mcal_gpsf'


        super(MaskFracBinner,self).__init__(
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

        kw['xlabel']='masked fraction'
        kw['fitlines']=False
        super(MaskFracBinner,self).doplot(d, **kw)



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
    columns= [
        'id',
        'nimage_tot',
        'flags',
        'box_size',
        'nimage_use',
        'mask_frac',
        'psfrec_T',
        'psfrec_g',

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
    ]


    """
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
    ]
    """

    first=True

    """
    tmp=fitsio.read(flist[0], rows=[0])
    for mod in ['gauss','exp']:
        n='%s_pars' % mod
        if n in tmp.dtype.names:
            print("adding",mod,"columns")
            for n in ['pars','s2n_r','T_r','T_err']:
                name='%s_%s' % (mod, n)
                columns += [name]
    """


    pprint(columns)

    nf=len(flist)
    print("cacheing to:",filename)
    with fitsio.FITS(filename,'rw',clobber=True) as fits:
        for i,f in enumerate(flist):
            print("%d/%d %s" % (i+1,nf,f))

            try:
                data=fitsio.read(f, columns=columns)

                if first:
                    fits.write(data)
                    first=False
                else:
                    fits[-1].append(data)
            except IOError as err:
                print(str(err))

# quick line fit pulled from great3-public code
def _calculateSvalues(xarr, yarr, sigma2=1.):
    """Calculates the intermediate S values required for basic linear regression.

    See, e.g., Numerical Recipes (Press et al 1992) Section 15.2.
    """
    if len(xarr) != len(yarr):
        raise ValueError("Input xarr and yarr differ in length!")
    if len(xarr) <= 1:
        raise ValueError("Input arrays must have 2 or more values elements.")

    S = len(xarr) / sigma2
    Sx = numpy.sum(xarr / sigma2)
    Sy = numpy.sum(yarr / sigma2)
    Sxx = numpy.sum(xarr * xarr / sigma2)
    Sxy = numpy.sum(xarr * yarr / sigma2)
    return (S, Sx, Sy, Sxx, Sxy)

def fitline(xarr, yarr):
    """Fit a line y = a + b * x to input x and y arrays by least squares.

    Returns the tuple (a, b, Var(a), Cov(a, b), Var(b)), after performing an internal estimate of
    measurement errors from the best-fitting model residuals.

    See Numerical Recipes (Press et al 1992; Section 15.2) for a clear description of the details
    of this simple regression.
    """
    # Get the S values (use default sigma2, best fit a and b still valid for stationary data)
    S, Sx, Sy, Sxx, Sxy = _calculateSvalues(xarr, yarr)
    # Get the best fit a and b
    Del = S * Sxx - Sx * Sx
    a = (Sxx * Sy - Sx * Sxy) / Del
    b = (S * Sxy - Sx * Sy) / Del
    # Use these to estimate the sigma^2 by residuals from the best-fitting model
    ymodel = a + b * xarr
    sigma2 = numpy.mean((yarr - ymodel)**2)
    # And use this to get model parameter error estimates
    var_a  = sigma2 * Sxx / Del
    cov_ab = - sigma2 * Sx / Del
    var_b  = sigma2 * S / Del

    a_err = numpy.sqrt(var_a)
    b_err = numpy.sqrt(var_b)
    return {'offset':a,
            'offset_err':a_err,
            'slope':b,
            'slope_err':b_err,
            'cov':cov_ab}

    #return a, a_err, b, b_err, cov_ab


