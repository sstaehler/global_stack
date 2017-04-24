#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Use instaseis to generate a global stack
:copyright:
    Simon Stähler
:license:
    GNU General Public License, Version 3
'''
import matplotlib.pyplot as plt
import numpy as np
import instaseis
import pickle
from tqdm import tqdm

__all__ = ["global_stack",
           "load"]


class global_stack():

    def __init__(self, db_name, depth_in_m=0.0, ndist=1024, dt=1.0):
        """
        Create global stack from an Instaseis database

        Keywords:
        :type  db_name: string
        :param db_name: filename for the instaseis database
        :type depth_in_m: float, optional
        :param depth_in_m: source depth
        :type ndist: int, optional
        :param ndist: Number of distances at which to calculate seismogram. Is
                      equivalent to X-resolution of resulting stack
        :type dt: float, optional
        :param dt: Time step of seismograms. Y-resolutions is approximately
                   (length of database / dt)
        """
        self.db_name = db_name
        self.depth_in_m = depth_in_m
        self.ndist = ndist
        self.dt = dt

        self.stack_R, self.stack_T, self.stack_Z, \
            self.stats, self.db_info = self._calc()

    def __str__(self):
        string = 'db_name:             %s\n' % self.db_name
        string += 'depth:               %d m\n' % self.depth_in_m
        string += 'number of distances: %d\n' % self.ndist
        string += 'dt:                  %5.3f\n' % self.dt
        string += 'stack size:          %dx%d' % (self.stack_R.shape[0],
                                                  self.stack_R.shape[2])

        return string

    def _calc(self):
        db = instaseis.open_db(self.db_name)

        nazi = 8

        dt = min([self.dt, db.info.dt])

        lats = np.linspace(start=-90., stop=90., num=self.ndist)
        lons = np.linspace(start=-180, stop=180., num=nazi, endpoint=False)

        src = instaseis.Source(latitude=90.0,
                               longitude=0.0,
                               depth_in_m=self.depth_in_m,
                               m_rr=-1.670000e+28 / 1e7,
                               m_tt=3.820000e+27 / 1e7,
                               m_pp=1.280000e+28 / 1e7,
                               m_rt=-7.840000e+27 / 1e7,
                               m_rp=-3.570000e+28 / 1e7,
                               m_tp=1.550000e+27 / 1e7)

        npts = _get_npts(db, dt)

        stack_R = np.zeros(shape=(self.ndist, nazi, npts))
        stack_T = np.zeros(shape=(self.ndist, nazi, npts))
        stack_Z = np.zeros(shape=(self.ndist, nazi, npts))

        for ilat in tqdm(range(0, int(self.ndist))):
            lat = lats[ilat]

            for ilon in range(0, nazi):
                lon = lons[ilon]
                rec = instaseis.Receiver(latitude=lat, longitude=lon,
                                         network="AB", station="%d" % lon)

                st = db.get_seismograms(src, rec, components='RTZ',
                                        kind='velocity', dt=dt,
                                        remove_source_shift=True)

                stack_R[ilat, ilon, :] = st.select(channel='*R')[0].data
                stack_T[ilat, ilon, :] = st.select(channel='*T')[0].data
                stack_Z[ilat, ilon, :] = st.select(channel='*Z')[0].data

        return stack_R, stack_T, stack_Z, st[0].stats, db.info

    def save(self, fnam):
        """
        Save global stack to pickle on disk

        Keywords:
        :type  fnam: string
        :param fnam: filename to save stack into
        """
        pickle.dump(self, open(fnam, 'wb'))

    def plot(self, waterlevel=1e-5, postfac=2,
             fnam=None, dpi=96, figsize=(10, 10),
             fontsize=16, ax=None, **kwargs):
        """
        Plot global stack

        Keywords:
        :type  waterlevel: float or tuple of float
        :param waterlevel: minimum value of plot
                           If tuple, the order is Z R T
        :type postfac: float or tuple of float
        :param postfac: Factor with which stack is multiplied after
                        normalization. Saturates strong values.
                        If tuple, the order is Z R T
        :type fnam: string, optional
        :param fnam: Filename in which stack is saved. If not given, the stack
                     is shown on screen. Choose type by extension!
        :type dpi: int
        :param dpi: Resolution of file to write.
        :type figsize: tuple of float
        :param figsize: Size of figure in centimeters
        :type fontsize: float
        :param fontsize: Font size for axis labels
        :type ax: matplotlib.axes.Axes
        :param ax: matplotlib axis to draw into. If None, a new figure is 
                   created

        kwargs are legal :class:`~matplotlib.axes.Axes` kwargs

        returns :class:`~matplotlib.figure.Figure` object
        """

        npts = self.stack_R.shape[0]
        ndist = self.stack_R.shape[2]

        stack = np.zeros(shape=(ndist, npts, 3))
        
        if type(waterlevel) == tuple:
            wl = waterlevel
        else:
            wl = (waterlevel, waterlevel, waterlevel)
 
        if type(postfac) == tuple:
            pf = postfac
        else:
            pf = (postfac, postfac, postfac)
            
        stack[:, :, 0] = np.sum(abs(self.stack_T), axis=1).T + wl[2]
        stack[:, :, 1] = np.sum(abs(self.stack_R), axis=1).T + wl[1]
        stack[:, :, 2] = np.sum(abs(self.stack_Z), axis=1).T + wl[0]

        stack = np.log10(stack)

        stack[:, :, 0] = _normalize(stack[:, ::-1, 0]) * pf[2] # T
        stack[:, :, 1] = _normalize(stack[:, ::-1, 1]) * pf[1] # R
        stack[:, :, 2] = _normalize(stack[:, ::-1, 2]) * pf[0] # Z

        
        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, **kwargs)

        ax.imshow(stack, aspect='auto',
                  interpolation='bicubic',
                  origin='lower',
                  extent=(0, 180, 0, float(self.stats.endtime)))

        ax.tick_params('both', labelright=True)
        ax.set_xlabel('distance / degree', fontsize=fontsize)
        ax.set_ylabel('time / seconds', fontsize=fontsize)

        ax2 = ax.twiny()
        planet_circ = self.db_info.planet_radius * 2 * np.pi / 1e3
        ax2.set_xlim(0, planet_circ / 2.)
        ax2.set_xlabel('distance / km', fontsize=fontsize)

        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize*0.8)

        if not ax:
            if not fnam:
                plt.show()
            else:
                plt.savefig(fnam, dpi=dpi)
                plt.close('all')

        return ax


def load(fnam):
    """
    Load global stack from pickle on disk

    Keywords:
    :type  fnam: string
    :param fnam: filename from which to read stack
    """
    return pickle.load(open(fnam, 'rb'))


def _normalize(array):
    minval = np.amin(array)
    maxval = np.amax(array)

    return (array - minval) / (maxval - minval)


def _get_npts(db, dt):
    src = instaseis.Source(latitude=90.0,
                           longitude=0.0,
                           depth_in_m=0.0,
                           m_rr=-1.670000e+28 / 1e7,
                           m_tt=3.820000e+27 / 1e7,
                           m_pp=1.280000e+28 / 1e7,
                           m_rt=-7.840000e+27 / 1e7,
                           m_rp=-3.570000e+28 / 1e7,
                           m_tp=1.550000e+27 / 1e7)

    rec = instaseis.Receiver(latitude=10, longitude=10)

    st = db.get_seismograms(src, rec, components='RTZ',
                            kind='displacement', dt=dt,
                            remove_source_shift=True)

    return st[0].stats.npts
