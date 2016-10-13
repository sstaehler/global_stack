'''
Use instaseis to generate a global stack
'''
import matplotlib.pyplot as plt
import numpy as np
import instaseis
import pickle

__all__ = ["global_stack",
           "load"]


class global_stack():

    def __init__(self, db_name, depth_in_m=0.0, ndist=1024, dt=1.0):
        self.db_name = db_name
        self.depth_in_m = depth_in_m
        self.ndist = ndist
        self.dt = dt

        self.stack_R, self.stack_T, self.stack_Z, \
            self.stats, self.db_info = self._calc()

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

        for ilat in range(0, int(self.ndist)):
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
        pickle.dump(self, open(fnam, 'wb'))

    def plot(self, waterlevel=1e-5, postfac=2,
             fnam=None, dpi=96, figsize=(10, 10)):

        npts = self.stack_R.shape[0]
        ndist = self.stack_R.shape[2]

        stack = np.zeros(shape=(ndist, npts, 3))
        stack[:, :, 0] = np.sum(abs(self.stack_T), axis=1).T
        stack[:, :, 1] = np.sum(abs(self.stack_R), axis=1).T
        stack[:, :, 2] = np.sum(abs(self.stack_Z), axis=1).T

        stack = np.log10(stack + waterlevel)

        stack[:, :, 0] = _normalize(stack[:, ::-1, 0]) * postfac
        stack[:, :, 1] = _normalize(stack[:, ::-1, 1]) * postfac
        stack[:, :, 2] = _normalize(stack[:, ::-1, 2]) * postfac

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

        ax.imshow(stack, aspect='auto',
                  interpolation='bicubic',
                  origin='lower',
                  extent=(0, 180, 0, float(self.stats.endtime)))

        ax.set_xlabel('distance / degree')
        ax.set_ylabel('time / seconds')

        ax2 = ax.twiny()
        planet_circ = self.db_info.planet_radius * 2 * np.pi / 1e3
        ax2.set_xlim(0, planet_circ / 2.)
        ax2.set_xlabel('distance / km')

        if not fnam:
            plt.show()
        else:
            fig.savefig(fnam, dpi=dpi)
            plt.close('all')

        return fig


def load(fnam):
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
