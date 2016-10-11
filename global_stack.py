'''
Use instaseis to generate a global stack
'''
import matplotlib.pyplot as plt
import numpy as np
import instaseis
import collections
import functools


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def normalize(array):
    minval = np.amin(array)
    maxval = np.amax(array)

    return (array - minval) / (maxval - minval)


def get_npts(db, dt):
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


def plot_global_stack(db_name, depth=0.0, xres=1024, dt=1.0,
                      waterlevel=1e-5, postfac=2,
                      fnam=None, dpi=96, figsize=(10, 10)):

    stack_R, stack_T, stack_Z, stats = calc_global_stack(db_name,
                                                         depth,
                                                         xres,
                                                         dt)

    npts = stack_R.shape[0]
    ndist = stack_R.shape[2]

    stack = np.zeros(shape=(ndist, npts, 3))
    stack[:, :, 0] = np.log10(np.sum(abs(stack_T), axis=1).T + waterlevel)
    stack[:, :, 1] = np.log10(np.sum(abs(stack_R), axis=1).T + waterlevel)
    stack[:, :, 2] = np.log10(np.sum(abs(stack_Z), axis=1).T + waterlevel)

    stack[:, :, 0] = normalize(stack[:, ::-1, 0]) * postfac
    stack[:, :, 1] = normalize(stack[:, ::-1, 1]) * postfac
    stack[:, :, 2] = normalize(stack[:, ::-1, 2]) * postfac

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

    ax.imshow(stack, aspect='auto',
              interpolation='bicubic',
              origin='lower',
              extent=(0, 180, 0, float(stats.endtime)))

    ax.set_xlabel('distance / degree')
    ax.set_ylabel('time / seconds')

    if not fnam:
        plt.show()
    else:
        fig.savefig(fnam, dpi=dpi)
        plt.close('all')


@memoized
def calc_global_stack(db_name, depth=0.0, ndist=1024, dt=1.0):
    db = instaseis.open_db(db_name)

    nazi = 8

    dt = min([dt, db.info.dt])

    lats = np.linspace(start=-90., stop=90., num=ndist)
    lons = np.linspace(start=-180, stop=180., num=nazi, endpoint=False)

    src = instaseis.Source(latitude=90.0,
                           longitude=0.0,
                           depth_in_m=0.0,
                           m_rr=-1.670000e+28 / 1e7,
                           m_tt=3.820000e+27 / 1e7,
                           m_pp=1.280000e+28 / 1e7,
                           m_rt=-7.840000e+27 / 1e7,
                           m_rp=-3.570000e+28 / 1e7,
                           m_tp=1.550000e+27 / 1e7)

    npts = get_npts(db, dt)

    stack_R = np.zeros(shape=(ndist, nazi, npts))
    stack_T = np.zeros(shape=(ndist, nazi, npts))
    stack_Z = np.zeros(shape=(ndist, nazi, npts))

    for ilat in range(0, ndist):
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

    return stack_R, stack_T, stack_Z, st[0].stats
