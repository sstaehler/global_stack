'''
Use instaseis to generate a global stack
'''
# import matplotlib.pyplot as plt
import numpy as np
import obspy
import instaseis
# import Image

def plot_global_stack(db_name,
                      resolution_x = 1024,
                      resolution_y = 768,
                      starttime    = -1.,
                      endtime      = -1.,
                      filename     = 'global_stack.png',
                      freqs        = (1./250., 1./10.),
                      running_mean_window_length = 200.,
                      waterlevel_percentile = 50.,
                      depth        = 10):
    """
    Plot a global stack with a given resolution for a source at a given depth

    :param db_name: filename for the instaseis database
    :type  db_name: string
    :param resolution_x: Size of the produced Image (x-component)
    :type resolution_x: np.int
    :param resolution_y: Size of the produced Image (y-component)
    :type resolution_y: np.int
    :param starttime: Starting time of the plot, default=event time
    :param endtime: End time of the plot, default=length of db seismograms
    :param freqs: Filtering frequencies in Hz.
    :type  freqs: Either scalar for lowpass filter, or 2-element tuple for
        bandpass
    :param running_mean_window_length: Length of window for AGC in seconds,
       default: 200. Can be adapted for different frequencies, is a hassle.
    :param waterlevel_percentile: Percentile for the waterlevel of the AGC
       default: 50. Higher values suppress noise, but also weaker phases.
    :param depth: Depth of event in kilometers, default: 10.
    """

    stf = np.array([0., 1., 0.])

    # Number of azimuths at which to calc. seismograms for the
    # moment
    nazi = 8

    # Open database
    db = instaseis.open_db(db_name)

    # Since we're removing the STF, the seismograms are slightly
    # shorter than the database
    len_db = db.info.length - db.info.src_shift

    # Set starttime and endtime to 0 and DB length, if not set
    if starttime==-1.:
        starttime = 0
    if endtime ==-1.:
        endtime = len_db

    print('len_db : %f' % len_db)

    if endtime > len_db:
        print('WARNING: endtime is larger als database length!')
        print('endtime:        %f'%endtime)
        print('db.info.length: %f'%len_db)
        print('Setting endtime to database length')
        endtime = len_db

    # Set Resolution in latitude and time
    dt = db.info.dt # (endtime - starttime) / (resolution_y)
    print('Selected dt:                 %f s'%dt)
    print('len_db / dt:                 %f' % (len_db / dt))

    # Set number of azimuths to resolution_x/2, since we calculate from
    # 0 to 180 degree and then mirror
    ndist = int(resolution_x * 0.5)
    print('Number of steps in latitude: %d'%ndist)


    sgs_data_Z = []
    sgs_data_R = []
    sgs_data_T = []

    running_mean_window = np.ones((int(running_mean_window_length/dt),)) / \
                          (running_mean_window_length / dt)

    #running_mean_window = np.hanning(int(running_mean_window_length / dt))

    for i in range(0, ndist):

        src = instaseis.Source(latitude=          90.0,
                               longitude=          0.0,
                               depth_in_m=   depth*1E3,
                               m_rr=      -1.670000e+28 / 1e7,
                               m_tt=       3.820000e+27 / 1e7,
                               m_pp=       1.280000e+28 / 1e7,
                               m_rt=      -7.840000e+27 / 1e7,
                               m_rp=      -3.570000e+28 / 1e7,
                               m_tp=       1.550000e+27 / 1e7,
                               origin_time=obspy.UTCDateTime(0),
                               sliprate =  stf)

        receiver = instaseis.Receiver(latitude=10, longitude=10,
                                      network="AB", station="%d"%i)
        component = 'R'
        test = load_smgr_with_agc(db, src, receiver, component, dt,
                                  freqs, running_mean_window,
                                  waterlevel_percentile)

        agc_R = np.zeros(test.shape)

        agc_T = np.zeros(test.shape)

        agc_Z = np.zeros(test.shape)

        deg = (90.-i*(180./ndist))
        print("distance: %d degree"%deg)
        print(' moment source')
        for j in range(0, nazi):
            lon = j * (360./nazi) - 180.
            print("  azimuth: %d degree"%lon)
            receiver = instaseis.Receiver(latitude=deg, longitude=lon,
                                          network="AB", station="%d"%i)

            ## R-component
            component = 'R'
            agc_R += load_smgr_with_agc(db, src, receiver, component, dt,
                                        freqs, running_mean_window,
                                        waterlevel_percentile)

            # T-component
            component = 'T'
            agc_T += load_smgr_with_agc(db, src, receiver, component, dt,
                                        freqs, running_mean_window,
                                        waterlevel_percentile)

            # Z-component
            component = 'Z'
            agc_Z += load_smgr_with_agc(db, src, receiver, component, dt,
                                        freqs, running_mean_window,
                                        waterlevel_percentile)

        print(' explosion source')
        # Add one more explosion source to enhance P-waves
        src = instaseis.Source(latitude=          90.0,
                               longitude=          0.0,
                               depth_in_m=   depth*1E3,
                               m_rr=       1.000000e+28 / 1e7,
                               m_tt=       1.000000e+28 / 1e7,
                               m_pp=       1.000000e+28 / 1e7,
                               m_rt=       0.000000e+27 / 1e7,
                               m_rp=       0.000000e+28 / 1e7,
                               m_tp=       0.000000e+27 / 1e7,
                               origin_time=obspy.UTCDateTime(0),
                               sliprate =  stf)

        receiver = instaseis.Receiver(latitude=deg, longitude=lon,
                                      network="AB", station="%d" % i)

        # Z-component
        component = 'Z'
        agc_Z = agc_Z + load_smgr_with_agc(db, src, receiver, component, dt,
                                           freqs, running_mean_window,
                                           waterlevel_percentile) * 4

        # R-component
        component = 'R'
        agc_R = agc_R + load_smgr_with_agc(db, src, receiver, component, dt,
                                           freqs, running_mean_window,
                                           waterlevel_percentile) * 4

        sgs_data_R.append(agc_R)

        sgs_data_T.append(agc_T)

        sgs_data_Z.append(agc_Z)

    # R-component
    im_R = plot_to_image(sgs_data_R)

    # T-component
    im_T = plot_to_image(sgs_data_T)

    # Z-component
    im_Z = plot_to_image(sgs_data_Z)

    # Merge the three components into one RGB image
    # im_half = Image.merge('RGB', (im_T, im_R, im_Z))

    # Mirror to get second half of globe, from 180 to 360 degrees
    # im_full = Image.new('RGB', (resolution_x, resolution_y))
    # im_full.paste(im_half, (0,0))
    # im_full.paste(im_half.transpose(Image.FLIP_LEFT_RIGHT), (resolution_x/2, 0))

    # im_full.save(filename)

    return im_R, im_T, im_Z;


def load_smgr_with_agc(db, src, receiver, component, dt, freqs,
                       running_mean_window, waterlevel_percentile):

    st = db.get_seismograms(source=src, receiver=receiver,
                            components=component, dt=dt)

    try:
        if len(np.array(freqs)) == 2:
            st.filter("bandpass", freqmin=freqs[0], freqmax=freqs[1])
        else:
            print('freqs should have 1 or 2 elements')

    except TypeError:
        st.filter("lowpass", freq=freqs)

    running_mean = np.convolve(abs(st[0].data),
                               running_mean_window,
                               mode='same')

    waterlevel = np.percentile(running_mean, waterlevel_percentile)

    running_mean[running_mean<waterlevel] = waterlevel

    agc = abs(st[0].data/running_mean)

    return agc;


def plot_to_image(sgs_data):
    data = abs(np.asarray(sgs_data))
    norm = (data - data.min()) / (data.max() - data.min())

    return norm * 256.



