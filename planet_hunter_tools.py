import numpy as np
from matplotlib import pyplot as plt
import lightkurve as lk
from lightkurve.search import SearchError


# grid functions for transit searching
# constants from transitleastsquares
#
# astrophysical constants
import astropy.units as u
from astropy import constants as const
R_sun = u.R_sun.to(u.cm) # in cm
M_sun = u.M_sun.to(u.g) # in grams
G = const.G.cgs.value #cm^3 per g per s^2
R_earth = u.R_earth.to(u.cm) # in cm
R_jup = u.R_jupiter.to(u.cm) # in cm
SECONDS_PER_DAY = u.day.to(u.second)


# For the duration grid
FRACTIONAL_TRANSIT_DURATION_MAX = 0.12
M_STAR_MIN = 0.1
M_STAR_MAX = 1.0
R_STAR_MIN = 0.13
R_STAR_MAX = 3.5
DURATION_GRID_STEP = 1.1
OVERSAMPLING_FACTOR = 5
N_TRANSITS_MIN = 3
MINIMUM_PERIOD_GRID_SIZE = 100

def T14(
    R_s, M_s, P, upper_limit=FRACTIONAL_TRANSIT_DURATION_MAX, small=False
):
    """Input:  Stellar radius and mass; planetary period
               Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""
    import numpy
    pi = numpy.pi

    P = P * SECONDS_PER_DAY
    R_s = R_sun * R_s
    M_s = M_sun * M_s

    if small:  # small planet assumption
        T14max = R_s * ((4 * P) / (pi * G * M_s)) ** (1 / 3)
    else:  # planet size 2 R_jup
        T14max = (R_s + 2 * R_jup) * (
            (4 * P) / (pi * G * M_s)
        ) ** (1 / 3)

    result = T14max / P
    if result > upper_limit:
        result = upper_limit
    return result


def duration_grid(periods, shortest, log_step=DURATION_GRID_STEP):
    import numpy    
    duration_max = T14(
        R_s=R_STAR_MAX,
        M_s=M_STAR_MAX,
        P=numpy.min(periods),
        small=False  # large planet for long transit duration
    )
    duration_min = T14(
        R_s=R_STAR_MIN,
        M_s=M_STAR_MIN,
        P=numpy.max(periods),
        small=True  # small planet for short transit duration
    )

    durations = [duration_min]
    current_depth = duration_min
    while current_depth * log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(duration_max)  # Append endpoint. Not perfectly spaced.
    return durations


def period_grid(
    R_star,
    M_star,
    time_span,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=OVERSAMPLING_FACTOR,
    n_transits_min=N_TRANSITS_MIN,
):
    """Returns array of optimal sampling periods for transit search in light curves
       Following Ofir (2014, A&A, 561, A138)"""
    import numpy
    pi = numpy.pi

    if R_star < 0.01:
        text = (
            "Warning: R_star was set to 0.01 for period_grid (was unphysical: "
            + str(R_star)
            + ")"
        )
        warnings.warn(text)
        R_star = 0.1

    if R_star > 10000:
        text = (
            "Warning: R_star was set to 10000 for period_grid (was unphysical: "
            + str(R_star)
            + ")"
        )
        warnings.warn(text)
        R_star = 10000

    if M_star < 0.01:
        text = (
            "Warning: M_star was set to 0.01 for period_grid (was unphysical: "
            + str(M_star)
            + ")"
        )
        warnings.warn(text)
        M_star = 0.01

    if M_star > 1000:
        text = (
            "Warning: M_star was set to 1000 for period_grid (was unphysical: "
            + str(M_star)
            + ")"
        )
        warnings.warn(text)
        M_star = 1000

    R_star = R_star * R_sun
    M_star = M_star * M_sun
    time_span = time_span * SECONDS_PER_DAY  # seconds

    # boundary conditions
    f_min = n_transits_min / time_span
    f_max = 1.0 / (2 * pi) * np.sqrt(G * M_star / (3 * R_star) ** 3)

    # optimal frequency sampling, Equations (5), (6), (7)
    A = (
        (2 * pi) ** (2.0 / 3)
        / pi
        * R_star
        / (G * M_star) ** (1.0 / 3)
        / (time_span * oversampling_factor)
    )
    C = f_min ** (1.0 / 3) - A / 3.0
    N_opt = (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3) * 3 / A

    X = numpy.arange(N_opt) + 1
    f_x = (A / 3 * X + C) ** 3
    P_x = 1 / f_x

    # Cut to given (optional) selection of periods
    periods = P_x / SECONDS_PER_DAY
    selected_index = numpy.where(
        numpy.logical_and(periods > period_min, periods <= period_max)
    )

    number_of_periods = numpy.size(periods[selected_index])

    if number_of_periods > 10 ** 6:
        text = (
            "period_grid generates a very large grid ("
            + str(number_of_periods)
            + "). Recommend to check physical plausibility for stellar mass, radius, and time series duration."
        )
        warnings.warn(text)

    if number_of_periods < MINIMUM_PERIOD_GRID_SIZE:
        if time_span < 5 * SECONDS_PER_DAY:
            time_span = 5 * SECONDS_PER_DAY
        warnings.warn(
            "period_grid defaults to R_star=1 and M_star=1 as given density yielded grid with too few values"
        )
        return period_grid(
            R_star=1, M_star=1, time_span=time_span / SECONDS_PER_DAY
        )
    else:
        return periods[selected_index]  # periods in [days]
    
# some loose mass-radius relationships for main-sequence stars    
def radius_from_mass(M_star):
    if M_star<=1:
        R_star = M_star**0.8
    if M_star>1:
        R_star = M_star**0.57
    return R_star
def mass_from_radius(R_star):
    if R_star<=1:
        M_star = R_star**(1/0.8)
    if R_star>1:
        M_star = R_star**(1/0.57)        
    return M_star


def download_data(starname,mission,quarter_number,cadence):
    from lightkurve.search import _search_products

    degrees = 21/3600 #size of TESS pixels in degrees on the sky
    
    if mission=='TESS':
        if (cadence=='long') or (cadence=='30 minute') or (cadence=='10 minute'):
                ffi_or_tpf='FFI'
        if (cadence=='short') or (cadence=='2 minute') or (cadence=='20 second') or (cadence=='fast'):
            ffi_or_tpf='Target Pixel'
    if mission=='Kepler':
        if (cadence=='long') or (cadence=='30 minute') or (cadence=='10 minute'):
            ffi_or_tpf='FFI'
        if (cadence=='short') or (cadence=='2 minute') or (cadence=='20 second') or (cadence=='fast'):
            ffi_or_tpf='Target Pixel'        
    
    if mission=='TESS':
        try:
            search_string=_search_products(starname, radius=degrees,\
                                           filetype=ffi_or_tpf, \
                                           cadence=cadence,\
                                           mission=mission,sector=quarter_number)
        except SearchError as e:
            print('No ',cadence,' cadence ',mission,' data for ',starname,' in Sector ',quarter_number,'!')
            return None
    if mission=='Kepler':
        search_string = lk.search_targetpixelfile(starname,author=mission, \
                              quarter=quarter_number,\
                              exptime=cadence)
        search_string=search_string[search_string.author==mission]
    #
    # #NEW
    if mission=='TESS':        
        print('Number of data products for ',starname,' in ',mission,' with ',cadence,' cadence and in Sector ',quarter_number,':',len(search_string.exptime.value))
    if mission=='Kepler':        
        print('Number of data products for ',starname,' in ',mission,' with ',cadence,' cadence and in Quarter ',quarter_number,':',len(search_string.exptime.value))        
    # # NEW
    #
    # filter search result by cadence input argument
    if (cadence=='30 minute') or (cadence=='long'):
        mask = np.where((search_string.exptime.value>600) & (search_string.exptime.value<=1800) )[0]
        search_string_filtered = search_string[mask]
    if (cadence=='10 minute'):
        mask = np.where((search_string.exptime.value>120) & (search_string.exptime.value<=600) )[0]
        search_string_filtered = search_string[mask]
    if (cadence=='2 minute') or (cadence=='short'):
        mask = np.where((search_string.exptime.value>20) & (search_string.exptime.value<=120) )[0]
        search_string_filtered = search_string[mask]
    if (cadence=='20 second') or (cadence=='fast'):
        mask = np.where((search_string.exptime.value>0) & (search_string.exptime.value<=20) )[0]
        search_string_filtered = search_string[mask]
        
    #make sure there are no duplicate quarters/sectors
    u, indices = np.unique(search_string_filtered.mission, return_index=True)
    search_string_filtered = search_string_filtered[indices]
    # # NEW    
    print('Filtered Data Product list:')
    # # NEW    
    print(search_string_filtered)
    print('')
    data = search_string_filtered.download()
    return data


def process_single_lightcurve(starname,mission,quarter_number,cadence,plot_tpf=False):
    
    tpf = download_data(starname,mission,quarter_number,cadence)    
    if tpf == None:
        print('No TargetPixelFile return from search!')
        print('TPF type:',type(tpf))
        if cadence=='short':
            new_cadence='long'
            print('Switching from',cadence,'to',new_cadence)
            tpf = download_data(starname,mission,quarter_number,new_cadence)
            if tpf==None:
                print('ERROR: see earlier error messages. No data for this target in this sector/quarter')
                lc = None
    else:
        print('TPF type:',type(tpf))    
    # use default mission pipeline aperture mask
    lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    #
    # mask out "bad" data points of poorer photometric quality
    quality_mask = tpf.quality==0
    lc = lc[quality_mask] 
    #
    
    #check if aperture_mask is a good mask to use:
    nanmask = np.where(np.isfinite(lc.flux.value))[0]
    if len(nanmask)<1: # if there aren't any finite values then we need to do a new mask
        new_aperture_mask = tpf.create_threshold_mask(threshold=1, reference_pixel='center')
        lc = tpf.to_lightcurve(aperture_mask=new_aperture_mask)
        if plot_tpf==True:
            median_mask = np.where(tpf.flux.value==(np.nanmedian(tpf.flux.value,axis=0)))[0]
            if len(median_mask)>0:
                median_frame=np.min(median_mask)
            else:
                median_frame=0
                tpf.plot(frame=median_frame,aperture_mask=new_aperture_mask)
            plt.show()        
        from lightkurve.correctors import PLDCorrector
        try:
            tpf = PLDCorrector(tpf)
            lc = tpf.correct(pca_components=10)
        except ValueError:
            print('Bad Pixel mask. Could not do PLD!')
            print('')
            lc=lc
    else:
        if plot_tpf==True:
            median_mask = np.where(tpf.flux.value==(np.nanmedian(tpf.flux.value,axis=0)))[0]
            if len(median_mask)>0:
                median_frame=np.min(median_mask)
            else:
                median_frame=0
            if cadence=='short':
                tpf.plot(frame=median_frame,aperture_mask='pipeline')
            plt.show()             
    #
    return tpf, lc


def process_multiple_lightcurves(starname,mission,quarter_number,cadence):
    if type(quarter_number)==list:
        all_times = []
        all_fluxes =[]
        all_errors =[]
        #
        for q in quarter_number:
            quarter = int(q)                
            if q==int(quarter_number[0]):
                plot_tpf=True
            else:
                plot_tpf=False
            tpf,lc = process_single_lightcurve(starname,mission,quarter,cadence,plot_tpf)
            if type(lc) is None:
                if (lc==None):
                    continue
            if q==int(quarter_number[0]):
                first_tpf=tpf
            all_times  = np.append(all_times ,lc.time.value)
            all_fluxes = np.append(all_fluxes,lc.flux.value/np.nanmedian(lc.flux.value))# normalize each sector
            all_errors = np.append(all_errors,lc.flux_err.value)
        final_lc = lk.LightCurve(time=all_times,flux=all_fluxes,flux_err=all_errors)
    else:
        quarter=int(quarter_number[0])
        tpf,lc = process_single_lightcurve(starname,mission,quarter,cadence)
        if q==int(quarter_number[0]):
            first_tpf=tpf        
        final_lc = lk.LightCurve(time=lc.time.value,flux=lc.flux.value/np.nanmedian(lc.flux.value),\
                                 flux_err=lc.flux_err.value/np.nanmedian(lc.flux.value))
        
    return first_tpf, final_lc
        



def phasefold(time,flux,period,T0):
    """
    This function will phase-fold the input light curve (time, flux)
    using a Mid-transit time and orbital period. The resulting phase
    is centered on the input Mid-transit time so that the transit
    occurs at phase 0.

    Input Parameters
    ----------
    time: array
        An array of timestamps from TESS observations.
    TO : float
        The Mid-transit time of a periodic event.
    period : float
        An orbital period of a periodic event.
    flux : array
        An array of flux values from TESS observations.
    Returns
    -------
        * phase : array
            An array of Orbital phase of the phase-folded light curve.
        * flux : array
            An array of flux values from TESS observations of the
            phase-folded light curve.
    """
    phase=(time- T0 + 0.5*period) % period - 0.5*period
    ind=np.argsort(phase, axis=0)
    return phase[ind],flux[ind] 

def everything(savepath,starname, M_star,R_star, mission, quarter_number, cadence, \
               smoothing_window, Nsigma, minP, maxP, Nfreq):
    import os
    # checking if savepath exists
    if os.path.exists(savepath)==True:
        pass
    else:
        # if it doesn't exist, create it
        os.makedirs(savepath)
    
    import lightkurve as lk
    from matplotlib import pyplot as plt
    
    
    # check stellar mass and radius:
    if (np.isnan(M_star)==True) & (np.isnan(R_star)==False):
        M_star = mass_from_radius(R_star)
    if (np.isnan(R_star)==True) & (np.isnan(M_star)==False):
        R_star = radius_from_mass(M_star)        
        
    # use download function to get light curve images:
#     tpf = download_data(starname,mission,quarter_number,cadence)    
#     if type(tpf) is None:
#         print('No TargetPixelFile return from search!')
#         print('TPF type:',type(tpf))
#     else:
#         print('TPF type:',type(tpf))

    # use default mission pipeline aperture mask
    #lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    #
    # mask out "bad" data points of poorer photometric quality
    #quality_mask = tpf.quality==0
    #lc = lc[quality_mask] 
    #
    
    #check if aperture_mask is a good mask to use:
    #nanmask = np.where(np.isfinite(lc.flux.value))[0]
    #if len(nanmask)<1: # if there aren't any finite values then we need to do a new mask
    #    new_aperture_mask = tpf.create_threshold_mask(threshold=1, reference_pixel='center')
#         lc = tpf.to_lightcurve(aperture_mask=new_aperture_mask)
                
#         tpf.plot(frame=0,aperture_mask=new_aperture_mask)
#         plt.show()
        
#         from lightkurve.correctors import PLDCorrector
#         tpf = PLDCorrector(tpf)
#         lc = tpf.correct(pca_components=10)
#     else:
#         tpf.plot(frame=0,aperture_mask='pipeline')
#         plt.show()  


    tpf,lc = process_multiple_lightcurves(starname,mission,quarter_number,cadence)
    
    
    def convert_window_size_to_Npts(lc,smoothing_window):
        #calculating the exposure time
        cad = np.nanmedian(np.diff(lc.time.value)) 
        print('cadence',np.round(cad*24*60,3),'minutes')
        def round_up_to_odd(f):
            return int(np.ceil(f) // 2 * 2 + 1)
        Npts = round_up_to_odd(int((smoothing_window)/cad))
        return Npts
        
        
    Npts_smoothing_window = convert_window_size_to_Npts(lc,smoothing_window)
    print('Npts for smoothing:',Npts_smoothing_window)
    
    try:
    
        flat, trend = lc.flatten(polyorder=2,window_length=Npts_smoothing_window, return_trend=True)
        
    except ValueError as E:
        print(E)
        return lc
    
    # now for outlier removal
    if type(Nsigma)==list: # if Nsigma is a list [a,b], then unpack the values
        Nsigma_low, Nsigma_high = Nsigma[0], Nsigma[1] 
    if (type(Nsigma)==int) or (type(Nsigma)==float):
        Nsigma_low, Nsigma_high = Nsigma, Nsigma #same as above as is below
        
    flat = flat.remove_outliers(sigma_lower = Nsigma_low, sigma_upper = Nsigma_high)
    
    
    # now do transit search and fold light curve on best fit 
    # transit time (T0) and orbital period:
    if maxP==None:
        maxP = (np.max(flat.time.value)-np.min(flat.time.value))
        maxP = maxP/3 #to allow at least 3 transits for detection
    #
    # NEW
    # define duration and period grids
    import requests
    #from transitleastsquares import period_grid,duration_grid, catalog_info
    ID = int(starname[4:])
    #
    # First lets make the grid
    LCduration = (np.max(flat.time.value)-np.min(flat.time.value)) #duration of light curve
    #
    #
    oversampling_factor = 5
    duration_grid_step=1.05
    periods = period_grid(R_star=R_star, M_star=M_star, time_span=LCduration,\
                          period_min=minP, period_max=maxP,oversampling_factor=oversampling_factor)
    durations= duration_grid(periods,shortest=None,log_step=duration_grid_step)
    #
    import time as clock
    start = clock.time()
    flat_periodogram = flat.to_periodogram(method="bls", period=periods, duration=durations)
    end = clock.time()
    runtime = end-start
    if runtime > 60:
        print('BLS took ',runtime/60,'minutes')
    if runtime < 60:
        print('BLS took ',runtime,'seconds')
    # # NEW    
    best_fit_period = flat_periodogram.period_at_max_power.value
    best_fit_T0 = flat_periodogram.transit_time_at_max_power.value
    best_fit_Duration = flat_periodogram.duration_at_max_power.value
    best_fit_Depth = flat_periodogram.depth_at_max_power.value    
    # from the transit depth, we can approximate the planet size if
    # we know the stellar radius:
    #
    # we can query the TESS Input Catalog to get stellar parameters:
    
    from astroquery.mast import Catalogs
    # we'll do a radial cone query to search for every star near our target star
    # within 21/3600 degrees or about 21 arcseconds on the sky
    # TESS has a pixel size of 21 arcseconds
    catalogData = Catalogs.query_object(starname, radius = 21/3600, catalog = "TIC")
    #
    R_star=catalogData[0]['rad']
    R_star_uncertainty=catalogData[0]['e_rad']
    from astropy import units as u
    R_earth = u.R_earth.to(u.cm) #this is an Earth radius in units of centimeters
    R_sun = u.R_sun.to(u.cm) #this is a Solar radius in units of centimeters
    # with these, we can measure our best fit planet radius in units
    # of Earth radii
    best_fit_planet_radius = np.sqrt(best_fit_Depth)*R_star*R_sun/R_earth
    
    print('Best fit period: ',str(np.round(best_fit_period,3))+' days')
    print('Best fit planet Radius: ',str(np.round(best_fit_planet_radius,3))+' Earth Radii')
    # # NEW       
    
    def plot_transit_times(flat,T0,P,ax2):
        import numpy as np
        #time array
        t = np.array(flat.time.value)
        
        #calculating transit times from T0 and P from BLS
        if T0 < np.min(t):
            transit_times = [T0 + P]
        else:
            transit_times = [T0]
        previous_transit_time = transit_times[0]
        transit_number = 0
        while True:
            transit_number = transit_number + 1
            next_transit_time = previous_transit_time + P
            if next_transit_time < (np.min(t) + (np.max(t) - np.min(t))):
                transit_times.append(next_transit_time)
                previous_transit_time = next_transit_time
            else:
                break
        
        for T in transit_times:
            # for every calculated transit time
            #plot a vertical grey line in the background (zorder=-100)
            ax2.axvline(x=T, color='grey',\
                        linewidth=3,zorder=-100)
            
        
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)     
    
    ax1.plot(lc.time.value, lc.flux.value/np.nanmedian(lc.flux.value),'r.',label='Raw LC')
    ax2.plot(flat.time.value, flat.flux.value/np.nanmedian(flat.flux.value),'k.',label='Flattened LC')
    
    plot_transit_times(flat,best_fit_T0,best_fit_period,ax1)
    plot_transit_times(flat,best_fit_T0,best_fit_period,ax2)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Normalized Flux')   
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Normalized Flux')        
    
    ax1.legend(loc='upper right',framealpha=1,fancybox=True)
    ax2.legend(loc='upper right',framealpha=1,fancybox=True)
    
    fig.tight_layout(pad=1)
    plt.show()
    
    
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)  
    
    #flat_periodogram.plot(ax=ax1,color='black')
    SDE = (flat_periodogram.power.value - np.nanmedian(flat_periodogram.power.value)) / np.nanstd(flat_periodogram.power.value)
    ax1.plot(flat_periodogram.period.value,SDE,'k-')
    ax1.set_xlabel('Period [days]')
    ax1.set_ylabel('BLS Signal Detection Efficiency')
    
    
    # plot vertical line near peak BLS period 
    ax1.axvline(x = best_fit_period,color='lightgreen',\
                alpha=0.5,zorder=-100,linewidth=5)
    # # NEW    
    # plotting aliases aka harmonics
    for a in range(2,100):
        if best_fit_period*a < np.max(flat_periodogram.period.value):
            ax1.axvline(x = best_fit_period*a ,color='lightgreen',\
                        alpha=0.5,zorder=-100,linewidth=3,linestyle='--')
        if best_fit_period/a > np.min(flat_periodogram.period.value):
            ax1.axvline(x = best_fit_period/a ,color='lightgreen',\
                        alpha=0.5,zorder=-100,linewidth=3,linestyle='--')            
    ax1.set_title('Best fit orbital period: '+str(np.round(best_fit_period,3))+' days')
    # # NEW    
    phasefolded, fluxfolded = phasefold(np.array(flat.time.value),np.array(flat.flux.value),best_fit_period,best_fit_T0)
    ax2.plot(24*phasefolded, fluxfolded/np.nanmedian(fluxfolded),'r.')       
    
    
    # let's use the best fit orbital period, T0 and transit duration
    # to calculate the best fit box model:
    BLS_model=flat_periodogram.get_transit_model(period=best_fit_period,
                                       transit_time=best_fit_T0,
                                       duration=best_fit_Duration)
    #lets fold our box model:
    #folded_BLS = BLS_model.fold(period=best_fit_period, epoch_time=best_fit_T0)
    BLS_phasefolded, BLS_modelfolded = phasefold(np.array(BLS_model.time.value), np.array(BLS_model.flux.value),\
                                             best_fit_period,best_fit_T0)
    #
    ax2.plot(24*BLS_phasefolded, BLS_modelfolded/np.nanmedian(BLS_modelfolded),\
             'b-',linewidth=3)       
    # # NEW    
    if mission=='Kepler':
        ax2.set_xlabel('Phase [Hours since '+str(np.round(best_fit_T0,3))+' BKJD]')
    if mission=='TESS':
        ax2.set_xlabel('Phase [Hours since '+str(np.round(best_fit_T0,3))+' BTJD]')        
    ax2.set_ylabel('Normalized Flux')
    ax2.set_title('Best fit Planet Radius: '+str(np.round(best_fit_planet_radius,3))+' Earth Radii ; Best fit Transit Duration: '+str(np.round(24*best_fit_Duration,3))+' Hours')
    # # NEW    
    
    N_durations = 3.5
    # set a xlim based on N transit duration widths 
    # of the folded light curve
    if best_fit_period > 1:
        ax2.set_xlim(-24*N_durations*best_fit_Duration,\
                     +24*N_durations*best_fit_Duration)
    else:
        ax2.set_xlim(-8,8)
    
    # let's also plot a horizontal line to show how deep our box model is
    ax2.axhline(y=np.nanmedian(fluxfolded/np.nanmedian(fluxfolded)) - best_fit_Depth,color='green',zorder=-100)
    #
    fig.tight_layout(pad=1)
    fig.savefig(savepath+starname+'_BLS_result.png',bbox_inches='tight')
    plt.show()
    
    #saving our data
    import pandas as pd
    bls_result = pd.DataFrame({'T0':best_fit_T0,'Period':best_fit_period,'Depth':best_fit_Depth,'Planet Radius':best_fit_planet_radius,'Stellar Radius':R_star},index=[0])
    bls_result.to_csv(savepath+starname+'_BLS_result.csv', index=False)
    
    raw_lightcurve = pd.DataFrame({'Time':lc.time.value, 'Flux':lc.flux.value/np.nanmedian(lc.flux.value), 'Error':lc.flux_err.value})
    raw_lightcurve.to_csv(savepath+starname+'_Raw_lightcurve.csv')
    
    
    flat_lightcurve = pd.DataFrame({'Time':flat.time.value,'Flux':flat.flux.value/np.nanmedian(flat.flux.value),'Error':flat.flux_err.value})
    flat_lightcurve.to_csv(savepath+starname+'_Smoothed_lightcurve.csv')