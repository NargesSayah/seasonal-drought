import argparse
import os
import xarray as xr
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd, MonthBegin
from datetime import datetime
import yaml
from datetime import datetime, timedelta

def calculate_seasonal_thresholds(dataset, seasons, start_date, end_date):
    """
    Calculate the drought event threshold (Q0) for each grid point as the 0.05th quantile 
    of streamflow values for each season between the start and end dates.
    """
    seasonal_Q0s = {}
    for season, months in seasons.items():
        # Select data within the overall date range
        ds_filtered = dataset.sel(time=slice(start_date, end_date))
        
        # Further filter the data for the specific season
        ds_seasonal = ds_filtered.sel(time=ds_filtered.time.dt.month.isin(months))

        
        # Convert specified fill values to NaN (if necessary, ensure this step is needed)
        ds_seasonal['streamflow'] = ds_seasonal['streamflow'].where(ds_seasonal['streamflow'] != 9.96920997e+36, np.nan)

        # Calculate the 0.05th quantile for each grid point. Ensure there's enough data for this calculation
        Q0 = ds_seasonal['streamflow'].quantile(0.05, dim='time', skipna=True)
        
        seasonal_Q0s[season] = Q0
        # print("season, seasonal_Q0s[season]: ",season, seasonal_Q0s[season])
    print("seasonal_Q0s: ", seasonal_Q0s['JJA'][0,100])
    return seasonal_Q0s

def save_Q0s(seasonal_Q0s, output_directory, ensemble_name, year_range):
    """
    Save the calculated seasonal Q0 values to a text file in the specified directory.
    """
    output_folder = os.path.join(output_directory, "seasonal_Q0", ensemble_name, year_range)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "seasonal_Q0s.txt")
    
    with open(output_path, 'w') as f:
        for season, Q0 in seasonal_Q0s.items():
            f.write(f"{season}: {Q0}\n")


def find_continuous_drought_events(streamflow, Q0s):
    """
    Identify continuous drought events based on streamflow data and per-grid point Q0 thresholds.

    Parameters:
    - streamflow: xarray DataArray with dimensions ('time', 'lat', 'lon')
    - Q0s: xarray DataArray with dimensions ('lat', 'lon') representing drought thresholds

    Returns:
    - Drought start indices, end indices, and durations for each grid point
    """
    # Create a boolean DataArray indicating where streamflow is below the Q0 threshold
    drought_conditions = streamflow < Q0s

    # Initialize containers for drought event information
    drought_starts = np.empty(streamflow.shape[1:], dtype=object)  # Removing 'time' dimension
    drought_ends = np.empty_like(drought_starts)
    durations = np.empty_like(drought_starts)

    # Iterate over each grid point
    for lat_idx in range(streamflow.shape[1]):
        for lon_idx in range(streamflow.shape[2]):
            drought_starts[lat_idx, lon_idx] = []
            drought_ends[lat_idx, lon_idx] = []
            durations[lat_idx, lon_idx] = []

            # Extract time series of drought conditions for the current grid point
            ts_drought_conditions = drought_conditions[:, lat_idx, lon_idx].values

            start = None
            for day_idx in range(len(ts_drought_conditions)):
                # Check for the start of a drought event
                if ts_drought_conditions[day_idx] and start is None:
                    start = day_idx
                # Check for the end of a drought event
                elif not ts_drought_conditions[day_idx] and start is not None:
                    end = day_idx - 1
                    duration = end - start + 1
                    drought_starts[lat_idx, lon_idx].append(start)
                    drought_ends[lat_idx, lon_idx].append(end)
                    durations[lat_idx, lon_idx].append(duration)
                    start = None  # Reset for the next event

            # Check if the last event extends to the end of the series
            if start is not None:
                end = len(ts_drought_conditions) - 1
                duration = end - start + 1
                drought_starts[lat_idx, lon_idx].append(start)
                drought_ends[lat_idx, lon_idx].append(end)
                durations[lat_idx, lon_idx].append(duration)
                
    # Debugging prints for a specific grid point
    # print("\nDurations for a specific grid point\n:", durations[0, 87])
    # print("\nStart Dates for a specific grid point\n", drought_starts[0, 87])
    # print("\nEnd Dates for a specific grid point\n", drought_ends[0, 87])
    
    return drought_starts, drought_ends, durations


def calculate_drought_magnitude(streamflow, drought_starts, drought_ends, durations, Q0s):
    # Initialize containers for the output
    magnitudes = np.empty(streamflow.shape[1:], dtype=object)  # Similar structure to drought_starts/ends without the time dimension
    start_dates = np.empty_like(magnitudes)
    event_durations = np.empty_like(magnitudes)

    # Loop over latitude and longitude
    for lat_idx in range(streamflow.shape[1]):
        for lon_idx in range(streamflow.shape[2]):
            magnitudes[lat_idx, lon_idx] = []
            start_dates[lat_idx, lon_idx] = []
            event_durations[lat_idx, lon_idx] = []

            # Access the drought events for the current location
            for event_idx, (start_idx, end_idx, duration) in enumerate(zip(drought_starts[lat_idx, lon_idx], drought_ends[lat_idx, lon_idx], durations[lat_idx, lon_idx])):
                if duration > 2:  # Consider events longer than 2 days
                    # Retrieve start and end times for the event
                    start_time = streamflow.isel(time=start_idx).time.values
                    end_time = streamflow.isel(time=end_idx).time.values
                    # print("start_time:\n", start_time, "end_time:\n", end_time)
                    # Select the event period
                    event_streamflow = streamflow.sel(time=slice(start_time, end_time))
                    event_streamflow = event_streamflow.where(event_streamflow != 9.96920997e+36, np.nan)
                    
                    # Calculate event magnitude
                    print("event_streamflow: ", event_streamflow)
                    event_deficit = Q0s[lat_idx, lon_idx] - event_streamflow
                    print("Q is:", Q0s[lat_idx, lon_idx])
                    print("HEEEERE:", event_deficit[lat_idx, lon_idx])
                    event_magnitude = abs(event_deficit[lat_idx, lon_idx].sum().item() / duration)
                    print("event_magnitude: ", event_magnitude[lat_idx, lon_idx])
                    magnitudes[lat_idx, lon_idx].append(event_magnitude[lat_idx, lon_idx])
                    start_dates[lat_idx, lon_idx].append(pd.to_datetime(start_time))
                    event_durations[lat_idx, lon_idx].append(duration)
    # print("start_dates, magnitudes, event_durations:\n", start_dates[0, 87], magnitudes[0, 87], event_durations[0, 87])
    return start_dates, magnitudes, event_durations


def find_max_magnitude_droughts(dataset, seasonal_Q0s, window_start, window_end, mid_season_dates, seasons):
    """
    Identifies the maximum magnitude drought events for each grid point and season.
    """
    drought_data = xr.Dataset({
        'magnitude': (('time', 'lat', 'lon'), np.nan * np.zeros((len(mid_season_dates), len(dataset.lat), len(dataset.lon)))),
        'startDate': (('time', 'lat', 'lon'), np.nan * np.zeros((len(mid_season_dates), len(dataset.lat), len(dataset.lon)))),
        'duration': (('time', 'lat', 'lon'), np.nan * np.zeros((len(mid_season_dates), len(dataset.lat), len(dataset.lon)))),
    }, coords={'time': mid_season_dates, 'lat': dataset.lat, 'lon': dataset.lon})

    reference_date = np.datetime64('1950-01-01')
    
    for year in range(int(window_start[:4]), int(window_end[:4]) + 1):
        for season, months in seasons.items():
            Q0 = seasonal_Q0s[season]
            print(f"Processing season: {season} for year range {year}-{year+1}")
            mid_season_date = np.datetime64(f"{year}-{months[len(months) // 2]:02d}-15")
            # Locate the corresponding mid-season date in the `drought_data` dataset
            print("Mid season date:", mid_season_date)
            mid_season_idx = np.argwhere(mid_season_dates.values == np.datetime64(mid_season_date))[0][0]

            # Adjust the start date
            if season == 'DJF': 
                if year == 1950: # beginning of the dataset
                    season_start_str = f"1950-01-01"
                else:
                    season_start_str = f"{year-1}-{str(months[0]).zfill(2)}-01"
            else:
                season_start_str = f"{year}-{str(months[0]).zfill(2)}-01"
            season_start_date = pd.to_datetime(season_start_str)

            
            season_end_str = f"{year}-{str(months[-1]).zfill(2)}-01"
            season_end_date = pd.to_datetime(season_end_str) + MonthEnd()
            print("season start date:", season_start_date, "season end date:", season_end_date)

            streamflow_season = dataset['streamflow'].sel(time=slice(season_start_date, season_end_date))
            
            # Identify and evaluate drought events within this seasonal data
            if streamflow_season.size > 0:
                drought_starts, drought_ends, drought_durations = find_continuous_drought_events(streamflow_season, Q0)
                # Convert each pandas Timestamp in the event_start_date list to a NumPy datetime64 object
                event_start_date, magnitude, duration = calculate_drought_magnitude(streamflow_season, 
                                                                                    drought_starts, drought_ends, drought_durations, Q0)
                event_start_date_np = np.array([np.datetime64(ts) for ts in event_start_date])
                if len(magnitude) > 0:
                    # print("magnituuuuude:", magnitude)
                    # Identify the index of the maximum magnitude event
                    max_magnitude_index = np.argmax(magnitude)
        
                    # Extract the maximum magnitude event's details
                    max_magnitude = magnitude[max_magnitude_index]
                    max_event_start_date = event_start_date_np[max_magnitude_index]
                    max_event_duration = duration[max_magnitude_index]

                    days_since_reference = (max_event_start_date - reference_date).astype('timedelta64[D]').astype(float)

                    # Convert startDate to "days since the reference date"
                    magnitude *= 86400  # Convert from m3/s to m3/d
                    
                    drought_data['magnitude'].loc[mid_season_dates[mid_season_idx], :, :] = max_magnitude
                    drought_data['startDate'].loc[mid_season_dates[mid_season_idx], :, :] = days_since_reference
                    drought_data['duration'].loc[mid_season_dates[mid_season_idx], :, :] = max_event_duration
                else:
                    # Handle seasons without drought events
                    mid_season_idx = np.argwhere(mid_season_dates.values == np.datetime64(mid_season_date))[0][0]
                    # Set magnitude to 0 and start date to mid-season date for no drought event seasons
                    print("mid season datttte:", mid_season_date, type(mid_season_date))
                    print("reference_dateeeee", reference_date, type(reference_date))
                    print("no drought, days from the mid season date:", (mid_season_date - reference_date).astype('timedelta64[D]').astype(float))
                    drought_data['magnitude'].loc[mid_season_dates[mid_season_idx], :, :] = 0
                    drought_data['startDate'].loc[mid_season_dates[mid_season_idx], :, :] = (mid_season_date - reference_date).astype('timedelta64[D]').astype(float)
                    drought_data['duration'].loc[mid_season_dates[mid_season_idx], :, :] = 0
            else:
                print(f"Warning: No data available for season '{season}' in year {year}. Skipping this season.")

    return drought_data


def generate_mid_season_dates(year_range):
    # Split the year_range to get start and end years
    start_year, end_year = map(int, year_range.split('-'))
    dates = []
    for year in range(start_year, end_year + 1):
        for date in ['01-15', '04-15', '07-15', '10-15']:
            dates.append(f"{year}-{date}")
    return pd.to_datetime(dates)


def adjust_attributes(drought_data):
    # dataset.lon.attrs['units'] = "degrees_east"
    dataset.lon.attrs['long_name'] = "Longitude"
    # dataset.lat.attrs['units'] = "degrees_north"
    dataset.lat.attrs['long_name'] = "Latitude"
    # dataset.time.attrs['units'] = "standard time"
    dataset.time.attrs['long_name'] = "Time"
    
    drought_data['magnitude'].attrs['units'] = "m3/d"
    drought_data['magnitude'].attrs['long_name'] = "Deficit Magnitude"
    
    drought_data['startDate'].attrs['units'] = "days since 1950-01-01 0:0:0"
    drought_data['startDate'].attrs['long_name'] = "Event Start Date"
    
    drought_data['duration'].attrs['units'] = "days"
    drought_data['duration'].attrs['long_name'] = "Event Duration"


def set_global_attributes(dataset, start_date, end_date):
    """
    Adds global attributes to the dataset from a YAML template.
    """
    # Define the path to the YAML file
    yaml_file_path = '/storage/home/nargessayah/narges_scripts/drought/global_attribute_template_deficit.yml'
    
    # Read the YAML file
    with open(yaml_file_path, 'r') as file:
        global_attrs = yaml.safe_load(file)
    
    # Update dynamic attributes
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    global_attrs['creation_date'] = current_time
    # Ensure start_date and end_date are formatted as strings
    global_attrs['climo_start_time'] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(start_date, datetime) else start_date
    global_attrs['climo_end_time'] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(end_date, datetime) else end_date
    
    # Convert any datetime.datetime objects in global_attrs to strings
    for key, value in global_attrs.items():
        if isinstance(value, datetime):
            dataset.attrs[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            dataset.attrs[key] = value

    return dataset


def save_max_magnitude_droughts(drought_data, output_directory, ensemble_name, year_range):
     # Define the bounds for valid startDate values, assuming they're already in the desired numeric format.
    min_valid_date = 0  # Assuming 0 represents the minimum valid start date in your numeric format
    max_valid_date = (np.datetime64('2100-12-31') - np.datetime64('1950-01-01')).astype('timedelta64[D]').astype(int)

    # Correct startDate values that are out of bounds
    # invalid_startDate_mask = (drought_data['startDate'].values < min_valid_date) | (drought_data['startDate'].values > max_valid_date)
    # drought_data['startDate'].values[invalid_startDate_mask] = 0

     # Check if any startDate values are outside the valid range
    startDate_values = drought_data['startDate'].values
    if ((startDate_values < min_valid_date) | (startDate_values > max_valid_date)).any():
        raise ValueError("Error: Found 'startDate' values outside the valid range. Processing halted.")

    output_folder = os.path.join(output_directory, "max_magnitude_droughts", ensemble_name)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"max_magnitude_drought_{ensemble_name}_{year_range}.nc")
    
    # Ensure the dataset is adjusted for attributes before saving
    adjust_attributes(drought_data)
   # Specify encoding for startDate to ensure NaT values are handled correctly
    encoding = {
        # 'startDate': {
        #     '_FillValue': None,  # This tells the NetCDF library to use NaN or an appropriate fill value for NaT
        #     'dtype': 'double'  
        # },
        'time': {
            'dtype': 'double',
            'units': 'days since 1950-01-01 0:0:0',
            'calendar': 'standard'
        }
    }
    
    # Saving the dataset with specified encoding
    drought_data.to_netcdf(output_path, encoding=encoding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate seasonal Q0s and most extreme drought events.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Domain name")
    parser.add_argument("-read", "--input_directory", type=str, required=True, help="Directory path to the input NetCDF files.")
    parser.add_argument("-write", "--output_directory", type=str, required=True, help="Directory path to the output files.")
    args = parser.parse_args()
    
    seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    start_dates = ['1950-01-01']#, '2001-01-01', '2011-01-01', '2021-01-01', '2031-01-01', '2041-01-01', '2051-01-01', '2061-01-01', '2071-01-01']
    end_dates = ['2000-12-31']#, '2030-01-01', '2040-12-31', '2050-01-01', '2060-01-01', '2070-12-31', '2080-01-01', '2090-01-01', '2100-12-31']
    # mid_window = [1975]#, 2015, 2025, 2035, 2045, 2055, 2065, 2075, 2085] 

    for window_start, window_end in zip(start_dates, end_dates):
        year_range = f"{window_start[:4]}-{window_end[:4]}"
        for run in range(1, 2): #6):
            for init in range(1, 2): #11):
                file_name = f"CanESM2-LE_historical-r{run}_r{init}i1p1/flow/streamflow_day_RVIC_CanESM2-LE_historical-r{run}_r{init}i1p1_1950-2100_fraser.nc"
                file_path = os.path.join(args.input_directory, file_name)
                if os.path.exists(file_path):
                    print(f"Processing {file_name}...")
                    # Open the NetCDF file as an xarray Dataset, ensuring time values are decoded
                    dataset = xr.open_dataset(file_path, decode_times=True)
                    seasonal_Q0s = calculate_seasonal_thresholds(dataset, seasons, window_start, window_end)
                    ensemble_name = f"RVIC_CanESM2-LE_historical-r{run}_r{init}i1p1"
                    # save_Q0s(seasonal_Q0s, args.output_directory, ensemble_name, year_range)
                    mid_season_dates = generate_mid_season_dates(year_range)
                    drought_data = find_max_magnitude_droughts(dataset, seasonal_Q0s, window_start, window_end, mid_season_dates, seasons)
                    drought_data = set_global_attributes(drought_data, window_start, window_end)
                    save_max_magnitude_droughts(drought_data, args.output_directory, ensemble_name, year_range)
                    dataset.close()