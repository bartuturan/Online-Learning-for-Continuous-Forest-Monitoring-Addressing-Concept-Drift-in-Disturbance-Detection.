import json
import numpy as np

def restore_dataset_attrs(cube_data):
    """
        Restore cube data attributes from JSON strings back to their original formats.
    Parameters:
        - cube_data: The xarray Dataset containing the cube data.
    """
    if "history" in cube_data.attrs:
        cube_data.attrs['history'] = json.loads(cube_data.attrs['history'])
    
    #restore dumped variables
    for v in cube_data.attrs.get("dumped", []):
        if v != "":
            cube_data.attrs[v] = json.loads(cube_data.attrs[v])
    cube_data.attrs.pop("dumped", None)

    for var in cube_data.data_vars:
        for v in cube_data[var].attrs.get("dumped", []):
            if v != "":
                cube_data[var].attrs[v] = json.loads(cube_data[var].attrs[v])
        cube_data[var].attrs.pop("dumped", None)

        #restore original dates
        if "original_dates" in cube_data[var].attrs:
            per_year_dates = cube_data[var].attrs["original_dates"]
            for year in per_year_dates:
                per_year_dates[year] = np.asarray(per_year_dates[year], dtype='datetime64[s]')
            cube_data[var].attrs["original_dates"] = per_year_dates
    
        if "sources" in cube_data[var].attrs:
            cube_data[var].attrs["sources"] = json.loads(cube_data[var].attrs['sources'])
        if "flag_values" in cube_data[var].attrs:
            cube_data[var].attrs["flag_dict"] = dict(zip(cube_data[var].attrs["flag_values"], cube_data[var].attrs["flag_meanings"]))

    if "weather_stats" in cube_data.data_vars:
        coords_wstats = cube_data["weather_stats"].coords
        cube_data["weather_stats"] = cube_data["weather_stats"].stack(stat_weather=("statistic", "weather_band"))
        cube_data["weather_stats_norm"] = cube_data["weather_stats_norm"].stack(stat_weather=("statistic", "weather_band"))
        cube_data.coords["stat_weather"] = [f"{stat}#{band}" for stat in coords_wstats["statistic"].values for band in coords_wstats["weather_band"].values]
        return cube_data
    else:
        return cube_data
    

def train_test_splitting(cube_data, MIN_, MAX_, LIM_TEST):
    """
        Split the cube data into training, validation, and test sets based on years.
    Parameters:
        - cube_data: The xarray Dataset containing the cube data.
        - MIN_: Minimum year for training data.
        - MAX_: Maximum year for test data.
        - LIM_TEST: Year that separates training and test data.
    Returns:
        - train_cube: xarray Dataset for training data.
        - val_cube: xarray Dataset for validation data.
        - test_cube: xarray Dataset for test data.
    """
    LIM_VAL = LIM_TEST-1 #last training year for validation

    train_years = np.arange(MIN_, LIM_VAL)
    LEN_ = min(len(train_years),5) -1
    val_years =  np.arange(LIM_VAL-LEN_, LIM_TEST)
    test_years = np.arange(LIM_TEST-LEN_, MAX_ +1)

    return cube_data.sel(year=train_years), cube_data.sel(year=val_years), cube_data.sel(year=test_years)



def clean_nondeforested_val(data_cube):
    #drop cubes that have deforestation different from 1 in all image
    if "x_30" not in data_cube.dims:
        mask_nondeforested = (data_cube['disturbances'].isel(year=-1) != 1).all(dim=['x', 'y'])
    else:
        mask_nondeforested = (data_cube['disturbances'].isel(year=-1) != 1).all(dim=['x_30', 'y_30'])
    cubes_with_deforestation = data_cube['cube'].where(mask_nondeforested, drop=True).values
    return data_cube.sel(cube=cubes_with_deforestation)