import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler
    
    
class CubeLoader(Dataset):
    def __init__(self, cube, lag_years, validation=False, augmentation=True, transforms={}, s2_extra_bands=["SCL", "forest_mask"]): #cloudmask
        self.dataset = cube.copy()
        self.years_train = self.dataset.year.values
        self.lag_years = lag_years
        self.min_year = self.years_train[0]
        self.weather_ver = [v for v in transforms.keys() if "weather" in v][0]
        self.validation = validation
        self.augmentation = augmentation
        self.transforms = transforms
        self.s2_extra_bands = s2_extra_bands

        self.samples = []
        for i_sample in range(len(self.dataset.cube)):
            if self.validation:
                self.samples.append((i_sample, self.years_train[-1]))
            else:
                for year in self.years_train[1:]: #avoid first year as no lag data available
                    self.samples.append((i_sample, year))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        i_idx, i_year = self.samples[item]
        
        data_idx = self.dataset.isel(cube=i_idx).copy()
        back_year = max(i_year-self.lag_years+1, self.min_year)
        #DATA AUGMENTATION FOR S2 DATA DURING TRAINING
        topK = np.random.choice(data_idx["topK"].values, size=1)[0] if (not self.validation and self.augmentation) else 0

        s2_products = data_idx.sel(topK=topK, year=slice(back_year,i_year))
        sensors_sample = { #covariates (input data)
            "S2": s2_products["S2"].values,
            "SCL": s2_products["SCL"].values,
            #"cloudmask": s2_products["cloudmask"].values,
            self.weather_ver: data_idx[self.weather_ver].sel(**{"year" if "stats" in self.weather_ver else "time": slice(str(back_year), str(i_year))}).values,
            "dem": data_idx["dem"].values,
            "lccs_class": data_idx["lccs_class"].values,
            "position": data_idx["position"].sel(pos=["lon", "lat"]).values,
        }

        labels_data = data_idx.sel(year=i_year)
        labels_sample = { #reference data (labels)
            "forest_mask": labels_data["forest_mask"].values,
            "disturbances": labels_data["disturbances"].values,
            "disturbance_agent": labels_data["disturbance_agent"].values,
        }
        
        for key in sensors_sample:
            tensor_sample = torch.tensor(sensors_sample[key])
            sensors_sample[key] = self.transforms[key](tensor_sample)

        for key in labels_sample:
            labels_sample[key] = torch.tensor(labels_sample[key], dtype=torch.long)

        if len(self.s2_extra_bands) > 0:
            attach = [sensors_sample.pop(band) for band in self.s2_extra_bands if band in sensors_sample]
            if "forest_mask" in self.s2_extra_bands:
                attach += [labels_sample["forest_mask"].repeat(sensors_sample["S2"].shape[0], 1, 1).unsqueeze(-1)]
            sensors_sample["S2"] = torch.concat([sensors_sample["S2"]] + attach, dim=-1)

        return {"input": sensors_sample, "label": labels_sample, "cube_name": str(data_idx["cube"].values), "idx": i_idx, "year": i_year}

class YearGroupedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Group indices by year
        self.year_to_indices = {}
        for i, (_, i_year) in enumerate(dataset.samples):
            self.year_to_indices.setdefault(i_year, []).append(i)
        
    def __iter__(self):
        all_batches = []
        for year, indices in self.year_to_indices.items():
            np.random.shuffle(indices)
            # Create batches for that year
            year_batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            all_batches.extend(year_batches)
        np.random.shuffle(all_batches) # Shuffle the order of batches across years
        
        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum( len(indices) // self.batch_size for indices in self.year_to_indices.values() )
        