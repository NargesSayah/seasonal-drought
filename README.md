# Most Severe Seasonal Drought Events Analysis Using Thresholds

## Introduction

This project employs a comprehensive methodology to analyze extreme low-flow conditions in hydrological datasets, focusing on drought events that surpass predefined seasonal thresholds. Using streamflow data, the analysis starts by establishing baseline thresholds, then progresses to identify and quantify significant drought events. This study will provide insight into the temporal dynamics and impacts of droughts across different seasons, which is crucial for water resource management and planning.


## Implementation

### Prerequisites

Ensure the following tools are installed and configured:

- Python 3.6+   

- Xarray, Numpy, and Pandas libraries   

- YAML for configuration management   


### Script Execution

The main script is encapsulated in a shell wrapper, which sets up the environment and handles batch processing via an HPC job scheduler. The entire workflow is structured to facilitate reusability and modifications as required by the dataset or specific applications.


## Technical Overview

### Baseline Threshold Estimation 

- Seasonal analysis:    
Streamflow data are segmented by meteorological seasons to reflect natural variations in water availability, ensuring that the drought assessment is sensitive to seasonal dynamics.
  - DJF   
  - MAM    
  - JJA
  - SON   

- Seasonal thresholds (Q0):   
Each grid point's seasonal drought threshold is defined as the 0.05th quantile of streamflow values for each season within the initial period (1950-2000). This quantile serves as a benchmark to detect when streamflow levels indicate drought conditions.

### Drought Event Extraction

- Continuous drought identification:    
Using the established Q0 thresholds, the methodology continuously monitors streamflow data to pinpoint instances where it remains below these thresholds for two consecutive days or more, indicating a drought event.

- Drought Characteristics:    
Each identified drought is characterized by its start date, magnitude (m), and duration (d), which are calculated based on how long and how much streamflow remains below the seasonal thresholds.

  - Volume or Severity of the event $(d_{i})$: The cumulative deficit of streamflow below the threshold, i.e., the area under the threshold line and above the streamflow curve.
  - Intensity or Magnitude $m_{i} =  \frac{v_{i}}{d_{i}}$


### Maximum Magnitude Drought Identification 

For every season, the single most severe drought event per grid point is identified, informing the most impactful droughts that could influence ecosystem and resource management strategies.


## Data Organization and Output

### Structured output:
The outputs are systematically organized into structured formats that include the magnitude, start date, and duration of each drought event, tagged with appropriate metadata to ensure clarity and usability in further analyses.

### Global Attributes:
The dataset includes global attributes set from a YAML template of standard attributes to ensure comprehensive metadata. As a result, data can be more easily interpreted and used in scientific and operational contexts.
