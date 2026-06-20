# ai-datacenters

The respository is inspired by the data heat island effect, which was first introduced in the following paper: 

https://arxiv.org/pdf/2603.20897

The current project focuses on 187 global AI data centers and their corresponding 92 nearby or enclosed city centers. More specifically, data centers here refer to sites with large-scale computations or cloud operations. For each category, the following datasets are collected: 

* Geographic coordinates: latitudes and longitudes for spatial alignment
* Average yearly temperatures from 2006 untill mid 2026
* Static hydrological risk indicators (baseline water stress and drought risk)

Data Sources

1. OpenStreetMap / Nominatim
   © OpenStreetMap contributors
   https://www.openstreetmap.org/copyright

2. Open-Meteo
   Weather data by Open-Meteo.com
   https://open-meteo.com/
   Licensed under CC BY 4.0

3. Aqueduct 4.0 (World Resources Institute)
   Kuzma et al. (2023), Aqueduct 4.0: Updated Decision-Relevant Global Water Risk Indicators
   World Resources Institute (WRI)
   Licensed under CC BY 4.0
   https://www.wri.org/aqueduct

The above data collection was performed using several libraraies in Python, which are included under src. 

Methods and techniques: 

The first step includes constructing a spatial-temporal environmental datasets by merging the static hydrological risk indicators with the temperature time series using a coordinate-based spatial join. Accordingly, four variables are then defined. These are:

* Year-over-year temperature change
* Long-term cumulative temperature deviation
* Baseline water stress
* Drought risk index

For analysing these variables, a combination of the following statistical and machine learning methods was applied:

* Non-parametric testing (Mann-Whitney U) to compare the distributions between data centers and city centers.
* Correlation analysis
* PCA for dimensionality reduction
* Random forest classification (GroupKFold validation)
* HDBSCAN clustering to identify environmental risk clusters.

Findings and Conclusions: 

1- Significant statistical differences were observed between the two location types.
<table>
  <tr>
    <th align="center">Warming Rate</th>
    <th align="center">Temperature Cumulative Change</th>
  </tr>
  <tr>
    <td><img src="figures/warming_rate_comparison.png" width="100%" alt="Caption 1" /></td>
    <td><img src="figures/temp_cumulative_change_comparison.png" width="100%" alt="Caption 2" /></td>
  </tr>
</table>


All 4 variables exhibit statistically significant distributional differences between the two groups, with the strongest effect been observed in baseline water stress. In particular, data centers demonstrate higher median, higher upper quartile, and high maximum for cumulative warming.

2- Thermal and hydrological risks are largly uncorrelated.

![result](figures/correlation_heatmap.png)

Correlation analysis shows near-zero relationships between temperatures variables and hydrological indicators.

3- Temperature variables dominate feature importance analysis predictions.

![result](figures/rf_feature_importance.png)

ML models identify temperature-based variables as the primary driver for classification, accounting for roughly two thirds of total feature importance. Therefore, feature importance reflects model-specific predictive contribution.

4- Distinct environmental risk groups are present based on HDBSCAN analysis

![result](figures/pca_clusters.png)

HDBSCAN identfies multiple environmental risk clusers. These are: 

* an extreme hydrolofical-risk cluster  
* a large dominate cluster with moderate-risk conditions
* a large outlier cluster that contains mixed risk conditiosn

5- Random forest classification provided limited ability to separate classes. 

With Accuracy = 87% and City F1 = 0.08, it is clear that class imbalance was observed with the data centers class domintating, which affected the classification performance on the other group.


Author 

Dr. Basel Ali



