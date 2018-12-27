# SF-Parking-Prediction

In this project, we worked with parking datasets given to us from Parknav. We used an ensemble of random forest models and gradient boosting models to determine where parking would be available. We found ensembling many models together would give an improvement to our accuracy.

We also found cross validation to be a much better approach to validating our predictions than one carefully constructed set. After moving to cross validation, the results of the test set were much more in line with our validation set.

- Methods: Stacking Ensemble, Random Forests, XGBoost
- Tools: Python, Scikit-Learn, XGBoost, ETL, Google Geocoding API, AWS S3, Github