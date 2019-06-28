# Disaster Message Classification

This project utilizes NLP and machine learning in order to categorize messages
incoming during time of crisis. Attaching appropriate label to the message
would allow it to reach organization responsible for providing help.
It would speed up the incoming help as messages could be redirected
automatically after being received as no one would need to read them beforehand
and decide which service should be alarmed.

There are 35 different labels.

It includes following scripts:
- [download_data.py](download_data.py) - script that fetches the data,
- [process_data.py](process_data.py) - ETL script that cleans preprocesses the data and saves it to sqlite database,
- [train_classifier.py](train_classifier.py) - scripts that loads data from .db file, performs feature engineering, performs hyperparameter tuning via GridSearch to tune LGBM Classifier and saves whole process in .pkl file,
- [start_server.py](start_server.py) - hosts flask webpage that allows to insert a query and receive classifications.

## Getting Started

### Prerequisites

Project requires Python 3.6.6 with installed [virtualenv](https://pypi.org/project/virtualenv/).

### Setup

1. Pull the project.
2. Setup new virtualenv with Python 3.6.6.
3. Install requirements.txt file via pip.
4. Move to project directory.
5. Run the scripts in the following order: `download_data.py`, `preprocess_data.py`, `train_data.py`, `start_server.py`.
6. Access Flask server webpage at `http://0.0.0.0:3001/`,

## Running scripts

All parameters are optional and have default values compatible with project root directory.

1. Downloading project `messages.csv` and `categories.csv`
```
$ python download_data.py
```

2. Running ETL pipeline
```
$ python process_data.py --messages_path <input_data_path> --categories_path <input_data_path> --db_path <output_db_file_path>
```

3. Training classifier
```
$ python train_classifier.py --db_path <input_db_file_path> --model_path <output_pkl_path>
```

4. Starting Flash server
```
$ python start_server.py --db_path <input_db_file_path> --model_path <input_pkl_path>
```

## Model details

#### Build time
GridSearch process will take around 7 hours to finish. Final model is not included in repo as it takes 159mb.

#### Model parameters
```
MultiOutputClassifier(estimator=LGBMClassifier(
        boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=25,
        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
        n_estimators=100, n_jobs=-1, num_leaves=20, objective=None,
        random_state=None, reg_alpha=0.0, reg_lambda=0.0, seed=1500,
        silent=True, subsample=1.0, subsample_for_bin=200000,
        subsample_freq=0),
           n_jobs=None)
```

#### Model score per class
```
                        f1_score	precision       recall
               related	0.822350	0.830450	0.838070
               request	0.908440	0.909965	0.913790
                 offer	0.998820	0.998805	0.998856
           aid_related	0.809726	0.811906	0.811940
          medical_help	0.938891	0.941192	0.947358
      medical_products	0.966633	0.969808	0.971200
     search_and_rescue	0.975727	0.977441	0.979783
              security	0.979746	0.981891	0.984169
                 water	0.980340	0.980155	0.980736
                  food	0.973298	0.973298	0.973298
               shelter	0.960914	0.960562	0.962235
              clothing	0.995552	0.995712	0.995804
                 money	0.990127	0.990311	0.990845
        missing_people	0.989553	0.990584	0.991417
              refugees	0.974522	0.976757	0.978257
                 death	0.983988	0.984256	0.984932
             other_aid	0.867964	0.884424	0.893382
infrastructure_related	0.931154	0.942047	0.947740
             transport	0.957973	0.964462	0.965859
             buildings	0.968422	0.969536	0.971581
           electricity	0.990637	0.990656	0.991226
                 tools	0.998802	0.998857	0.998856
             hospitals	0.994580	0.994812	0.995232
                 shops	0.999608	0.999619	0.999619
           aid_centers	0.993060	0.993487	0.993897
  other_infrastructure	0.949567	0.955573	0.962426
       weather_related	0.896038	0.897199	0.898531
                floods	0.961480	0.963530	0.964715
                 storm	0.963817	0.963479	0.964333
                  fire	0.992343	0.993204	0.993324
            earthquake	0.980145	0.980042	0.980355
                  cold	0.991472	0.991880	0.992180
         other_weather	0.952121	0.956218	0.961282
         direct_report	0.858519	0.864495	0.871066
```
#### Model mean score
```
f1_score     0.956492
precision    0.958807
recall       0.960884
```

#### Examples

Input:
```
query = ["I like pancakes!"]
category_indices = [i for i, result in enumerate(model.predict(query).ravel()) if result == 1]
print("Attached labels: {}".format(categories[category_indices]))
```
Output:
```
Attached labels: []
```
---
Input:
```
query = ["It's heavy raining in xyz. Whole area is flooded and we are stuck on the roof of our house."]
category_indices = [i for i, result in enumerate(model.predict(query).ravel()) if result == 1]
print("Attached labels: {}".format(categories[category_indices]))
```
Output:
```
Attached labels: ['related' 'shelter' 'buildings' 'weather_related' 'floods' 'storm']
```
---
Input:
```
query = ["The building in front of my house is on fire!"]
category_indices = [i for i, result in enumerate(model.predict(query).ravel()) if result == 1]
print("Attached labels: {}".format(categories[category_indices]))
```
Output:
```
Attached labels: ['related' 'fire']
```
---
Input:
```
query = ["We got cut off the electricity. It's heavy snowing and we are freezing cold."]
category_indices = [i for i, result in enumerate(model.predict(query).ravel()) if result == 1]
print("Attached labels: {}".format(categories[category_indices]))
```
Output:
```
Attached labels: ['related' 'electricity' 'weather_related' 'cold']
```
---
Input:
```
query = ["My friend broke the leg. Please send medical help to xyz."]
category_indices = [i for i, result in enumerate(model.predict(query).ravel()) if result == 1]
print("Attached labels: {}".format(categories[category_indices]))
```
Output:
```
Attached labels: ['related' 'request' 'aid_related' 'medical_help' 'direct_report']
```
---

### Flask webpage

- interactive text input field and model result display

![Flask webpage example 1](images/flask_example.png?raw=true "Flask Example 1")

- interactive training dataset overview

![Flask webpage example 2](images/plot_1.png?raw=true "Flask Example 2")

![Flask webpage example 3](images/plot_2.png?raw=true "Flask Example 3")

## Built With

* [Udacity](https://www.udacity.com/) - project for passing Data Engineering section and Data Science Nanodegree
* [Figure Eight](https://www.figure-eight.com/) - data contributors

## License

This project is licensed under the MIT License.
