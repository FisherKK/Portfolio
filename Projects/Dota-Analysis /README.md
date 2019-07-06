# How to write data driven blog post? - Dota 2 Analysis

This project was created as a part of [Data Scientist Nanodegree](https://eu.udacity.com/course/data-scientist-nanodegree--nd025)
at [Udacity](https://eu.udacity.com/?cjevent=d1a59cbeab1111e9834e02630a18050b), for the first project “Write A Data
Science Blog Post”. Goal of this project is to present my ability of getting access to data, analyzing it, understanding 
and presenting in elegant way. 

Backend side:
- [ETL pipeline.ipynb](notebook/ETL%20Pipeline.ipynb) - notebook where data scraping and cleaning process occurs, saves d
downloaded data to [data](data) directory
- [Data Analysis.ipynb](notebook/Data%20Analysis.ipynb) - notebook where data is analysed and visualised, saves graphs to
[image](/image) directory

Fronted side:
- blog post available under following [LINK](https://medium.com/@krzyk.kamil/dota-2-valid-career-path-or-just-extraordinary-form-of-entertainment-91c456ea82fc)

Other:
- As an exception, downloaded data (in folder [data](/data)) and generated graphs (in folder [image](/image)) used in
blog post were kept in this repository. This is because data changes in realtime and same results might not be reproduced in the future.

## Getting Started

### Prerequisites

Project requires Python 3.6.6 with installed [virtualenv](https://pypi.org/project/virtualenv/).

### Setup

1. Pull the project to `<project_dir>`.
2. Setup new virtualenv with Python 3.6.6.
3. Install requirements.txt file via pip.
4. Run jupyter notebook instance from terminal by invoking `jupyter notebook` command. (available by default 
at http://localhost:8888 in any browser)
5. Navigate to `<project_dir>/notebook`.
6. Run notebooks in the following order `ETL Pipeline.ipynb`, `Data Analysis.ipynb`.


## Built With

* [Udacity](https://www.udacity.com/) - project for passing Data Engineering section and Data Science Nanodegree
* [OpenDota](https://www.opendota.com/) - data contributors

## License

This project is licensed under the MIT License.