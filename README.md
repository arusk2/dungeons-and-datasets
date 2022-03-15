# Dungeons & Datasets
```
   (  )   /\   _                 (     
    \ |  (  \ ( \.(               )                      _____
  \  \ \  `  `   ) \             (  ___                 / _   \
 (_`    \+   . x  ( .\            \/   \____-----------/ (o)   \_
- .-               \+  ;          (  O                           \____
                          )        \_____________  `              \  /
(__                +- .( -'.- <. - _  VVVVVVV VV V\                 \/
(_____            ._._: <_ - <- _  (--  _AAAAAAA__A_/                  |
  .    /./.+-  . .- /  +--  - .     \______________//_              \_______
  (__ ' /x  / x _/ (                                  \___'          \     /
 , x / ( '  . / .  /                                      |           \   /
    /  /  _/ /    +                                      /              \/
   '  (__/                                             /                  \
```
An NLP dataset for creating rooms for use in World's Most Popular Roleplaying Game. This alpha release has 2,000 examples of 
location/room descriptions sourced from handpicked PDF adventures, random generators, and a processed version of the 
[Critical Role dataset](https://github.com/RevanthRameshkumar/CRD3).

A Project by:  
Zach Grow  
Andrew Ruskamp-White  
Joel Williams

## Installation
1. Clone the repository or download the release package. In order to clone the spaCy processed CRD3 dataset, you must
have [Git LFS](https://git-lfs.github.com/) installed.
2. We recommend a Python 3.8 interpreter (in our experience spaCy and CUDA have issues with 3.10). Your mileage with other
interpreter versions may vary.
3. Create a python virtual environment either [manually](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/) or through your IDE.
4. Install dependencies `pip install -r requirements.txt`

## Releases
For wizards that just want the dataset, cast "Magic Jar" on your computer then...
1. Choose a release version.
2. Download the csv or `curl https://github.com/arusk2/dungeons-and-datasets/releases/download/v0.0.2-alpha/dungeons-dataset.pkl -o dungeons-dataset.pkl`
in your repository for the pickled dataframe.
3. When reading in the raw csv, change the deliminator to '|', otherwise normally occurring commas will be interpreted
as separate columns.
