Implementation for the paper "LGFat-RGCN: Faster Attention with Heterogeneous RGCN for Medical Label Graph Generation"
-----

1、Configure environment
-----
Please refer to the following way to configure the development environment:

```conda create -n LGFat-RGCN python=3.7```.

```conda activate LGFat-RGCN```.

```pip install -r requirements.txt```.

2、Reorganize the directory structure
-----
Please reorganize the directory structure of this project as follows: 

a.After downloading all the code of this project.

b.Take out all the source code files in all the directories except the data directory, __pycache__ directory and elmo directory.

c.Place them in the main LGFat-RGCN directory.

3、Train and test
-----
Run ```python main.py ```. (Due to the existence of abnormal situations such as server disconnection, it is recommended to use tools such as nohup or tmux to protect the stable operation of the program.)


4、Acknowledgement
-----
We are sincerely grateful to the reviewers for your precious time and effort in reviewing our article, and we will adopt your valuable review comments seriously!
