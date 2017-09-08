# FaceRec
A simple working facial recognition program.


## Installation:
    1. Install dependencies

    2. Download the pretrained models here: https://drive.google.com/file/d/0Bx4sNrhhaBr3TDRMMUN3aGtHZzg/view?usp=sharing
    
        Then extract those files into models

    3. Run main.py

## Requirements:
    Python3 (3.5 ++ is recommended)

## Dependencies:

    opencv3

    numpy

    tensorflow


## Howto:
    `python3 main.py` to run the program
    `python3 main.py --mode "input"` to add new user. Start turning left, right, up, down after inputting a new name. Turn slowly to avoid blurred images

### Flags:
   `--mode "input"` to add new user into the data set
    

## General information:

Architecture: Inception Resnet V1 

More information on the model: https://arxiv.org/abs/1602.07261

![GIF Demo](https://media.giphy.com/media/l378mx3j8ZsWlOuze/giphy.gif)

Live demo: https://www.youtube.com/watch?v=6CeCheBN0Mg


@Author: David Vu

## Credits:
    -  Pretrained models from: https://github.com/davidsandberg/facenet
