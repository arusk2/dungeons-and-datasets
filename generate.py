# This file is a cmdline script for generating descriptions from a pretrained model
# it takes:
#   1: number of sentences
#   2: a seed word/phrase
import sys

if len(sys.argv) == 1:
    print("Please enter a number of sentences to generate followed by a seed word or phrase!")
else:
    print(f"I should make {sys.argv[1]} sentences")
    sentence = ' '.join(sys.argv[2:])
    print(f"The seed is: {sentence}")