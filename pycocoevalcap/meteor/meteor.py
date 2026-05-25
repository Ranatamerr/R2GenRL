import subprocess
import os

METEOR_JAR = 'meteor-1.5.jar'

class Meteor:
    def __init__(self):
        self.meteor_p = None

    def compute_score(self, gts, res):
        return 0.0, [0.0] * len(gts)

    def _stat(self, hypothesis_str, reference_list):
        return ""

    def __del__(self):
        pass
