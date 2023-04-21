"""
Recipes available to data with tags ['IGRINS', 'CAL', 'ARC'].
"""

recipe_tags = {'IGRINS', 'CAL', 'ARC'}

def makeProcessedArc(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.referencePixelsCorrect()
    p.stackArcs()    # or stackFrames if there's nothing special about the stacking.

    # The old IGRINS pipeline way looks like this
    p.registerArc()
    p.determine2DWavelengthSolution()

    # The DRAGONS way would maybe be more like this
    # p.determineWavelengthSolution()  # equilavent to registerArc
    # p.determineDistortion()
    # p.determineDeepWavelengthSolution()

    p.storeProcessedArc()
