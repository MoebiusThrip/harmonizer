

from importlib import reload
from matplotlib import pyplot as plt
from time import sleep





def wheel(name=''):
    """"Plot a pie chart of the chromatic wheel

    Arguments:
        name: str

    Returns:
        None
    """

    # default name
    presets = {}
    presets['ionian'] = 'C D E F G A B'
    presets['major'] = 'C E G'
    presets['minor'] = 'C Eb G'
    presets['minor7th'] = 'C Eb G Bb'
    presets['dominant7th'] = 'C E G Bb'
    presets['minor6th'] = 'C Eb G A'
    presets['harmonic'] = 'C D Eb F G Ab B'
    presets['diminished'] = 'C Eb Gb A'
    presets['dorian'] = 'C D Eb F G A Bb'
    presets['phrygian'] = 'C Db Eb F G Ab Bb'
    presets['locrian'] = 'C Db Eb F Gb Ab Bb'
    presets['lydian'] = 'C D E Gb G A B'
    presets['mixolydian'] = 'C D E F G A Bb'
    presets['aeolian'] = 'C D Eb F G Ab Bb'
    presets['whole'] = 'C D E Gb Ab Bb'
    presets['augmented'] = 'C E Ab'

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']
    labels = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    labels = ['G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb']
    sizes = [1] * 12
    colors = ['lightgray', 'black', 'magenta', 'crimson', 'darkorange', 'green', 'midnightblue', 'dodgerblue', 'purple', 'cyan', 'yellow', 'chartreuse']

    # set default colors scheme
    scheme = {label: color for label, color in zip(labels, colors)}
    chord = {label: 'white' for label in labels}


    # try to get inclusions from defaults
    try:

        # get from defaults
        inclusions = presets[name]

    # otherwise
    except KeyError:

        # name by default
        inclusions = name

    # split inclusions
    inclusions = inclusions.split()

    # set default inclusions
    if len(inclusions) < 1:

        # set to all
        inclusions = labels

    # apply inclusions
    for inclusion in inclusions:

        # set color with scheme
        chord[inclusion] = scheme[inclusion]

    # reset colors
    colors = [chord[tone] for tone in labels]

    # reverse lists to go clockwise
    labels.reverse()
    colors.reverse()

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, startangle=105, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(name)
    plt.show()
    plt.close()

    return None


wheel('G D')
wheel('Eb G D')
wheel('D A')
wheel('Bb D A')
wheel('F C')
wheel('Db F C')
wheel('G D')
