__author__ = 'TheOz'

#  harmonizers to learn where notes are on a page and analyze them

# import system tools
from importlib import reload
import json, csv, os, sys
import numpy
from time import clock, time
from random import random, choice
from datetime import datetime
from math import sqrt

# import pil and skimage
from PIL import Image, ImageFont, ImageDraw

# import sklearn
from sklearn.cluster import AffinityPropagation, MeanShift


# import tensorflow
import tensorflow

# import keras
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences

# status
print('imported modules.')


# Harmonizer class for an LSTM to predict an airn context vector on the context vectors of all conversation utterances prior
class Harmonizer(object):
    """Harmonizer class to analyze and color a piece of sheet music.

    Inherits from:
        object
    """

    def __init__(self, directory='pieces/concerto', key='D', signature='F', mode='flats', clef='bass'):
        """Initialize an Harmonizer instance.

        Arguments:
            None
        """

        # image parameters
        self.gray = 0.2
        self.darkness = 20
        self.height = 39
        self.width = 39

        # ingested data
        self.data = {}
        self.samples = {}
        self.categories = []
        self.mirror = {}

        # training and holdouts
        self.fraction = 0.1
        self.training = []
        self.holdouts = []

        # network hidden layer sizes
        self.hiddens = 200
        self.hiddensii = 200
        self.hiddensiii = 9600
        self.hiddensiv = 96
        self.hiddensv = 1

        # activation function
        self.activation = 'relu'
        self.activationii = 'tanh'
        self.activationiii = 'softmax'

        # set model references
        self.model = None
        self.path = 'harmonizer.h5'

        # set training parameters
        self.eras = 5
        self.epochs = 10
        self.validation = 0.1

        # prediction threshold
        self.threshold = 0.1

        # colors
        self.colors = {}
        self.palette = {}

        # staff properties
        self.positions = (-2, 16)

        # discover properties
        self.increment = 12
        self.smearing = 4
        self.criteria = 0.98
        self.reports = []

        # current sheet harmonic properties
        self.key = key
        self.clef = clef
        self.signature = signature
        self.mode = mode

        # current sheet properties
        self.directory = directory
        self.sheet = None
        self.silhouette = None
        self.original = None
        self.painting = None
        self.original = None
        self.staff = None
        self.measures = None
        self.spacing = None
        self.notes = None
        self.tiles = None
        self.chords = []

        # general harmonic properties
        self.clefs = {}
        self.wheel = {}
        self.spectrum = {}
        self.signatures = {}
        self.lexicon = {}
        self.enharmonics = {}

        # annotation properties
        self.font = 'Arial Black'
        self.size = 30

        # define general harmonic properties
        self._define()
        self._tabulate()

        # construct sheet
        self._glue()

        return

    def _balance(self, samples):
        """Balance each list of samples by duplicating entries.

        Arguments:
            samples: dist of lists

        Returns:
            dict of lists
        """

        # get maximum length and expand each category
        maximum = max([len(members) for members in samples.values()])
        expansions = {}
        for category, members in samples.items():

            # shuffle members
            members.sort(key=lambda member: random())

            # expand
            ratio = int(maximum / len(members))
            expansion = members * (ratio + 1)
            expansions[category] = expansion[:maximum]

        return expansions

    def _bound(self, point, width=None, height=None):
        """Determine standard sized-boundary around point.

        Arguments:
            point: (int, int) tuple
            width=None: int, the width
            height=None: int, the height

        Returns:
            (int, int, int, int) tuple
        """

        # set default width
        if not width:

            # set width
            width = self.width

        # set default height

            # set height
            height = self.height

        # set center
        horizontal, vertical = point

        # calculate bounding points
        top = vertical - int(height / 2)
        bottom = vertical + 1 + int(height / 2)
        left = horizontal - int(width / 2)
        right = horizontal + 1 + int(width / 2)

        return top, bottom, left, right

    def _build(self):
        """Build the neural network model.

        Arguments:
            None

        Returns:
            None

        Sets
            self.model
        """

        # status
        print('building...')

        # Initialize the Model
        model = Sequential()

        # add model layers
        size = len(self.categories)
        model.add(Conv2D(size * 2, kernel_size=3, activation='relu', input_shape=(self.width, self.height, 1)))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(size * 4, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(size, activation='softmax'))

        # compile
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

        # summarize and set attribute
        model.summary()
        self.model = model

        return None

    def _center(self, shadow):
        """Find the center point of a shadow.

        Arguments:
            shadow: numpy array

        Returns:
            (int, int) tuple
        """

        # get the shape
        height, width = shadow.shape

        # divide by two
        vertical = int(height / 2)
        horizontal = int(width / 2)
        center = horizontal, vertical

        return center

    def _compare(self, vector, vectorii):
        """Compare two vectors by cosine similarity.

        Arguments:
            vector: list of floats
            vectorii: list of floats

        Returns:
            float, the cosine similarity
        """

        # make into numpy arrays
        vector = numpy.array(vector)
        vectorii = numpy.array(vectorii)

        # compute the cosine similarity
        similarity = vector.dot(vectorii) / sqrt(vector.dot(vector) * vectorii.dot(vectorii))

        return similarity

    def _deepen(self, shadow):
        """Deepen the shadow by one layer to pass to CNN.

        Arguments:
            shadow: two 2 array

        Returns:
            3-d array
        """

        # deepen shadow
        prism = [[[float(entry)] for entry in row] for row in shadow]
        prism = numpy.array(prism)

        return prism

    def _define(self):
        """Populate harmonic information.

        Arguments:
            None

        Returns:
            None

        Populates:
            self.clefs
            self.spectrum
            self.wheel
            self.signatures
        """

        # make notes
        notes = 'ABCDEFG'

        # define clefs
        self.clefs['treble'] = {position: notes[(position + 4) % 7] for position in range(*self.positions)}
        self.clefs['bass'] = {position: notes[(position + 6) % 7] for position in range(*self.positions)}

        # for flats
        if self.mode == 'flats':
            
            # define pitches and mirror
            pitches = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
            mirror = {pitch: index for index, pitch in enumerate(pitches)}

        # or sharps
        if self.mode == 'sharps':
            
            # define pitches and mirror
            pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            mirror = {pitch: index for index, pitch in enumerate(pitches)}

        # define intervals and spectrum
        intervals = ['1', 'b9', '9', 'b3', '3', '11', 'b5', '5', 'b13', '13', 'b7', '7']
        rainbow = ['white', 'black', 'magenta', 'red', 'orange', 'green', 'indigo', 'blue', 'violet', 'cyan', 'yellow', 'acid']
        spectrum = {interval: color for interval, color in zip(intervals, rainbow)}
        self.spectrum = spectrum

        # define wheel
        indexing = lambda pitch, pitchii: (mirror[pitchii] - mirror[pitch]) % len(spectrum)
        self.wheel = {pitch: {pitchii: intervals[indexing(pitch, pitchii)] for pitchii in pitches} for pitch in pitches}

        # define signatures
        self.signatures['C'] = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        self.signatures['F'] = ['F', 'G', 'A', 'Bb', 'C', 'D', 'E']

        # define flat enharmonics
        flats = {note: note for note in notes}
        flats.update({note: note for note in ('Db', 'Eb', 'Gb', 'Ab', 'Bb')})
        flats.update({pair.split()[0]: pair.split()[1] for pair in ['C# Db', 'D# Eb', 'F# Gb', 'G# Ab', 'A# Bb']})
        flats.update({pair.split()[0]: pair.split()[1] for pair in ['E# F', 'B# C']})

        # define sharp enharmonics
        sharps = {note: note for note in notes}
        sharps.update({note: note for note in ('C#', 'D#', 'F#', 'G#', 'A#')})
        sharps.update({pair.split()[0]: pair.split()[1] for pair in ['Db C#', 'Eb D#', 'Gb F#', 'Ab G#', 'Bb A#']})
        sharps.update({pair.split()[0]: pair.split()[1] for pair in ['Fb E', 'Cb B']})

        # set enharmonics
        self.enharmonics = {'flats': flats, 'sharps': sharps}

        return None

    def _describe(self, element):
        """Describe an element in terms of its harmonic properties.

        Arguments:
            element: dict

        Returns:
            dict
        """

        # make note
        note = {'center': element['center']}
        note['pitch'] = self.clefs[self.clef][element['position']]
        note['interval'] = self.wheel[self.key][note['pitch']]
        note['color'] = self.spectrum[note['interval']]

        return note

    def _detect(self, row, number=100, fraction=0.5, criterion=None):
        """Detect a horizontal line by checking a number of points at random.

        Arguments:
            row: list of int, a shadow row
            number=100: how many rows to check
            fraction=0.5: how much of the page to check.
            criterion=None: function object

        Returns:
            boolean, black line?
        """

        # set default criterion
        if not criterion:

            # set criterion:
            criterion = lambda x: x > -0.3

        # get row indices
        indices = [index for index, _ in enumerate(row)]

        # calculate indent
        indent = int(fraction * len(row) / 2)
        indices = indices[indent: -indent]

        # pick points at random
        black = True
        for trial in range(number):

            # get index
            index = choice(indices)
            if criterion(row[index]):

                # break, not a black line
                black = False
                break

        return black

    def _fix(self, element):
        """Fix the accidentals on a pitch or interval.

        Arguments:
            element: str

        Returns:
            str
        """

        # remove sharps and flats
        fixation = element.replace('b', '').replace('#', '')

        return fixation

    def _glue(self):
        """Glue together all subscores.

        Arguments:
            None

        Returns:
            None

        Populates:
            self.sheet
        """

        # get all png files from directory
        paths = os.listdir(self.directory)
        paths = [path for path in paths if 'png' in path]
        paths.sort()

        # collect images
        images = [Image.open('{}/{}'.format(self.directory, path)).convert('RGBA') for path in paths]
        images = [numpy.array(image) for image in images]

        # get longest row
        width = max([len(image[0]) for image in images])
        halves = [1 + int((width - len(image[0])) / 2) for image in images]

        # pad the images with margins
        sections = []
        for image, half in zip(images, halves):

            # make margin
            margin = [[[255] * 4] * half] * len(image)
            margin = numpy.array(margin)

            # make section
            section = numpy.concatenate((margin, image, margin), axis=1)
            section = numpy.array([row[:width] for row in section])
            sections.append(section)

        # make sheet
        sheet = numpy.concatenate(sections, axis=0)
        sheet = numpy.array(sheet, dtype='uint8')
        self.sheet = sheet

        return None

    def _ingest(self):
        """Ingest the source material, training data conversations.

        Arguments:
            None

        Returns:
            None

        Sets:
            self.notes
            self.nonsense
            self.training
            self.holdouts
        """

        # status
        print('ingesting...')

        # set categories
        categories = ['quarters', 'halves', 'sharps', 'flats', 'naturals', 'rests', 'clefs', 'numbers', 'bars', 'blanks']
        self.categories = categories

        # set mirror
        mirror = {category: index for index, category in enumerate(categories)}
        self.mirror = mirror

        # populate data with images
        print('importing images...')
        data = {category: [] for category in categories}
        for category in categories:

            # get all images
            for name in os.listdir(category):

                # get image file
                image = Image.open(category + '/' + name).convert('RGBA')
                image = numpy.array(image)

                # add to data
                data[category].append(image)

        # set data
        self.data = data
        
        # create samples 
        print('creating samples...')
        samples = {category: [] for category in categories}
        for category in categories:
            
            # go through each image
            for image in data[category]:
                
                # create shadow
                shadow = self.backlight(image)
                
                # smear the shadow into multiple offset samples
                smears = self.smear(shadow, number=self.smearing)
                samples[category] += smears

        # balance the sample lists
        print('balancing...')
        samples = self._balance(samples)
        self.samples = samples

        # split into training and holdouts
        print('splitting into training and holdouts')
        training = []
        holdouts = []
        for index, category in enumerate(categories):

            # go through each image
            for shadow in samples[category]:

                # calculate target
                target = [0.0] * len(categories)
                target[index] = 1.0

                # add to training randomly by fraction
                if random() > self.fraction:

                    # add to training
                    training.append((shadow, target))

                # otherwise
                else:

                    # add to holdouts
                    holdouts.append((shadow, target))

        # shuffle
        training.sort(key=lambda sample: random())
        holdouts.sort(key=lambda sample: random())

        # set attributes
        self.training = training
        self.holdouts = holdouts

        # status
        print('{} training samples'.format(len(training)))
        print('{} holdout samples'.format(len(holdouts)))

        return None

    def _mask(self, shadow):
        """Surround a small shadow with zeroes to fill in the frames.

        Arguments:
            shadow: numpy array

        Returns:
            numpy array
        """

        # create array of zeros
        mask = numpy.zeros((self.width, self.height))

        # get height and width of shadow
        width = len(shadow[0])
        height = len(shadow)

        # find margins
        margin = int((self.width - width) / 2)
        marginii = int((self.height - height) / 2)

        # replace rows
        for row in range(marginii, height + marginii):

            # replace columns
            for column in range(margin, width + margin):

                # replace entry
                mask[row][column] = shadow[row - marginii][column - margin]

        return mask

    def _measure(self, vector, vectorii):
        """Compare two vectors by euclidean distance.

        Arguments:
            vector: list of floats
            vectorii: list of floats

        Returns:
            float, the euclidiean distance
        """

        # make into numpy arrays
        vector = numpy.array(vector)
        vectorii = numpy.array(vectorii)

        # compute the euclidean distance
        #distance = sum([(entry - entryii) ** 2 for entry, entryii in zip(vector, vectorii)])
        distance = numpy.linalg.norm(vector-vectorii)

        return distance

    def _normalize(self, image):
        """Normalize an image by resize and converting to black and white.

        Arguments:
            image: numpy array

        Returns:
            numpy array
        """

        # convert image to black and white shadow
        shadow = self.backlight(image).tolist()

        # buffer the image by framing with white
        shadow = [[1] * self.width + row + [1] * self.width for row in shadow]
        shadow = [[1] * len(shadow[0])] * self.height + shadow + [[1] * len(shadow[0])] * self.height
        center = (int(len(shadow[0]) / 2), int(len(shadow) / 2), )

        # reframe
        shadow = shadow[center[1] - int(self.height / 2): center[1] + 1 + int(self.height / 2)]
        shadow = [row[center[0] - int(self.width / 2): center[0] + 1 + int(self.width / 2)] for row in shadow]
        self.shadow = shadow

        # reconstruct as RGBA array
        reconstruction = []
        for row in shadow:

            # start new row
            rowii = []
            for point in row:

                # check value (on 0-1 scale)
                if point < self.gray:

                    # append black
                    rowii.append([0, 0, 0, 255])

                # otherwise
                else:

                    # append white
                    rowii.append([255, 255, 255, 255])

            # append to image
            reconstruction.append(rowii)

        # convert to numpy.array
        reconstruction = numpy.array(reconstruction, dtype='uint8')

        #print(reconstruction.shape)

        return reconstruction

    def _pad(self, shadow):
        """Pad a shadow with a margin of white.

        Arguments:
            shadow: numpy array

        Returns:
            numpy array
        """

        # make vertical margins
        margin = numpy.zeros((self.height, shadow.shape[1]))
        pad = numpy.vstack((margin, shadow, margin))

        # make horizontal margins
        margin = numpy.ones((pad.shape[0], self.width))
        pad = numpy.hstack((margin, pad, margin))

        return pad

    def _punch(self, shadow, point, width=None, height=None):
        """Punch out a standard sized shadow from a bigger shadow at specified point.

        Arguments:
            shadow: numpy array
            point: (int, int) tuple of indices
            width=None: int, width of punchout
            height=None: int, height of punchout

        Returns:
            numpy array
        """

        # set default width
        if not width:

            # set width
            width = self.width

        # set default height

            # set height
            height = self.height

        # figure out boundary
        top, bottom, left, right = self._bound(point, width, height)
        punchout = shadow[top:bottom, left:right]

        return punchout

    def _scale(self, data, lowest=-0.5, highest=0.5):
        """Scale data to a particular range.

        Arguments:
            data: list of floats
            lowest: float, lower limit
            highest: float, upper limit

        Returns:
            list of floats
        """

        # get max and min
        minimum = min(data)
        maximum = max(data)
        spread = maximum - minimum

        # normalize
        normalization = [((entry - minimum) / spread) - 0.5 for entry in data]

        return normalization

    def _square(self, data, number):
        """Square linear data into rows and columns based on the number of rows.

        Arguments:
            data: list of floats
            number: int, number of rows

        Returns:
            list of lists of floats
        """

        # copy data
        data = [entry for entry in data]

        # break into rows
        length = int(len(data) / number)
        rows = []
        while len(data) > 0:

            # append row
            row = data[:length]
            data = data[length:]
            rows.append(row)

        return rows

    def _tabulate(self):
        """Define chords.

        Arguments:
            None

        Returns:
            None

        Populates:
            self.lexicon
        """

        # define major chords
        self.lexicon[''] = ['3', '5']
        self.lexicon['maj7'] = ['3', '5', '7']
        self.lexicon['maj9'] = ['3', '5', '7', '9']
        self.lexicon['maj11'] = ['3', '5', '7', '9', '11']
        self.lexicon['maj13'] = ['3', '5', '7', '9', '11', '13']

        # define minor chords
        self.lexicon['m'] = ['b3', '5']
        self.lexicon['m7'] = ['b3', '5', 'b7']
        self.lexicon['m9'] = ['b3', '5', 'b7', '9']
        self.lexicon['m11'] = ['b3', '5', 'b7', '9', '11']
        self.lexicon['m13'] = ['b3', '5', 'b7', '9', '11', '13']

        # define dominant chords
        self.lexicon['7'] = ['3', '5', 'b7']
        self.lexicon['9'] = ['3', '5', 'b7', '9']
        self.lexicon['11'] = ['3', '5', 'b7', '9', '11']
        self.lexicon['13'] = ['3', '5', 'b7', '9', '11', '13']

        # define other chords
        self.lexicon['dim'] = ['b3', 'b5']
        self.lexicon['m7b5'] = ['b3', 'b5', 'b7']

        return None

    def _tesselate(self, horizontal, vertical, silhouette, measure):
        """Create a tile from a center point.

        Arguments:
            horizontal: int
            vertical: int
            silhouette: numpy array
            measure: int

        Returns:
            dict
        """

        # begin tile
        tile = {}

        # nudge vertical to closest staff line
        staves = [value for stave in self.staff for value in stave.values()]
        squares = [(vertical - verticalii) ** 2 for verticalii in staves]
        zipper = [pair for pair in zip(staves, squares)]
        zipper.sort(key=lambda pair: pair[1])
        vertical = zipper[0][0]

        # calculate center
        center = (horizontal, vertical)

        # determine position
        positions = {height: position for stave in self.staff for position, height in stave.items()}
        position = positions[vertical]
        position = max([position - 1, self.positions[0]])

        # add attributes
        tile['position'] = position
        tile['measure'] = measure
        tile['center'] = center
        tile['shadow'] = self._punch(silhouette, center)
        tile['color'] = self.colors[position]

        return tile

    def annotate(self, measure, *chords):
        """Annotate measures with chords.

        Arguments:
            measure: int, beginning measure
            *chord: unpacked tuple of strings

        Returns:
            None

        Populates:
            self.chords
        """

        # populate chords
        index = measure
        for chord in chords:

            # annotate
            self.chords[index] = chord

            # advance index
            index += 1

        return None

    def ask(self, number, category):
        """Ask about the top discoveries for a category.

        Arguments:
            number=10: number of top discoveries
            category: str, the category

        Returns:
            None
        """

        # sort discoveries
        index = self.mirror[category]
        discoveries = self.tiles[category]
        discoveries.sort(key=lambda discovery: discovery['prediction'][index], reverse=True)

        # clip to requested number
        discoveries = discoveries[:number]

        # get chunk size
        chunk = int(number / 4) + 1
        colors = ('orange', 'magenta', 'green', 'blue')
        for index, color in zip(range(4), colors):

            # paint the chunk
            self.spot(discoveries[chunk * index: chunk * (index + 1)], color)
            self.keep()

        # see painting
        self.see()

        return None

    def backlight(self, image):
        """Convert an RGBA image into a black and white image.

        Arguments:
            image: numpy arry

        Returns:
            numpy array
        """

        # convert image to black and white shadow
        shadow = []
        for row in image:

            # start new row
            rowii = []
            for point in row:

                # add three primary hexadecimal colors and create grayscale
                gray = sum(point[:3]) / (255 * 3)
                rowii.append(-0.5 + gray)

            # append row
            shadow.append(rowii)

        # make into numpy array
        shadow = numpy.array(shadow)

        return shadow

    def box(self, image, center, color, thickness):
        """Draw a box around the center point.

        Arguments:
            image: numpy array, the image
            center: (int * 2), the center coordinates
            color: str, key to the palette dict
            thickness: int, thickness of framing box

        Returns:
            None
        """

        # get color from the palette
        color = self.palette[color]

        # unpack boundary
        up, down, left, right = self._bound(center)

        # make copy so as not to disturb the training data
        image = numpy.copy(image)

        # paint the top and bottom
        for horizontal in range(left, right):

            # at each thickness
            for offset in range(thickness):

                # paint points
                image[up + offset][horizontal] = color
                image[down - offset][horizontal] = color

        # paint the sides
        for vertical in range(up, down):

            # at each thickness
            for offset in range(thickness):

                # paint the points
                image[vertical][left + offset] = color
                image[vertical][right - offset] = color

        return image

    def clear(self):
        """Reset current painting to the original.

        Arguments:
            None

        Returns:
            None
        """

        # reset sheets
        self.sheet = self.original
        self.painting = self.sheet

        return None

    def coalesce(self, tiles, silhouette, category, measure, width=25, height=25):
        """Coalesce several overlapping detections into single centered detections.

        Arguments:
            tiles: list of dicts
            silhouette: greater silhouette
            category: str, the category
            width=25: width of insert
            height=25: height of insert

        Returns:
            list of dicts
        """

        # vertical and horizontal margins
        margin = int((self.width - width) / 2)
        marginii = int((self.height - height) / 2)

        # begin centers
        centers = []
        for tile in tiles:

            # get center
            center = tile['center']

            # go through all horizontal points
            for horizontal in range(-margin, margin + 1):

                # go through all vertical points
                for vertical in range(-marginii, marginii + 1):

                    # add point to centers
                    point = (center[0] + horizontal, center[1] + vertical)
                    centers.append(point)

        # remove duplicates
        centers = list(set(centers))

        # get new punchouts and mask them
        punches = [self._punch(silhouette, center, width, height) for center in centers]
        masks = [self._mask(punch) for punch in punches]

        # set default condensations to empty
        condensations = []
        if len(masks) > 0:

            # predict from masks
            predictions = self.predict(masks)

            # get category index
            index = self.mirror[category]

            # get all predicted centers
            points = [center for center, prediction in zip(centers, predictions) if prediction[index] == max(prediction)]
            weights = [float(prediction[index]) for prediction in predictions if prediction[index] == max(prediction)]

            # check for points
            if len(points) > 0:

                # make matrix and cluster
                matrix = numpy.array(points)
                propagation = MeanShift(bandwidth=10, bin_seeding=True).fit(matrix)

                # get the cluster centers
                condensations = []
                clusters = propagation.cluster_centers_
                for indexii, cluster in enumerate(clusters):

                    # get average points
                    horizontals = [point[0] for point, label in zip(points, propagation.labels_) if label == indexii]
                    verticals = [point[1] for point, label in zip(points, propagation.labels_) if label == indexii]
                    weighting = [weight for weight, label in zip(weights, propagation.labels_) if label == indexii]

                    # make condensation
                    horizontal = int(numpy.average(horizontals, weights=weighting))
                    vertical = int(numpy.average(verticals, weights=weighting))
                    condensation = self._tesselate(horizontal, vertical, silhouette, measure)
                    condensations.append(condensation)

        return condensations

    def correct(self, measure, *corrections):
        """Correct the notes in a measure.

        Arguments:
            measure: int, index of measure
            *corrections: unpacked tuple of int and str

        Returns:
            None

        Populates:
            self.notes
        """

        # arrange corrections into pairs
        indices = [entry for index, entry in enumerate(corrections) if index % 2 == 0]
        pitches = [entry for index, entry in enumerate(corrections) if index % 2 == 1]
        pairs = [pair for pair in zip(indices, pitches)]

        # sort pairs by biggest index first
        pairs.sort(key=lambda pair: pair[0], reverse=True)

        # apply each pair
        for index, pitch in pairs:

            # check for pitch
            if pitch:

                # tune
                self.tune(index, measure, pitch)

            # otherwise
            else:

                # snuff out the note
                self.snuff(index, measure)

        return None

    def discover(self, extent=None):
        """Discover the notes in an image file.

        Arguments:
            extent: int, number of measures to scan

        Returns:
            None

        Sets:
            self.notes
            self.sheet
            self.painting
            self.tiles
        """

        # establish sheet
        self.original = self.sheet
        self.sheet = self.sheet
        self.painting = self.sheet
        sheet = self.sheet

        # making silhouette
        print('making silhouette...')
        silhouette = self.backlight(sheet)
        self.silhouette = silhouette

        # make staff
        print('finding staff...')
        self.notch(silhouette)

        # get measures
        print('finding measures...')
        self.measure(silhouette)

        # begin discoveries
        discoveries = {category: [] for category in self.categories}

        # check extent
        measures = self.measures
        if extent:

            # shorten to extent
            measures = self.measures[:extent]

        # check along each staff line
        print('finding notes...')
        reports = []
        notes = []
        chords = []
        for number, measure in enumerate(measures):

            # status
            print('measure {} of {}...'.format(number, len(self.measures)))

            # begin notes of the measure
            members = []

            # for each pixel
            tiles = []
            for index in range(measure['left'], measure['right']):

                # mark boundaries
                if self.width < index < len(silhouette[0]) - self.width and index % self.increment == 0:

                    # for each position
                    for position in range(*self.positions):

                        # make tile
                        row = measure[position]
                        tile = self._tesselate(index, row, silhouette, measure)
                        tiles.append(tile)

            # make predictions
            print('making predictions...')
            shadows = [tile['shadow'] for tile in tiles]
            predictions = self.predict(shadows)
            reports.append(predictions)
            for tile, prediction in zip(tiles, predictions):

                # add prediction
                tile['prediction'] = [float(entry) for entry in prediction]

            # find elements
            for category in self.categories:

                # find elements
                elements = self.select(tiles, category)

                # coalesce for certain elements
                if category in ('quarters'):

                    # coelsece elements
                    elements = self.coalesce(elements, silhouette, category, measure)

                    # add member to notes for each element
                    [members.append(self._describe(element)) for element in elements]

                # add to discoveries
                print('{}: {}'.format(category, len(elements)))
                discoveries[category] += elements

            # sort members by center and append
            members.sort(key=lambda member: member['center'][0])
            notes.append(members)

            # append blank chord
            chords.append('')

        # set discoveries
        self.reports = reports
        self.tiles = discoveries
        self.notes = notes
        self.chords = chords

        # report
        print(' ')
        for category in self.categories:

            # print report
                print('{} {}'.format(len(discoveries[category]), category))

        return None

    def enharmonize(self, pitch):
        """Get the enharmonic match for a pitch.

        Arguments:
            pitch: str

        Returns:
            str
        """

        # get enharmonic note
        enharmonic = self.enharmonics[self.mode][pitch]

        return enharmonic

    def evaluate(self):
        """Evaluate the model against the holdout set.

        Arguments:
            None

        Returns:
            None
        """

        # construct matrix and targets
        matrix = numpy.array([self._deepen(shadow) for shadow, _ in self.holdouts])
        truths = numpy.array([truth for _, truth in self.holdouts])

        # grade holdout set
        self.grade(self.holdouts, name='holdout set')

        # perform model evaluation
        evaluation = self.model.evaluate(matrix, truths, verbose=1)
        print('Holdout score {:.2f}'.format(evaluation[1]))

        return None

    def grade(self, samples=None, name='training set'):
        """Grade the scores of all training samples.

        Arguments:
            samples: list of samples, training by default
            name: str, name for printout

        Returns:
            None
        """

        # set default sequences
        if samples is None:

            # set to training sequences
            samples = self.training

        # make predictions off of each training sequence
        tallies = {}
        predictions = self.predict([sample[0] for sample in samples])
        truths = [sample[1] for sample in samples]
        for prediction, truth in zip(predictions, truths):

            # determine if truths match predictions
            # score = self._compare(actual, prediction)
            score = int(self._compare(truth, prediction) * 100)
            tallies[score] = tallies.setdefault(score, 0) + 1

        # calcuate precentages
        total = sum([count for count in tallies.values()])
        percents = [(score, round(float(count) / total, 2)) for score, count in tallies.items()]
        percents.sort(key=lambda pair: pair[0], reverse=True)

        # print
        print(' ')
        print(name + ':')
        for score, percent in percents:

            # print
            print('{}: {}%'.format(score, round(percent * 100)))

        return None

    def harmonize(self):
        """Determine chords based on melody notes.

        Arguments:
            None

        Returns:
            None

        Populates:
            self.chords
        """

        # go through each measure
        for index, measure in enumerate(self.notes):

            # print
            print('measure {} of {}'.format(index, len(self.notes)))

            # get all pitches in the measure
            pitches = [note['pitch'] for note in measure]
            pitches = list(set(pitches))

            # get chord and annotate
            chord = self.theorize(*pitches)
            self.annotate(index, chord)

        return None

    def holograph(self, shadow):
        """Convert a two dimension shadow to a three dimensional grayscale image.

        Arguments:
            shadow: numpy.array

        Returns:
            numpy.array
        """

        # expanding into rgb function
        expanding = lambda gray: [int(gray * 255), int(gray * 255), int(gray * 255), 255]

        # construct hologram
        hologram = [[expanding(entry + 0.5) for entry in row] for row in shadow]
        hologram = numpy.array(hologram,  dtype=numpy.uint8)

        return hologram

    def inspect(self, index):
        """Inspect the training conversation at the particular index

        Arguments:
            index: int, the conversation index

        Returns:
            None
        """

        # inspect training example
        image, truth = self.training[index]

        # print to screen
        print('truth: {}'.format(truth))

        # view image
        self.see(image)

        return None

    def introspect(self, index, see=True):
        """Introspect the weights in a neuron.

        Arguments:
            index: int
            see=False: boolean, use see method?

        Returns:
            None
        """

        # print category
        print(self.categories[index])

        # get weights
        evaluation = self.model.layers[1].get_weights()[0]

        # separate per neuron
        evaluation = evaluation.tolist()
        neurons = [neuron for neuron in zip(*evaluation)]
        neuron = neurons[index]

        # normalize and square into image
        normalization = self._scale(neuron)
        square = self._square(normalization, self.height)
        image = numpy.array(square)

        # pixelate
        if not see:

            # pixelate
            self.pixelate(image)

        # otherwise
        if see:

            # see
            self.see(image)

        return None

    def keep(self):
        """Keep the current painting as the base sheet.

        Arguments:
            None

        Returns:
            None
        """

        # set painting as sheet
        self.sheet = self.painting

        return None

    def load(self, path=None):
        """Load in model from file.

        Arguments:
            path: str, file path

        Returns:
            None
        """

        # default path
        if path is None:
            path = self.path

        # message
        print('loading...')

        # try to load in saved model
        try:

            # load from path
            model = load_model(path)
            self.model = model

            # message
            print(path + ' loaded.')

        # otherwise train
        except OSError:

            # message
            print(path + ' not found.  Training...')
            self.train()

        return None

    def measure(self, silhouette):
        """Break the staff into measures by detecting vertical lines along each staff.

        Arguments:
            None

        Returns:
            None

        Sets:
            self.measures
        """

        # set left and right
        left = 0
        right = len(silhouette[0])

        # go through each stave
        measures = []
        for stave in self.staff:

            # set top and bottom
            top = stave[9]
            bottom = stave[1]

            # get all columns
            columns = []
            for horizontal in range(left, right):

                # get column
                column = [float(silhouette[vertical][horizontal]) for vertical in range(top, bottom)]
                columns.append(column)

            # detect all possible columns
            candidates = [(index, column) for index, column in enumerate(columns) if self._detect(column, 40, 0.9)]

            # check all points for each
            pipes = [pair[0] for pair in candidates if all([entry < -0.3 for entry in pair[1]])]

            # get rid of duplicates
            pipes = [pipe for pipe in pipes if pipe + 1 not in pipes]

            # get all vertical indices for white space not at staff lines
            indices = [index for index in range(top, bottom)]
            indices = [index for index in indices if all([abs(index - row) > 3 for row in stave.values()])]

            # verify that pipes are legitamate
            verified = []
            for pipe in pipes:

                # check for whites
                lefts = [silhouette[index][pipe - 3] for index in indices]
                rights = [silhouette[index][pipe + 3] for index in indices]
                if all([float(entry) > 0.4 for entry in lefts + rights]):

                    # add to verified
                    verified.append(pipe)

            # detect all possible blanks
            criterion = lambda x: x < 0.4
            blanks = [index for index, column in enumerate(columns) if self._detect(column, 100, 0.95, criterion=criterion)]
            blanks = [index for index in blanks if index < verified[0]]
            blanks.sort()
            start = blanks[-1]

            # add a measure for each pipe
            for pipe in verified:

                # make measure
                measure = {'left': start, 'right': pipe}
                measure.update(stave)
                measures.append(measure)
                start = pipe

        # set attribue
        self.measures = measures

        return None

    def notch(self, silhouette):
        """Find all the horizontal black lines.

        Arguments:
            silhouette: numpy array

        Returns:
            None

        Sets:
            self.spacing
            self.staff
        """

        # go through each row and check for a black line
        lines = []
        for index, row in enumerate(silhouette):

            # check for line
            if self._detect(row):

                # add notch
                lines.append(index)

        # pick first index if no white space in between
        notches = []
        for line in lines:

            # check for one less
            if line - 1 not in notches:

                # add to list
                notches.append(line)

        # break into staves by fives
        staff = []
        while len(notches) > 0:

            # get five lines
            lines = notches[:5]
            notches = notches[5:]

            # determine average spacing
            spacings = [second - first for first, second in zip(lines[:4], lines[1:])]
            spacing = float(sum(spacings)) / 4

            # begin stave with odd lines
            stave = {position: {'position': position, 'row': row} for position, row in zip((9, 7, 5, 3, 1), lines)}

            # add spaces at evens
            averaging = lambda first, second: int(float(stave[first]['row'] + stave[second]['row']) / 2)
            stave.update({position: {'position': position, 'row': averaging(position - 1, position + 1)} for position in (8, 6, 4, 2)})

            # add lower ledger lines
            ledging = lambda number: stave[1]['row'] + int((1 - number) * spacing / 2)
            ledges = (number for number in range(0, self.positions[0] - 1, -1))
            stave.update({position: {'position': position, 'row': ledging(position)} for position in ledges})

            # add lower ledger lines
            ledging = lambda number: stave[9]['row'] - int((number - 9) * spacing / 2)
            ledges = (number for number in range(10, self.positions[1]))
            stave.update({position: {'position': position, 'row': ledging(position)} for position in ledges})

            # add to staff
            stave = {value['position']: value['row'] for value in stave.values()}
            staff.append(stave)

        # set staff
        self.staff = staff

        return None

    def paint(self, monocolor=None, criterion=8):
        """Paint the discovered notes onto the sheet.

        Arguments:
            monocolor=None: specific color to use
            criterion=8: the shading criterion

        Returns:
            None
        """

        # coalesce notes
        notes = [note for measure in self.notes for note in measure]

        # go through each category
        print('painting {} notes...'.format(len(notes)))
        painting = self.sheet
        for note in notes:

            # shade it
            box = self._bound(note['center'])
            color = monocolor or note['color']
            painting = self.shade(painting, box, color, criterion)

        # annotate chords
        for index, chord in enumerate(self.chords):

            # get coordinates
            left = self.measures[index]['left']
            top = self.measures[index][self.positions[1] - 1] - 50
            bottom = self.measures[index][0]

            # convert to image and begin draw mode
            xerox = Image.fromarray(painting)
            draw = ImageDraw.Draw(xerox)

            # add measure number to painting
            font = ImageFont.truetype('/Library/Fonts/{}.ttf'.format(self.font), 15)
            draw.text((left, bottom), str(index), font=font, fill='black')

            # if there is a chord name
            if chord:

                # determine chord root
                pitches = [key for key in self.wheel.keys() if chord.startswith(key)]
                pitches.sort(key=lambda pitch: len(pitch), reverse=True)
                pitch = pitches[0]

                # get color
                interval = self.wheel[self.key][pitch]
                color = self.spectrum[interval]

                # add chord to painting
                font = ImageFont.truetype('/Library/Fonts/{}.ttf'.format(self.font), self.size)
                draw.text((left, top), chord, font=font, fill='black')

                # convert back to numpy array
                painting = numpy.array(xerox)

                # shade chord box
                up = top
                down = top + self.size
                right = self.measures[index]['right']
                box = (up, down, left, right)
                painting = self.shade(painting, box, color, criterion)

            # otherwise
            else:

                # convert back to numpy array
                painting = numpy.array(xerox)

        # set it
        self.painting = painting

        return None

    def pinpoint(self, tiles, category):
        """Pinpoint the maximum values for each category.

        Arguments:
            tiles: list of dicts, the tile objects
            category: int, index of relevant category

        Returns:
            list of dicts, the elements
        """

        # arrange tiles in a grid
        grid = {}
        for tile in tiles:

            # add to list by row
            row = tile['center'][1]
            members = grid.setdefault(row, [])
            members.append(tile)

        # sort each group of members by horizontal coordinate
        for members in grid.values():

            # sort
            members.sort(key=lambda tile: tile['center'][0])

        # organize grid by position
        grid = [item for item in grid.items()]
        grid.sort(key=lambda item: item[0])
        grid = [item[1] for item in grid]

        # find maxima
        maxima = []
        for index, row in enumerate(grid):

            # go through each entry
            for indexii, tile in enumerate(row):

                # get prediction score
                score = tile['prediction'][category]

                # get neighbor scores
                neighbors = []
                offsets = [(vertical, horizontal) for vertical in (-1, 0, 1) for horizontal in (-1, 0, 1)]
                for offset in offsets:

                    # skip center
                    if offset != (0, 0):

                        # try to get score
                        try:

                            # get neighboring score
                            neighbor = grid[index + offset[0]][indexii + offset[1]]['prediction'][category]
                            neighbors.append(neighbor)

                        # unless at edges
                        except IndexError:

                            # pass
                            pass

                # check for maximum
                if all([score > neighbor for neighbor in neighbors]) and score > self.criteria:

                    # add to maxima
                    maxima.append(tile)

        # select only those that are also the chosen category
        maxima = self.select(maxima, category)

        return maxima

    def pixelate(self, shadow):
        """Print a pixel by pixel account of a shadow.

        Arguments:
            shadow: numpy array

        Returns:
            None
        """

        # set markers
        markers = {number: ' ' for number in range(-5, -3)}
        markers.update({number: '-' for number in range(-3, -1)})
        markers.update({number: '~' for number in range(-1, 2)})
        markers.update({number: '+' for number in range(2, 4)})
        markers.update({number: '*' for number in range(4, 6)})

        # create list
        shadow = shadow.tolist()

        # construct rows
        rows = []
        for line in shadow:

            # make row
            row = [int(entry * 10) for entry in line]
            row = ''.join([markers[entry] for entry in row])
            rows.append(row)

        # print
        print(' ')
        for row in rows:

            # print row
            print(row)

        return None

    def predict(self, shadows):
        """Predict whether an image is a note or not.

        Arguments:
            shadows: numpy array, list of shadows

        Returns:
            list of lists of floats
        """

        # make a matrix from the image
        # vectors = [shadow.ravel() for shadow in shadows]
        matrix = numpy.array([self._deepen(shadow) for shadow in shadows])

        # make prediction
        predictions = self.model.predict(matrix)

        return predictions

    def prepare(self):
        """Execute the full model build, training and test.

        Arguments:
            None

        Returns:
            None
        """

        # ingest data and build model
        self._ingest()
        self._build()

        # prepare palette
        self.squeeze()

        return None

    def probe(self, element, number=5, compare=False):
        """Probe the training data for the closest matches.

        Arguments:
            element: dict
            number=5: int, number of matches to print
            compare=False: boolean, use compare mode?

        Returns:
            None
        """

        # look at element
        shadow = element['shadow']
        self.see(shadow)
        print(element['prediction'])
        print(' ')

        # compare
        if not compare:

            # use euclidean method
            similarities = [(training, self._measure(shadow.ravel(), training[0].ravel())) for training in self.training]

        # otherwise
        else:

            # use similarity method
            similarities = [(training, 1 - self._compare(shadow.ravel(), training[0].ravel())) for training in self.training]

        # sort by scores
        similarities.sort(key=lambda pair: pair[1], reverse=False)

        # print results
        print('{} matches:'.format(number))
        for pair in similarities[:number]:

            # see image
            self.see(pair[0][0])
            print(pair[0][1])
            print(pair[1])
            print(' ')

        return None

    def publish(self, chunk=4):
        """Publish a painting into a pdf file.

        Arguments:
            chunk=4: number of staff lines per page

        Returns:
            None
        """

    def remember(self, path):
        """Remember the picture for later.

        Arguments:
            None

        Returns:
            None
        """

        # transform painting into image and save
        image = Image.fromarray(self.painting)
        image.save(path)

        return None

    def report(self, line, index):
        """Report the result of predictions for a line and category.

        Arguments:
            line: int, index of line
            index: int, category index

        Returns:
            None
        """

        # print category
        print(self.categories[index])

        # get predictions
        predictions = [float(entry[index]) for entry in self.reports[line]]
        predictions = self._scale(predictions)
        height = self.positions[1] - self.positions[0]
        predictions = self._square(predictions, height)
        image = numpy.array(predictions)

        # view
        self.see(image)

        return None

    def save(self, path=None):
        """Save the model to a file.

        Arguments:
            path: str, file path

        Returns:
            None
        """

        # default path
        if path is None:
            path = self.path

        # message
        print(' \n' + path + ' saving...')

        # save model
        model = self.model
        model.save(path)

        # message
        print(path + ' saved.')

        return None

    def see(self, image=None):
        """See an image based on a np array.

        Arguments:
            image: numpy array

        Returns:
            None
        """

        # default image
        if image is None:

            # to painting
            image = self.painting

        # check if it is a shadow
        if len(image.shape) < 3:

            # reproject
            image = self.holograph(image)

        # view the image
        Image.fromarray(image).save('display.png')

        return None

    def select(self, tiles, category):
        """Pinpoint the maximum values for each category.

        Arguments:
            tiles: list of dicts, the tile objects
            category: str, the category

        Returns:
            list of dicts, the elements
        """

        # get those tiles with highest score
        index = self.mirror[category]
        elements = [tile for tile in tiles if max(tile['prediction']) == tile['prediction'][index]]

        # add attributes
        [element.update({'category': category, 'score': element['prediction'][index]}) for element in elements]

        return elements

    def shade(self, image, box, color, criterion=8):
        """Shade all the dark region in an image with the color of choice.

        Arguments:
            image: numpy array, the image
            box: (int * 4), the bounding box
            color: str, key to the palette dict
            box=None: tuple of ints, the bounding box
            criterion=8: int, number of dark neighboring pixels at which to shade

        Returns:
            None
        """

        # get color from the palette
        color = self.palette[color]

        # unpack box
        up, down, left, right = box

        # make copy so as not to disturb the training data
        image = numpy.copy(image)

        # get the tiling subset (still pointing to image)
        tile = [row[left:right] for row in image[up:down]]

        # make a shadow and determine shade points
        shadow = self.backlight(tile)
        reckoning = self.weigh(shadow, criterion)

        # shade all points
        for vertical, horizontal in reckoning:

            # color point
            tile[vertical][horizontal] = color

        return image

    def smear(self, shadow, number=None):
        """Smear a shadow by generating multiple offset copies.

        Arguments:
            shadow: numpy array
            number: int, maximum smear in any direction

        Returns:
            list of numpy arrays
        """

        # default number
        if not number:

            # set to smearing
            number = self.smearing

        # pad with rows of white
        pad = self._pad(shadow)

        # find center point
        center = self._center(pad)

        # punch out at all offsets
        offsets = [(row, column) for row in range(-number, number + 1) for column in range(-number, number + 1)]
        points = [(center[1] + vertical, center[0] + horizontal) for vertical, horizontal in offsets]
        smears = [self._punch(pad, point) for point in points]

        return smears

    def snuff(self, note, measure):
        """Snuff out a note.

        Arguments:
            note: int, note number
            measure: int, measure number

        Returns:
            None

        Populates:
            self.notes
        """

        # snuff out the note
        notes = [member for index, member in enumerate(self.notes[measure]) if index != note]
        self.notes[measure] = notes

        return None

    def spot(self, notes, monocolor=None, thickness=3):
        """Paint the discovered objects onto the sheet.

        Arguments:
            notes: list of note to paint.
            monocolor=None: specific color to use
            thickness=3: int, thickness of framing box

        Returns:
            None
        """

        # go through each category
        print('spotting {} notes...'.format(len(notes)))
        painting = self.sheet
        for note in notes:

            # shade it
            center = note['center']
            color = monocolor or note['color']
            painting = self.box(painting, center, color, thickness)

        # set it
        self.painting = painting

        return None

    def squeeze(self):
        """Squeeze out the palette.

        Argument:
            None

        Returns:
            None

        Sets:
            self.palette
            self.colors
        """

        # set primaries of palette in rgb coordinates
        self.palette['red'] = [200, 0, 0, 255]
        self.palette['green'] = [0, 200, 0, 255]
        self.palette['blue'] = [70, 140, 255, 255]

        # set secondaries
        self.palette['yellow'] = [255, 255, 0, 255]
        self.palette['magenta'] = [255, 0, 255, 255]
        self.palette['cyan'] = [0, 255, 255, 255]

        # set black and white
        self.palette['white'] = [220, 220, 220, 255]
        self.palette['black'] = [60, 60, 60, 255]

        # set the rest of roygbiv
        self.palette['orange'] = [255, 180, 0, 255]
        self.palette['indigo'] = [0, 20, 200, 255]
        self.palette['violet'] = [160, 0, 160, 255]

        # set the twelvth
        self.palette['acid'] = [200, 250, 0, 255]

        # assign to positions
        positions = [position for position in range(*self.positions)]
        labels = ['white', 'magenta', 'red', 'green', 'blue', 'cyan', 'yellow', 'white', 'magenta', 'red', 'green', 'blue', 'cyan', 'yellow', 'white', 'magenta', 'red', 'green']
        colors = {position: label for position, label in zip(positions, labels)}
        self.colors = colors

        return None

    def stack(self, index):
        """View the average of training examples.

        Arguments:
            index: int

        Returns:
            None
        """

        # print category
        print(self.categories[index])

        # stack all training images
        shadows = [train[0] for train in self.training if train[1][index] > 0]
        collection = shadows[0]
        for shadow in shadows[1:]:

            # add shadow
            collection = collection + shadow

        # normalize and view
        raveled = collection.ravel()
        scaled = self._scale(raveled)
        squared = self._square(scaled, self.height)

        # make image
        image = numpy.array(squared)
        self.see(image)

        return None

    def test(self, number=10):
        """Test a random selection of a number of training samples to verify.

        Arguments:
            number=5: int, number of samples

        Returns:
            None
        """

        # test random training samples
        total = len(self.training)
        indices = [int(random() * total) for _ in range(number)]
        for index in indices:

            # verify
            self.verify(index)

        return None

    def theorize(self, *pitches):
        """Theorize about the chord given the pitches.

        Arguments:
            *pitches: unpacked tuple of strings

        Returns:
            list of tuples
        """

        # get enharmonic pitches
        pitches = [self.enharmonize(pitch) for pitch in pitches]

        # find signature pitches not represented
        signature = self.signatures[self.signature]
        notes = [self._fix(pitch) for pitch in pitches]
        signature = [member for member in signature if self._fix(member) not in notes]

        # print
        print('pitches: {} + ({})'.format(list(pitches), signature))

        # use each pitch as a root
        scores = []
        for root in pitches:

            # get all intervals
            harmony = [pitch for pitch in pitches if pitch != root]
            intervals = [self.wheel[root][pitch] for pitch in harmony]
            fixations = [int(self._fix(interval)) for interval in intervals]

            # get all implied intervals lower than the highest pitch but at least until the fifth
            maximum = max([7, max(fixations or [7])])
            implications = [self.wheel[root][pitch] for pitch in signature]
            implications = [interval for interval in implications if int(self._fix(interval)) < maximum]

            # compute scores for each chord in the lexicon
            for chord, structure in self.lexicon.items():

                # start score and compare intervals
                score = 0
                penalty = 0.5
                for interval in structure:

                    # check for match
                    if interval in intervals:

                        # add to score by the inverse of the interval
                        score += 1 / float(int(self._fix(interval)) + penalty * int('b' in interval))

                    # check against implications
                    if interval in implications:

                        # add some portion to the score
                        weight = 100
                        score += 1 / (weight * float(int(self._fix(interval)) + penalty * int('b' in interval)))

                # add score
                scores.append((root + chord, score, structure))

        # sort scores and annotate
        scores.sort(key=lambda triplet: len(triplet[2]))
        scores.sort(key=lambda triplet: triplet[1], reverse=True)

        # print
        [print(score) for score in scores[:7]]

        # return chord
        chord = ''
        if len(scores) > 0:

            # get top choice
            chord = scores[0][0]

        return chord

    def train(self, eras=None, epochs=None, grade=100):
        """Train the model.

        Arguments:
            epochs=None: int, number of epochs
            eras=None: int, number of eras, each of many epochs
            grade=1: how often to grade

        Returns:
            None
        """

        # set default epochs and eras
        if epochs is None:

            # use self
            epochs = self.epochs

        # set default eras
        if eras is None:

            # use self
            eras = self.eras

        # construct matrix and targets
        # matrix = numpy.array([shadow.ravel() for shadow, _ in self.training])
        matrix = numpy.array([self._deepen(shadow) for shadow, _ in self.training])
        targets = numpy.array([target for _, target in self.training])

        # print status
        print('training {} eras of {} epochs each...'.format(eras, epochs))
        print('matrix: {}'.format(matrix.shape))
        print('targets: {}'.format(targets.shape))

        # for each era
        for era in range(eras):

            # shuffle training indices
            indices = [index for index, _ in enumerate(matrix)]
            indices.sort(key=lambda index: random())

            # reconstitute shuffled matrices
            matrix = numpy.array([matrix[index] for index in indices])
            targets = numpy.array([targets[index] for index in indices])

            # and each epoch, only verbose on last
            verbosity = [False] * (epochs - 1) + [True]
            for epoch in range(epochs):

                # take timepoint
                start = time()
                self.model.fit(matrix, targets, epochs=1, verbose=verbosity[epoch], batch_size=None, validation_split=self.validation)
                final = time()
                print('epoch {} of {}: {} seconds'.format(epoch + 1, epochs, round((final - start), 2)))

            # print evaluation
            if era % grade == 0 and era > 0:

                # report score for all positives
                self.grade()
                self.evaluate()
                pass

            # save model
            self.save()

            # print status
            print('era {} of {} eras complete.'.format(era + 1, eras))
            print(' ')

        return None

    def tune(self, note, measure, pitch):
        """Tune a note to the correct pitch.

        Arguments:
            note: int, the note index
            measure: int, the measure index
            pitch: str, the correct pitch

        Returns:
            None

        Populates:
            self.notes
        """

        # correct pitch
        pitch = self.enharmonize(pitch)
        self.notes[measure][note]['pitch'] = pitch
        self.notes[measure][note]['color'] = self.spectrum[self.wheel[self.key][pitch]]

        return None

    def verify(self, index, threshold=None):
        """Verify the result from a test sample

        Arguments:
            index: int, index of sequence
            threshold=None: float, threshold at which to view remnants

        Returns:
            dict, predicted context
        """

        # set threshold
        if threshold is None:

            # set to actual threhhold
            threshold = self.threshold

        # print test sample
        self.inspect(index)

        # make prediction
        image, truth = self.training[index]
        prediction = self.predict([image])[0]
        print('prediction: {}'.format([entry for entry in prediction]))

        return None

    def weigh(self, shadow, criterion):
        """Weigh each pixel based on the criterion for number of surrounding dark pixels.

        Arguments:
            shadow: numpy array
            criterion: int, the number of surrounding neighbors

        Returns:
            list of list of (int, int) tuples, coordinates of pixels with enough neighbors
        """

        # get list of all dark points
        darks = []
        for vertical, row in enumerate(shadow):

            # and amongst each pixel
            for horizontal, pixel in enumerate(row):

                # check for dark pixel
                if pixel < -0.2:

                    # add to darks
                    point = (vertical, horizontal)
                    darks.append(point)

        # find which meet criteria
        surrounded = []
        for point in darks:

            # check vertical neighbors
            count = 0
            for vertical in (point[0] - 1, point[0], point[0] + 1):

                # and horizontal neigbors
                for horizontal in (point[1] - 1, point[1], point[1] + 1):

                    # check for dark pixel, excluding middle
                    if (vertical, horizontal) in darks and (vertical, horizontal) != point:

                        # add count
                        count += 1

            # add to surrounded points if criteria is met
            if count >= criterion:

                # add to surround points
                surrounded.append(point)

        return surrounded


# status
print('imported harmonizers.')

# # load harmonizer
harmo = Harmonizer('pieces/concerto')
harmo.prepare()
harmo.load()

# script
harmo.discover(3)
harmo.harmonize()
harmo.paint()
harmo.see()





