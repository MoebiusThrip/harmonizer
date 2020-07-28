__author__ = 'TheOz'

#  harmonizers to learn where notes are on a page and analyze them

# import system tools
from importlib import reload
import json, csv, os, sys
import numpy as np
from time import clock, time
from random import random, choice
from datetime import datetime
from math import sqrt

# import pil and skimage
from PIL import Image

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

    def __init__(self):
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
        self.positions = (-2, 15)

        # discover properties
        self.increment = 12
        self.smearing = 4
        self.criteria = 0.98
        self.tiles = {}
        self.reports = []

        # current sheet properties
        self.sheet = None
        self.painting = None
        self.orginal = None
        self.staff = None
        self.measures = None
        self.spacing = None
        self.notes = None

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

    def _bound(self, point):
        """Determine standard sized-boundary around point.

        Arguments:
            point: (int, int) tuple

        Returns:
            (int, int, int, int) tuple
        """

        # figure out boundary around point
        horizontal, vertical = point
        top = vertical - int(self.height / 2)
        bottom = vertical + 1 + int(self.height / 2)
        left = horizontal - int(self.width / 2)
        right = horizontal + 1 + int(self.width / 2)

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
        vector = np.array(vector)
        vectorii = np.array(vectorii)

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
        prism = np.array(prism)

        return prism

    def _detect(self, row, number=100, fraction=0.5):
        """Detect a horizontal line by checking a number of points at random.

        Arguments:
            row: list of int, a shadow row
            number=100: how many rows to check
            fraction=0.5: how much of the page to check.

        Returns:
            boolean, black line?
        """

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
            if row[index] > -0.3:

                # break, not a black line
                black = False
                break

        return black

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
                image = np.array(image)

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

    def _measure(self, vector, vectorii):
        """Compare two vectors by euclidean distance.

        Arguments:
            vector: list of floats
            vectorii: list of floats

        Returns:
            float, the euclidiean distance
        """

        # make into numpy arrays
        vector = np.array(vector)
        vectorii = np.array(vectorii)

        # compute the euclidean distance
        #distance = sum([(entry - entryii) ** 2 for entry, entryii in zip(vector, vectorii)])
        distance = np.linalg.norm(vector-vectorii)

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

        # convert to np.array
        reconstruction = np.array(reconstruction, dtype='uint8')

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
        margin = np.zeros((self.height, shadow.shape[1]))
        pad = np.vstack((margin, shadow, margin))

        # make horizontal margins
        margin = np.ones((pad.shape[0], self.width))
        pad = np.hstack((margin, pad, margin))

        return pad

    def _punch(self, shadow, point):
        """Punch out a standard sized shadow from a bigger shadow at specified point.

        Arguments:
            shadow: numpy array
            point: (int, int) tuple of indices

        Returns:
            numpy array
        """

        # figure out boundary
        top, bottom, left, right = self._bound(point)
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
        shadow = np.array(shadow)

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
        image = np.copy(image)

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

    def discover(self, name='concerto.png'):
        """Discover the notes in an image file.

        Arguments:
            name: str, the filename

        Returns:
            None

        Sets:
            self.notes
            self.sheet
            self.painting
        """

        # get file and convert
        sheet = Image.open(name).convert('RGBA')
        sheet = np.array(sheet)
        self.original = sheet
        self.sheet = sheet
        self.painting = sheet

        # making silhouette
        print('making silhouette...')
        silhouette = self.backlight(sheet)

        # make staff
        print('finding staff...')
        self.notch(silhouette)

        # get measures
        print('finding measures...')
        self.measure(silhouette)

        # check along each staff line
        print('finding notes...')
        reports = []
        discoveries = {category:[] for category in self.categories}
        for number, stave in enumerate(self.staff):

            # status
            print('stave {} of {}...'.format(number, len(self.staff)))

            # for each pixel
            tiles = []
            for index, _ in enumerate(silhouette[0]):

                # mark boundaries
                if self.width < index < len(silhouette[0]) - self.width and index % self.increment == 0:

                    # for each position
                    for line in stave:

                        # make tile
                        tile = {}
                        row = line['row']
                        position = line['position']
                        center = (index, row)
                        shadow = self._punch(silhouette, center)
                        tile['position'] = position
                        tile['center'] = center
                        tile['shadow'] = shadow
                        tile['stave'] = number
                        tile['color'] = self.colors[position]
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
            for index, category in enumerate(self.categories):

                # find elements
                # elements = self.pinpoint(tiles, index)
                elements = self.select(tiles, index)
                discoveries[category] += elements

        # set discoveries
        self.reports = reports
        self.tiles = discoveries

        # report
        print(' ')
        for category in self.categories:

            # print report
                print('{} {}'.format(len(discoveries[category]), category))

        return None

    def evaluate(self):
        """Evaluate the model against the holdout set.

        Arguments:
            None

        Returns:
            None
        """

        # construct matrix and targets
        matrix = np.array([self._deepen(shadow) for shadow, _ in self.holdouts])
        truths = np.array([truth for _, truth in self.holdouts])

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

    def holograph(self, shadow):
        """Convert a two dimension shadow to a three dimensional grayscale image.

        Arguments:
            shadow: np.array

        Returns:
            np.array
        """

        # expanding into rgb function
        expanding = lambda gray: [int(gray * 255), int(gray * 255), int(gray * 255), 255]

        # construct hologram
        hologram = [[expanding(entry + 0.5) for entry in row] for row in shadow]
        hologram = np.array(hologram,  dtype=np.uint8)

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
        image = np.array(square)

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

                # check for whiles
                lefts = [silhouette[index][pipe - 3] for index in indices]
                rights = [silhouette[index][pipe + 3] for index in indices]
                if all([float(entry) > 0.4 for entry in lefts + rights]):

                    # add to verified
                    verified.append(pipe)

            # add a measure for each pipe
            start = left
            for pipe in verified:

                # make measure
                measure = {'positions': stave, 'left': start, 'right': pipe}
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
            ledges = (number for number in range(10, self.positions[1] + 1))
            stave.update({position: {'position': position, 'row': ledging(position)} for position in ledges})

            # add to staff
            stave = {value['position']: value['row'] for value in stave.values()}
            staff.append(stave)

        # set staff
        self.staff = staff

        return None

    def paint(self, notes, monocolor=None):
        """Paint the discovered objects onto the sheet.

        Arguments:
            notes: list of note to paint.
            monocolor=None: specific color to use

        Returns:
            None
        """

        # go through each category
        print('painting {} notes...'.format(len(notes)))
        painting = self.sheet
        for note in notes:

            # shade it
            center = note['center']
            color = monocolor or note['color']
            painting = self.shade(painting, center, color, 8)

        # set it
        self.painting = painting

        return None

    def pinpoint(self, tiles, category):
        """Pinpoint the maxium values for each category.

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
        matrix = np.array([self._deepen(shadow) for shadow in shadows])

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
        height = self.positions[1] - self.positions[0] + 1
        predictions = self._square(predictions, height)
        image = np.array(predictions)

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
        Image.fromarray(image).show()

        return None

    def select(self, tiles, index):
        """Pinpoint the maxium values for each category.

        Arguments:
            tiles: list of dicts, the tile objects
            index: int, index of relevant category

        Returns:
            list of dicts, the elements
        """

        # get those tiles with highest score
        elements = [tile for tile in tiles if max(tile['prediction']) == tile['prediction'][index]]

        # add attributes
        category = self.categories[index]
        [element.update({'category': category, 'score': element['prediction'][index]}) for element in elements]

        return elements

    def shade(self, image, center, color, criterion=8):
        """Shade all the dark region in an image with the color of choice.

        Arguments:
            image: numpy array, the image
            center: (int * 2), the center coordinates
            color: str, key to the palette dict
            criterion=8: int, number of dark neighboring pixels at which to shade

        Returns:
            None
        """

        # get color from the palette
        color = self.palette[color]

        # unpack boundary
        up, down, left, right = self._bound(center)

        # make copy so as not to disturb the training data
        image = np.copy(image)

        # get the tiling subset (still pointing to image)
        tile = [row[left:right] for row in image[up:down]]

        # check image at random
        if random() < 0.00:

            # view image
            self.see(np.array(tile))

        # make a shadow
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
        self.palette['red'] = [150, 0, 0, 255]
        self.palette['green'] = [0, 200, 0, 255]
        self.palette['blue'] = [50, 120, 255, 255]

        # set secondaries
        self.palette['yellow'] = [255, 255, 0, 255]
        self.palette['magenta'] = [255, 0, 255, 255]
        self.palette['cyan'] = [0, 255, 255, 255]

        # set black and white
        self.palette['white'] = [220, 220, 220, 255]
        self.palette['black'] = [40, 40, 40, 255]

        # set the rest of roygbiv
        self.palette['orange'] = [255, 180, 0, 255]
        self.palette['indigo'] = [50, 0, 230, 255]
        self.palette['violet'] = [170, 0, 170, 255]

        # set the twelvth
        self.palette['acid'] = [220, 240, 0, 255]

        # assign to positions
        positions = [position for position in range(self.positions[0], self.positions[1] + 1)]
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
        image = np.array(squared)
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
        # matrix = np.array([shadow.ravel() for shadow, _ in self.training])
        matrix = np.array([self._deepen(shadow) for shadow, _ in self.training])
        targets = np.array([target for _, target in self.training])

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
            matrix = np.array([matrix[index] for index in indices])
            targets = np.array([targets[index] for index in indices])

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
                if pixel < self.gray:

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


# # load harmonizer
harmo = Harmonizer()
harmo.prepare()
harmo.load()

# status
print('imported harmonizers.')







