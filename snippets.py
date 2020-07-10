def weigh(self, silhouette, criterion):
    """Weigh each pixel based on the criterion for number of surrounding dark pixels.

    Arguments:
        silhouette: numpy array
        criterion: int, the number of surrounding neighbors

    Returns:
        list of list of (int, int) tuples, coordinates of pixels with enough neighbors
    """

    # get list of all dark points
    darks = []
    for vertical, _ in enumerate(silhouette):

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