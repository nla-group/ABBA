def dtw(x, y, *, dist=lambda a, b: (a-b)*(a-b), return_path=False, filter_redundant=False):

    x = np.array(x)
    y = np.array(y)

    if filter_redundant:
        if return_path:
            warning.warn('return path not supported when filter_redundant=True')
            return_path = False

        # remove points
        if len(x) > 2:
            xdiff = np.diff(x)
            x = x[np.hstack((True,(xdiff[1:] - xdiff[0:-1]) >= 1e-14, True))]
        if len(y) > 2:
            ydiff = np.diff(y)
            y = y[np.hstack((True,(ydiff[1:] - ydiff[0:-1]) >= 1e-14, True))]

    len_x, len_y = len(x), len(y)
    window = [(i+1, j+1) for i in range(len_x) for j in range(len_y)]
    D = defaultdict(lambda: (float('inf'),))

    if return_path:
        D[0, 0] = (0, 0, 0)
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                          (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])

        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((i-1, j-1))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return (D[len_x, len_y][0], path)

    else:
        D[0, 0] = 0
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min(D[i-1, j]+dt, D[i, j-1]+dt, D[i-1, j-1]+dt)
        return D[len_x, len_y]
