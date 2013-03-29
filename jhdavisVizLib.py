import math
import matplotlib.pyplot as pylab
import scipy.cluster.hierarchy as sch

def findIndices(g):
    """findIndices is a  helper function that likely should be deleted
    
    """
    change = [0]
    seen = [g[0]]
    for i in range(1, len(g)):
        if not g[i] in seen:
            change.append(i)
            seen.append(g[i])
    return change

def mapGroups(groupList, letters):
    """mapGroups is a  helper function that maps groups numbers to letters - likely should be deleted

    :param groupList: a list to be mapped
    :type d: list
    :param letters: a list to be mapped onto
    :type letters: list
    :returns: a list with elements of groupList mapped onto the letters
    
    """
    changeList = findIndices(groupList)
    i = 0
    for index in changeList:
        toReplace = groupList[index]
        groupList = listReplace(groupList, toReplace, letters[i])
        i = i+1
    return list(groupList)

def listReplace(l, to, rv):
    """listReplace is a helper function replaces all occurances of to with rv in a list l

    :param l: list to be replaced
    :type l: list
    :param to: item to be replaced
    :type to: string
    :param rv: item to replace with
    :type rv: string
    :returns: a list with all occurances of to replaced with rv
    
    """
    tr = []
    for i in l:
        if i == to:
            tr.append(rv)
        else:
            tr.append(i)
    return tr

def printSortedDict(d):
    """printSortedDict is a helper function to print a dictionary

    :param d: a dictionary to be printed
    :type d: dict
    :returns: a string of the dictionary
    
    """
    k = d.keys()
    k.sort()
    tp = ''
    for i in k:
        tp = tp + str(i) + ":" + str(d[str(i)]) + ", "
    return tp

def drawHeatMap(xdat, name="unnamed", colors=pylab.cm.RdBu, dendro=False, protColors=None, cIndex=None, km=None):
    """drawHeatMap produces a colored heatmap in a new figure window

    :param xdat: a data object (must contain fields 'data', 'fractions', 'proteins')
    :type xdat: dict
    :param colors: a color scale (a cmap)
    :type colors: cmap
    :param name: figure name and title
    :type name: str.
    :param dendro: a boolean to draw the dendrogram on the left
    :type dendro: bool.
    :param protColors: a color map used to label the protein names with group colors
    :type protColors: cmap
    :param cIndex: a list of groupIds for the proteins
    :type cIndex: list
    :param km: if present, will draw the kmeans cluster profiles at the top of the figure- input is a 2d-matrix - rowVectors for each centroid, each column is a fraction
    :type km: matrix
    :returns:  int -- the return code.
    :raises: AttributeError, KeyError
    :returns: a figure object

    """

    data = xdat['data']
    vecs = xdat['dimensions']
    ls = xdat['ls']
    fig = pylab.figure()
    fig.suptitle(name)
    ##Draw heatmap
    offset = 0.1
    yStart = 0.1
    yLength = 0.8
    if dendro:
        xStart = 0.35
        xLength = 0.5
    else:
        xStart = 0.1
        xLength = 0.8
    
    figData = heatMapAxes(data, dims = [xStart, yStart, xLength, yLength], columns=vecs, rows=ls, protColors=protColors, cIndex=cIndex, fig=fig)
    ##Draw colorbar
    fig.colorbar(figData)

    if dendro:
        ax2Data = fig.add_axes([offset, offset, xLength-0.3, yLength])
        sch.dendrogram(xdat['rightDendro'], orientation='right', color_threshold=8)
        ax2Data.set_xticks([])
        ax2Data.set_yticks([])
        
    return fig

def heatMapAxes(data, dims=[0.1, 0.1, 0.7, 0.7], colors=pylab.cm.RdBu, columns=None, rows=None, protColors=None, cIndex=None, fig=None):
    """heatMapAxes draws a heatmap

    :param data: a datamatrix to draw
    :type xdat: a 2D Matrix
    :param dims: the size of the plot to draw - defaults to full window
    :type dims: list (4 eements)
    :param colors: color index to use - defaults to redblue
    :type colors: cmap
    :param fractions: fraction names
    :type fractions: list
    :param proteins: protein names
    :type proteins: list
    :param protColors: a color map used to label the protein names with group colors
    :type protColors: cmap
    :param cIndex: a list of groupIds for the proteins
    :type cIndex: list
    :param fig: where to plot the axes (which figure); defaults to new figure
    :type fig: matplotlib figure
    :returns:  an axes

    """
    if fig is None:
        fig = pylab.figure()
    axData = fig.add_axes(dims)
    for i in range(len(columns)):
        axData.text(i, -0.5 , ' '+str(columns[i]), rotation=270, verticalalignment="top", horizontalalignment="center", fontsize=12)
    if protColors == None:
        for i in range(len(rows)):
            axData.text(-0.525, i, '  '+str(rows[i]), verticalalignment="center", horizontalalignment="right", fontsize=8)
    else:
        for i in range(len(rows)):
            axData.text(-0.525, i, '  '+str(rows[i]), verticalalignment="center", horizontalalignment="right", fontsize=8, color=cIndex(float(protColors[i])/(protColors.max()+1)))
    small = data.min()
    big = data.max()
    if math.fabs(small) > math.fabs(big):
        big = 0-small
    else:
        small = 0-big
    figData = axData.matshow(data, aspect='auto', origin='lower', cmap=colors, vmin=small, vmax=big)
    #fig.colorbar(figData)
    axData.set_xticks([])
    axData.set_yticks([])

    return figData