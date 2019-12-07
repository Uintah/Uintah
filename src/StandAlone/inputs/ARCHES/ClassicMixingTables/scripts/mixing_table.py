# -*- coding: utf-8 -*-
from numpy import array, empty, s_, index_exp, float64, shape

class ClassicTable:
    """Import an Arches file and provide a utility for interpolation."""
    def __init__(self, file_name):
        """
        Parse the table file into the classic table structure.
        
        WARNINGS
        --------
        Will likely not work for 1 independent variable!
        The independent variables must be monotonically increasing.
        """
        self.__call__ = self.interpolate
        self.KEYS = {}  # container for any #KEY constants
        file = open(file_name, 'r')
        def next_line(file, split=False):
            while (True):
                line = file.readline()
                if line.strip(' ') == '\n':
                    continue  # skip empty lines
                elif line[:4] == '#KEY':
                    name, value = line[5:].split('=', 1)
                    self.KEYS[name] = float(value)
                    continue
                elif line[0] == '#':
                    continue  # skip comment lines
                elif split:
                    return line.split()
                else:
                    return line
        
        # Independent variables (number, names, and lengths):
        self.ind_n = int( next_line(file) )
        self.ind_names = next_line(file, split=True)
        self.ind_len = [ int(n) for n in next_line(file, split=True) ]
        self.ind = [None]*self.ind_n
        # Dependent variables (number, names, and units):
        self.dep_n = int( next_line(file) )
        self.dep_names = next_line(file, split=True)
        self.dep_units = next_line(file, split=True)
        self.dep = [ empty(self.ind_len) for i in range(self.dep_n) ]
        # Independent variable values (for the first, only allocate):
        self.ind[0] = empty((self.ind_len[0], self.ind_len[-1]))
        for i in range(self.ind_n-1, 0, -1):
            self.ind[i] = array(next_line(file, split=True), dtype=float64)
        
        # Recursive loop to read arbitrary number of dimensions:
        def rfor(dep, index, dim, i_max):
            if dim > 0:
                # For this dim, prepend index with a loop iterator and recurse.
                for i in range(i_max[dim]):
                    dep = rfor(dep, (i,)+index, dim-1, i_max)
            else:
                # Read the file's next line and place it in dep based on index.
                str_list = next_line(file, split=True)
                dep[index_exp[:]+index] = array(str_list, dtype=float64)
            return dep
        # Dependent variable values (& values of first ind. var.):
        for d in range(self.dep_n):
            for ilast in range(self.ind_len[-1]):
                str_list = next_line(file, split=True)
                if d == 0:
                    self.ind[0][:,ilast] = array(str_list, dtype=float64)
                self.dep[d] = rfor(self.dep[d], (ilast,), self.ind_n-2,
                                   self.ind_len)
        file.close()
    
    
    def interpolate(self, x, spec_dep=None):
        """
        Interoplate each dependent variable to the input value of the
        independent variable. Use multilinear interpolation in all
        dimensions.
        
        Arguments
        ---------
        x:  iterable of floats (1-D array or list),
            independent variable values - where to evaluate the interpolant.
        spec_dep:  iterable of ints (1-D array or list) optional,
            integers indicating which dependent variables to output.
            It will default to outputting all dependent variables.
        
        Returns
        -------
        interpolant:  1-D array of floats,
            dependent variable values interpolated to x.
        
        WARNING
        -------
        Currently doesn't check for out of bounds value of x!
        """
        if spec_dep is None:
            spec_dep = tuple(range(self.dep_n))

        # Binary search for nearest indices in each independent variable:
        index, dx = [None,]*self.ind_n, [None]*self.ind_n
        for j in range(1, self.ind_n):
            ilow, ihigh = 0, self.ind_len[j]-1
            if x[j] == self.ind[j][ilow]:
                ihigh = 1
            elif x[j] == self.ind[j][ihigh]:
                ilow = ihigh - 1
            else:
                while ihigh - ilow > 1:
                    imid = (ilow + ihigh) // 2
                    if x[j] < self.ind[j][imid]:
                        ihigh = imid
                    elif x[j] == self.ind[j][imid]:
                        ilow, ihigh = imid, imid + 1
                    else:
                        ilow = imid
            index[j] = [ilow, ihigh]
            dx[j] = ( (x[j] - self.ind[j][ilow]) /
                      (self.ind[j][ihigh] - self.ind[j][ilow]) )
        # Two separate binary searches for first independent variable, since
        # its values depend on the index of the last independent vaiable:
        index[0], dx[0] = [None]*2, [None]*2
        for side, ilast in enumerate(index[-1]):
            ilow, ihigh = 0, self.ind_len[0]-1
            if x[0] == self.ind[0][ilow, ilast]:
                ihigh = 1
            elif x[0] == self.ind[0][ihigh, ilast]:
                ilow = ihigh - 1
            else:
                while ihigh - ilow > 1:
                    imid = (ilow + ihigh) // 2
                    if x[0] < self.ind[0][imid, ilast]:
                        ihigh = imid
                    elif x[0] == self.ind[0][imid, ilast]:
                        ilow, ihigh = imid, imid+1
                    else:
                        ilow = imid
            index[0][side] = [ilow, ihigh]
            dx[0][side] = ( (x[0] - self.ind[0][ilow, ilast]) /
                            (self.ind[0][ihigh, ilast] - self.ind[0][ilow, ilast]) )

        # Multilinear interpolation -
        # make sure to interpolate across the first dimension before the last:
        index[0][0] = s_[index[0][0][0]:index[0][0][1]+1]
        index[0][1] = s_[index[0][1][0]:index[0][1][1]+1]
        for j in range(1, self.ind_n-1):
            index[j] = s_[index[j][0]:index[j][1]+1]
        index0 = (index[0][0],) + tuple(index[1:-1]) + (index[-1][0],)
        index1 = (index[0][1],) + tuple(index[1:-1]) + (index[-1][1],)
        interpolant = empty(len(spec_dep))
        box = empty((2,)*self.ind_n)
        box_index0 = index_exp[:]*(self.ind_n-1)+(0,)
        box_index1 = index_exp[:]*(self.ind_n-1)+(1,)
        for k, d in enumerate(spec_dep):
            box[ box_index0 ] = self.dep[d][ index0 ].copy()
            box[ box_index1 ] = self.dep[d][ index1 ].copy()
            # Interpolate across the first dimension in view of the last:
            i0 = (0,) + index_exp[:]*(self.ind_n-2) + (0,)
            i1 = (1,) + index_exp[:]*(self.ind_n-2) + (0,)
            box[i0] = (1.0-dx[0][0])*box[i0] + dx[0][0]*box[i1]
            i0 = (0,) + index_exp[:]*(self.ind_n-2) + (1,)
            i1 = (1,) + index_exp[:]*(self.ind_n-2) + (1,)
            box[i0] = (1.0-dx[0][1])*box[i0] + dx[0][1]*box[i1]
            # All the other dimensions are simple multi-linear:
            for j in range(1, self.ind_n):
                i0 = (0,)*(j + 1)  + index_exp[:]*(self.ind_n-j-1)
                i1 = (0,)*j + (1,) + index_exp[:]*(self.ind_n-j-1 )
                box[i0] = (1.0-dx[j])*box[i0] + dx[j]*box[i1]
            interpolant[k] = box[(0,)*self.ind_n]
        return interpolant


if __name__ == "__main__":
    mixing_file = 'test.mix'
    myTable = ClassicTable(mixing_file)
    x0 = array([0.0050674086, -0.875, 0.050000000699999998])
    selected_dep = range(myTable.dep_n)
    y0 = myTable(x0, selected_dep)
    print( ' ')
    print( 'Independent Variables:')
    print( '   ', ', '.join(myTable.ind_names))
    print( '   ', ', '.join(map(str, x0)))
    print( ' ')
    print( 'Dependent Variables:')
    i0 = 0
    for i in selected_dep:
        print(str(myTable.dep_names[i]).rjust(30), repr(y0[i0]).rjust(25), \
              str(myTable.dep_units[i]).ljust(15))
        i0 += 1
    print(' ')
