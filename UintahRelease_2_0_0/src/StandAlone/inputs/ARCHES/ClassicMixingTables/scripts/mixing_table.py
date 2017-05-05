# -*- coding: utf-8 -*-
from numpy import array, empty, zeros, index_exp
from math import copysign

class ClassicTable(object):
    
    def __init__(self,file_name):
        # Parse the table file into the classic table structure:
        # WARNING: Will likely not work for 1 independent variable!
        self.KEYS = {}  # KEYS stored in dict
        file_obj = open(file_name, 'r')
        
        def read_next_line(file_obj, split=False):
            while (True):
                line = file_obj.readline()
                if (line.strip(' ') == '\n'): continue  # skip empty lines
                elif (line[:4] == '#KEY'):
                    (name, value) = line[5:].split('=',1)
                    self.KEYS[name] = float(value)
                elif (line[0] == '#'): continue  # skip comment lines
                else:
                    if (split): return line.split()
                    else: return line
                    break
        
        # Independent variables (number, names, and lengths):
        self.Nind = int( read_next_line(file_obj) )
        self.ind_names = read_next_line(file_obj, split=True)
        str_list = read_next_line(file_obj, split=True)
        self.ind_len = [ int(str_list[i]) for i in range(self.Nind) ]
        self.ind = [ empty((self.ind_len[0],self.ind_len[-1])) ]
        for i in range(1,self.Nind):
            self.ind.append( empty(self.ind_len[i]) )
        
        # Dependent variables (number, names, and units):
        self.Ndep = int( read_next_line(file_obj) )
        self.dep_names = read_next_line(file_obj, split=True)
        self.dep_units = read_next_line(file_obj, split=True)
        self.dep = [ empty(self.ind_len) for i in range(self.Ndep) ]
        
        # Independent variable values (excluding the first):
        for i in range(self.Nind-1,0,-1):
            str_list = read_next_line(file_obj, split=True)
            self.ind[i] = array([ float(str_list[j])
                                     for j in range(self.ind_len[i]) ])
        
        # Recursive loop to read arbitrary number of dimensions:
        def recursive_for(index,dim, i_max):
            if (dim > 0):
                for i in range(i_max[dim]):
                    recursive_for(index[:dim]+[i]+index[dim+1:],dim-1, i_max)
            else:
                str_list = read_next_line(file_obj, split=True)
                self.dep[d][index_exp[:]+tuple(index[1:])] =  \
                              [ float(str_list[i]) for i in range(i_max[0]) ]
        
        # Dependent variable values (& values of first ind. var.):
        for d in range(self.Ndep):
            # sections are grouped by dependent variable
            for ilast in range(self.ind_len[-1]):
                # sub-sections for each value of the last ind. var.
                # but starts with one row of the first independent variable
                str_list = read_next_line(file_obj, split=True)
                if (d == 1):
                    self.ind[0][:,ilast] = array([ float(str_list[j])
                                           for j in range(self.ind_len[0]) ])
                # depenent variable values (for this value of last ind. var.)
                index = [0]*(self.Nind-1) + [ilast]
                recursive_for(index, self.Nind-2, self.ind_len)
        
        file_obj.close()
    
    
    def interpolate(self, ind0, *spec_dep):
        # Interoplates each dependent variable to the input value of the
        # independent variable. Uses multilinear interpolation in all
        # dimensions exept the last - which is nearest neighbor.
        # WARNING: currently doesn't check for out of bounds value of ind0!
        
        # Binary search for nearest indecies in each independent variable:
        ilows  = [0]*self.Nind
        ihighs = [ self.ind_len[i]-1 for i in range(self.Nind) ]
        for i in range(1,self.Nind):
            (ilow, ihigh) = (ilows[i], ihighs[i])
            if (self.ind[i][ilow] == ind0[i]):
                ihighs[i] = ilow+1
            elif (self.ind[i][ihigh] == ind0[i]):
                ilows[i] = ihigh-1
            else:
                while (ihigh-ilow > 1):
                    imid = (ilow+ihigh)//2
                    if (self.ind[i][imid] == ind0[i]):
                        (ilow, ihigh) = (imid, imid+1)
                    elif( copysign(1,self.ind[i][imid]-ind0[i]) ==
                          copysign(1,self.ind[i][ilow]-ind0[i]) ):
                        ilow = imid
                    else:
                        ihigh = imid
            (ilows[i], ihighs[i]) = (ilow, ihigh)
        # For the last variable, move both ilows & ihighs to the nearest:
        if ( abs(self.ind[-1][ilows[-1]] - ind0[-1]) < 
             abs(self.ind[-1][ihighs[-1]] - ind0[-1]) ):
            ihighs[-1] = ilows[-1]
        else:
            ilows[-1] = ihighs[-1]
        # Separate binary search for first independent variable since its
        # values depend on the index of the last independent vaiable:
        (ilow, ihigh) = (ilows[0], ihighs[0])
        if (self.ind[0][ilow,ilows[-1]] == ind0[0]):
            ihighs[0] = ilow+1
        elif (self.ind[0][ihigh,ilows[-1]] == ind0[0]):
            ilows[0] = ihigh-1
        else:
            while (ihigh-ilow > 1):
                imid = (ilow+ihigh)//2
                if (self.ind[0][imid,ilows[-1]] == ind0[0]):
                    (ilow, ihigh) = (imid, imid+1)
                elif( copysign(1,self.ind[0][imid,ilows[-1]]-ind0[0]) ==
                      copysign(1,self.ind[0][ilow,ilows[-1]]-ind0[0]) ):
                    ilow = imid
                else:
                    ihigh = imid
        (ilows[0], ihighs[0]) = (ilow, ihigh)
        
        # Multilinear interpolation (except last - nearest neighbor):
        dist = zeros(self.Nind-1)
        dist[0] = ( ind0[0] - self.ind[0][ilows[0],ilows[-1]] ) \
                  / ( self.ind[0][ihighs[0],ilows[-1]] \
                    - self.ind[0][ilows[0], ilows[-1]] )
        for i in range(1,self.Nind-1):
            dist[i] = ( ind0[i] - self.ind[i][ilows[i]] ) \
                      / ( self.ind[i][ihighs[i]] - self.ind[i][ilows[i]] )

        if len(spec_dep) == 0:
            spec_dep  = tuple(range(self.Ndep))
        interpolant = ()
        index = reduce( lambda x,y:x+y, [ index_exp[ilows[i]:ihighs[i]+1]
                               for i in range(self.Nind-1) ] ) + (ilows[-1],)
        for j in range(len(spec_dep)):
            d = spec_dep[j]
            tmp = self.dep[d][ index ]
            for i in range(self.Nind-2,-1,-1):
                index0 = index_exp[:]*i+(0,)*(self.Nind-1-i)
                index1 = index_exp[:]*i+(1,)+(0,)*(self.Nind-2-i)
                tmp[index0] = (1.-dist[i])*tmp[index0] + dist[i]*tmp[index1]
            interpolant += ( tmp[(0,)*(self.Nind-1)], )
        return interpolant


if __name__ == "__main__":
    mixing_file = 'lrgo_prediction.mix'
    myTable = ClassicTable(mixing_file)
    x0 = array([0.1, 0.26, 0.51])
    y0 = myTable.interpolate(x0)
    selected_dep = range(myTable.Ndep)
    print ' '
    print 'Independent Variables:'
    print '   ', ', '.join(myTable.ind_names)
    print '   ', ', '.join(map(str,x0))
    print ' '
    print 'Dependent Variables:'
    i0 = 0
    for i in selected_dep:
        print str(myTable.dep_names[i]).rjust(30), repr(y0[i0]).rjust(25), \
              str(myTable.dep_units[i]).ljust(15)
        i0 += 1