from xml.etree import ElementTree as ET
from numpy import *
import os,sys
import re,operator


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def IntVector(iv):
    s = iv[1:-1]
    s_array = s.split(',')
    int_vector = map(int,s_array)
    return int_vector


def Vector(iv):
    s = iv[1:-1]
    s_array = s.split(',')
    vector = map(float,s_array)
    return vector

def find_type(type):
    m = re.search("<[a-zA-Z]+[\d]*>",type)
    t = m.group()
    return t[1:-1]


def generate_coordinates(high_index,low_index,upper,lower):
    num_x = high_index[0] - low_index[0]
    num_y = high_index[1] - low_index[1]
    num_z = high_index[2] - low_index[2]
    space_x = (upper[0] - lower[0])/num_x
    space_y = (upper[1] - lower[1])/num_y
    space_z = (upper[2] - lower[2])/num_z
    x_coord = []
    y_coord = []
    z_coord = []
    for i in range(num_x+1):
        x_coord.append(lower[0] + float(i)*space_x)

    for i in range(num_y+1):
        y_coord.append(lower[1] + float(i)*space_y)

    for i in range(num_z+1):
        z_coord.append(lower[2] + float(i)*space_z)

    return (x_coord,y_coord,z_coord)

def create_vtkfile_element(vtk_type,version=None,endianness=None):
    vtkfile_element = ET.Element('VTKFile')
    vtkfile_element.attrib['type'] = vtk_type
    if version is not None:
        vtkfile_element.attrib['version'] = version
    else:
        vtkfile_element.attrib['version'] = '0.1'

    if endianness is not None:
        vtkfile_element.attrib['byte_order'] = endianness
    else:
        vtkfile_element.attrib['byte_order'] = 'LittleEndian'
        
    return vtkfile_element
    

class Uda:

    "Uintah uda reader"

    def __init__(self,uda=None):
        self.name = uda
        self.output_file_names = []
        self.output_particle_file_names = []

    def read(self,uda=None):

        if uda is not None:
            self.name = uda
        elif self.name is None:
            print 'Please specify an Uda directory name'
            return
            
        print self.name
        print os.getcwd()
        os.chdir(self.name)
        print os.getcwd()
        
        try:
            self.index_tree = ET.parse("index.xml")
            print "Reading the index.xml file from %s" % self.name
        except Exception:
            print "Unexpected error opening %s" % "index.xml" 
            return

        self.read_globals()
        self.read_timesteps()

        os.chdir('../')
        return None

            
    def read_globals(self):
        self.global_variables = {}
        elems = self.index_tree.getroot()
        global_vars = elems.find('globals')
        global_variables_iter = global_vars.getiterator('variable')
        for global_variables in global_variables_iter:
            variable = global_variables.get('href')
            name = global_variables.get('name')
            self.global_variables[name] = variable
            
        return None


    def read_timesteps(self):
        self.timesteps = []
        elems = self.index_tree.getroot()
        timestep_iter = elems.getiterator('timestep')
        for ts in timestep_iter:
            timestep_href = ts.get('href')
            timestep = timestep_href.split('/')
            # returns the t***** of the t****/timestep.xml
            time_value = timestep[0]
            self.timesteps.append(Timestep(timestep_href))


    def output_vtk(self,filename = None):
        root_element = self.create_root_element(vtk_type = 'Collection')
        self.timestep_vtk()
        self.output_vtk_file(root_element)

    def output_vtk_particle(self,filename = None):
        root_element = self.create_root_element(vtk_type = 'Collection')
        self.timestep_vtk_particle()
        self.output_vtk_file_particle(root_element)


    def create_root_element(self,vtk_type):
        elems = self.index_tree.getroot()

        meta_tag = elems.find('Meta')
        endianness = meta_tag.findtext('endianness')
        if endianness == 'little_endian':
            endian_tag = 'LittleEndian'
        elif endianness == 'big_endian' :
            endian_tag = 'BigEndian'

        root_element = create_vtkfile_element(vtk_type,'0.1',endian_tag)

        return root_element

    def timestep_vtk(self):
        name = self.name.split('.')[0]
        for ts in self.timesteps:
            for level in ts.grid.levels:
                filename = name + '.L' + repr(level.id) + '_' +\
                    str(ts.timestepNumber) + '.vti'
                # ts.output_vtk(filename)
                self.output_file_names.append(filename)
                ts.output_vtk(filename)

    def timestep_vtk_particle(self):
        name = self.name.split('.')[0]
        for ts in self.timesteps:
            for level in ts.grid.levels:
                filename_particle = name + '.L' + repr(level.id) + '_' +\
                    str(ts.timestepNumber) + '.vtu'
                # ts.output_vtk(filename)
                self.output_particle_file_names.append(filename_particle)
                ts.output_vtk_particle(filename_particle)
            

    def output_vtk_file(self,root_element):
        collection = ET.SubElement(root_element,"Collection")
        for i in range(len(self.output_file_names)):
            data_set = ET.SubElement(collection,"DataSet")
            data_set.attrib['timestep'] = str(self.timesteps[i].currentTime)
            data_set.attrib['group'] = ''
            data_set.attrib['part'] = '0'
            data_set.attrib['file'] = str(self.output_file_names[i])
            
        indent(root_element)         
        # ET.dump(root_element)
        tree = ET.ElementTree(root_element)
        name = self.name.split('.')[0]
        tree.write(name + '.pvd')

    def output_vtk_file_particle(self,root_element):
        collection = ET.SubElement(root_element,"Collection")
        for i in range(len(self.output_particle_file_names)):
            data_set_particle = ET.SubElement(collection,"DataSet")
            data_set_particle.attrib['timestep'] = str(self.timesteps[i].currentTime)
            data_set_particle.attrib['group'] = ''
            data_set_particle.attrib['part'] = '0'

            data_set_particle.attrib['file'] = str(self.output_particle_file_names[i])
                  
        indent(root_element)         
        # ET.dump(root_element)
        tree = ET.ElementTree(root_element)
        name = self.name.split('.')[0]
        tree.write(name + '.pvd')




class Timestep:

    def __init__(self,timestep_xml=None):
        self.dir = ''
        self.datafiles = []
        if timestep_xml is not None:
            Uintah_timestep = self.read_timestep(timestep_xml)
        else:
            return

        meta = Uintah_timestep.find('Meta')
        self.endianness = meta.findtext('endianness')
        self.nBits = int(meta.findtext('nBits'))
        self.numProcs = int(meta.findtext('numProcs'))
        time = Uintah_timestep.find('Time')
        self.timestepNumber = int(time.findtext('timestepNumber'))
        self.currentTime = float(time.findtext('currentTime'))
        self.oldDelt = float(time.findtext('oldDelt'))
        self.grid = Grid(Uintah_timestep.find('Grid'))
        # All the grid data structions are created, i.e. levels,patches
        # actually read in the l*/p*.xml to read in the actual variable data
        # and store it in each patch's list of variables

        self.read_datafile(Uintah_timestep)

    def read_datafile(self,timestep):
        datafile_iter = timestep.getiterator('Datafile')
        for data in datafile_iter:
            da = data.get('href')
            self.datafiles.append(da)

        for files in self.datafiles:
            datafile_xml = self.dir + '/' + files
            datafile_split = datafile_xml.split('/')
            level_id = datafile_split[1]
            tree = ET.parse(datafile_xml)
            elem = tree.getroot()
            for var in elem.getiterator('Variable'):
                v = Variable(var)
#                v.print_variable()
                datafile_name = self.dir + '/' + level_id + '/' + v.filename
                v.read_data(datafile_name)
                patch = self.grid.find_patch(v.patch)
                patch.add_variable(v)

    def read_timestep(self,timestep_xml):
        self.dir = timestep_xml.split('/')[0]
            
        try:
            tree = ET.parse(timestep_xml)
            print "Reading the timestep file %s" % timestep_xml
        except Exception:
            print "Unexpected error opening %s" % timestep_xml 
            return
            
        return tree.getroot()

    def output_vtk(self,filename):
        self.grid.output_vtk(filename)

    def output_vtk_particle(self,filename):
        self.grid.output_vtk_particle(filename)

class Grid:
    def __init__(self,grid):
        self.numLevels = int(grid.findtext('numLevels'))
        self.levels = []
        level_iter = grid.getiterator('Level')
        self.read_level(level_iter)

    def print_grid(self):
        print "Grid number of levels = %s" % self.numLevels

    def read_level(self,level_iter):
        for level in level_iter:
            self.add_level(Level(level))

    def add_level(self,level):
        self.levels.append(level)
#        print "Number of current levels stored = %s" % len(self.levels)

        
    def get_levels(self):
        return self.levels

    def get_grid_extents(self):
        extent = self.get_extent()
        print "Grid extent = %s %s" % (extent[0],extent[1])
        return extent

    def get_extent(self):
        ex = self.levels[0].get_extent()
        for level in self.levels:
            extent = level.get_extent()
            # print "Level %s extent = %s" % (level.id,extent)
            ex_lo = [min(ex[0][0],extent[0][0]),min(ex[0][1],extent[0][1]), \
                         min(ex[0][2],extent[0][2])]
            ex_hi =[max(ex[1][0],extent[1][0]), max(ex[1][1],extent[1][1]), \
                        max(ex[1][2],extent[1][2])]
            # print "Current ex_lo = %s" % ex_lo
            # print "Current ex_hi = %s" % ex_hi
            ex = (ex_lo,ex_hi)

        return ex


    def find_patch(self,id):
        for level in self.levels:
            for patch in level.patches:
                if id == patch.id:
                    return patch
            
        return None

    def vtk_element(self,endianness=None,nbits=None):
        extent = self.get_grid_extents()
        lo = extent[0]
        hi = extent[1]
        string_extent = repr(lo[0]) + ' ' + repr(hi[0]) \
            + ' ' + repr(lo[1]) + ' ' + repr(hi[1]) \
            + ' ' + repr(lo[2]) + ' ' + repr(hi[2]) 
        
        # rectilinear_elem = create_vtkfile_element(vtk_type='RectilinearGrid')
        # grid_elem = ET.Element('RectilinearGrid')
        rectilinear_elem = create_vtkfile_element(vtk_type='ImageData')
        grid_elem = ET.Element('ImageData')

        grid_elem.attrib['WholeExtent'] = string_extent

        for level in self.levels:
            level.vtk_element(grid_elem)

        rectilinear_elem.append(grid_elem)
        indent(rectilinear_elem)
        # ET.dump(rectilinear_elem)
        return rectilinear_elem

    # def output_vtk(self,filename):
    #     grid_elem = self.vtk_element()
    #     grid_tree = ET.ElementTree(grid_elem)
    #     grid_tree.write(filename)


    def output_vtk(self,filename):
        for level in self.levels:
            elem = level.vtk_element()
            elem_tree = ET.ElementTree(elem)
            elem_tree.write(filename)

    def output_vtk_particle(self,filename):
        for level in self.levels:
            elem = level.vtk_element_particle()
            elem_tree = ET.ElementTree(elem)
            elem_tree.write(filename)


class Level:
    def __init__(self,level):
        self.numPatches = int(level.findtext('numPatches'))
        self.totalCells = int(level.findtext('totalCells'))
        self.extraCells = IntVector(level.findtext('extraCells'))
        self.anchor = IntVector(level.findtext('anchor'))
        self.id = int(level.findtext('id'))
        self.cellspacing = Vector(level.findtext('cellspacing'))
        self.patches = []
        patch_iter = level.getiterator('Patch')
        self.read_patch(patch_iter)
        for p in self.patches:
            p.find_neighbors(self.patches)


    def print_level(self):
        print "Level number of patches = %s" % self.numPatches
        print "Level number of total cells  = %s" % self.totalCells
        print "Level extra cells = %s" % self.extraCells
        print "Level anchor = %s" % self.anchor
        print "Level id = %s" % self.id
        print "Level cell spacing = %s" % self.cellspacing

    def read_patch(self,patch_iter):
        for patch in patch_iter:
            self.add_patch(Patch(patch))

        
    def add_patch(self,patch):
        self.patches.append(patch)
#        print "Number of current patches stored = %s" % len(self.patches)
    
    def get_patches(self):
        return self.patches

    def get_extent(self):
        ex = self.patches[0].get_extent()
        for patch in self.patches:
            extent = patch.get_extent()
            # print "Patch %s extent = %s" % (patch.id,extent)
            ex_lo = [min(ex[0][0],extent[0][0]),min(ex[0][1],extent[0][1]), \
                         min(ex[0][2],extent[0][2])]
            ex_hi =[max(ex[1][0],extent[1][0]), max(ex[1][1],extent[1][1]), \
                        max(ex[1][2],extent[1][2])]
            # print "Current ex_lo = %s" % ex_lo
            # print "Current ex_hi = %s" % ex_hi
            ex = (ex_lo,ex_hi)

        return ex

    def vtk_element(self):

        extent = self.get_extent()
        lo = extent[0]
        hi = extent[1]
        string_extent = repr(lo[0]) + ' ' + repr(hi[0]) \
            + ' ' + repr(lo[1]) + ' ' + repr(hi[1]) \
            + ' ' + repr(lo[2]) + ' ' + repr(hi[2]) 

        vtkfile_elem = create_vtkfile_element(vtk_type='ImageData')
        image_data_elem = ET.Element('ImageData')

        image_data_elem.attrib['WholeExtent'] = string_extent
        image_data_elem.attrib['Origin'] = repr(self.anchor[0]) + ' ' +\
            repr(self.anchor[1]) + ' ' + repr(self.anchor[2])
        image_data_elem.attrib['Spacing'] = repr(self.cellspacing[0]) + ' ' +\
            repr(self.cellspacing[1]) + ' ' + repr(self.cellspacing[2])
        

        for patch in self.patches:
            elem = patch.vtk_element(image_data_elem)
            image_data_elem.append(elem)
                
        vtkfile_elem.append(image_data_elem)
        indent(vtkfile_elem)
        # ET.dump(vtkfile_elem)
        return vtkfile_elem


    def vtk_element_particle(self):

        vtkfile_elem = create_vtkfile_element(vtk_type='UnstructuredGrid')
        unstructured_data_elem = ET.Element('UnstructuredGrid')

        for patch in self.patches:
            elem = patch.vtk_element_particle(unstructured_data_elem)
            unstructured_data_elem.append(elem)
                
        vtkfile_elem.append(unstructured_data_elem)
        indent(vtkfile_elem)
        # ET.dump(vtkfile_elem)
        return vtkfile_elem


class Patch:
    def __init__(self,patch):
        self.id = int(patch.findtext('id'))
        self.proc = int(patch.findtext('proc'))
        self.lowIndex = IntVector(patch.findtext('lowIndex'))
        self.highIndex = IntVector(patch.findtext('highIndex'))
        self.interiorLowIndex = IntVector(patch.findtext('interiorLowIndex'))
        self.interiorHighIndex = IntVector(patch.findtext('interiorHighIndex'))
        self.nnodes = int(patch.findtext('nnodes'))
        self.lower = Vector(patch.findtext('lower'))
        self.upper = Vector(patch.findtext('upper'))
        self.totalCells = int(patch.findtext('totalCells'))
        self.variables = []
        self.plus_neighbor = [0,0,0]

    def print_patch(self):
        print "Patch id = %s" % self.id
        print "Patch proc = %s" % self.proc
        print "Patch lowIndex = %s" % self.lowIndex
        print "Patch highIndex = %s" % self.highIndex
        print "Patch interiorLowIndex = %s" % self.interiorLowIndex
        print "Patch interiorHighIndex = %s" % self.interiorHighIndex
        print "Patch nnodes = %s" % self.nnodes
        print "Patch lower = %s" % self.lower
        print "Patch upper = %s" % self.upper
        print "Patch totalCells = %s" % self.totalCells

    def add_variable(self,variable):
        self.variables.append(variable)
#        print "Number of current variables stored = %s" % len(self.variables)
    
    def get_variables(self):
        return self.variables

    def generate_grid(self):
        pass

    def find_neighbors(self,neighbors):
        for n in neighbors:
            if self.id == n.id:
                continue
            else:
                if self.highIndex[0] == n.lowIndex[0]:
                    self.plus_neighbor[0] = 1
                if self.highIndex[1] == n.lowIndex[1]:
                    self.plus_neighbor[1] = 1
                if self.highIndex[2] == n.lowIndex[2]:
                    self.plus_neighbor[2] = 1
                


    def get_extent(self):
        return (self.lowIndex, self.highIndex)


    def get_num_materials(self):
        num_mat = 0
        for v in variables:
            if v.type == 'ParticleVariable<Point>':
                num_mat = num_mat + 1

        return num_mat

    def get_num_particles(self, mat_index):
        for v in variables:
            if v.index == mat_index:
                if v.type == 'ParticleVariable<Point>':
                    return v.numParticles

                

    def vtk_element(self,root_elem):
        extent = self.get_extent()
#        num_points = self.get_num_particles(mat_id)
        lo = extent[0]
        # subtract off the plus_neighbor values from the hi extent
        # hi = extent[1] - plus_neighbor
        hi = map(operator.sub,extent[1],self.plus_neighbor)
        
        string_extent = repr(lo[0]) + ' ' + repr(hi[0]) \
            + ' ' + repr(lo[1]) + ' ' + repr(hi[1]) \
            + ' ' + repr(lo[2]) + ' ' + repr(hi[2]) 
        patch_elem = ET.Element('Piece')
        patch_elem.attrib['Extent'] = string_extent
        
        # coord_elem = ET.Element('Coordinates')
        # patch_elem.append(coord_elem)

        # (x,y,z) = generate_coordinates(self.highIndex,self.lowIndex,\
        #                                    self.upper,self.lower)
        # data_elem = ET.Element('DataArray')
        # data_elem.attrib['type'] = 'Float64'
        # data_elem.attrib['Name'] = 'x_component'
        # data_elem.attrib['NumberOfComponents'] = '1'
        # data_elem.attrib['format'] = 'ascii'

        # string_data_elem = str()
        # for i_x in x:
        #     string_data_elem = string_data_elem + ' ' + repr(i_x)
        # data_elem.text = string_data_elem

        # coord_elem.append(data_elem)

        # data_elem = ET.Element('DataArray')
        # data_elem.attrib['type'] = 'Float64'
        # data_elem.attrib['Name'] = 'y_component'
        # data_elem.attrib['NumberOfComponents'] = '1'
        # data_elem.attrib['format'] = 'ascii'

        # string_data_elem = str()
        # for i_y in y:
        #     string_data_elem = string_data_elem + ' ' + repr(i_y)
        # data_elem.text = string_data_elem

        # coord_elem.append(data_elem)

        # data_elem = ET.Element('DataArray')
        # data_elem.attrib['type'] = 'Float64'
        # data_elem.attrib['Name'] = 'z_component'
        # data_elem.attrib['NumberOfComponents'] = '1'
        # data_elem.attrib['format'] = 'ascii'

        # string_data_elem = str()
        # for i_z in z:
        #     string_data_elem = string_data_elem + ' ' + repr(i_z)
        # data_elem.text = string_data_elem

        # coord_elem.append(data_elem)


        cell_elem = ET.Element('CellData')
        point_elem = ET.Element('PointData')

        for variable in self.variables:
            variable_type = variable.type.split('<')[0]
            elem = variable.vtk_element()
            
            if variable_type == 'CCVariable':
                cell_elem.append(elem)
            if variable_type == 'NCVariable':
                point_elem.append(elem)

        patch_elem.append(cell_elem)
        patch_elem.append(point_elem)

        indent(patch_elem)
        # ET.dump(patch_elem)
        return patch_elem

    def vtk_element_particle(self,root_elem):
#        num_points = self.get_num_particles(mat_id)
        num_points = 0
        patch_elem = ET.Element('Piece')
        patch_elem.attrib['NumberOfPoints'] = repr(num_points)
        patch_elem.attrib['NumberOfCells'] = '0'
        

        celldata_elem = ET.Element('CellData')
        cells_elem = ET.Element('Cells')
        pointdata_elem = ET.Element('PointData')
        points_elem = ET.Element('Points')

        for variable in self.variables:
            variable_type = variable.type.split('<')[0]
            elem = variable.vtk_element()
            
            if variable_type == 'ParticleVariable':
                data_type = variable.data_type
                if data_type == 'Point':
                    points_elem.append(elem)
                else:
                    pointdata_elem.append(elem)

        patch_elem.append(celldata_elem)
        patch_elem.append(pointdata_elem)
        patch_elem.append(points_elem)
        patch_elem.append(cells_elem)

        indent(patch_elem)
        # ET.dump(patch_elem)
        return patch_elem

class Variable:
    def __init__(self,variable):
        self.type = variable.get('type')
        self.data_type = find_type(self.type)
        self.variable_name = variable.findtext('variable')
        self.index = int(variable.findtext('index'))
        self.patch = int(variable.findtext('patch'))
        self.start = int(variable.findtext('start'))
        numParticles = variable.findtext('numParticles')
        if numParticles is not None:
            numParticles = int(numParticles)
        self.numParticles = numParticles
        self.end = int(variable.findtext('end'))
        self.filename = variable.findtext('filename')


    def print_variable(self):
        print "Variable type = %s" % self.type
        print "Variable data type = %s" % self.data_type
        print "Variable name = %s" % self.variable_name
        print "Variable index = %s" % self.index
        print "Variable patch = %s" % self.patch
        print "Variable start = %s" % self.start
        print "Variable numParticles = %s" % self.numParticles
        print "Variable end = %s" % self.end
        print "Variable filename = %s" % self.filename

    def read_data(self,filename):
        fd = open(filename,'rb')
#        print 'location of file descriptor = %s' % fd.tell()
        fd.seek(self.start)
#        print 'location of file descriptor after start = %s' % fd.tell()
        bytes_per_data = 8
        num_data = (self.end - self.start)/bytes_per_data
#        print 'Number of data items to load = %d' % num_data
        data_per_node = 1
        if self.data_type == 'Vector' or self.data_type == 'Point':
            data_per_node = 3
        if self.data_type == 'Matrix3':
            data_per_node = 9
        self.data = fromfile(fd,float,num_data)
        fd.close()
#        print self.data
#        print self.data.shape
#        print num_data/data_per_node,data_per_node
        self.data = self.data.reshape(num_data/data_per_node,data_per_node)
#        print self.data
#        print self.data.shape
        
    def get_data(self):
        return self.data

    def vtk_element(self):

        variable_type = self.type.split('<')[0]
        # print variable_type

#        if variable_type != 'CCVariable' and variable_type != 'NCVariable':
#            return None
                    
        var_elem = ET.Element('DataArray')
        var_elem.attrib['type'] = 'Float64'
        var_elem.attrib['Name'] = self.variable_name + '_' + repr(self.index)
        var_elem.attrib['NumberOfComponents'] = str(self.data.shape[1])
        var_elem.attrib['format'] = 'ascii'

        string_var_elem = str()
        for val in self.data.flatten():
            string_var_elem = string_var_elem + ' ' + repr(val)
        # Comment out below to not dump data for debugging purposes
        var_elem.text = string_var_elem

        indent(var_elem)
        # ET.dump(var_elem)
        return var_elem
