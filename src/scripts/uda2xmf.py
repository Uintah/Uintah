import numpy as np
import h5py
import struct
import itertools
import os
import xml.etree.ElementTree as ET
import ast
import sys
import types
import argparse
"""
Filename:   uda2xmf.py

Summary:    
            This is a python script for the post-processing of Uintah *.uda output files.
            Approximate particle domains are visualized for GIMP, CPDI and CPTI 
            interpolators, using the XDMF *.xmf file format, which can be visualized 
            using tools like ParaView or VisIt. 

Usage:      python uda2xmf.py [-flages] <uda_directory>

Input:      Uintah *.uda directory
Output:     XDMF *.xmf and HDF5 *.h5 files for visualization 
            
Revisions:  YYMMDD    Author            Comments
-------------------------------------------------------------------------------------
            171201    Cody Herndon, Brian Phung, Ashley Spear Speed up of python code 
            151201    Brian Leavy       Initial visualization of particle domains 
""" 
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    comm = None
    rank = 0
    size = 1

class Data_parser:
    """!
    @brief A factory class that returns variable data and descriptors.

    Parses input from binary files, and generates descriptors for that data.
    """
    parse_all=False

    @staticmethod
    def parse_ParticleVariable_Point(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses three floats for point coordinates.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        for node in range(nodes):
            point.append(struct.unpack(endian_flag+"ddd", binfile_handle.read(24)))

        return np.array(point)

    @staticmethod
    def parse_ParticleVariable_Vector(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses three floats for a vector.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        for node in range(nodes):
            point.append(struct.unpack(endian_flag+"ddd", binfile_handle.read(24)))

        return np.array(point)

    @staticmethod
    def parse_ParticleVariable_double(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses a single double for a scalar.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        for node in range(nodes):
            point.append(struct.unpack(endian_flag+"d", binfile_handle.read(8)))

        return np.array(point)

    @staticmethod
    def parse_ParticleVariable_Matrix3(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses nine doubles for a 3x3 matrix.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        for node in range(nodes):
            point.append(struct.unpack(endian_flag+"ddddddddd", binfile_handle.read(72)))

        return np.array(point)

    @staticmethod
    def parse_ParticleVariable_long64(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses one long long for a scalar.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        for node in range(nodes):
            point.append(struct.unpack(endian_flag+"Q", binfile_handle.read(8)))

        return np.array(point)

    @staticmethod
    def parse_NCVariable_double(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses one double for Node Centered scalar.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        end = nodes*8
        here = binfile_handle.tell()

        if Data_parser.parse_all:
            for node in range(nodes):
                point.append(struct.unpack(endian_flag+"d", binfile_handle.read(8)))

        else:
            binfile_handle.seek(here+end)



        return np.array(point)

    @staticmethod
    def parse_NCVariable_Vector(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses three doubles for a Node Centered vector.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        end = nodes*(8*3)
        here = binfile_handle.tell()

        if Data_parser.parse_all:
            for node in range(nodes):
                point.append(struct.unpack(endian_flag+"ddd", binfile_handle.read(24)))

        else:
            binfile_handle.seek(here+end)

        return np.array(point)

    @staticmethod
    def parse_NCVariable_Matrix3(binfile_handle, nodes, endian_flag):
        """!
        @brief Parses nine doubles for Node Centered 3x3 matrix.

        @param binfile_handle binary file to read from, already at the start point
        @param nodes the number of nodes to be read
        @param endian_flag the endianness of the data, as determine by the struct lib
        @return a numpy array of data
        """
        point = []

        end = nodes*(8*9)
        here = binfile_handle.tell()

        if Data_parser.parse_all:
            for node in range(nodes):
                point.append(struct.unpack(endian_flag+"ddddddddd", binfile_handle.read(72)))

        else:
            binfile_handle.seek(here+end)

        return np.array(point)

    @staticmethod
    def generate_xmf_ParticleVariable_Point(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends particle point descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        pass

    @staticmethod
    def generate_xmf_ParticleVariable_Vector(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends particle vector descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        variable = h5_file_path.split("/")[-1]

        attribute = ET.SubElement(xml_root, "Attribute")
        attribute.attrib["Name"] = variable
        attribute.attrib["Center"] = "Cell"
        attribute.attrib["Type"] = "Vector"

        data_item = ET.SubElement(attribute, "DataItem")
        data_item.attrib["Format"] = "HDF"
        data_item.attrib["Dimensions"] = str(h5_data_dims[0])+" "+str(h5_data_dims[1])
        data_item.attrib["Precision"] = "8"
        data_item.attrib["DataType"] = "Float"
        data_item.text = h5_file_path

    @staticmethod
    def generate_xmf_ParticleVariable_Double(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends particle double descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        variable = h5_file_path.split("/")[-1]

        attribute = ET.SubElement(xml_root, "Attribute")
        attribute.attrib["Name"] = variable
        attribute.attrib["Center"] = "Cell"
        attribute.attrib["Type"] = "Scalar"

        data_item = ET.SubElement(attribute, "DataItem")
        data_item.attrib["Format"] = "HDF"
        data_item.attrib["Dimensions"] = str(h5_data_dims[0])+" "+str(h5_data_dims[1])
        data_item.attrib["Precision"] = "8"
        data_item.attrib["DataType"] = "Float"
        data_item.text = h5_file_path

    @staticmethod
    def generate_xmf_ParticleVariable_Matrix3(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends particle 3x3 matrix descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        variable = h5_file_path.split("/")[-1]

        attribute = ET.SubElement(xml_root, "Attribute")
        attribute.attrib["Name"] = variable
        attribute.attrib["Center"] = "Cell"
        attribute.attrib["Type"] = "Tensor"

        data_item = ET.SubElement(attribute, "DataItem")
        data_item.attrib["Format"] = "HDF"
        data_item.attrib["Dimensions"] = str(h5_data_dims[0])+" "+str(h5_data_dims[1])
        data_item.attrib["Precision"] = "8"
        data_item.attrib["DataType"] = "Float"
        data_item.text = h5_file_path

    @staticmethod
    def generate_xmf_ParticleVariable_Long64(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends particle long long descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        variable = h5_file_path.split("/")[-1]

        attribute = ET.SubElement(xml_root, "Attribute")
        attribute.attrib["Name"] = variable
        attribute.attrib["Center"] = "Cell"
        attribute.attrib["Type"] = "Scalar"

        data_item = ET.SubElement(attribute, "DataItem")
        data_item.attrib["Format"] = "HDF"
        data_item.attrib["Dimensions"] = str(h5_data_dims[0])+" "+str(h5_data_dims[1])
        data_item.attrib["Precision"] = "8"
        data_item.attrib["DataType"] = "Int"
        data_item.text = h5_file_path

    @staticmethod
    def generate_xmf_NCVariable_Double(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends Node Centered double descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        pass

    @staticmethod
    def generate_xmf_NCVariable_Vector(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends Node Centered vector descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        pass

    @staticmethod
    def generate_xmf_NCVariable_Matrix3(xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Appends Node Centered 3x3 matrix descriptor to the xmf xml root.

        @param xml_root xmf descriptor root
        @param h5_file_path file and h5 path to data
        @param h5_data_dims data dimensions as returned from data.shape
        @return None
        """
        pass

class Variable_data_factory:
    """!
    @brief Front end class for static Data_parser variables.
    """
    _data_lookup = {}
    _data_lookup["ParticleVariable<Point>"] = Data_parser.parse_ParticleVariable_Point
    _data_lookup["ParticleVariable<Vector>"] = Data_parser.parse_ParticleVariable_Vector
    _data_lookup["ParticleVariable<double>"] = Data_parser.parse_ParticleVariable_double
    _data_lookup["ParticleVariable<Matrix3>"] = Data_parser.parse_ParticleVariable_Matrix3
    _data_lookup["ParticleVariable<long64>"] = Data_parser.parse_ParticleVariable_long64
    _data_lookup["NCVariable<double>"] = Data_parser.parse_NCVariable_double
    _data_lookup["NCVariable<Vector>"] = Data_parser.parse_NCVariable_Vector
    _data_lookup["NCVariable<Matrix3>"] = Data_parser.parse_NCVariable_Matrix3

    _generate_xmf = {}
    _generate_xmf["ParticleVariable<Point>"] = Data_parser.generate_xmf_ParticleVariable_Point
    _generate_xmf["ParticleVariable<Vector>"] = Data_parser.generate_xmf_ParticleVariable_Vector
    _generate_xmf["ParticleVariable<double>"] = Data_parser.generate_xmf_ParticleVariable_Double
    _generate_xmf["ParticleVariable<Matrix3>"] = Data_parser.generate_xmf_ParticleVariable_Matrix3
    _generate_xmf["ParticleVariable<long64>"] = Data_parser.generate_xmf_ParticleVariable_Long64
    _generate_xmf["NCVariable<double>"] = Data_parser.generate_xmf_NCVariable_Double
    _generate_xmf["NCVariable<Vector>"] = Data_parser.generate_xmf_NCVariable_Vector
    _generate_xmf["NCVariable<Matrix3>"] = Data_parser.generate_xmf_NCVariable_Matrix3

    endianness = ""

    @staticmethod
    def parse(type_string, binfile_handle, start, end, nodes, endian_flag=None):
        """!
        @brief Resolves the proper factory function to parse binary data.

        @param type_string data type as determined from the processor xml file (PXXXXX.xml)
        @param binfile_handle open binary file to extract dat from, will seek to data start
        @param start start point of the data to be read in the binary file
        @param end endpoint of the data, will be checked at the end of the factory function
        @param nodes number of variables to be read
        @param endian_flag endianness of the data, defaults to not override if not defined
        @return a numpy array of the data
        """
        if not endian_flag:
            if Variable_data_factory.endianness is "little-endian":
                endian_flag = "<"

            elif Variable_data_factory.endianness is "big-endian":
                endian_flag = ">"

            else:
                # default to native encoding
                endian_flag = "="

        binfile_handle.seek(start)

        result =  Variable_data_factory._data_lookup[type_string](binfile_handle, int(nodes), endian_flag)

        result_end = binfile_handle.tell()

        if result_end != end:
            raise IndexError("Attempted to read " +str(nodes)+ " elements at "+str(start)+" to "+str(end)+" and ended up at "+str(result_end)+" from the file "+str(binfile_handle.name)+".")

        return result

    @staticmethod
    def generate_xmf(type_string, xml_root, h5_file_path, h5_data_dims):
        """!
        @brief Resolves the proper factory to use to generate data descriptors.

        @param type_string data type as determined from the processor xml file (PXXXXX.xml)
        @param xml_root the root of the xml descriptor
        @param h5_file_path the filename and h5 path to the data
        @param h5_data_dims the dimensions of the data, as given by data.shape
        @return None
        """
        Variable_data_factory._generate_xmf[type_string](xml_root, h5_file_path, h5_data_dims)

class Domain_calculators():
    """!
    @brief Class to contain and generate corner points from particle center points based on extrapolation method.
    """
    corners_per_particle = {}
    corners_per_particle['cpti'] = 4
    corners_per_particle['linear'] = 8
    corners_per_particle['cpdi'] = 8
    corners_per_particle['gimp'] = 8

    @staticmethod
    def _calculate_cpti_domain(center, r):
        """!
        @brief Static function to generate cpti points.

        @param center numpy array particle center.
        @param r rvalues to use in extrapolating points
        @return numpy array of corner points
        """
        dim = center.shape[0]
        cpp = Domain_calculators.corners_per_particle['cpti']
        xpc = np.zeros((cpp, dim), float)

        # adapted from uda2vtk
        xpc[0] = (
                center[0] - (r[0][0]+r[1][0]+r[2][0])/4.0,
                center[1] - (r[0][1]+r[1][1]+r[2][1])/4.0,
                center[2] - (r[0][2]+r[1][2]+r[2][2])/4.0,
        )

        for i in range(cpp-1):
            xpc[i+1] = (
                xpc[0, 0]+r[i][0],
                xpc[0, 1]+r[i][1],
                xpc[0, 2]+r[i][2],
            )

        return xpc

    @staticmethod
    def _calculate_linear_domain(center, r):
        """!
        @brief Static function to generate linear interpolator corners.

        Not implemented, as gimp is the same with different rvalue packing.

        @param center numpy array particle center.
        @param r rvalues to use in extrapolating points
        @return numpy array of corner points
        """
        pass

    @staticmethod
    def _calculate_cpdi_domain(center, r):
        """!
        @brief Static function to generate cpdi points.

        Not implemented, as gimp is the same with different rvalue packing.

        @param center numpy array particle center.
        @param r rvalues to use in extrapolating points
        @return numpy array of corner points
        """
        pass

    @staticmethod
    def _calculate_gimp_domain(center, r):
        """!
        @brief Static function to generate gimp and cpdi points.

        @param center numpy array particle center.
        @param r rvalues to use in extrapolating points
        @return numpy array of corner points
        """
        dim = center.shape[0]
        cpp = Domain_calculators.corners_per_particle['gimp']
        xpc = np.zeros((cpp, dim), float)

        xpc[0]=(
                center[0]-r[0][0]/2.-r[1][0]/2.-r[2][0]/2.,
                center[1]-r[0][1]/2.-r[1][1]/2.-r[2][1]/2.,
                center[2]-r[0][2]/2.-r[1][2]/2.-r[2][2]/2.
                )

        xpc[1]=(
                center[0]+r[0][0]/2.-r[1][0]/2.-r[2][0]/2.,
                center[1]+r[0][1]/2.-r[1][1]/2.-r[2][1]/2.,
                center[2]+r[0][2]/2.-r[1][2]/2.-r[2][2]/2.
                )

        xpc[2]=(
                center[0]+r[0][0]/2.+r[1][0]/2.-r[2][0]/2.,
                center[1]+r[0][1]/2.+r[1][1]/2.-r[2][1]/2.,
                center[2]+r[0][2]/2.+r[1][2]/2.-r[2][2]/2.
                )

        xpc[3]=(center[0]-r[0][0]/2.+r[1][0]/2.-r[2][0]/2.,
                center[1]-r[0][1]/2.+r[1][1]/2.-r[2][1]/2.,
                center[2]-r[0][2]/2.+r[1][2]/2.-r[2][2]/2.
                )

        xpc[4]=(
                center[0]-r[0][0]/2.-r[1][0]/2.+r[2][0]/2.,
                center[1]-r[0][1]/2.-r[1][1]/2.+r[2][1]/2.,
                center[2]-r[0][2]/2.-r[1][2]/2.+r[2][2]/2.
                )

        xpc[5]=(
                center[0]+r[0][0]/2.-r[1][0]/2.+r[2][0]/2.,
                center[1]+r[0][1]/2.-r[1][1]/2.+r[2][1]/2.,
                center[2]+r[0][2]/2.-r[1][2]/2.+r[2][2]/2.
                )

        xpc[6]=(
                center[0]+r[0][0]/2.+r[1][0]/2.+r[2][0]/2.,
                center[1]+r[0][1]/2.+r[1][1]/2.+r[2][1]/2.,
                center[2]+r[0][2]/2.+r[1][2]/2.+r[2][2]/2.
                )

        xpc[7]=(
                center[0]-r[0][0]/2.+r[1][0]/2.+r[2][0]/2.,
                center[1]-r[0][1]/2.+r[1][1]/2.+r[2][1]/2.,
                center[2]-r[0][2]/2.+r[1][2]/2.+r[2][2]/2.
                )

        return xpc

class Domain_factory():
    """!
    @brief Class to resolve proper factory methods, and extrapolate corners.
    """
    interpolator = "cpti"
    _interpolator = {}
    _interpolator['cpti'] = Domain_calculators._calculate_cpti_domain
    _interpolator['linear'] = Domain_calculators._calculate_gimp_domain
    _interpolator['cpdi'] = Domain_calculators._calculate_gimp_domain
    _interpolator['gimp'] = Domain_calculators._calculate_gimp_domain

    @staticmethod
    def calculate_domain(center, r, interpolator='cpti'):
        """!
        @brief Static front end to generate extrapolated points.

        @param center numpy array of particle center
        @param r packed rvalues for extrapolating points
        @param interpolator method to use extracting corners, defaults to cpti
        @return numpy array of extrapolated points
        """
        return Domain_factory._interpolator[interpolator](center, r)

class Variable:
    """!
    @brief A container for variable data, as described in processor xml files (PXXXXX.xml).
    """
    def __init__(self, variable_element, file_path):
        """!
        @brief Initializes variable data using the processor files.

        @param variable_element etree element describing the variable
        @param file_path h5 data path
        @return None
        """
        self.name = variable_element.find(".//variable").text
        self.var_patch = int(variable_element.find(".//patch").text)
        self.var_type = variable_element.attrib["type"]
        self.index = variable_element.find(".//index").text
        self.start = int(variable_element.find(".//start").text)
        self.end = int(variable_element.find(".//end").text)
        self.datafile = file_path+variable_element.find(".//filename").text

        try:
            self.particles = int(variable_element.find(".//numParticles").text)

        except AttributeError:
            # no 'numParticles', is probably 'NCVariable' type, use nnodes
            self.particles = Patch_lightweight.nnodes[self.var_patch]

        f = open(self.datafile, "r")

        self.dataset = Variable_data_factory.parse(self.var_type, f, int(self.start), int(self.end), int(self.particles))
        f.close()

    def get_element(self):
        """!
        @brief Returns a tuple of the variable name, data, and data type for further processing.

        @return tuple of h5 data name, numpy array data, and string data type
        """
        return(self.name, self.dataset, self.var_type)

    def get_variables(self):
        """!
        @brief Returns the variable name

        Depreciated by get element

        @return string h5 variable name
        """
        return self.name

    def get_h5_path(self):
        """!
        @brief Returns the variable name

        Depreciated by get element

        @return string h5 variable name
        """
        return self.name

    def get_variable_data(self):
        """!
        @brief Returns the variable name

        Depreciated by get element

        @return string h5 variable name
        """
        return self.dataset

    def get_variable_index(self):
        """!
        @brief Returns the variable material index

        @return int variable material index
        """
        return self.index

class Lightweight_container(object):
    """!
    @brief A base class for storing meta, paths, and subcontainers.
    """
    def __init__(self):
        """!
        @brief initializer for the container

        @return None
        """
        self.name = ""
        self.contents = []

    def get_element(self):
        """!
        @brief Collects and modifies variable data for each sub element.

        @return array of tuples as described in Variable class, appends data path with container name
        """
        result = []

        for item in self.contents:
            item_res = item.get_element()

            if type(item_res) is tuple:
                if self.name:
                    result.append((self.name+"/"+item_res[0], item_res[1]))

                else:
                    result.append(item_res)

            else:
                updated_elements = []
                for element in item_res:
                    if self.name:
                        updated_elements.append((self.name+"/"+element[0], element[1]))

                    else:
                        updated_elements.append(element)

                result.extend(updated_elements)

        return result

    def get_variables(self):
        """!
        @brief Returns variable names without modification.

        @return string of varaible name
        """
        result = []

        for item in self.contents:
            item_res = item.get_variables()

            if type(item_res) is str:
                result.append(item_res)

            else:
                result.extend(item_res)

        return result

class Patch_lightweight(Lightweight_container):
    """!
    @brief A derived class of the container for patch level data.
    """
    nnodes = {}
    patchfiles = {}

    def __init__(self, patch_element, rel_path):
        """!
        @brief Derived initializer for the container.

        @param patch_element patch data xml element
        @param rel_path h5 file and path to the patch level (e.g. 'data.h5:dataset/T0/')
        @return None
        """
        super(Patch_lightweight, self).__init__()

        self.pid = int(patch_element.find(".//id").text)
        p_num = str(self.pid).zfill(5)
#        self.name = "p"+p_num
        self.proc = int(patch_element.find(".//proc").text)

#        self.low_index = ast.literal_eval(patch_element.find(".//lowIndex").text)
#        self.high_index = ast.literal_eval(patch_element.find(".//highIndex").text)
#        self.interior_low = ast.literal_eval(patch_element.find(".//interiorLowIndex").text)
#        self.interior_high = ast.literal_eval(patch_element.find(".//interiorHighIndex").text)
#        self.lower = ast.literal_eval(patch_element.find(".//lower").text)
#        self.upper = ast.literal_eval(patch_element.find(".//upper").text)

#        Patch_lightweight.nnodes[self.pid] = int(patch_element.find(".//nnodes").text)

        xml_file = rel_path+Patch_lightweight.patchfiles[self.proc]
        data_path = os.path.dirname(xml_file)+"/"

        patch_tree = ET.parse(xml_file)
        self.patch_root = patch_tree.getroot()

        for ele in self.patch_root.findall(".//Variable"):
            self.contents.append(Variable(ele, data_path))

class Level_lightweight(Lightweight_container):
    """!
    @brief A derived class for level data.
    """
    def __init__(self, level_element, rel_path):
        """!
        @brief Derived initializer for the container.

        @param level_element etree level element
        @param rel_path h5 filename and data path
        @return None
        """
        super(Level_lightweight, self).__init__()

#        self.name = "l"+level_element.find(".//id").text
        self.cellspacing = ast.literal_eval(level_element.find("./cellspacing").text)

        patch_elements = level_element.findall(".//Patch")

        # Patches in p00000 are controlled by processor id 00000
        for ele in patch_elements:
            Patch_lightweight.nnodes[int(ele.find(".//id").text)] = int(ele.find(".//nnodes").text)

        for ele in patch_elements:
            self.contents.append(Patch_lightweight(ele, rel_path))

class Grid_lightweight(Lightweight_container):
    """!
    @brief A derived class for grid data.
    """
    def __init__(self, grid_element, patch_xml):
        """!
        @brief Derived initializer for the container.

        @param grid_element etree level element
        @param rel_path h5 filename and data path
        @return None
        """
        super(Grid_lightweight, self).__init__()

        self.root = grid_element

        for ele in grid_element.findall("Level"):
            self.contents.append(Level_lightweight(ele, patch_xml))

    def get_level_cellspacing(self):
        result = None

        for level in self.contents:
            if level.cellspacing:
                result = level.cellspacing
                break

        return result

    def generate_xmf(self, root, h5_handle, h5_path, h5_root):
        """!
        @brief Used to add data elements from each patch to the xmf descriptor.

        @param root xmf descriptor root
        @param h5_handle generated h5 file handle
        @param h5_path relative path to the grid (e.g. dataset/T0/)
        @param h5_root path including the file name (e.g. data.h5:dataset/T0/)
        @return None
        """
        elements = self.get_element()

        # create attributes for each variable
        for variable in Uda_lightweight.variables:
            for element in elements:
                variable_name = element[0].split('/')[-1]
                if variable == variable_name:
                    try:
                        dims = h5_handle[h5_path+element[0]].shape

                        Variable_data_factory.generate_xmf(
                                Uda_lightweight.variables[variable_name],
                                root, h5_root+element[0], dims
                            )

                    except KeyError:
                        pass

                    break

class Timestep_lightweight(Lightweight_container):
    """!
    @brief A minimal container for timestep meta.
    """
    verbose = False

    def __init__(self, timestep_element, path):
        """!
        @brief Constructs a timestep from an etree element.

        @param timestep_element timestep xml element
        @param path path to h5 data
        """
        super(Timestep_lightweight, self).__init__()

        self.xyz = None
        self.topology = None
        self.material = None

        self.name = "T"+timestep_element.text
        self.path = path+timestep_element.attrib["href"]
        self.time = timestep_element.attrib["time"]
        self.delta = timestep_element.attrib["oldDelt"]

        tree = ET.parse(self.path)
        self.root = tree.getroot()

        self.res = ast.literal_eval(self.root.find(".//geom_object/res").text)
        self.interpolator = self.root.find(".//MPM/interpolator").text
        Domain_factory.interpolator = self.interpolator

        patch_xml = self.root.findall(".//Data/Datafile")

        for patch in patch_xml:
            Patch_lightweight.patchfiles[int(patch.attrib["proc"])] = patch.attrib["href"]

        self.rel_path = os.path.dirname(self.path)+"/"

        if self.verbose:
            print("reading timestep "+self.name)

    def parse(self):
        """!
        @brief Parses contained data after initialization.

        @return None
        """
        if self.verbose:
            print("parsing timestep "+self.name)

        for ele in self.root.findall(".//Grid"):
            self.contents.append(Grid_lightweight(ele, self.rel_path))

    def get_element(self):
        """!
        @brief Returns element data and derives and appends xyz and topology data.

        @return array of element tuples
        """
        result = super(Timestep_lightweight, self).get_element()

        centroids = []
        scalefactors = []
        domains = []

        if self.xyz is not None and self.topology is not None:
            result.append((self.name+"/xyz", self.xyz))
            result.append((self.name+"/topology", self.topology))
            return result

        # append coordinate data and scalefactor for entire timestep
        for element in result:
            name = element[0].split('/')[-1]

            if name == "p.x":
                centroids.append(element)

            if name == "p.scalefactor":
                scalefactors.append(element)

        if len(scalefactors) is 0:
            for centroid in centroids:
                domains.append(( centroid, ("",[]) ))

        # associate centroids and scalefactors
        else:
            for centroid in centroids:
                cent_path = os.path.dirname(centroid[0])

                for i, scalefactor in enumerate(scalefactors):
                    scale_path = os.path.dirname(scalefactor[0])

                    if scale_path == cent_path and centroid[1].shape[0] == scalefactor[1].shape[0]:
                        domains.append((centroid, scalefactor))
                        scalefactors.pop(i)
                        break

        # calculate xyz coordinates for each centroid/scalefactor combination
        xyz = []

        if self.verbose:
            print("extrapolating corners using "+self.interpolator+" method")

        for domain in domains:
            # domain[0][1] = p.x coord
            # domain[1][1] = p.scalefactor argument

            for i, element in enumerate(domain[0][1]):
                # CPTI/CPDI
                if self.interpolator == 'cpti' or self.interpolator == 'cpdi':
                    # pack rvectors with scalefactors
                    # scalefactor vectors passed as a len=9 array
                    scale = domain[1][1][i]
                    r = []
                    r.append([scale[0], scale[3], scale[6]]) # r1 = r[0]
                    r.append([scale[1], scale[4], scale[7]]) # r2 = r[1]
                    r.append([scale[2], scale[5], scale[8]]) # r3 = r[2]

                    xyz.append(Domain_factory.calculate_domain(
                        element,
                        r,
                        interpolator=self.interpolator))

                # GIMP
                else:
                    for grid in self.contents:
                        cellspacing = grid.get_level_cellspacing()
                        if cellspacing:
                            break

                    # pack rvectors
                    r = []
                    r.append([cellspacing[0]/float(self.res[0]), 0.0, 0.0]) # r1 = r[0]
                    r.append([0.0, cellspacing[1]/float(self.res[1]), 0.0]) # r2 = r[1]
                    r.append([0.0, 0.0, cellspacing[2]/float(self.res[2])]) # r3 = r[2]

                    xyz.append(Domain_factory.calculate_domain(
                        element,
                        r,
                        interpolator=self.interpolator
                        )
                    )

        xyz_data = np.array([])

#        for data in xyz:
#            if len(data) is not 0:
#                if len(xyz_data) is 0:
#                    xyz_data = data
#                else:
#                    xyz_data = np.concatenate((xyz_data, data), axis=0)


#        print("concatenating...")
        xyz_data = np.concatenate(xyz)
#        print("done")
        self.xyz = xyz_data

        xyz_name = self.name+"/xyz"

        result.append((xyz_name, xyz_data))

        # generate topology for the coordinates
        cpp = Domain_calculators.corners_per_particle[self.interpolator]

        topology = []
        topo_item = []
        for i in range(xyz_data.shape[0]):
            topo_item.append(i)

            if (i+1)%cpp == 0:
                topology.append(topo_item)
                topo_item = []

        topo_data = np.array(topology)

        self.topology = topo_data

        topo_name = self.name+"/topology"
        result.append((topo_name, topo_data))

        return result

    def generate_h5(self):
        """!
        @brief Generates h5 data from elements.

        @return appends and returns h5 paths and data
        """
        if self.verbose:
            print("generating h5 elements")

        datasets = self.get_element()

        paths = []
        path_data = []

        for element in datasets:
            if element[0] not in paths:
                if len(element[1]) is not 0:
                    paths.append(element[0])
                    path_data.append(element[1])

            else:
                index = paths.index(element[0])

                if len(element[1]) is not 0:
                    path_data[index] = np.append(path_data[index], element[1], 0)

        return (paths, path_data)

    def generate_xmf(self, root, h5_handle, h5_path, h5_root):
        """!
        @brief Generates xmf descriptors to data for the timestep.

        @param root etree parent root
        @param h5_handle file handle for the h5 data

        @param h5_root root of the h5 data path (name of the dataset)
        @return None
        """
        if self.verbose:
            print("generating xmf descriptor")

        elements = self.get_element()

        timestep = ET.SubElement(root, "Grid")
        timestep.attrib["Name"] = self.name
        timestep.attrib["GridType"]="Uniform"

        topology = ET.SubElement(timestep, "Topology")

        for i, path in enumerate(elements):
            name = path[0].split('/')[-1]
            if name == "topology":
                topo_data_path = h5_root+path[0]
                topo_data_element = h5_handle[h5_path+path[0]]

        if self.interpolator == 'cpti':
            topology.attrib["TopologyType"] = "Tetrahedron"
            if args.xdmf2:
              topology.attrib[ "NumberOfElements"] = str(topo_data_element.shape[0])

        else:
            topology.attrib["TopologyType"] = "Hexahedron"
            if args.xdmf2:
              topology.attrib[ "NumberOfElements"] = str(topo_data_element.shape[0])

        topo_data = ET.SubElement(topology, "DataItem")
        topo_data.attrib["Format"] = "HDF"
        topo_data.attrib["DataType"] = "UInt"
        topo_data.attrib["Precision"] = "8"

        topo_data.text = topo_data_path
        topo_shape = h5_handle[h5_path+path[0]].shape
        topo_data.attrib["Dimensions"] = str(topo_data_element.shape[0])+" "+str(topo_data_element.shape[1])


        geometry = ET.SubElement(timestep, "Geometry")
        geometry.attrib["GeometryType"] = "XYZ"

        geo_data = ET.SubElement(geometry, "DataItem")
        geo_data.attrib["Format"] = "HDF"

        geo_data.attrib["DataType"] = "Float"
        geo_data.attrib["Precision"] = "8"

        for i, path in enumerate(elements):
            name = path[0].split('/')[-1]
            if name == "xyz":
                geo_data.text = h5_root+path[0]
                geo_shape = h5_handle[h5_path+path[0]].shape
                geo_data.attrib["Dimensions"] = str(geo_shape[0])+" "+str(geo_shape[1])


        time = ET.SubElement(timestep, "Time")
        time.attrib["Value"] = self.time

        for level in self.contents:
            level.generate_xmf(timestep, h5_handle, h5_path+self.name+"/", h5_root+self.name+"/")

class Uda_lightweight(Lightweight_container):
    """!
    @brief A minimal proof of concept for reading in Uda data.
    """
    variables = {}
    static_name = ""

    def __init__(self, root_folder, target_timestep=None):
        """!
        @brief Derived initializer for the container.

        @param root_folder folder containing timestep folders, index and input xmls
        """
        super(Uda_lightweight, self).__init__()

        read_error = True

        if os.path.isdir(root_folder):
            self.root_folder = root_folder

            if self.root_folder[-1] is not "/":
                self.root_folder += "/"

            read_error = self._read_input()
            read_error = self._read_index(target_timestep=target_timestep) and read_error

        if read_error:
            print("Failed to read. Please pass Uda folder containing the index.xml and input.xml files.")
            exit()

    def _read_index(self, target_timestep=None):
        """!
        @brief Initalizes data from the index xml.

        @param target_timestep array of the timesteps to be parsed
        @return return conditional, true if exit on error
        """
        index_file = self.root_folder+"index.xml"

        if not os.path.exists(index_file):
            return True

        index_tree = ET.parse(index_file)
        index_root = index_tree.getroot()

        self.particle_position = index_root.find(".//ParticlePosition").text
        self.data_endianness = index_root.find(".//endianness").text

        Variable_data_factory.endianness = self.data_endianness

        var_elems = index_root.findall(".//variables/variable")
        for ele in var_elems:
            Uda_lightweight.variables[ele.attrib["name"]] = ele.attrib["type"]

        timestep_elems = index_root.findall(".//timesteps/timestep")

        for ele in timestep_elems:
            self.contents.append(Timestep_lightweight(ele, self.root_folder))

            if target_timestep:
                if self.contents[-1].name not in target_timestep:
                    del self.contents[-1]

        return False

    def _read_input(self):
        """!
        @brief Initializes data from the input xml.

        @return return conditional, true if exit on error
        """
        input_file = self.root_folder+"input.xml"

        if not os.path.exists(input_file):
            return True

        input_tree = ET.parse(self.root_folder+"input.xml")
        input_root = input_tree.getroot()

        self.name = input_root.find(".//Meta/title").text
        Uda_lightweight.static_name = self.name
        self.time_min = input_root.find(".//Time/initTime").text
        self.time_max = input_root.find(".//Time/maxTime").text
        self.delta_min = input_root.find(".//Time/delt_min").text
        self.delta_max = input_root.find(".//Time/delt_max").text

        return False

    def generate_descriptors(self, h5_handle, xmf_handle):
        """!
        @brief Generates h5 and xmf files in order.

        @param h5_handle handle of the opened h5 file
        @param xmf_handle handle of the opened xmf file
        @return None
        """
#        self.generate_h5(h5_handle)
#        self.generate_xmf(h5_handle, xmf_handle)

        root = self.generate_xmf(h5_handle.filename, xmf_handle)
        timeseries_grid = root.find(".//Grid[@Name='TimeSeries']")

        h5_path = self.name+"/"
        h5_root = h5_handle.filename+":"+h5_path

        if size == 1:
            # parse contents
            while len(self.contents) > 0:
                item = self.contents[0]

                item.parse()

                h5_paths, h5_data_sets = item.generate_h5();

                for i, path in enumerate(h5_paths):
                    h5_handle.create_dataset(h5_path+path, data=h5_data_sets[i])

                item.generate_xmf(timeseries_grid, h5_handle, h5_path, h5_root)

                del self.contents[0]

        else:
            # split items by processor
            proc_items = []

            for proc_index in range(size-1):
                proc_items.append([])

            # round robin
            for i, item in enumerate(self.contents):
                proc_items[i%(size-1)].append(item)

            # send static info
            for i in range(size-1):
                comm.send((Patch_lightweight.patchfiles, Domain_factory.interpolator), dest=(i+1), tag=1)

            for i, item in enumerate(proc_items):
                comm.send(item, dest=(i+1), tag=2)

#            del proc_items

            proc_items_parsed = []
            done_threads = []
            done = False

            for i in range(size-1):
                proc_items_parsed.append([])

            while not done:
                for i in range(size-1):
                    index = (i+1)
                    if comm.Iprobe(source=index, tag=3):
                        proc_items_parsed[i].append(comm.recv(source=index, tag=3))

                        if proc_items_parsed[i][-1] == None:
                            done_threads.append(index)

                        else:
                            item = proc_items_parsed[i].pop()
                            h5_paths, h5_data_sets = item.generate_h5()

                            for i, path in enumerate(h5_paths):
                                h5_handle.create_dataset(h5_path+path, data=h5_data_sets[i])

                            item.generate_xmf(timeseries_grid, h5_handle, h5_path, h5_root)

#                            del item

                        comm.send(0, dest=(index), tag=4)


                for i in range(size-1):
                    index = (i+1)
                    if index not in done_threads:
                        done = False
                        break

                    else:
                        done = True

        xmf_handle.write(ET.tostring(root))

    @staticmethod
    def generate_descriptor_parallel(item):
        result = []

        item.parse()

        # generating the dataset the first time caches the results
        h5_paths, h5_datasets = item.generate_h5()

        return item


    def generate_h5(self, h5_handle):
        """!
        @brief Generates h5 file from data and paths.

        Depreciated by generate_descriptors

        @param h5_handle the handle of the opened h5 file
        @return None
        """
        paths = []
        path_data = []

        results = []

        for item in self.contents:
            results.append(item.generate_h5())

        for result in results:
            item_paths, item_path_data = result

            paths.extend(item_paths)
            path_data.extend(item_path_data)

        for i, path in enumerate(paths):
            h5_handle.create_dataset(self.name+"/"+path, data=path_data[i])

        h5_handle.flush()

    def generate_xmf(self, h5_filename, xmf_handle):
        """!
        @brief Generates xmf file root and timestep grid for timesteps.

        @param xmf_handle the handle of the opened xmf file
        @return xml root node
        """
        h5_path = self.name+"/"
        h5_root = h5_filename+":"+h5_path

        root = ET.Element("Xdmf")
        root.attrib["Version"] = "2.0"

        domain = ET.SubElement(root, "Domain")

        # primary data container, timeseries grid
        timeseries_grid = ET.SubElement(domain, "Grid")
        timeseries_grid.attrib["Name"] = "TimeSeries"
        timeseries_grid.attrib["GridType"] = "Collection"
        timeseries_grid.attrib["CollectionType"] = "Temporal"

        return root

if __name__=="__main__":
    if rank == 0:
        parser = argparse.ArgumentParser()

        parser.add_argument("uda", help="uda directory with index.xml and input.xml files", type=str)
        #parser.add_argument("xml", help="target output xml name", type=str)
        #parser.add_argument("h5", help="target output h5 name", type=str)
        parser.add_argument("--timestep", help="target timestep names (e.g. T1,T2,T10)", type=str)
        parser.add_argument("--parse_all", help="parses non-particle centered variables, drastically increases runtime", action="store_true")
        parser.add_argument("--verbose", help="prints debug messages", action="store_true")
        parser.add_argument("--xdmf2", help="create XMF2 compatible files", action="store_true")

        args = parser.parse_args()

        if args.timestep:
            timesteps = args.timestep.split(',')

        else:
            timesteps = None

        Timestep_lightweight.verbose = args.verbose
        Data_parser.parse_all = args.parse_all

        uda = Uda_lightweight(args.uda, target_timestep=timesteps)

        prefix=args.uda.split('/')[-1].split('.')[0]
        xml_root = prefix+'.xmf' 
        h5_root = prefix+'.h5' 

        h = h5py.File(h5_root, "w")
        x = open(xml_root, "w")



        uda.generate_descriptors(h, x)
        h.close()
        x.close()

    else:
        # synch patchfiles
        Patch_lightweight.patchfiles, Domain_factory.interpolator = comm.recv(source=0, tag=1)

        # wait for comms
        items = comm.recv(source=0, tag=2)
        ack = 0

        while len(items) > 0:
            if items[0].verbose:
                print("Parsing timestep "+items[0].name+" in thread "+str(rank))
            dataset = Uda_lightweight.generate_descriptor_parallel(items[0])
            comm.send(dataset, dest=0, tag=3)
            # poor man's semaphore
            ack = comm.recv(source=0, tag=4)
            del items[0]

        comm.send(None, dest=0, tag=3)
