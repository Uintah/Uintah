/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
  In addition this code was derived from h5dump.c which is part of the
  HDF5 tools distribution. As such this copyright notice also applys:



  Copyright Notice and Statement for NCSA Hierarchical Data Format (HDF)
  Software Library and Utilities

  NCSA HDF5 (Hierarchical Data Format 5) Software Library and Utilities 
  Copyright 1998, 1999, 2000, 2001, 2002, 2003 by the Board of Trustees 
  of the University of Illinois.  All rights reserved.

  Contributors: National Center for Supercomputing Applications (NCSA) at the
  University of Illinois at Urbana-Champaign (UIUC), Lawrence Livermore 
  National Laboratory (LLNL), Sandia National Laboratories (SNL), Los Alamos 
  National Laboratory (LANL), Jean-loup Gailly and Mark Adler (gzip library).

  Redistribution and use in source and binary forms, with or without
  modification, are permitted for any purpose (including commercial purposes)
  provided that the following conditions are met:

  1.  Redistributions of source code must retain the above copyright notice,
  this list of conditions, and the following disclaimer.

  2.  Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions, and the following disclaimer in the documentation
  and/or materials provided with the distribution.

  3.  In addition, redistributions of modified forms of the source or binary
  code must carry prominent notices stating that the original code was
  changed and the date of the change.

  4.  All publications or advertising materials mentioning features or use of
  this software are asked, but not required, to acknowledge that it was 
  developed by the National Center for Supercomputing Applications at the 
  University of Illinois at Urbana-Champaign and to credit the contributors.

  5.  Neither the name of the University nor the names of the Contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission from the University or the Contributors,
  as appropriate for the name(s) to be used.

  6.  THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY AND THE CONTRIBUTORS "AS IS"
  WITH NO WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED.  In no event
  shall the University or the Contributors be liable for any damages
  suffered by the users arising out of the use of this software, even if
  advised of the possibility of such damage.

  --------------------------------------------------------------------------
  Portions of HDF5 were developed with support from the University of 
  California, Lawrence Livermore National Laboratory (UC LLNL).
  The following statement applies to those portions of the product
  and must be retained in any redistribution of source code, binaries,
  documentation, and/or accompanying materials:

  This work was partially produced at the University of California,
  Lawrence Livermore National Laboratory (UC LLNL) under contract no.
  W-7405-ENG-48 (Contract 48) between the U.S. Department of Energy 
  (DOE) and The Regents of the University of California (University) 
  for the operation of UC LLNL.

  DISCLAIMER:
  This work was prepared as an account of work sponsored by an agency 
  of the United States Government.  Neither the United States 
  Government nor the University of California nor any of their 
  employees, makes any warranty, express or implied, or assumes any 
  liability or responsibility for the accuracy, completeness, or 
  usefulness of any information, apparatus, product, or process 
  disclosed, or represents that its use would not infringe privately-
  owned rights.  Reference herein to any specific commercial products, 
  process, or service by trade name, trademark, manufacturer, or 
  otherwise, does not necessarily constitute or imply its endorsement, 
  recommendation, or favoring by the United States Government or the 
  University of California.  The views and opinions of authors 
  expressed herein do not necessarily state or reflect those of the 
  United States Government or the University of California, and shall 
  not be used for advertising or product endorsement purposes.
  --------------------------------------------------------------------------

*/

/*
 *  HDF5Dump.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include "HDF5Dump.h"

#ifdef HAVE_HDF5

namespace DataIO {

using namespace std;

herr_t
hdf5_attr_iter(hid_t group_id, const char *name, void *op_data) {

  HDF5Dump *hdf = (HDF5Dump*) op_data;

  return hdf->attr( group_id, name );
}

herr_t
hdf5_group_iter(hid_t group_id, const char *name, void *op_data) {

  HDF5Dump *hdf = (HDF5Dump*) op_data;

  return hdf->all( group_id, name );
}

HDF5Dump::HDF5Dump( ostream *iostr ) :
  indent_(0)
{
  iostr_ = iostr;
}

void
HDF5Dump::tab()
{
  for( unsigned int i=0; i<indent_; i++ )
    *iostr_ << "   ";
}


herr_t
HDF5Dump::file(const string fname) {

  herr_t status = 0;

  /* Open the file using default properties. */
  hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  if (file_id < 0) {
    error_msg_ = "Unable to open file: " + fname;
    status = -1;
  }

  *iostr_ << "HDF5 \"" << fname << "\" {" << endl;

  hid_t group_id = H5Gopen(file_id, "/");

  if (group_id < 0) {
    error_msg_ = string("Unable to open root group");
    status = 1;
  } else if( group(group_id, "/") < 0 ) {
    error_msg_ = string("Unable to dump root group");
    status = -1;
  }
  
  *iostr_ << "}" << endl;

  if (H5Gclose(group_id) < 0) {
    error_msg_ = "Unable to close root group";
    status = -1;
  }

  if (H5Fclose(file_id ) < 0) {
    error_msg_ = "Unable to close file: " + fname;
    status = -1;
  }

  return status;
}


herr_t
HDF5Dump::group(hid_t group_id, const char *name ) {

  herr_t status = 0;

  tab();
  *iostr_ << "GROUP \"" << name << "\" {" << endl;
  indent_++;

  H5Aiterate(group_id,      NULL, hdf5_attr_iter,  this);
  H5Giterate(group_id, ".", NULL, hdf5_group_iter, this);

  indent_--;
  tab();
  *iostr_ << "}" << endl;

  return status;
}


herr_t
HDF5Dump::all(hid_t obj_id, const char *name) {

  herr_t status = 0;

  H5G_stat_t  statbuf;
  hid_t group_id;
  hid_t dataset_id;

  H5Gget_objinfo(obj_id, name, 0, &statbuf);
  
  switch (statbuf.type) {
  case H5G_GROUP:
    if ((group_id = H5Gopen(obj_id, name)) < 0) {
      error_msg_ = string("Unable to open group \"") + name + "\"";
      status = -1;
    } else if( group(group_id, name) < 0 ) {
      error_msg_ = string("Unable to dump group \"") + name + "\"";
      status = -1;
    } else {
      H5Gclose(group_id);
      status = 0;
    }

    break;
  case H5G_DATASET:
    if ((dataset_id = H5Dopen(obj_id, name)) < 0) {
      error_msg_ = string("Unable to open dataset \"") + name + "\"";
      status = -1;
    } else if( dataset(dataset_id, name) < 0) {
      error_msg_ = string("Unable to dump dataset \"") + name + "\"";
      status = -1;
    } else {
      H5Dclose(dataset_id);
      status = 0;
    }

    break;

  default:
    break;
  }

  return status;
}


herr_t
HDF5Dump::attr(hid_t group_id, const char *name) {

  herr_t status = 0;

  tab();
  *iostr_ << "ATTRIBUTE \"" << name << "\" {" << endl;
  indent_++;

  hid_t attr_id = H5Aopen_name(group_id, name);

  if (attr_id < 0) {
    error_msg_ = string("Unable to open attribute \"") + name + "\"";
    status = -1;
  } else {

    /* Open the file space in the file.
    hid_t file_space_id = H5Aget_space( attr_id );

    if( file_space_id < 0 ) {
      error_msg_ =
        string("Unable to open file space \"") + name + "\"";
      return -1;
    } else if( dataspace(file_space_id) < 0 ) {
      error_msg_ =
	string("Unable to dump attribute data \"") + name + "\"";
      return -1;
    }

    H5Sclose(file_space_id);
    */

    if( datatype(attr_id, 0) < 0) {
      error_msg_ =
	string("Unable to dump attribute type \"") + name + "\"";
      status = -1;
    } else if( data(attr_id, 0) < 0 ) {
      error_msg_ =
	string("Unable to dump attribute data \"") + name + "\"";
      status = -1;
    }

    H5Aclose(attr_id);
  }    

  indent_--;
  tab();
  *iostr_ << "}" << endl;
  
  return status;
}


herr_t
HDF5Dump::dataset(hid_t dataset_id, const char *name) {

  herr_t status = 0;

  tab();
  *iostr_ << "DATASET \"" << name << "\" {" << endl;
  indent_++;

  hid_t file_space_id = H5Dget_space( dataset_id );

  /* Open the data space in the file. */
  if( datatype( dataset_id, H5G_DATASET ) < 0) {
    error_msg_ = string("Unable to dump datatype \"") + name + "\"";
    status = -1;
  } else if( file_space_id < 0 ) {
    error_msg_ = string("Unable to open dataspace \"") + name + "\"";
    status = -1;
  } else if( dataspace( file_space_id ) < 0) {
    error_msg_ = string("Unable to dump dataspace \"") + name + "\"";
    status = -1;
  } else {
    H5Sclose(file_space_id);
    /*    
    if( data(dataset_id, H5G_DATASET, iostr_) < 0 ) {
      error_msg_ = "Unable to dump dataset data \"" + name + "\"";
      status = -1;
    }
    */
  }

  H5Aiterate(dataset_id, NULL, hdf5_attr_iter, this);

  indent_--;
  tab();
  *iostr_ << "}" << endl;

  return status;
}


herr_t
HDF5Dump::datatype(hid_t obj_id, hid_t type)
{
  herr_t status = 0;

  hid_t type_id;

  // Get the data type.
  if( type == H5G_DATASET ) {
    type_id = H5Dget_type(obj_id);
  } else {
    type_id = H5Aget_type(obj_id);
  }

  tab();
  *iostr_ << "DATATYPE \"";

  switch (H5Tget_class(type_id)) {
  case H5T_INTEGER:
    *iostr_ << "Integer";
    break;

  case H5T_FLOAT:
    if (H5Tequal(type_id, H5T_IEEE_F32BE) ||
	H5Tequal(type_id, H5T_IEEE_F32LE) ||
	H5Tequal(type_id, H5T_NATIVE_FLOAT)) {
      // Float
      *iostr_ << "Float";

    } else if (H5Tequal(type_id, H5T_IEEE_F64BE) ||
	       H5Tequal(type_id, H5T_IEEE_F64LE) ||
	       H5Tequal(type_id, H5T_NATIVE_DOUBLE) ||
	       H5Tequal(type_id, H5T_NATIVE_LDOUBLE)) {
      // Double
      *iostr_ << "Double";

    } else {
      *iostr_ << "Undefined HDF5 float.";
    }
    break;

  case H5T_STRING:
    *iostr_ << "String";
    if( type == H5G_DATASET )
      *iostr_ << " - Unsupported";
    break;

  case H5T_COMPOUND:
    *iostr_ << "Compound - Unsupported";
    break;
      
  case H5T_REFERENCE:
    *iostr_ << "Reference";
    if( type == H5G_DATASET )
      *iostr_ << " - Unsupported";
    break;
      
  default:
    *iostr_ << "Unsupported or unknown data type";
    break;
  }

  *iostr_ << "\"" << endl;

  H5Tclose(type_id);

  return status;
}


herr_t
HDF5Dump::dataspace(hid_t file_space_id) {

  herr_t status = 0;

  /* Get the rank (number of dims) in the space. */
  int ndims = H5Sget_simple_extent_ndims(file_space_id);

  if (H5Sis_simple(file_space_id)) {

    if (ndims == 0) {
      /* scalar dataspace */

      tab();
      *iostr_ << "DATASPACE  SCALAR { ( 1 ) }" << endl;
    } else {
      /* simple dataspace */

      hsize_t *dims = new hsize_t[ndims];

      /* Get the dims in the space. */
      int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

      if( ndim != ndims ) {
	error_msg_ = "Data dimensions not match.";
	return -1;
      }

      tab();
      *iostr_ << "DATASPACE  SIMPLE { ( " << dims[0];

      for( int i = 1; i < ndims; i++ )
	*iostr_ << ", " << dims[i];
      
      *iostr_ << " ) }" << endl;

      delete dims;
    }
  }

  return status;
}


herr_t
HDF5Dump::data(hid_t obj_id, hid_t type) {

  hid_t type_id, file_space_id, mem_type_id;

  /* Get the data type and open the coordinate space. */
  if( type == H5G_DATASET ) {
    type_id = H5Dget_type(obj_id);
    file_space_id = H5Dget_space( obj_id );
  } else {
    type_id = H5Aget_type(obj_id);
    file_space_id = H5Aget_space( obj_id );
  }

  if( file_space_id < 0 ) {
    error_msg_ = "Unable to open data ";
    return -1;
  }

  switch (H5Tget_class(type_id)) {
  case H5T_STRING:
    // String
    if(H5Tis_variable_str(type_id)) {                    
      mem_type_id = H5Tcopy(H5T_C_S1);                        
      H5Tset_size(mem_type_id, H5T_VARIABLE);                 
    } else {                                      
      mem_type_id = H5Tcopy(type_id);
      H5Tset_cset(mem_type_id, H5T_CSET_ASCII);
    }

    break;

  case H5T_INTEGER:
    // Integer
    mem_type_id = H5T_NATIVE_INT;
    break;

  case H5T_FLOAT:
    if (H5Tequal(type_id, H5T_IEEE_F32BE) ||
	H5Tequal(type_id, H5T_IEEE_F32LE) ||
	H5Tequal(type_id, H5T_NATIVE_FLOAT)) {
      // Float
      mem_type_id = H5T_NATIVE_FLOAT;

    } else if (H5Tequal(type_id, H5T_IEEE_F64BE) ||
	       H5Tequal(type_id, H5T_IEEE_F64LE) ||
	       H5Tequal(type_id, H5T_NATIVE_DOUBLE) ||
	       H5Tequal(type_id, H5T_NATIVE_LDOUBLE)) {
      // Double
      mem_type_id = H5T_NATIVE_DOUBLE;

    } else {
      error_msg_ = "Undefined HDF5 float";
      return -1;
    }
    break;
  case H5T_REFERENCE:
    mem_type_id = H5T_STD_REF_OBJ;
    break;

  default:
    error_msg_ = "Unknown or unsupported HDF5 data type";
    return -1;
  }

    
  /* Get the rank (number of dims) in the space. */
  int ndims = H5Sget_simple_extent_ndims(file_space_id);

  hsize_t *dims = new hsize_t[ndims];

  /* Get the dims in the space. */
  int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

  if( ndim != ndims ) {
    error_msg_ = "Data dimensions not match.";
    return -1;
  }


  int cc = 1;

  for( int ic=0; ic<ndims; ic++ )
    cc *= dims[ic];

  int size, element_size;

  if( H5Tget_size(type_id) > H5Tget_size(mem_type_id) )
    element_size = H5Tget_size(type_id);
  else
    element_size = H5Tget_size(mem_type_id);

  size = cc * element_size;

  char *data = new char[size];

  if( data == NULL ) {
    error_msg_ = "Can not allocate enough memory for the data";
    return -1;
  }

  herr_t  status;

  if( type == H5G_DATASET )
    status = H5Dread(obj_id, mem_type_id,
		     H5S_ALL, H5S_ALL, H5P_DEFAULT, 
		     data);
  else
    status = H5Aread(obj_id, mem_type_id, data);

  if( status < 0 ) {
    error_msg_ = "Can not read data";
    return -1;
  }

  tab();
  *iostr_ << "DATA {" << endl;

  indent_++;

  unsigned int *counters = new unsigned int[ndims];

  for( int ic=0; ic<ndims; ic++ )
    counters[ic] = 0;


  tab();
  for( int ic=0; ic<cc; ic++ ) {
    if (mem_type_id == H5T_NATIVE_INT)
      *iostr_ << ((int*) data)[ic];
    else if (mem_type_id == H5T_NATIVE_FLOAT)
      *iostr_ << ((float*) data)[ic];
    else if (mem_type_id == H5T_NATIVE_DOUBLE)
      *iostr_ << ((double*) data)[ic];
    else if (mem_type_id == H5T_STD_REF_OBJ) {

      // Find out the type of the object the reference points to.
      hid_t ref_type =
	H5Rget_obj_type(obj_id, H5R_OBJECT, data+ic*element_size);
      // Dereference the reference points to.
      hid_t ref_obj_id =
	H5Rdereference(obj_id, H5R_OBJECT, data+ic*element_size);
      // Get the object info.
      H5G_stat_t  sb;
      H5Gget_objinfo(ref_obj_id, ".", 0, &sb);
      // Get the object name.
      char name[256];
      name[0] ='\0';
      H5Iget_name(ref_obj_id, name, 256);

      /* Print object type and close object */
      switch (ref_type) {
      case H5G_GROUP:
	*iostr_ << "GROUPNAME ";
	H5Gclose(ref_obj_id);
	break;
      case H5G_DATASET:
	*iostr_ << "DATASET ";
	H5Dclose(ref_obj_id);
	break;
      case H5G_TYPE:
	*iostr_ << "DATATYPE ";
	H5Tclose(ref_obj_id);
	break;
      case H5G_LINK:
	*iostr_  << "LINK ";
	break;
      }

      *iostr_ << name << " ";
//      *iostr_ << sb.objno[1] << ":" << sb.objno[0];
    } else if( H5Tget_class(type_id) == H5T_STRING ) {
      if(H5Tis_variable_str(type_id))
	*iostr_ << ((char*) data)[ic];
      // For non variable types all of the string data is read together
      // so parse it out based on the element size.
      else {
	*iostr_ << "\"";
	for( int jc=0; jc<element_size; jc++ )
	  if( !iscntrl( ((char*) data)[ic*element_size+jc] ) )
	    *iostr_ << ((char*) data)[ic*element_size+jc];
	  else
	    break;

	*iostr_ << "\"";
      }
    }

    if( cc > 1 && ic<cc-1)
      *iostr_ << ", ";

    if( ndims ) {
      counters[ndims-1]++;

      for( int jc=ndims-1; jc>0; jc-- ) {
	if( counters[jc] == dims[jc] ) {
	  counters[jc] = 0;
	  counters[jc-1]++;
	  *iostr_ << endl;
	  tab();
	}
      }
    }
  }

  *iostr_ << endl;

  indent_--;
  tab();
  *iostr_ << "}" << endl;

  H5Tclose(type_id);

  return 0;
}

} // End namespace SCITeem

#endif
