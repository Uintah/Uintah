/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.


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

#include <sci_defs.h>

#ifdef HAVE_HDF5

#include "HDF5Dump.h"

namespace DataIO {

using namespace std;

static int HDF5Dump_indent = 0;

void HDF5Dump_tab( ostream* iostr ) {

  for( int i=0; i<HDF5Dump_indent; i++ )
    *iostr << "   ";
}


herr_t HDF5Dump_file(const char * fname, ostream *iostr) {

  herr_t status = 0;

  /* Open the file using default properties. */
  hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  if (file_id < 0) {
    cerr << "Unable to open file: " << fname << endl;
    status = -1;
  }

  *iostr << "HDF5 \"" << fname << "\" {" << endl;

  hid_t group_id = H5Gopen(file_id, "/");

  if (group_id < 0) {
    cerr << "Unable to open root group" << endl;
    status = 1;
  } else if( HDF5Dump_group(group_id, "/", iostr) < 0 ) {
    cerr << "Unable to dump root group" << endl;
    status = -1;
  }
  
  *iostr << "}" << endl;

  if (H5Gclose(group_id) < 0) {
    cerr << "Unable to close root group" << endl;
    status = -1;
  }

  if (H5Fclose(file_id ) < 0) {
    cerr << "Unable to close file: " << fname << endl;
    status = -1;
  }

  return status;
}

herr_t HDF5Dump_attr(hid_t group_id, const char * name, void* op_data) {

  herr_t status = 0;

  ostream* iostr = (ostream*) op_data;

  HDF5Dump_tab( iostr );
  *iostr << "ATTRIBUTE \"" << name << "\" {" << endl;
  HDF5Dump_indent++;

  hid_t attr_id = H5Aopen_name(group_id, name);

  if (attr_id < 0) {
    cerr << "Unable to open attribute \"" << name << "\"" << endl;
    status = -1;
  } else {

    /* Open the file space in the file.
    hid_t file_space_id = H5Aget_space( attr_id );

    if( file_space_id < 0 ) {
      cerr << "Unable to open file space \"" << name << "\"" << endl;
      return -1;
    } else if( HDF5Dump_dataspace(file_space_id) < 0 ) {
      cerr << "Unable to dump attribute data \"" << name << "\"" << endl;
      return -1;
    }

    H5Sclose(file_space_id);
    */
    if( HDF5Dump_data(attr_id, 0, iostr) < 0 ) {
      cerr << "Unable to dump attribute data \"" << name << "\"" << endl;
      status = -1;
    }

    H5Aclose(attr_id);
  }    

  HDF5Dump_indent--;
  HDF5Dump_tab( iostr );
  *iostr << "}" << endl;
  
  return status;
}

herr_t HDF5Dump_all(hid_t obj_id, const char * name, void* op_data) {

  herr_t status = 0;

  ostream* iostr = (ostream*) op_data;

  H5G_stat_t  statbuf;
  hid_t group_id;
  hid_t dataset_id;

  H5Gget_objinfo(obj_id, name, 0, &statbuf);
  
  switch (statbuf.type) {
  case H5G_GROUP:
    if ((group_id = H5Gopen(obj_id, name)) < 0) {
      cerr << "Unable to open group \"" << name << "\"" << endl;
      status = -1;
    } else if( HDF5Dump_group(group_id, name, iostr) < 0 ) {
      cerr << "Unable to dump group \"" << name << "\"" << endl;
      status = -1;
    } else {
      H5Gclose(group_id);
      status = 0;
    }

    break;
  case H5G_DATASET:
    if ((dataset_id = H5Dopen(obj_id, name)) < 0) {
      cerr << "Unable to open dataset \"" << name << "\"" << endl;
      status = -1;
    } else if( HDF5Dump_dataset(dataset_id, name, iostr) < 0) {
      cerr << "Unable to dump dataset \"" << name << "\"" << endl;
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

herr_t HDF5Dump_group(hid_t group_id, const char * name, ostream* iostr ) {

  herr_t status = 0;

  HDF5Dump_tab( iostr );
  *iostr << "GROUP \"" << name << "\" {" << endl;
  HDF5Dump_indent++;

  H5Aiterate(group_id, NULL, HDF5Dump_attr, iostr);
  H5Giterate(group_id, ".", NULL, HDF5Dump_all, iostr);

  HDF5Dump_indent--;
  HDF5Dump_tab( iostr );
  *iostr << "}" << endl;

  return status;
}



herr_t HDF5Dump_dataset(hid_t dataset_id, const char * name, ostream* iostr) {

  herr_t status = 0;

  HDF5Dump_tab( iostr );
  *iostr << "DATASET \"" << name << "\" {" << endl;
  HDF5Dump_indent++;

  hid_t file_space_id = H5Dget_space( dataset_id );

  /* Open the data space in the file. */
  if( HDF5Dump_datatype( dataset_id, iostr ) < 0) {
    cerr << "Unable to dump datatype \"" << name << "\"" << endl;
    status = -1;
  } else if( file_space_id < 0 ) {
    cerr << "Unable to open dataspace \"" << name << "\"" << endl;
    status = -1;
  } else if( HDF5Dump_dataspace( file_space_id, iostr ) < 0) {
    cerr << "Unable to dump dataspace \"" << name << "\"" << endl;
    status = -1;
  } else {
    H5Sclose(file_space_id);
    /*    
    if( HDF5Dump_data(dataset_id, H5G_DATASET, iostr) < 0 ) {
      cerr << "Unable to dump attribute data \"" << name << "\"" << endl;
      status = -1;
    }
    */
  }

  H5Aiterate(dataset_id, NULL, HDF5Dump_attr, iostr);


  HDF5Dump_indent--;
  HDF5Dump_tab( iostr );
  *iostr << "}" << endl;

  return status;
}


herr_t HDF5Dump_datatype(hid_t dataset_id, ostream* iostr)
{
  herr_t status = 0;

  hid_t type_id = H5Dget_type(dataset_id);

  HDF5Dump_tab( iostr );
  *iostr << "DATATYPE \"";

  switch (H5Tget_class(type_id)) {
  case H5T_INTEGER:
    *iostr << "Integer";
    break;

  case H5T_FLOAT:
    if (H5Tequal(type_id, H5T_IEEE_F32BE) ||
	H5Tequal(type_id, H5T_IEEE_F32LE) ||
	H5Tequal(type_id, H5T_NATIVE_FLOAT)) {
      // Float
      *iostr << "Float";

    } else if (H5Tequal(type_id, H5T_IEEE_F64BE) ||
	       H5Tequal(type_id, H5T_IEEE_F64LE) ||
	       H5Tequal(type_id, H5T_NATIVE_DOUBLE) ||
	       H5Tequal(type_id, H5T_NATIVE_LDOUBLE)) {
      // Double
      *iostr << "Double";

    } else {
      *iostr << "Undefined HDF5 float.";
    }
    break;

  case H5T_STRING:
    *iostr << "String - Unsupported";
    break;

  case H5T_COMPOUND:
    *iostr << "Compound - Unsupported";
    break;
      
  default:
    *iostr << "Unsupported or unknown data type";
    break;
  }

  *iostr << "\"" << endl;

  H5Tclose(type_id);

  return status;
}


herr_t HDF5Dump_dataspace(hid_t file_space_id, ostream* iostr) {

  herr_t status = 0;

  /* Get the rank (number of dims) in the space. */
  int ndims = H5Sget_simple_extent_ndims(file_space_id);

  if (H5Sis_simple(file_space_id)) {

    if (ndims == 0) {
      /* scalar dataspace */

      HDF5Dump_tab( iostr );
      *iostr << "DATASPACE  SCALAR { ( 1 ) }" << endl;
    } else {
      /* simple dataspace */

      hsize_t *dims = new hsize_t[ndims];

      /* Get the dims in the space. */
      int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

      if( ndim != ndims ) {
	cerr << "Data dimensions not match." << endl;
	return -1;
      }

      HDF5Dump_tab( iostr );
      *iostr << "DATASPACE  SIMPLE { ( " << dims[0];

      for( int i = 1; i < ndims; i++ )
	*iostr << ", " << dims[i];
      
      *iostr << " ) }" << endl;

      delete dims;
    }
  }

  return status;
}


herr_t HDF5Dump_data(hid_t obj_id, hid_t type, ostream* iostr) {

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
    cerr << "Unable to open data " << endl;
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
      cerr << "Undefined HDF5 float" << endl;
      return -1;
    }
    break;
  default:
    cerr << "Unknown or unsupported HDF5 data type" << endl;
    return -1;
  }

    
  /* Get the rank (number of dims) in the space. */
  int ndims = H5Sget_simple_extent_ndims(file_space_id);

  hsize_t *dims = new hsize_t[ndims];

  /* Get the dims in the space. */
  int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

  if( ndim != ndims ) {
    cerr << "Data dimensions not match." << endl;
    return -1;
  }


  int cc = 1;

  for( int ic=0; ic<ndims; ic++ )
    cc *= dims[ic];

  int size;

  if( H5Tget_size(type_id) > H5Tget_size(mem_type_id) )
    size = cc * H5Tget_size(type_id);
  else
    size = cc * H5Tget_size(mem_type_id);

  void *data = new char[size+1];

  if( data == NULL ) {
    cerr << "Can not allocate enough memory for the data" << endl;
    return -1;
  }

  herr_t  status;

  if( type == H5G_DATASET )
    status = H5Dread(obj_id, mem_type_id,
		     H5S_ALL, H5S_ALL, H5P_DEFAULT, 
		     data);
  else
    status = H5Aread(obj_id, mem_type_id, data);

  ((char*) data)[size] = '\0';

  if( status < 0 ) {
    cerr << "Can not read data" << endl;
    return -1;
  }

  HDF5Dump_tab( iostr );
  *iostr << "DATA {" << endl;

  HDF5Dump_indent++;

  unsigned int *counters = new unsigned int[ndims];

  for( int ic=0; ic<ndims; ic++ )
    counters[ic] = 0;


  HDF5Dump_tab( iostr );
  for( int ic=0; ic<cc; ic++ ) {
    if (mem_type_id == H5T_NATIVE_INT)
      *iostr << ((int*) data)[ic];
    else if (mem_type_id == H5T_NATIVE_FLOAT)
      *iostr << ((float*) data)[ic];
    else if (mem_type_id == H5T_NATIVE_DOUBLE)
      *iostr << ((double*) data)[ic];
    else if( H5Tget_class(type_id) == H5T_STRING ) {
      if(H5Tis_variable_str(type_id))
	*iostr << ((char*) data)[ic];
      else
	*iostr << "\"" << (char*) data  << "\"";
    }

    if( cc > 1 && ic<cc-1)
      *iostr << ", ";

    if( ndims ) {
      counters[ndims-1]++;

      for( int jc=ndims-1; jc>0; jc-- ) {
	if( counters[jc] == dims[jc] ) {
	  counters[jc] = 0;
	  counters[jc-1]++;
	  *iostr << endl;
	  HDF5Dump_tab( iostr );
	}
      }
    }
  }

  *iostr << endl;

  HDF5Dump_indent--;

  HDF5Dump_tab( iostr );
  *iostr << "}" << endl;

  H5Tclose(type_id);

  return 0;
}

} // End namespace SCITeem

#endif
