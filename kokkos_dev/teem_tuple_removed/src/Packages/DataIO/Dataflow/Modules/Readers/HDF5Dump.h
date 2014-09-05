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
*/

//    File   : HDF5Dump.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2003

#include <sci_defs.h>

#include <iostream>

#ifdef HAVE_HDF5

#include "hdf5.h"

namespace DataIO {

using namespace std;

void HDF5Dump_tab( ostream *iostr );

herr_t HDF5Dump_file(const char *fname, ostream *iostr);
herr_t HDF5Dump_attr(hid_t group_id, const char *name, void *op_data);
herr_t HDF5Dump_all(hid_t obj_id, const char *name, void *op_data);
herr_t HDF5Dump_group(hid_t group_id, const char *name, ostream *iostr );
herr_t HDF5Dump_dataset(hid_t dataset_id, const char *name, ostream *iostr);
herr_t HDF5Dump_datatype(hid_t dataset_id, ostream *iostr);
herr_t HDF5Dump_dataspace(hid_t file_space_id, ostream *iostr);
herr_t HDF5Dump_data(hid_t obj_id, hid_t type, ostream *iostr);

}

#endif
