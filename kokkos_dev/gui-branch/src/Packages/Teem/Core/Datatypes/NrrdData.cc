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

// NrrdData.cc - Interface to Gordon's Nrrd package
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#include <Teem/Core/Datatypes/NrrdData.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cout;
using std::cerr;

namespace SCITeem {

static Persistent* make_NrrdData() {
  return scinew NrrdData;
}

PersistentTypeID NrrdData::type_id("NrrdData", "Datatype", make_NrrdData);

NrrdData::NrrdData() : nrrd(0) {}

NrrdData::NrrdData(const NrrdData &copy) :
  fname(copy.fname) 
{
  nrrd = nrrdNew();
  nrrdCopy(nrrd, copy.nrrd);
}

NrrdData::~NrrdData() {
  nrrdNuke(nrrd);
}

#define NRRDDATA_VERSION 1

//////////
// PIO for NrrdData objects
void NrrdData::io(Piostream& stream) {
  /*  int version = */ stream.begin_class("NrrdData", NRRDDATA_VERSION);

  if (stream.reading()) {
    Pio(stream, fname);
    if (!(nrrdLoad(nrrd=nrrdNew(), strdup(fname.c_str())))) {
      char *err = biffGet(NRRD);
      cerr << "Error reading nrrd "<<fname<<": "<<err<<"\n";
      free(err);
      biffDone(NRRD);
      return;
    }
    fname="";
  } else { // writing
    if (fname == "") {   // if fname wasn't set up stream, just append .nrrd
      fname = stream.file_name + string(".nrrd");
    }
    Pio(stream, fname);
    if (nrrdSave(strdup(fname.c_str()), nrrd, 0)) {
      char *err = biffGet(NRRD);      
      cerr << "Error writing nrrd "<<fname<<": "<<err<<"\n";
      free(err);
      biffDone(NRRD);
      return;
    }
  }
  stream.end_class();
}
}  // end namespace SCITeem
