// NrrdData.cc - Interface to Gordon's Nrrd package
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#include <Nrrd/Core/Datatypes/NrrdData.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cout;
using std::cerr;

namespace SCINrrd {

static Persistent* make_NrrdData() {
  return scinew NrrdData;
}

PersistentTypeID NrrdData::type_id("NrrdData", "Datatype", make_NrrdData);

NrrdData::NrrdData() : nrrd(0) {}

NrrdData::NrrdData(const NrrdData &copy) :
  fname(copy.fname) 
{
  nrrd = nrrdNewCopy(copy.nrrd);
}

NrrdData::~NrrdData() {
  nrrdNuke(nrrd);
}

#define NRRDDATA_VERSION 1

//////////
// PIO for NeumannBC objects
void NrrdData::io(Piostream& stream) {
  /*  int version = */ stream.begin_class("NrrdData", NRRDDATA_VERSION);

  FILE *f;
  if (stream.reading()) {
    Pio(stream, fname);
    if (!(f = fopen(fname(), "rt"))) {
      cerr << "Error opening file "<<fname<<"\n";
      return;
    }
    if (!(nrrd=nrrdNewRead(f))) {
      char *err = biffGet(NRRD);
      cerr << "Error reading nrrd "<<fname<<": "<<err<<"\n";
      free(err);
      biffDone(NRRD);
      fclose(f);
      return;
    }
    fclose(f);
    fname="";
  } else { // writing
    if (fname == "") {   // if fname wasn't set up stream, just appdend .nrrd
      if (stream.file_name()) {
	fname = stream.file_name() + clString(".nrrd");
      }
    }
    Pio(stream, fname);
    if (!(f = fopen(fname(), "wt"))) {
      cerr << "Error opening file "<<fname<<"\n";
      return;
    }
    if (nrrdWrite(f, nrrd)) {
      char *err = biffGet(NRRD);      
      cerr << "Error writing nrrd "<<fname<<": "<<err<<"\n";
      free(err);
      biffDone(NRRD);
      fclose(f);
      return;
    }
    fclose(f);
  }
  stream.end_class();
}

}  // end namespace SCINrrd
