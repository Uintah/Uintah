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

// ITKImage.cc 
//
//  Written by:
//   Darby J Brown
//   School of Computing
//   University of Utah
//   January 2003
//
//  Copyright (C) 2001 SCI Institute

#include <Insight/Core/Datatypes/ITKImage.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;

namespace Insight {

static Persistent* make_ITKImage() {
  return scinew ITKImage;
}

PersistentTypeID ITKImage::type_id("ITKImage", "Datatype", make_ITKImage);

ITKImage::ITKImage() {
  to_float_ = itk::CastImageFilter<ShortImageType, ImageType>::New();
  to_short_ = itk::CastImageFilter<ImageType, ShortImageType>::New();
}
  
ITKImage::ITKImage(const ITKImage &copy) :
  fname(copy.fname) {
  to_float_ = itk::CastImageFilter<ShortImageType, ImageType>::New();
  to_short_ = itk::CastImageFilter<ImageType, ShortImageType>::New();
  cerr << "*** ITKImage Copy Constructor needs to be FINISHED!\n";
}

ITKImage::~ITKImage() {
}


#define ITKIMAGE_VERSION 1

//////////
// PIO for ITKImage objects
void ITKImage::io(Piostream& stream) {
  /*  int version = */ stream.begin_class("ITKImage", ITKIMAGE_VERSION);
  /*
  if (stream.reading()) {
    Pio(stream, fname);
    //if (!(nrrdLoad(nrrd=nrrdNew(), strdup(fname.c_str())))) {
    //char *err = biffGet(NRRD);
    //cerr << "Error reading nrrd "<<fname<<": "<<err<<"\n";
    //free(err);
    //biffDone(NRRD);
    //return;
    //}
    fname="";
  } else { // writing
    if (fname == "") {   // if fname wasn't set up stream, just append .nrrd
      fname = stream.file_name + string(".itkimage");
    }
    Pio(stream, fname);
    //if (nrrdSave(strdup(fname.c_str()), nrrd, 0)) {
    //char *err = biffGet(NRRD);      
    //cerr << "Error writing nrrd "<<fname<<": "<<err<<"\n";
    //free(err);
    //biffDone(NRRD);
    //return;
    //}
  }
  stream.end_class();
  */
}
}  // end namespace SCITeem
