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

// ITKImage.h 
//
//  Written by:
//   Darby J Brown
//   School of Computing
//   University of Utah
//   January 2003
//
//  Copyright (C) 2001 SCI Institute

#ifndef SCI_Insight_ITKImage_h
#define SCI_Insight_ITKImage_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>

#include "itkCastImageFilter.h"
#include "itkPNGImageIO.h"

namespace Insight {

using namespace SCIRun;

class ITKImage;
typedef LockingHandle<ITKImage> ITKImageHandle;
 
typedef itk::Image<unsigned char, 2>   ShortImageType; 
typedef itk::Image<float, 2>            ImageType; 

class ITKImage : public Datatype {
public:  
  string fname;
  itk::CastImageFilter<ShortImageType, ImageType>::Pointer to_float_;
  itk::CastImageFilter<ImageType, ShortImageType>::Pointer to_short_;

  ITKImage();
  ITKImage(const ITKImage&);
  ~ITKImage();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};
} // end namespace Insight

#endif
