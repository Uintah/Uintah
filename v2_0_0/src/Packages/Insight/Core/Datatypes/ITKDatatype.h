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

// itkDatatype.h - Base Insight Datatype
//
//  Written by:
//   Darby J Brown
//   School of Computing
//   University of Utah
//   January 2003
//
//  Copyright (C) 2001 SCI Institute

#ifndef SCI_Insight_itkDatatype_h
#define SCI_Insight_itkDatatype_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include "itkObject.h"

// TEMP
#include "itkImage.h"
#include "itkRGBPixel.h"
#include "itkVector.h"

namespace Insight {

using namespace SCIRun;


class ITKDatatype : public Datatype {
public:  
  string fname;
  itk::Object::Pointer data_;

  ITKDatatype();
  ITKDatatype(const ITKDatatype&);
  ~ITKDatatype();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

typedef LockingHandle<ITKDatatype> ITKDatatypeHandle;

} // end namespace Insight

#endif
