
/*
 *  CameraViewPort.cc
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Datatypes/CameraViewPort.h>

namespace PSECore {
  namespace Datatypes {

    using namespace SCICore::Containers;
    
    template<> clString SimpleIPort<CameraViewHandle>::port_type("CameraView");
    template<> clString SimpleIPort<CameraViewHandle>::port_color("chocolate1");

  } // End namespace Datatypes
} // End namespace PSECore
