
/*
 *  NektarVectorField.cc: The Vector Field Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Nektar/Datatypes/NektarVectorFieldPort.h>

namespace Nektar {
  namespace Datatypes {
    
    template<> clString SimpleIPort<NektarVectorFieldHandle>::port_type("NektarVectorField");
    template<> clString SimpleIPort<NektarVectorFieldHandle>::port_color("LightBlue");

  } // End namespace Datatypes
} // End namespace Nektar

