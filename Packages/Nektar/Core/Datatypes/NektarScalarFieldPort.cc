
/*
 *  NektarScalarFieldPort.cc: The Scalar Field Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Nektar/Datatypes/NektarScalarFieldPort.h>

namespace Nektar {
  namespace Datatypes {

    template<> clString SimpleIPort<NektarScalarFieldHandle>::port_type("NektarScalarField");
    template<> clString SimpleIPort<NektarScalarFieldHandle>::port_color("Blue");
    
  } // End namespace Datatypes
} // End namespace Nektar

