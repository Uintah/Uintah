
/*
 *  NektarScalarField.h: The Nektar Scalar Field Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef NektarScalarFieldPort_h
#define NektarScalarFieldPort_h 

#include <PSECore/Datatypes/SimplePort.h>
#include <Nektar/Datatypes/NektarScalarField.h>

namespace Nektar {
  namespace Datatypes {
    
    using namespace SCICore::Datatypes;
    using SCICore::Datatypes::ScalarFieldHandle;
    
    typedef SimpleIPort<NektarScalarFieldHandle> NektarScalarFieldIPort;
    typedef SimpleOPort<NertarScalarFieldHandle> NektarScalarFieldOPort;
    
  } // End namespace Datatypes
} // End namespace PSECore

