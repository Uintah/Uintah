
/*
 *  NektarVectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef NektarVectorFieldPort_h
#define NektarVectorFieldPort_h 

#include <PSECore/Datatypes/SimplePort.h>
#include <Nektar/Datatypes/NektarVectorField.h>

namespace Nektar {
  namespace Datatypes {

    using namespace SCICore::Datatypes;
    using Nektar::Datatypes::NektarVectorFieldHandle;

    typedef SimpleIPort<NektarVectorFieldHandle> NektarVectorFieldIPort;
    typedef SimpleOPort<NektarVectorFieldHandle> NektarVectorFieldOPort;

} // End namespace Datatypes
} // End namespace Nektar

#endif
