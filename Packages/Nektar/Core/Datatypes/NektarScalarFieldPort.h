
/*
 *  Packages/NektarScalarField.h: The Packages/Nektar Scalar Field Data type
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Packages/NektarScalarFieldPort_h
#define Packages/NektarScalarFieldPort_h 

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/Nektar/Core/Datatypes/NektarScalarField.h>

namespace Nektar {
using namespace SCIRun;

    typedef SimpleIPort<Packages/NektarScalarFieldHandle> Packages/NektarScalarFieldIPort;
    typedef SimpleOPort<Packages/NektarScalarFieldHandle> Packages/NektarScalarFieldOPort;
} // End namespace Nektar
    

#endif

