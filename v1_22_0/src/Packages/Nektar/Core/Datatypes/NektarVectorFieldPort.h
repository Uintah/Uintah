
/*
 *  Packages/NektarVectorField.h: The Packages/Nektar Vector Field Data type
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Packages/NektarVectorFieldPort_h
#define Packages/NektarVectorFieldPort_h 

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/Nektar/Core/Datatypes/NektarVectorField.h>

namespace Nektar {
using namespace SCIRun;
    
    typedef SimpleIPort<Packages/NektarVectorFieldHandle> Packages/NektarVectorFieldIPort;
    typedef SimpleOPort<Packages/NektarVectorFieldHandle> Packages/NektarVectorFieldOPort;
} // End namespace Nektar
    

#endif

