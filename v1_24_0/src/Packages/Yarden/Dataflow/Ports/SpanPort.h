/*
 *  SpanPort.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_SpanPort_h
#define SCI_project_SpanPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Packages/Yarden/Core/Datatypes/SpanSpace.h>

namespace Yarden {
    
using namespace SCIRun;    

    typedef SimpleIPort<SpanUniverseHandle> SpanUniverseIPort;
    typedef SimpleOPort<SpanUniverseHandle> SpanUniverseOPort;
    
} // End namespace Yarden

#endif
