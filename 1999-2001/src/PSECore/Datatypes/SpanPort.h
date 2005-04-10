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

#include <PSECore/Datatypes/SimplePort.h>
#include <PSECore/Datatypes/SpanSpace.h>

namespace PSECore {
  namespace Datatypes {
    
    using namespace SCICore::Datatypes;
    
    typedef SimpleIPort<SpanUniverseHandle> SpanUniverseIPort;
    typedef SimpleOPort<SpanUniverseHandle> SpanUniverseOPort;
    
  } // End namespace Datatypes
} // End namespace PSECore

#endif
