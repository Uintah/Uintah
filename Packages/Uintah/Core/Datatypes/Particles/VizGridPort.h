
/*
 *  VizGridPort.h
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_project_VizGridPort_h
#define SCI_project_VizGridPort_h 1

#include <CommonDatatypes/SimplePort.h>
#include <Datatypes/Particles/VizGrid.h>

namespace Uintah {
namespace Datatypes {

using namespace PSECommon::CommonDatatypes;

typedef SimpleIPort<VizGridHandle> VizGridIPort;
typedef SimpleOPort<VizGridHandle> VizGridOPort;

} // End namespace Datatypes
} // End namespace Uintah

//

#endif
