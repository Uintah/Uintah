
/*
 *  Texture3D.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ArchivePort_h
#define SCI_project_ArchivePort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include "Archive.h"

namespace PSECore {
namespace Datatypes {

using namespace Kurt::Datatypes;

typedef SimpleIPort<ArchiveHandle> ArchiveIPort;
typedef SimpleOPort<ArchiveHandle> ArchiveOPort;

} // End namespace Datatypes
} // End namespace PSECore

#endif
