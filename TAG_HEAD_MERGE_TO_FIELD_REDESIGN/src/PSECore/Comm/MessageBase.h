
/*
 *  MessageBase.h: Base class for messages
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MessageBase_h
#define SCI_project_MessageBase_h 1

#include <PSECore/share/share.h>
#include <PSECore/Comm/MessageTypes.h>

namespace PSECore {
namespace Comm {

class PSECORESHARE MessageBase {
public:
    MessageTypes::MessageType type;
    MessageBase(MessageTypes::MessageType type);
    virtual ~MessageBase();
};

} // End namespace Comm
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/26 23:56:07  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.2  1999/08/17 06:38:05  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:45  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:16:59  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif
