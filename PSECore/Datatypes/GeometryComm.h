
/*
 *  GeometryComm.h: Communication classes for Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_GeometryComm_h
#define SCI_Datatypes_GeometryComm_h 1

#include <PSECore/share/share.h>

#include <PSECore/Comm/MessageBase.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geom/View.h>

namespace SCICore {
  namespace GeomSpace {
    class GeomObj;
  }
  namespace Thread {
    template<class T> class FutureValue;
    class Semaphore;
  }
}

namespace PSECore {
namespace Datatypes {

using PSECore::Comm::MessageBase;
using PSECore::Comm::MessageTypes;
using SCICore::Thread::Semaphore;
using SCICore::Thread::FutureValue;

struct GeomReply {
    int portid;
    int* busy_bit;
    GeomReply();
    GeomReply(int, int*);
};

class PSECORESHARE GeometryComm : public MessageBase {
public:
    GeometryComm(Mailbox<GeomReply>*);
    GeometryComm(int, GeomID, GeomObj*, const clString&, CrowdMonitor* lock);
    GeometryComm(int, GeomID, int del);
    GeometryComm(MessageTypes::MessageType, int);
    GeometryComm(MessageTypes::MessageType, int, Semaphore* wait);
    GeometryComm(MessageTypes::MessageType, int, int, View);
    GeometryComm(MessageTypes::MessageType, int portid,
		 FutureValue<GeometryData*>* reply,
		 int which_roe, int datamask);
    GeometryComm(MessageTypes::MessageType, int portid,
		 FutureValue<int>* reply);
    virtual ~GeometryComm();

    Mailbox<GeomReply>* reply;
    int portno;
    GeomID serial;
    GeomObj* obj;
    clString name;
    CrowdMonitor* lock;
    Semaphore* wait;
    int del;
    View view;

    GeometryComm* next;

    int which_roe;
    int datamask;
    FutureValue<GeometryData*>* datareply;
    FutureValue<int>* nreply;
};

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.6  1999/12/03 00:36:08  dmw
// more files for the setView message
//
// Revision 1.5  1999/08/28 17:54:31  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/27 00:03:02  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.3  1999/08/25 03:48:20  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:00  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif /* SCI_Datatypes_GeometryComm_h */
