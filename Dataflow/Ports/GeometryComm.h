
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

#ifndef SCI_CommonDatatypes_GeometryComm_h
#define SCI_CommonDatatypes_GeometryComm_h 1

#include <SCICore/share/share.h>

#include <PSECore/Comm/MessageBase.h>
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <SCICore/Multitask/ITC.h>

namespace SCICore {
  namespace GeomSpace {
    class GeomObj;
  }
  namespace Multitask {
    template<class T> class AsyncReply;
  }
}

namespace PSECore {
namespace CommonDatatypes {

using PSECore::Comm::MessageBase;
using PSECore::Comm::MessageTypes;
using SCICore::Multitask::Semaphore;
using SCICore::Multitask::AsyncReply;

struct GeomReply {
    int portid;
    int* busy_bit;
    GeomReply();
    GeomReply(int, int*);
};

class SCICORESHARE GeometryComm : public MessageBase {
public:
    GeometryComm(Mailbox<GeomReply>*);
    GeometryComm(int, GeomID, GeomObj*, const clString&, CrowdMonitor* lock);
    GeometryComm(int, GeomID, int del);
    GeometryComm(MessageTypes::MessageType, int);
    GeometryComm(MessageTypes::MessageType, int, Semaphore* wait);
    GeometryComm(MessageTypes::MessageType, int portid,
		 AsyncReply<GeometryData*>* reply,
		 int which_roe, int datamask);
    GeometryComm(MessageTypes::MessageType, int portid,
		 AsyncReply<int>* reply);
    virtual ~GeometryComm();

    Mailbox<GeomReply>* reply;
    int portno;
    GeomID serial;
    GeomObj* obj;
    clString name;
    CrowdMonitor* lock;
    Semaphore* wait;
    int del;

    GeometryComm* next;

    int which_roe;
    int datamask;
    AsyncReply<GeometryData*>* datareply;
    AsyncReply<int>* nreply;
};

} // End namespace CommonDatatypes
} // End namespace PSECore

//
// $Log$
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

#endif /* SCI_CommonDatatypes_GeometryComm_h */
