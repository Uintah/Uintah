
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

#include <Dataflow/share/share.h>

#include <Dataflow/Comm/MessageBase.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/View.h>
#include <Core/Thread/FutureValue.h>

namespace SCIRun {
  class GeomObj;
  class Semaphore;


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
		 int which_viewwindow, int datamask);
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

    int which_viewwindow;
    int datamask;
    FutureValue<GeometryData*>* datareply;
    FutureValue<int>* nreply;
};

} // End namespace SCIRun


#endif /* SCI_Datatypes_GeometryComm_h */
