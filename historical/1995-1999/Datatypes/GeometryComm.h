
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

#include <Comm/MessageBase.h>
#include <Datatypes/GeometryPort.h>
#include <Multitask/ITC.h>
class GeomObj;
template<class T> class AsyncReply;

struct GeomReply {
    int portid;
    int* busy_bit;
    GeomReply();
    GeomReply(int, int*);
};

class GeometryComm : public MessageBase {
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

#endif /* SCI_Datatypes_GeometryComm_h */
