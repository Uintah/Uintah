
/*
 *  GeometryPort.h: Handle to the Geometry Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_GeometryPort_h
#define SCI_project_GeometryPort_h 1

#include <Classlib/String.h>
#include <Comm/MessageBase.h>
#include <Dataflow/Port.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>
#include <Multitask/ITC.h>

typedef int GeomID;
class GeomObj;

class GeometryIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01,
    };

protected:
    friend class GeometryOPort;
public:
    GeometryIPort(Module*, const clString& name, int protocol);
    virtual ~GeometryIPort();

    virtual void reset();
    virtual void finish();
};

class GeometryOPort : public OPort {
    GeometryIPort* in;
    int portid;
    GeomID serial;
    int dirty;
    int* busy_bit;

    virtual void reset();
    virtual void finish();

    Mailbox<MessageBase*>* outbox;
public:
    GeometryOPort(Module*, const clString& name, int protocol);
    virtual ~GeometryOPort();

    GeomID addObj(GeomObj*, const clString& name);
    void delObj(GeomID);
    void delAll();
    void flushViews();

    int busy();
};

struct GeomReply {
    int portid;
    int* busy_bit;
    GeomReply();
    GeomReply(int, int*);
};

class GeometryComm : public MessageBase {
public:
    GeometryComm(Mailbox<GeomReply>*);
    GeometryComm(int, GeomID, GeomObj*, const clString&);
    GeometryComm(int, GeomID);
    GeometryComm(int);
    GeometryComm();
    virtual ~GeometryComm();

    Mailbox<GeomReply>* reply;
    int portno;
    GeomID serial;
    GeomObj* obj;
    clString name;
};

#endif /* SCI_project_GeometryPort_h */
