
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

#include <Dataflow/Port.h>
#include <Multitask/ITC.h>

typedef int GeomID;
class clString;
class GeomObj;
class GeometryComm;
class MessageBase;
class Mutex;

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
    Mutex* lock;

    GeometryComm* save_msgs;
    GeometryComm* save_msgs_tail;
    void save_msg(GeometryComm*);

    virtual void reset();
    virtual void finish();

    Mailbox<MessageBase*>* outbox;
    virtual void attach(Connection*);
public:
    GeometryOPort(Module*, const clString& name, int protocol);
    virtual ~GeometryOPort();

    GeomID addObj(GeomObj*, const clString& name, CrowdMonitor* lock=0);
    void delObj(GeomID);
    void delAll();
    void flushViews();

    int busy();
};

#endif /* SCI_project_GeometryPort_h */
