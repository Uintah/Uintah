/*
 *  GeomPortTest.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Geom/Sphere.h>
#include <Geom/Group.h>
#include <CommonDatatypes/GeometryPort.h>
#include <CommonDatatypes/GeometryComm.h>
#include <Malloc/Allocator.h>

class GeomPortTest : public Module {
    virtual void do_execute();
    GeometryOPort* out;
    void process_event();
    int busy;
    int portid;
public:
    GeomPortTest(const clString& id);
    GeomPortTest(const GeomPortTest&, int deep);
    virtual ~GeomPortTest();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_GeomPortTest(const clString& id)
{
    return new GeomPortTest(id);
}
}

GeomPortTest::GeomPortTest(const clString& id)
: Module("GeomPortTest", id, SalmonSpecial)
{
    add_iport(scinew GeometryIPort(this, "Geometry", GeometryIPort::Atomic));
    out=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(out);

    have_own_dispatch=1;
    busy=0;
}

GeomPortTest::GeomPortTest(const GeomPortTest& copy, int deep)
: Module(copy, deep)
{
}

GeomPortTest::~GeomPortTest()
{
}

Module* GeomPortTest::clone(int deep)
{
    return new GeomPortTest(*this, deep);
}

void GeomPortTest::do_execute()
{
    update_state(Completed);
    for(;;){
	process_event();
    }
}

void GeomPortTest::process_event()
{
    MessageBase* msg=mailbox.receive();
    GeometryComm* gmsg=(GeometryComm*)msg;
    switch(gmsg->type){
    case MessageTypes::ExecuteModule:
	// We ignore these messages...
	break;
    case MessageTypes::GeometryAddObj:
	{
	    GeomGroup* group=new GeomGroup();
	    group->add(gmsg->obj);
	    group->add(new GeomSphere(Point(0,0,0), 5));
	    gmsg->obj=group;
	    out->forward(gmsg);
	}
	break;
    case MessageTypes::GeometryDelObj:
	out->forward(gmsg);
	break;
    case MessageTypes::GeometryDelAll:
	out->forward(gmsg);
	break;
    case MessageTypes::GeometryInit:
	gmsg->reply->send(GeomReply(portid++, &busy));
	break;	
    case MessageTypes::GeometryFlush:
	out->forward(gmsg);
	break;
    case MessageTypes::GeometryFlushViews:
	out->forward(gmsg);
	break;
    case MessageTypes::GeometryGetNRoe:
	out->forward(gmsg);
	break;
    case MessageTypes::GeometryGetData:
	out->forward(gmsg);
	break;
    default:
	cerr << "GeomPortTest: Illegal Message type: " << gmsg->type << endl;
	break;
    }
}

void GeomPortTest::execute()
{
    // Never gets called...
}

