/*
 *  GeomPortTest.cc:  Testing advanced Geometry port operations
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geom/GeomGroup.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/GeometryComm.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class GeomPortTest : public Module {
    virtual void do_execute();
    GeometryOPort* out;
    void process_event();
    int busy;
    int portid;
public:
    GeomPortTest(const clString& id);
    virtual ~GeomPortTest();
    virtual void execute();
};

extern "C" Module* make_GeomPortTest(const clString& id)
{
    return new GeomPortTest(id);
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

GeomPortTest::~GeomPortTest()
{
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  2000/03/17 09:26:48  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
//
