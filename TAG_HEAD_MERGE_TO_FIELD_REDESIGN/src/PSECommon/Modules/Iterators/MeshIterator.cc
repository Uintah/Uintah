//static char *id="@(#) $Id$";

/*
 *  MeshIterator.cc: Mesh Iterator class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Dataflow/Connection.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <PSECore/Datatypes/BooleanPort.h>
#include <SCICore/Datatypes/Boolean.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class MeshIterator : public Module {
    MeshIPort* initial_iport;
    MeshIPort* feedback_iport;
    sciBooleanIPort* condition_iport;
    MeshOPort* final_oport;
    MeshOPort* feedback_oport;
public:
    MeshIterator(const clString& id);
    virtual ~MeshIterator();
    virtual void execute();
    int first;
};

extern "C" Module* make_MeshIterator(const clString& id) {
   return new MeshIterator(id);
}

MeshIterator::MeshIterator(const clString& id)
: Module("MeshIterator", id, Source)
{
    // Create the output data handle and port
    initial_iport=scinew MeshIPort(this, "Initial data", MeshIPort::Atomic);
    add_iport(initial_iport);
    feedback_iport=scinew MeshIPort(this, "Input feedback", MeshIPort::Atomic);
    add_iport(feedback_iport);
    condition_iport=scinew sciBooleanIPort(this, "Finish condition",
					   sciBooleanIPort::Atomic);
    add_iport(condition_iport);
    final_oport=scinew MeshOPort(this, "Final Data", MeshIPort::Atomic);
    add_oport(final_oport);
    feedback_oport=scinew MeshOPort(this, "Output feedback", MeshIPort::Atomic);
    add_oport(feedback_oport);
    first=1;
}

MeshIterator::~MeshIterator()
{
}

void MeshIterator::execute()
{
  first=1;
  int count=0;
  for(;;){
    if(first){
      MeshHandle data;
      if(!initial_iport->get(data))
	return;
      first=0;
      feedback_oport->send_intermediate(data);
    } else {
      // Get the condition...
      count++;
      sciBooleanHandle cond;
      if(!condition_iport->get(cond))
	return;
      MeshHandle data;
      if(!feedback_iport->get(data))
	return;
      cerr << "Value is " << cond->value << endl;
      if(count > 6 || cond->value){
	// done...
	final_oport->send(data);
	feedback_oport->send(data);

	// One more get...
	if(!condition_iport->get(cond))
	  return;
	if(!feedback_iport->get(data))
	  return;
	
	first=1;
	return;
      } else {
	feedback_oport->send_intermediate(data);
      }
    }
  }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  2000/03/17 09:27:04  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:06:50  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:50  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:47  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:45  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:30  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:45  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:49  dav
// updates in Modules for Datatypes
//
// Revision 1.1  1999/04/25 00:05:37  dav
// initialish commit
//
//
