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

#include <Util/NotFinished.h>
#include <Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <CommonDatatypes/MeshPort.h>
#include <CoreDatatypes/Mesh.h>
#include <CommonDatatypes/BooleanPort.h>
#include <CoreDatatypes/Boolean.h>
#include <Malloc/Allocator.h>

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
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
    MeshIterator(const MeshIterator&, int deep=0);
    virtual ~MeshIterator();
    virtual Module* clone(int deep);
    virtual void execute();
    int first;
};

Module* make_MeshIterator(const clString& id) {
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

MeshIterator::MeshIterator(const MeshIterator& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("MeshIterator::MeshIterator");
}

MeshIterator::~MeshIterator()
{
}

Module* MeshIterator::clone(int deep)
{
    return scinew MeshIterator(*this, deep);
}

void MeshIterator::execute()
{
  first=1;
  int count=0;
  while(1){
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
// Revision 1.1  1999/07/27 16:57:45  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:49  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1  1999/04/25 00:05:37  dav
// initialish commit
//
//
