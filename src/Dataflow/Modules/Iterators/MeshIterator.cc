
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

#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Dataflow/Ports/BooleanPort.h>
#include <Core/Datatypes/Boolean.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {


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

} // End namespace SCIRun

