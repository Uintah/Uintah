
/*
 *  TYPEIterator.cc: TYPE Iterator class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1994 SCI Group
 *
 *  WARNING: The file "Iterator.cc" is never compiled by itself.  Instead
 *           TYPE is replaced with a type (using sed) and the
 *           new file is compiled.
 *  
 */

#include <SCICore/Util/NotFinished.h>
#include <PSECommon/Dataflow/Connection.h>
#include <Dataflow/Module.h>
#include <PSECommon/Datatypes/TYPEPort.h>
#include <PSECommon/Datatypes/TYPE.h>
#include <PSECommon/Datatypes/BooleanPort.h>
#include <PSECommon/Datatypes/Boolean.h>
#include <SCICore/Malloc/Allocator.h>

class TYPEIterator : public Module {
    TYPEIPort* initial_iport;
    TYPEIPort* feedback_iport;
    sciBooleanIPort* condition_iport;
    TYPEOPort* final_oport;
    TYPEOPort* feedback_oport;
public:
    TYPEIterator(const clString& id);
    virtual ~TYPEIterator();
    virtual void execute();
    int first;
};

Module* make_TYPEIterator(const clString& id) {
  return new TYPEIterator(id);
}

TYPEIterator::TYPEIterator(const clString& id)
: Module("TYPEIterator", id, Source)
{
    // Create the output data handle and port
    initial_iport=scinew TYPEIPort(this, "Initial data", TYPEIPort::Atomic);
    add_iport(initial_iport);
    feedback_iport=scinew TYPEIPort(this, "Input feedback", TYPEIPort::Atomic);
    add_iport(feedback_iport);
    condition_iport=scinew sciBooleanIPort(this, "Finish condition",
					   sciBooleanIPort::Atomic);
    add_iport(condition_iport);
    final_oport=scinew TYPEOPort(this, "Final Data", TYPEIPort::Atomic);
    add_oport(final_oport);
    feedback_oport=scinew TYPEOPort(this, "Output feedback", TYPEIPort::Atomic);
    add_oport(feedback_oport);
    first=1;
}

TYPEIterator::TYPEIterator(const TYPEIterator& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("TYPEIterator::TYPEIterator");
}

TYPEIterator::~TYPEIterator()
{
}

void TYPEIterator::execute()
{
  first=1;
  int count=0;
  while(1){
    if(first){
      TYPEHandle data;
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
      TYPEHandle data;
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

