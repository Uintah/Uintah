
/*
 *  ScalarFieldReader.cc: ScalarField Reader class
 *
 *  Written by:
 *   Steven G. Parker - hacked for multiple scalar fields by ppsloan...
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>

class MultiSFRGReader : public Module {
  ScalarFieldOPort* outport;
  TCLstring filename;
  ScalarFieldHandle handle;
  ScalarFieldHandle base;
  clString old_filename;
public:
  MultiSFRGReader(const clString& id);
  MultiSFRGReader(const MultiSFRGReader&, int deep=0);
  virtual ~MultiSFRGReader();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
  Module* make_MultiSFRGReader(const clString& id)
    {
      return scinew MultiSFRGReader(id);
    }
};

MultiSFRGReader::MultiSFRGReader(const clString& id)
: Module("MultiSFRGReader", id, Source), filename("filename", id, this)
{
  // Create the output data handle and port
  outport=scinew ScalarFieldOPort(this, "Output Data", ScalarFieldIPort::Atomic);
  add_oport(outport);
}

MultiSFRGReader::MultiSFRGReader(const MultiSFRGReader& copy, int deep)
: Module(copy, deep), filename("filename", id, this)
{
  NOT_FINISHED("MultiSFRGReader::MultiSFRGReader");
}

MultiSFRGReader::~MultiSFRGReader()
{
}

Module* MultiSFRGReader::clone(int deep)
{
  return scinew MultiSFRGReader(*this, deep);
}

#ifdef BROKEN
static void watcher(double pd, void* cbdata)
{
  MultiSFRGReader* reader=(MultiSFRGReader*)cbdata;
  if(TCLTask::try_lock()){
    // Try the malloc lock once before we call update_progress
    // If we can't get it, then back off, since our caller might
    // have it locked
    if(!Task::test_malloc_lock()){
      TCLTask::unlock();
      return;
    }
    reader->update_progress(pd);
    TCLTask::unlock();
  }
}
#endif

void MultiSFRGReader::execute()
{
  clString fn(filename.get());
  if(!handle.get_rep() || fn != old_filename){

    // now loop through all of the files - 000 to number of frames...

    ScalarFieldHandle* crap; // for later...

    ScalarFieldRG*    last_ptr=0;

    old_filename=fn;
    clString base_name(fn);

    clString work_name;

    char hun = '0',ten = '0', one = '1';
    int ndone=0;
    int finished=0;

    while (!finished) {
      work_name = base_name;
      work_name += hun;
      work_name += ten;
      work_name += one; // this is the file for this thing...

      Piostream* stream=auto_istream(work_name);
      
      if(!stream){
	if (!ndone) {
	  cerr << "Error, couldn't open file!\n";
	  return; // Can't open file...
	}
	finished=1; // done with all of the scalar fields...
      } else {
	
	// Read the file...
	Pio(*stream, base);
	crap = new ScalarFieldHandle;
	*crap = base; // add a fake reference...
	if (!ndone)
	  handle=base; // set the root
	if(!base.get_rep() || stream->error()){
	  error("Error reading ScalarField from file");
	  delete stream;
	  return;
	}
	delete stream;
	++ndone;

	cerr << "Did " << ndone << " " << work_name << "\n";

	if (last_ptr)
	  last_ptr->next = base.get_rep()->getRG();
	last_ptr = base.get_rep()->getRG();
      }
      one = one + 1;
      if (one > '9') {
	ten = ten + 1;
	if (ten > '9') {
	  hun = hun+1; // shouldn't go over...
	  ten = '0';
	}
	one = '0';
      }
    }
  }
  outport->send(handle);
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template void Pio(Piostream&, ScalarFieldHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/LockingHandle.cc>

static void _dummy_(Piostream& p1, ScalarFieldHandle& p2)
{
  Pio(p1, p2);
}

#endif
#endif

