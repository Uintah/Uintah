
/*
 *  ScalarFieldWriter.cc: ScalarField Writer class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/Pstreams.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarField.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

class MultiScalarFieldWriter : public Module {
    ScalarFieldIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    MultiScalarFieldWriter(const clString& id);
    MultiScalarFieldWriter(const MultiScalarFieldWriter&, int deep=0);
    virtual ~MultiScalarFieldWriter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_MultiScalarFieldWriter(const clString& id)
{
    return scinew MultiScalarFieldWriter(id);
}
}

MultiScalarFieldWriter::MultiScalarFieldWriter(const clString& id)
: Module("MultiScalarFieldWriter", id, Source), filename("filename", id, this),
  filetype("filetype", id, this)
{
    // Create the output data handle and port
    inport=scinew ScalarFieldIPort(this, "Input Data", ScalarFieldIPort::Atomic);
    add_iport(inport);
}

MultiScalarFieldWriter::MultiScalarFieldWriter(const MultiScalarFieldWriter& copy, int deep)
: Module(copy, deep), filename("filename", id, this),
  filetype("filetype", id, this)
{
    NOT_FINISHED("MultiScalarFieldWriter::MultiScalarFieldWriter");
}

MultiScalarFieldWriter::~MultiScalarFieldWriter()
{
}

Module* MultiScalarFieldWriter::clone(int deep)
{
    return scinew MultiScalarFieldWriter(*this, deep);
}

#if 0
static void watcher(double pd, void* cbdata)
{
    MultiScalarFieldWriter* writer=(MultiScalarFieldWriter*)cbdata;
    writer->update_progress(pd);
}
#endif

void MultiScalarFieldWriter::execute()
{
    ScalarFieldHandle handle;
    if(!inport->get(handle))
	return;
    clString fn(filename.get());
    if(fn == "")
	return;
    Piostream* stream;
    clString ft(filetype.get());

    char hun = '0',ten='0',one='1';

    ScalarFieldHandle  *temp_handle;
    ScalarFieldRG *RG = handle.get_rep()->getRG();

    if (!RG)
      return; // has to be a RG

    while (RG) {
      clString tmps = fn;
      temp_handle = scinew ScalarFieldHandle;
      *temp_handle = (ScalarField*)RG;
      
      tmps += hun;
      tmps += ten;
      tmps += one;

      cerr << "Trying "+tmps << "\n";

      if(ft=="Binary"){
	stream=scinew BinaryPiostream(tmps, Piostream::Write);
      } else {
	stream=scinew TextPiostream(tmps, Piostream::Write);
      }
      // Write the file
      //stream->watch_progress(watcher, (void*)this);
      Pio(*stream, *temp_handle);
      delete stream;
#ifdef NEEDAUGDATA
      RG = (ScalarFieldRG*)RG->next;
#endif
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

