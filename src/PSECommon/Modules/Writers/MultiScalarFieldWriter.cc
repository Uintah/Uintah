//static char *id="@(#) $Id$";

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

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Persistent/Pstreams.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarFieldRG.h>
#include <SCICore/CoreDatatypes/ScalarField.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

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

Module* make_MultiScalarFieldWriter(const clString& id) {
  return new MultiScalarFieldWriter(id);
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
    using SCICore::Containers::Pio;

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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:20  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:32  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:58:05  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
