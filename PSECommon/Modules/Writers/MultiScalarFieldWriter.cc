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

#include <SCICore/Persistent/Pstreams.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;

class MultiScalarFieldWriter : public Module {
    ScalarFieldIPort* inport;
    TCLstring filename;
    TCLstring filetype;
public:
    MultiScalarFieldWriter(const clString& id);
    virtual ~MultiScalarFieldWriter();
    virtual void execute();
};

extern "C" Module* make_MultiScalarFieldWriter(const clString& id) {
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

MultiScalarFieldWriter::~MultiScalarFieldWriter()
{
}

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
// Revision 1.7  2000/03/17 09:27:41  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:07:12  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:48:15  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:18:01  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:15  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
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
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
