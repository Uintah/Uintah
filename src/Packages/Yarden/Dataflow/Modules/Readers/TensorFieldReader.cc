/*
 *  TensorFieldReader.cc: TensorField Reader class
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Yarden/Datatypes/TensorFieldPort.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace Yarden {
  namespace Modules {

    using namespace SCICore::Datatypes;
    using namespace PSECore::Dataflow;
    using namespace PSECore::Datatypes;
    using namespace SCICore::TclInterface;
    using namespace SCICore::PersistentSpace;

    class TensorFieldReader : public Module {
      TensorFieldOPort* outport;
      TCLstring filename;
      TensorFieldHandle handle;
      clString old_filename;
    public:
      TensorFieldReader(const clString& id);
      virtual ~TensorFieldReader();
      virtual void execute();
    };
    
    extern "C" Module* make_TensorFieldReader(const clString& id) {
      return new TensorFieldReader(id);
    }
    
    TensorFieldReader::TensorFieldReader(const clString& id)
      : Module("TensorFieldReader", id, Source), filename("filename", id, this)
    {
      // Create the output data handle and port
      outport=scinew TensorFieldOPort(this, "Output Data", TensorFieldIPort::Atomic);
      add_oport(outport);
    }
    
    TensorFieldReader::~TensorFieldReader()
    {
    }
    
    void TensorFieldReader::execute()
    {
      using SCICore::Containers::Pio;
      
      clString fn(filename.get());
      if(!handle.get_rep() || fn != old_filename){
	old_filename=fn;
	Piostream* stream=auto_istream(fn);
	if(!stream){
	  error(clString("Error reading file: ")+filename.get());
	  return; // Can't open file...
	}
	// Read the file...
	cerr << "reading...";
	Pio(*stream, handle);
	cerr << "done" << endl;
	if(!handle.get_rep() || stream->error()){
	  error("Error reading TensorField from file");
	  delete stream;
	  return;
	}
	else
	  cerr << "Reader: " << handle->type_id.type << endl;
	delete stream;
      }
      outport->send(handle);
    }
    
  } // End namespace Modules
} // End namespace Yarden

//
// $Log$
// Revision 1.1.2.1  2000/10/26 23:55:04  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.1  2000/10/23 23:43:45  yarden
// initial commit
//
// Revision 1.2  2000/03/17 09:25:58  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  1999/09/01 07:19:53  dmw
// new DaveW modules
//
//
