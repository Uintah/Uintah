/*
 *  TensorFieldWriter.cc: TensorField Writer class
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <Yarden/Datatypes/TensorFieldPort.h>

namespace Yarden {
  namespace Modules {
    
    using namespace SCICore::Datatypes;
    using namespace PSECore::Dataflow;
    //using namespace PSECore::Datatypes;
    using namespace SCICore::TclInterface;
    using namespace SCICore::PersistentSpace;
    
    class TensorFieldWriter : public Module {
      TensorFieldIPort* inport;
      TCLstring filename;
      TCLstring filetype;
      TCLint    split;
    public:
      TensorFieldWriter(const clString& id);
      virtual ~TensorFieldWriter();
      virtual void execute();
    };
    
    extern "C" Module* make_TensorFieldWriter(const clString& id) {
      return new TensorFieldWriter(id);
    }
    
    TensorFieldWriter::TensorFieldWriter(const clString& id)
      : Module("TensorFieldWriter", id, Source), filename("filename", id, this),
	filetype("filetype", id, this),
	split("split", id, this)
    {
      // Create the output data handle and port
      inport=scinew TensorFieldIPort(this, "Input Data", TensorFieldIPort::Atomic);
      add_iport(inport);
    }
    
    TensorFieldWriter::~TensorFieldWriter()
    {
    }
    
    
    void TensorFieldWriter::execute()
    {
      using SCICore::Containers::Pio;
      
      TensorFieldHandle handle;
      if(!inport->get(handle))
	return;
      clString fn(filename.get());
      if(fn == "")
	return;
      Piostream* stream;
      clString ft(filetype.get());
      if(ft=="Binary"){
	stream=scinew BinaryPiostream(fn, Piostream::Write);
      } else {
	stream=scinew TextPiostream(fn, Piostream::Write);
      }
      // Write the file
      //stream->watch_progress(watcher, (void*)this);
      handle->set_raw( split.get() );
      Pio(*stream, handle);
      delete stream;
    }
    
  } // End namespace Modules
} // End namespace Yarden

//
// $Log$
// Revision 1.1.2.1  2000/10/26 23:55:06  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.1  2000/10/23 23:43:46  yarden
// initial commit
//
// Revision 1.2  2000/03/17 09:26:06  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.1  1999/09/01 07:21:01  dmw
// new DaveW modules
//
//
