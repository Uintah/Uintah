/*
 *  TensorFieldWriter.cc: TensorField Writer class
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Yarden/Dataflow/Ports/TensorFieldPort.h>

namespace Yarden {
using namespace SCIRun;
    //using namespace Dataflow::Datatypes;
    
    class TensorFieldWriter : public Module {
      TensorFieldIPort* inport;
      GuiString filename;
      GuiString filetype;
      GuiInt    split;
    public:
      TensorFieldWriter(const clString& get_id());
      virtual ~TensorFieldWriter();
      virtual void execute();
    };
    
    extern "C" Module* make_TensorFieldWriter(const clString& get_id()) {
      return new TensorFieldWriter(get_id());
    }
    
    TensorFieldWriter::TensorFieldWriter(const clString& get_id())
      : Module("TensorFieldWriter", get_id(), Source), filename("filename", get_id(), this),
	filetype("filetype", get_id(), this),
	split("split", get_id(), this)
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
} // End namespace Yarden
    

