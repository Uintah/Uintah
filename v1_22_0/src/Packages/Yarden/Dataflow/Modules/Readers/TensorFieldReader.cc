/*
 *  TensorFieldReader.cc: TensorField Reader class
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Yarden/Dataflow/Ports/TensorFieldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>

namespace Yarden {
using namespace SCIRun;

    class TensorFieldReader : public Module {
      TensorFieldOPort* outport;
      GuiString filename;
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
} // End namespace Yarden
    

