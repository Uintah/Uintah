/*
 *  ICNektarReader.cc: IC Nektar Reader class
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 200 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <Nektar/Datatypes/NektarScalarFieldPort.h>
#include <Nektar/Datatypes/NektarVectorFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace Nektar {
  namespace Modules {
    
    using namespace Nektar::Datatypes;
    using namespace PSECore::Dataflow;
    using namespace SCICore::TclInterface;
    using namespace SCICore::PersistentSpace;

    class ICNektarReader : public Module {
      NektarScalarFieldOPort* osport;
      NektarVectorFieldOPort* ovport;
      TCLstring filename;
      NektarScalarFieldHandle scalar_handle;
      NektarVectorFieldHandle vector_handle;
      clString old_filename;
    public:
      ICNektarReader(const clString& id);
      virtual ~ICNektarReader();
      virtual void execute();
    };
    
    extern "C" Module* make_ICNektarReader(const clString& id) {
      return new ICNektarReader(id);
    }
    
    ICNektarReader::ICNektarReader(const clString& id)
      : Module("ICNektarReader", id, Source), 
      filename("filename", id, this)
    {
      // Create the output data handle and port
      osport =scinew NektarScalarFieldOPort(this, 
				    "NektarScalarField", 
				    NektarScalarFieldIPort::Atomic);
      add_oport(osport);

      ovport=scinew NektarVectorFieldOPort(this, 
					    "NektarVectorField", 
					    NektarVectorFieldIPort::Atomic);
      add_oport(ovport);
    }
    
    ICNektarReader::~ICNektarReader()
    {
    }
    

    void ICNektarReader::execute()
    {
      using SCICore::Containers::Pio;
      
      clString fn(filename.get());
      if( fn != old_filename){
	old_filename=fn;
      }
    osport->send(scalar_handle);
    ovport->send(vector_handle);
    }
    
  } // End namespace Modules
} // End namespace PSECommon

