/*
 *  ReferenceGeom.cc: IC Nektar Reader class
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

    class ReferenceGeom : public Module {
      NektarScalarFieldOPort* osport;
      NektarVectorFieldOPort* ovport;
      TCLstring filename;
      NektarScalarFieldHandle scalar_handle;
      NektarVectorFieldHandle vector_handle;
      clString old_filename;
    public:
      ReferenceGeom(const clString& id);
      virtual ~ReferenceGeom();
      virtual void execute();
    };
    
    extern "C" Module* make_ReferenceGeom(const clString& id) {
      return new ReferenceGeom(id);
    }
    
    ReferenceGeom::ReferenceGeom(const clString& id)
      : Module("ReferenceGeom", id, Source), 
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
    
    ReferenceGeom::~ReferenceGeom()
    {
    }
    

    void ReferenceGeom::execute()
    {
      clString fn(filename.get());
      if( fn != old_filename){
	old_filename=fn;
	// read file
      }
    osport->send(scalar_handle);
    ovport->send(vector_handle);
    }
    
  } // End namespace Modules
} // End namespace PSECommon

