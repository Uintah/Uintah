/*
 *  ICPackages/NektarReader.cc: IC Packages/Nektar Reader class
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 200 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Packages/Nektar/Core/Datatypes/NektarScalarFieldPort.h>
#include <Packages/Nektar/Core/Datatypes/NektarVectorFieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>

namespace Nektar {
    using namespace Nektar::Datatypes;
using namespace SCIRun;

    class ICPackages/NektarReader : public Module {
      Packages/NektarScalarFieldOPort* osport;
      Packages/NektarVectorFieldOPort* ovport;
      GuiString filename;
      Packages/NektarScalarFieldHandle scalar_handle;
      Packages/NektarVectorFieldHandle vector_handle;
      clString old_filename;
    public:
      ICPackages/NektarReader(const clString& id);
      virtual ~ICPackages/NektarReader();
      virtual void execute();
    };
    
    extern "C" Module* make_ICPackages/NektarReader(const clString& id) {
      return new ICPackages/NektarReader(id);
    }
    
    ICPackages/NektarReader::ICPackages/NektarReader(const clString& id)
      : Module("ICPackages/NektarReader", id, Source), 
      filename("filename", id, this)
    {
      // Create the output data handle and port
      osport =scinew Packages/NektarScalarFieldOPort(this, 
				    "Packages/NektarScalarField", 
				    Packages/NektarScalarFieldIPort::Atomic);
      add_oport(osport);

      ovport=scinew Packages/NektarVectorFieldOPort(this, 
					    "Packages/NektarVectorField", 
					    Packages/NektarVectorFieldIPort::Atomic);
      add_oport(ovport);
    }
    
    ICPackages/NektarReader::~ICPackages/NektarReader()
    {
    }
    

    void ICPackages/NektarReader::execute()
    {
      clString fn(filename.get());
      if( fn != old_filename){
	old_filename=fn;
	// read file
      }
    osport->send(scalar_handle);
    ovport->send(vector_handle);
    }
} // End namespace Nektar
    

