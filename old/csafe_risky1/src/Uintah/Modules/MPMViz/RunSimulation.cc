
/*
 *  RunSimulation: Run a particle simulation
 *  $Id$
 *
 *  Written 
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <PSECore/Dataflow/Module.h>
#include <Uintah/Datatypes/Particles/PIDLObject.h>
#include <Uintah/Datatypes/Particles/PIDLObjectPort.h>
#include <Component/PIDL/URL.h>
#include <Uintah/Datatypes/Particles/Particles_sidl.h>
#include <iostream>
using std::cerr;
#include <sstream>

namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using Uintah::Datatypes::PIDLObject;
using Uintah::Datatypes::PIDLObjectHandle;
using Uintah::Datatypes::PIDLObjectIPort;
using Uintah::Datatypes::Particles::Simulation;
using std::ostringstream;

class RunSimulation : public Module {
    PIDLObjectIPort* iface;
 public:
    RunSimulation(const clString& id);
    virtual ~RunSimulation();

    virtual void execute();
};

RunSimulation::RunSimulation(const clString& id)
  : Module("RunSimulation", id, Filter)
{
    iface=scinew PIDLObjectIPort(this, "SimulationInterface",
			      PIDLObjectIPort::Atomic);
    add_iport(iface);
}

RunSimulation::~RunSimulation()
{
}

void RunSimulation::execute()
{
    PIDLObjectHandle h;
    if(!iface->get(h))
	return;

    if(h->getObject()){
	Simulation sobj=pidl_cast<Simulation>(h->getObject());
	if(!sobj){
	    cerr << "Wrong object type!\n";
	}
#if 0
	ostringstream cmd;
	cmd << "mpirun -np 2 ~/PSE/test1 " << sobj->getURL().getString();
	system(cmd.str().c_str());
#endif
    }
}

extern "C" Module* make_RunSimulation( const clString& id )
{
  return scinew RunSimulation( id );
}

} // End namespace Modules
} // End namespace Uintah


//
// $Log$
// Revision 1.4  2000/08/09 03:18:07  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.3  2000/03/17 09:30:12  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  1999/10/15 20:23:01  sparker
// Mostly working
//
// Revision 1.1  1999/10/07 02:08:28  sparker
// use standard iostreams and complex type
//
//
