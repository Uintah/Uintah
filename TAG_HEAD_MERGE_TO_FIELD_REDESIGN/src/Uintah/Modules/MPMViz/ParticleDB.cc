
/*
 *  ParticleDB: particle database
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
#include <Uintah/Modules/MPMViz/ParticleDatabase.h>
#include <Component/PIDL/Object.h>
#include <Component/PIDL/URL.h>
#include <iostream>
using std::cerr;

namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using Uintah::Datatypes::PIDLObject;
using Uintah::Datatypes::PIDLObjectHandle;
using Uintah::Datatypes::PIDLObjectIPort;
using Uintah::Datatypes::PIDLObjectOPort;

class ParticleDB : public Module {
    Component::PIDL::Object db;
    PIDLObjectHandle dbhandle;
    PIDLObjectOPort* osim;
    PIDLObjectOPort* oviz;

 public:
    ParticleDB(const clString& id);
    virtual ~ParticleDB();

    virtual void execute();
};

ParticleDB::ParticleDB(const clString& id)
  : Module("ParticleDB", id, Filter)
{
    osim=scinew PIDLObjectOPort(this, "SimulationInterface",
			     PIDLObjectIPort::Atomic);
    add_oport(osim);
    oviz=scinew PIDLObjectOPort(this, "VisualizationInterface",
			     PIDLObjectIPort::Atomic);
    add_oport(oviz);
}

ParticleDB::~ParticleDB()
{
}

void ParticleDB::execute()
{
    if(!db){
	db=scinew Uintah::Modules::ParticleDatabase(this);
	dbhandle=scinew PIDLObject(db);
	cerr << "ParticleDatabase instantated at:\n";
	std::cerr << db->getURL().getString() << '\n';
    }
    osim->send(dbhandle);
    oviz->send(dbhandle);
}

extern "C" Module* make_ParticleDB( const clString& id )
{
  return scinew ParticleDB( id );
}

} // End namespace Modules
} // End namespace Uintah


//
// $Log$
// Revision 1.4  2000/08/09 03:18:06  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.3  2000/03/17 09:30:11  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  1999/10/15 20:23:00  sparker
// Mostly working
//
// Revision 1.1  1999/10/07 02:08:27  sparker
// use standard iostreams and complex type
//
//
