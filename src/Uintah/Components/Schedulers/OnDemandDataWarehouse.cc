/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <SCICore/Exceptions/InternalError.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Exceptions/UnknownVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Thread/Guard.h>
#include <SCICore/Geometry/Point.h>
#include <iostream>

using namespace Uintah;

using SCICore::Exceptions::InternalError;
using SCICore::Thread::Guard;
using SCICore::Geometry::Point;
using std::cerr;

OnDemandDataWarehouse::OnDemandDataWarehouse( int MpiRank, int MpiProcesses )
  : d_lock("DataWarehouse lock"), DataWarehouse( MpiRank, MpiProcesses )
{
  d_allowCreation = true;
  position_label = new VarLabel("__internal datawarehouse position variable",
				ParticleVariable<Point>::getTypeDescription(),
				VarLabel::Internal);
}

void OnDemandDataWarehouse::setGrid(const GridP& grid)
{
  this->grid=grid;
}

OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
}

void OnDemandDataWarehouse::get(ReductionVariableBase&, const VarLabel*) const
{
#if 0
    dbType::const_iterator iter = d_reductions.find(name);
    if(iter == d_data.end())
	throw UnknownVariable("Variable not found: "+name);
    DataRecord* dr = iter->second;
    if(dr->region != 0)
	throw InternalError("Region not allowed here");
    if(dr->td != td)
	throw TypeMismatchException("Type mismatch");
    dr->di->get(result);
#endif
   cerr << "OnDemandDataWarehouse::get not finished\n";
}

void OnDemandDataWarehouse::allocate(ReductionVariableBase&,
				     const VarLabel*)
{
   cerr << "OnDemend DataWarehouse::allocate not finished\n";
}

void OnDemandDataWarehouse::put(const ReductionVariableBase&, const VarLabel*)
{
   cerr << "OnDemend DataWarehouse::put not finished\n";
}

void OnDemandDataWarehouse::get(ParticleVariableBase&, const VarLabel*,
				int, const Region*, int) const
{
   cerr << "OnDemend DataWarehouse::get not finished\n";
}

void OnDemandDataWarehouse::allocate(int numParticles,
				     ParticleVariableBase& var,
				     const VarLabel* label,
				     int matlIndex,
				     const Region* region)
{
   // Error checking
   if(particledb.exists(label, matlIndex, region))
      throw InternalError("Particle variable already exists: "+label->getName());
   if(!label->isPositionVariable())
      throw InternalError("Particle allocate via numParticles should only be used for position variables");
   if(particledb.exists(position_label, matlIndex, region))
      throw InternalError("Particle position already exists in datawarehouse");

   // Create the particle set and variable
   ParticleSet* pset = new ParticleSet(numParticles);
   ParticleSubset* psubset = new ParticleSubset(pset);
   ParticleVariable<Point> positions(psubset);

   // Put it in the database
   particledb.put(label, matlIndex, region, positions);
   particledb.put(position_label, matlIndex, region, positions);

   // Copy the pointer for return
   var.copyPointer(positions);
}

void OnDemandDataWarehouse::allocate(ParticleVariableBase&, const VarLabel*,
				     int, const Region*)
{
   cerr << "OnDemend DataWarehouse::allocate not finished\n";
}

void OnDemandDataWarehouse::put(const ParticleVariableBase&, const VarLabel*,
				int, const Region*)
{
   cerr << "OnDemend DataWarehouse::put not finished\n";
}

void OnDemandDataWarehouse::get(NCVariableBase&, const VarLabel*,
				int, const Region*, int) const
{
   cerr << "OnDemend DataWarehouse::get not finished\n";
}

void OnDemandDataWarehouse::allocate(NCVariableBase&, const VarLabel*,
				     int, const Region*)
{
   cerr << "OnDemend DataWarehouse::allocate not finished\n";
}

void OnDemandDataWarehouse::put(const NCVariableBase&, const VarLabel*,
				int, const Region*)
{
   cerr << "OnDemend DataWarehouse::put not finished\n";
}


//
// $Log$
// Revision 1.13  2000/04/28 07:35:34  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.12  2000/04/27 23:18:48  sparker
// Added problem initialization for MPM
//
// Revision 1.11  2000/04/26 06:48:33  sparker
// Streamlined namespaces
//
// Revision 1.10  2000/04/24 15:17:01  sparker
// Fixed unresolved symbols
//
// Revision 1.9  2000/04/20 22:58:18  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
// Revision 1.8  2000/04/20 18:56:26  sparker
// Updates to MPM
//
// Revision 1.7  2000/04/19 21:20:03  dav
// more MPI stuff
//
// Revision 1.6  2000/04/19 05:26:11  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.5  2000/04/13 06:50:57  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/04/11 07:10:40  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/17 01:03:17  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//
