/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <SCICore/Exceptions/InternalError.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Exceptions/UnknownVariable.h>
#include <SCICore/Thread/Guard.h>
#include <iostream>

using namespace Uintah;

using SCICore::Exceptions::InternalError;
using SCICore::Thread::Guard;
using std::cerr;

OnDemandDataWarehouse::OnDemandDataWarehouse( int MpiRank, int MpiProcesses )
  : d_lock("DataWarehouse lock"), DataWarehouse( MpiRank, MpiProcesses )
{
  d_allowCreation = true;
}

void OnDemandDataWarehouse::setGrid(const GridP& grid)
{
  this->grid=grid;
}

OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
    for(dbType::iterator iter = d_data.begin(); iter != d_data.end(); iter++){
	delete iter->second->di;
    }
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

void OnDemandDataWarehouse::allocate(int numParticles, ParticleVariableBase&,
				     const VarLabel*, const Region*) const
{
   cerr << "OnDemend DataWarehouse::allocate not finished\n";
}

void OnDemandDataWarehouse::allocate(ReductionVariableBase&, const VarLabel*) const
{
   cerr << "OnDemend DataWarehouse::allocate not finished\n";
}

void OnDemandDataWarehouse::put(const ReductionVariableBase&, const VarLabel*)
{
   cerr << "OnDemend DataWarehouse::put not finished\n";
}

void OnDemandDataWarehouse::get(ParticleVariableBase&, const VarLabel*,
				int matlIndex, const Region*, int numGhostCells) const
{
   cerr << "OnDemend DataWarehouse::get not finished\n";
}

void OnDemandDataWarehouse::allocate(ParticleVariableBase&, const VarLabel*,
				     int matlIndex, const Region*) const
{
   cerr << "OnDemend DataWarehouse::allocate not finished\n";
}

void OnDemandDataWarehouse::put(const ParticleVariableBase&, const VarLabel*,
				int matlIndex, const Region*)
{
   cerr << "OnDemend DataWarehouse::put not finished\n";
}

void OnDemandDataWarehouse::get(NCVariableBase&, const VarLabel*,
				int matlIndex, const Region*, int numGhostCells) const
{
   cerr << "OnDemend DataWarehouse::get not finished\n";
}

void OnDemandDataWarehouse::allocate(NCVariableBase&, const VarLabel*,
				     int matlIndex, const Region*) const
{
   cerr << "OnDemend DataWarehouse::allocate not finished\n";
}

void OnDemandDataWarehouse::put(const NCVariableBase&, const VarLabel*,
				int matlIndex, const Region*)
{
   cerr << "OnDemend DataWarehouse::put not finished\n";
}


#if 0
void
OnDemandDataWarehouse::getBroadcastData(DataItem& result,
					const std::string& name,
					const TypeDescription* td) const
{
    /* REFERENCED */
    //    Guard locker(&lock, Guard::Read);
    dbType::const_iterator iter = d_data.find(name);
    if(iter == d_data.end())
	throw UnknownVariable("Variable not found: "+name);
    DataRecord* dr = iter->second;
    if(dr->region != 0)
	throw InternalError("Region not allowed here");
    if(dr->td != td)
	throw TypeMismatchException("Type mismatch");
    dr->di->get(result);
}

void
OnDemandDataWarehouse::getRegionData(DataItem& result, 
				     const std::string& name,
				     const TypeDescription* td,
				     const Region* region,
				     int numGhostCells) const
{
    if(numGhostCells != 0)
	throw InternalError("ghostcells not implemented");
    /* REFERENCED */
    //    Guard locker(&lock, Guard::Read);
    dbType::const_iterator iter = d_data.find(name);
    if(iter == d_data.end())
	throw UnknownVariable("Variable not found: "+name);
    DataRecord* dr = iter->second;
    if(dr->region != region)
	throw InternalError("Mixed regions not implemented");
    if(dr->td != td)
	throw TypeMismatchException("Type mismatch");
    dr->di->get(result);
}

void
OnDemandDataWarehouse::getRegionData(DataItem& result, 
				     const std::string& name,
				     const TypeDescription* td,
				     const Region* region) const
{
    /* REFERENCED */
    //    Guard locker(&lock, Guard::Read);
    dbType::const_iterator iter = d_data.find(name);
    if(iter == d_data.end())
	throw UnknownVariable("Variable not found: "+name);
    DataRecord* dr = iter->second;
    if(dr->region != region)
	throw InternalError("Mixed regions not implemented");
    if(dr->td != td)
	throw TypeMismatchException("Type mismatch");
    dr->di->get(result);
}

void
OnDemandDataWarehouse::putRegionData(const DataItem& result, 
				     const std::string& name,
				     const TypeDescription* td,
				     const Region* region,
				     int numGhostCells)
{
    if(numGhostCells != 0)
	throw InternalError("ghostcells not implemented");
    /* REFERENCED */
    //Guard locker(&lock, Guard::Write);
    dbType::iterator iter = d_data.find(name);
    if(iter == d_data.end()){
	if(d_allowCreation){
	    //cerr << "Creating variable: " << name << '\n';
	    d_data[name]=new DataRecord(result.clone(), td, region);
	}
	iter = d_data.find(name);
    }
    DataRecord* dr = iter->second;
    if(dr->region != region)
	throw InternalError("Mixed regions not implemented");
    if(dr->td != td)
	throw TypeMismatchException("Type mismatch");
    result.get(*dr->di);
}

void
OnDemandDataWarehouse::putRegionData(const DataItem& result, 
				     const std::string& name,
				     const TypeDescription* td,
				     const Region* region)
{
    /* REFERENCED */
    //Guard locker(&lock, Guard::Write);
    dbType::iterator iter = d_data.find(name);
    if(iter == d_data.end()){
	if(d_allowCreation){
	    //cerr << "Creating variable: " << name << '\n';
	    d_data[name]=new DataRecord(result.clone(), td, region);
	}
	iter = d_data.find(name);
    }
    DataRecord* dr = iter->second;
    if(dr->region != region)
	throw InternalError("Mixed regions not implemented");
    if(dr->td != td)
	throw TypeMismatchException("Type mismatch");
    result.get(*dr->di);
}

void OnDemandDataWarehouse::allocateRegionData(DataItem& result, 
					       const std::string& name,
					       const TypeDescription* td,
					       const Region* region,
					       int numGhostCells)
{
    if(numGhostCells != 0)
	throw InternalError("ghostcells not implemented");
    /* REFERENCED */
    //    Guard locker(&lock, Guard::Write);
    //    lock.writeLock();
    dbType::iterator iter = d_data.find(name);
    if(iter == d_data.end()){
	//cerr << "Creating variable: " << name << '\n';
	DataItem* di = result.clone();
	d_data[name]=new DataRecord(di, td, region);
	di->allocate(region);
	di->get(result);
    } else {
	DataRecord* dr = iter->second;
	if(dr->region != region)
	    throw InternalError("Multi regions not implemented");
	if(dr->td != td)
	    throw TypeMismatchException("Type mismatch");
	dr->di->get(result);
    }
    //    lock.writeUnlock();
}

void OnDemandDataWarehouse::putBroadcastData(const DataItem& result, 
					     const std::string& name,
					     const TypeDescription* td)
{
    /* REFERENCED */
    //Guard locker(&lock, Guard::Write);
    dbType::iterator iter = d_data.find(name);
    if(iter == d_data.end()){
	if(d_allowCreation){
	    //cerr << "Creating variable: " << name << '\n';
	    d_data[name]=new DataRecord(result.clone(), td, 0);
	}
	iter = d_data.find(name);
    }
    DataRecord* dr = iter->second;
    if(dr->region != 0)
	throw InternalError("Have a region for broadcast data?");
    if(dr->td != td)
	throw TypeMismatchException("Type mismatch");
    result.get(*dr->di);
}
#endif

OnDemandDataWarehouse::DataRecord::DataRecord(DataItem* di,
					      const TypeDescription* td,
					      const Region* region)
    : di(di), td(td), region(region)
{
}

//
// $Log$
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
