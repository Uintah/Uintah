/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Exceptions/DataWarehouseException.h>
#include <SCICore/Thread/Guard.h>
#include <iostream>

namespace Uintah {
namespace Components {

using Uintah::Exceptions::DataWarehouseException;
using SCICore::Thread::Guard;
using std::cerr;

OnDemandDataWarehouse::OnDemandDataWarehouse()
    : d_lock("DataWarehouse lock")
{
    d_allowCreation = true;
}

OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
    for(dbType::iterator iter = d_data.begin(); iter != d_data.end(); iter++){
	delete iter->second->di;
    }
}

void
OnDemandDataWarehouse::getBroadcastData(DataItem& result,
					const std::string& name,
					const TypeDescription* td) const
{
    /* REFERENCED */
    //    Guard locker(&lock, Guard::Read);
    dbType::const_iterator iter = d_data.find(name);
    if(iter == d_data.end())
	throw DataWarehouseException("Variable not found: "+name);
    DataRecord* dr = iter->second;
    if(dr->region != 0)
	throw DataWarehouseException("Region not allowed here");
    if(dr->td != td)
	throw DataWarehouseException("Type mismatch");
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
	throw DataWarehouseException("ghostcells not implemented");
    /* REFERENCED */
    //    Guard locker(&lock, Guard::Read);
    dbType::const_iterator iter = d_data.find(name);
    if(iter == d_data.end())
	throw DataWarehouseException("Variable not found: "+name);
    DataRecord* dr = iter->second;
    if(dr->region != region)
	throw DataWarehouseException("Mixed regions not implemented");
    if(dr->td != td)
	throw DataWarehouseException("Type mismatch");
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
	throw DataWarehouseException("ghostcells not implemented");
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
	throw DataWarehouseException("Mixed regions not implemented");
    if(dr->td != td)
	throw DataWarehouseException("Type mismatch");
    result.get(*dr->di);
}

void OnDemandDataWarehouse::allocateRegionData(DataItem& result, 
					       const std::string& name,
					       const TypeDescription* td,
					       const Region* region,
					       int numGhostCells)
{
    if(numGhostCells != 0)
	throw DataWarehouseException("ghostcells not implemented");
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
	    throw DataWarehouseException("Multi regions not implemented");
	if(dr->td != td)
	    throw DataWarehouseException("Type mismatch");
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
	throw DataWarehouseException("Have a region for broadcast data?");
    if(dr->td != td)
	throw DataWarehouseException("Type mismatch");
    result.get(*dr->di);
}

OnDemandDataWarehouse::DataRecord::DataRecord(DataItem* di,
					      const TypeDescription* td,
					      const Region* region)
    : di(di), td(td), region(region)
{
}

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/17 01:03:17  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//
