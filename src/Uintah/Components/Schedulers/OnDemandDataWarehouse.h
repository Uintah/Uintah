#ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H

#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <map>

namespace Uintah {

namespace Grid {
    class DataItem;
    class TypeDescription;
    class Region;
}

namespace Components {

using Uintah::Interface::DataWarehouse;
using Uintah::Grid::DataItem;
using Uintah::Grid::TypeDescription;
using Uintah::Grid::Region;

/**************************************

CLASS
   OnDemandDataWarehouse
   
   Short description...

GENERAL INFORMATION

   OnDemandDataWarehouse.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   On_Demand_Data_Warehouse

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class OnDemandDataWarehouse : public DataWarehouse {
public:
    OnDemandDataWarehouse();
    virtual ~OnDemandDataWarehouse();

    //////////
    // Insert Documentation Here:
    virtual void getBroadcastData(DataItem& di, const std::string& name,
				  const TypeDescription*) const;

    //////////
    // Insert Documentation Here:
    virtual void getRegionData(DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*, int numGhostCells) const;

    //////////
    // Insert Documentation Here:
    virtual void getRegionData(DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*) const;

    //////////
    // Insert Documentation Here:
    virtual void putRegionData(const DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*, int numGhostCells);

    //////////
    // Insert Documentation Here:
    virtual void putRegionData(const DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*);

    //////////
    // Insert Documentation Here:
    virtual void putBroadcastData(const DataItem& di, const std::string& name,
				  const TypeDescription*);

    //////////
    // Insert Documentation Here:
    virtual void allocateRegionData(DataItem& di, const std::string& name,
				    const TypeDescription*,
				    const Region*, int numGhostCells);

private:
    struct DataRecord {
	DataItem* di;
	const TypeDescription* td;
	const Region* region;
	DataRecord(DataItem* di, const TypeDescription* td,
		   const Region* region);
    };
    typedef std::map<std::string, DataRecord*> dbType;


    //////////
    // Insert Documentation Here:
    mutable SCICore::Thread::CrowdMonitor  d_lock;
    dbType                                 d_data;
    bool                                   d_allowCreation;
};

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.4  2000/03/22 00:36:37  sparker
// Added new version of getRegionData
//
// Revision 1.3  2000/03/17 01:03:17  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//

#endif
