#ifndef UINTAH_HOMEBREW_DataWarehouse_H
#define UINTAH_HOMEBREW_DataWarehouse_H


#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/DataItem.h>
#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Exceptions/SchedulerException.h>
#include <iostream> // TEMPORARY
#include <string>

namespace SCICore {
    namespace Geometry {
	class Vector;
    }
}

namespace Uintah {

namespace Grid {
  class Region;
  class TypeDescription;
}

namespace Interface {

using Uintah::Grid::RefCounted;
using Uintah::Grid::DataItem;
using Uintah::Grid::Region;
using Uintah::Grid::TypeDescription;

/**************************************

CLASS
   DataWarehouse
   
   Short description...

GENERAL INFORMATION

   DataWarehouse.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   DataWarehouse

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class DataWarehouse : public RefCounted {
public:
    virtual ~DataWarehouse();

    DataWarehouseP getTop() const;

    // These need to be generalized.  Also do Handle<T>
    template<class T> void get(T& data, const std::string& name) const {
	getBroadcastData(data, name, T::getTypeDescription());
    }

    template<class T> void get(T& data, const std::string& name,
			       const Region* region) const {
	getRegionData(data, name, T::getTypeDescription(),
		      region);
    }

    template<class T> void get(T& data, const std::string& name,
			       const Region* region, int numGhostCells) const {
	getRegionData(data, name, T::getTypeDescription(),
		      region, numGhostCells);
    }

    template<class T> void allocate(T& data, const std::string& name,
				    const Region* region, int numGhostCells) {
	allocateRegionData(data, name, T::getTypeDescription(),
			   region, numGhostCells);
    }

    template<class T> void put(const T& data, const std::string& name,
			       const Region* region) {
	putRegionData(data, name, T::getTypeDescription(),
		      region);
    }

    template<class T> void put(const T& data, const std::string& name,
			       const Region* region, int numGhostCells) {
	putRegionData(data, name, T::getTypeDescription(),
		      region, numGhostCells);
    }

    template<class T> void put(const T& data, const std::string& name) {
	putBroadcastData(data, name, T::getTypeDescription());
    }

    bool exists(const std::string&, const Region*, int) {
	return true;
    }

    bool exists(const std::string&, const Region*) {
	return true;
    }

protected:
    DataWarehouse();

private:

    virtual void getBroadcastData(DataItem& di, const std::string& name,
				  const TypeDescription*) const = 0;
    virtual void getRegionData(DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*) const = 0;
    virtual void getRegionData(DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*, int numGhostCells) const = 0;
    virtual void putRegionData(const DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*) = 0;
    virtual void putRegionData(const DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*, int numGhostCells) = 0;
    virtual void allocateRegionData(DataItem& di, const std::string& name,
				    const TypeDescription*,
				    const Region*, int numGhostCells) = 0;
    virtual void putBroadcastData(const DataItem& di, const std::string& name,
				  const TypeDescription*) = 0;
    DataWarehouse(const DataWarehouse&);
    DataWarehouse& operator=(const DataWarehouse&);
};

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/03/22 00:37:17  sparker
// Added accessor for PerRegion data
//
// Revision 1.4  2000/03/17 18:45:43  dav
// fixed a few more namespace problems
//
// Revision 1.3  2000/03/16 22:08:22  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
