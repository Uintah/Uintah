
#ifndef UINTAH_HOMEBREW_DataWarehouse_H
#define UINTAH_HOMEBREW_DataWarehouse_H

#include "Handle.h"
#include "GridP.h"
#include "CCVariable.h"
#include "DataItem.h"
#include "RefCounted.h"
#include "DataWarehouseP.h"
#include "SchedulerException.h"
#include <iostream> // TEMPORARY
#include <string>
namespace SCICore {
    namespace Geometry {
	class Vector;
    }
}
class Region;
class TypeDescription;

class DataWarehouse : public RefCounted {
public:
    virtual ~DataWarehouse();

    // These need to be generalized.  Also do Handle<T>
    template<class T> void get(T& data, const std::string& name) const {
	getBroadcastData(data, name, T::getTypeDescription());
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

protected:
    DataWarehouse();

private:

    virtual void getBroadcastData(DataItem& di, const std::string& name,
				  const TypeDescription*) const = 0;
    virtual void getRegionData(DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*, int numGhostCells) const = 0;
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

#endif
