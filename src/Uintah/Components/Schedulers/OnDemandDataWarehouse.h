
#ifndef UINTAH_HOMEBREW_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_HOMEBREW_ONDEMANDDATAWAREHOUSE_H

#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <map>

class OnDemandDataWarehouse : public DataWarehouse {
public:
    OnDemandDataWarehouse();
    virtual ~OnDemandDataWarehouse();
    virtual void getBroadcastData(DataItem& di, const std::string& name,
				  const TypeDescription*) const;
    virtual void getRegionData(DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*, int numGhostCells) const;
    virtual void putRegionData(const DataItem& di, const std::string& name,
			       const TypeDescription*,
			       const Region*, int numGhostCells);
    virtual void putBroadcastData(const DataItem& di, const std::string& name,
				  const TypeDescription*);
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
    mutable SCICore::Thread::CrowdMonitor lock;
    typedef std::map<std::string, DataRecord*> dbType;
    dbType data;
    bool d_allowCreation;
};

#endif
