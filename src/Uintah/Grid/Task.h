
#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Handle.h>
#include <string>
#include <vector>

class ProcessorContext;
class Region;
class TypeDescription;

class Task {
    class ActionBase {
    public:
	virtual ~ActionBase();
 	virtual void doit(const ProcessorContext* pc,
			  const Region* region,
			  const DataWarehouseP& fromDW,
			  DataWarehouseP& toDW) = 0;
    };
    template<class T> class Action : public ActionBase {
	T* ptr;
	void (T::*pmf)(const ProcessorContext*, const Region*, const DataWarehouseP&, DataWarehouseP&);
    public:
	Action(T* ptr, void (T::*pmf)(const ProcessorContext*, const Region*, const DataWarehouseP&,
				      DataWarehouseP&))
	    : ptr(ptr), pmf(pmf) {}
	virtual ~Action() {}

 	virtual void doit(const ProcessorContext* pc,
			  const Region* region,
			  const DataWarehouseP& fromDW,
			  DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, region, fromDW, toDW);
	}
    };

public:
    template<class T>
    Task(const std::string& taskName, const Region* region,
	 const DataWarehouseP& fromDW, DataWarehouseP& toDW,
	 T* ptr, void (T::*pmf)(const ProcessorContext*, const Region*, const DataWarehouseP&,
				DataWarehouseP&))
	: taskName(taskName), region(region),
	  action(new Action<T>(ptr, pmf)),
          fromDW(fromDW), toDW(toDW) {
	      completed = false;
	      d_usesThreads = false;
	      d_usesMPI = false;
	      d_subregionCapable = false;
    }

    ~Task();

    void usesMPI(bool state=true);
    void usesThreads(bool state);
    bool usesThreads() const {
	return d_usesThreads;
    }
    void subregionCapable(bool state=true);
    void requires(const DataWarehouseP& ds, const std::string& name,
		  const TypeDescription* td);
    void requires(const DataWarehouseP& ds, const std::string& name,
		  const Region* region, int numGhostCells,
		  const TypeDescription* td);

    void computes(const DataWarehouseP& ds, const std::string& name,
		  const TypeDescription* td);
    void computes(const DataWarehouseP& ds, const std::string& name,
		  const Region* region, int numGhostCells,
		  const TypeDescription* td);

    void doit(const ProcessorContext* pc);
    const std::string& getName() const {
	return taskName;
    }
    bool isCompleted() const {
	return completed;
    }

    struct Dependency {
	Task* task;
	DataWarehouseP dw;
	std::string varname;
	const TypeDescription* vartype;

	const Region* region;
	int numGhostCells;
	Dependency(Task* task, const DataWarehouseP& dw, std::string varname,
		   const TypeDescription*, const Region*, int numGhostcells);
	bool operator<(const Dependency& c) const {
	    if(varname < c.varname) {
		return true;
	    } else if(varname == c.varname){
		if(region < c.region){
		    return true;
		} else if(region == c.region) {
		    if(dw.get_rep() < c.dw.get_rep()) {
			return true;
		    } else if(dw.get_rep() == c.dw.get_rep()){
			return false;
		    } else {
			return false;
		    }
		} else {
		    return false;
		}
	    } else {
		return false;
	    }
	}
    };

    void addComps(std::vector<Dependency*>&) const;
    void addReqs(std::vector<Dependency*>&) const;
private:
    std::string taskName;
    const Region* region;
    ActionBase* action;
    DataWarehouseP fromDW;
    DataWarehouseP toDW;
    bool completed;
    std::vector<Dependency*> reqs;
    std::vector<Dependency*> comps;

    bool d_usesMPI;
    bool d_usesThreads;
    bool d_subregionCapable;

    Task(const Task&);
    Task& operator=(const Task&);
};

#endif
