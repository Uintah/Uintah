#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/VarLabel.h>
#include <string>
#include <vector>

using std::vector;
using std::string;

namespace Uintah {

namespace Parallel {
  class ProcessorContext;
}

namespace Grid {

class VarLabel;
class Region;
class TypeDescription;

using Uintah::Parallel::ProcessorContext;
using Uintah::Interface::DataWarehouseP;

/**************************************

CLASS
   Task
   
   Short description...

GENERAL INFORMATION

   Task.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Task

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class Task {

  class ActionBase {
  public:
    virtual ~ActionBase();
    virtual void doit(const ProcessorContext* pc,
		      const Region* region,
		      const DataWarehouseP& fromDW,
			    DataWarehouseP& toDW) = 0;
  };

  template<class T>
  class Action : public ActionBase {

    T* ptr;
    void (T::*pmf)(const ProcessorContext*,
		   const Region*,
		   const DataWarehouseP&,
		   DataWarehouseP&);
  public:
    Action( T* ptr,
	    void (T::*pmf)(const ProcessorContext*, 
			   const Region*, 
			   const DataWarehouseP&,
			         DataWarehouseP&) )
      : ptr(ptr), pmf(pmf) {}
    virtual ~Action() {}

    //////////
    // Insert Documentation Here:
    virtual void doit(const ProcessorContext* pc,
		      const Region* region,
		      const DataWarehouseP& fromDW,
			    DataWarehouseP& toDW) {
      (ptr->*pmf)(pc, region, fromDW, toDW);
    }
  };

public:
  template<class T>
  Task(const string&         taskName,
       const Region*         region,
       const DataWarehouseP& fromDW,
       DataWarehouseP&       toDW,
       T*                     ptr,
       void (T::*pmf)(const ProcessorContext*,
		      const Region*,
		      const DataWarehouseP&,
			    DataWarehouseP&) )
    : d_taskName( taskName ), 
      d_region( region ),
      d_action( new Action<T>(ptr, pmf) ),
      d_fromDW( fromDW ),
      d_toDW( toDW )
  {
    d_completed = false;
    d_usesThreads = false;
    d_usesMPI = false;
    d_subregionCapable = false;
  }

  template<class T>
  Task(const string&         taskName,
       const DataWarehouseP& fromDW,
       DataWarehouseP&       toDW,
       T*                     ptr,
       void (T::*pmf)(const ProcessorContext*,
		      const Region*,
		      const DataWarehouseP&,
			    DataWarehouseP&) )
    : d_taskName( taskName ), 
      d_region( 0 ),
      d_action( new Action<T>(ptr, pmf) ),
      d_fromDW( fromDW ),
      d_toDW( toDW )
  {
    d_completed = false;
    d_usesThreads = false;
    d_usesMPI = false;
    d_subregionCapable = false;
  }

  template<class T, class Arg1>
  Task(const string&         taskName,
       const Region*         region,
       const DataWarehouseP& fromDW,
       DataWarehouseP&       toDW,
       T*                     ptr,
       void (T::*pmf)(const ProcessorContext*,
		      const Region*,
		      const DataWarehouseP&,
			    DataWarehouseP&,
		      Arg1),
       Arg1)
    : d_taskName( taskName ), 
      d_region( region ),
      //d_action( new Action<T>(ptr, pmf) ),
      d_fromDW( fromDW ),
      d_toDW( toDW )
  {
    d_completed = false;
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

  //////////
  // Insert Documentation Here:
  void subregionCapable(bool state=true);

  //////////
  // Insert Documentation Here:
  void requires(const DataWarehouseP& ds, const VarLabel*);

  //////////
  // Insert Documentation Here:
  void requires(const DataWarehouseP& ds, const VarLabel*,
		const Region* region, int numGhostCells);

  //////////
  // Insert Documentation Here:
  void computes(const DataWarehouseP& ds, const VarLabel*);

  //////////
  // Insert Documentation Here:
  void computes(const DataWarehouseP& ds, const VarLabel*,
		const Region* region);

  //////////
  // Insert Documentation Here:
  void doit(const ProcessorContext* pc);
  const string& getName() const {
    return d_taskName;
  }

  //////////
  // Insert Documentation Here:
  bool isCompleted() const {
    return d_completed;
  }

  struct Dependency {
          Task*            d_task;
          DataWarehouseP   d_dw;
          const VarLabel*  d_var;

    const Region*          d_region;
    int                    d_numGhostCells;

    Dependency(      Task*           task,
	       const DataWarehouseP& dw,
	       const VarLabel* d_var,
	       const Region*,
		     int numGhostcells );

    bool operator<(const Dependency& c) const {
      if(d_var->getName() < c.d_var->getName()) {
		return true;
	    } else if(d_var->getName() == c.d_var->getName()){
		if(d_region < c.d_region){
		    return true;
		} else if(d_region == c.d_region) {
		    if(d_dw.get_rep() < c.d_dw.get_rep()) {
			return true;
		    } else if(d_dw.get_rep() == c.d_dw.get_rep()){
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
    }; // end struct Dependency

  //////////
  // Insert Documentation Here:
  void addComps(std::vector<Dependency*>&) const;

  //////////
  // Insert Documentation Here:
  void addReqs(std::vector<Dependency*>&) const;

private:
    //////////
    // Insert Documentation Here:
          string              d_taskName;
    const Region*             d_region;
          ActionBase*         d_action;
          DataWarehouseP      d_fromDW;
          DataWarehouseP      d_toDW;
          bool                d_completed;
          vector<Dependency*> d_reqs;
          vector<Dependency*> d_comps;

          bool                d_usesMPI;
          bool                d_usesThreads;
          bool                d_subregionCapable;

    Task(const Task&);
    Task& operator=(const Task&);
};

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/04/20 18:56:31  sparker
// Updates to MPM
//
// Revision 1.6  2000/03/22 00:32:13  sparker
// Added Face-centered variable class
// Added Per-region data class
// Added new task constructor for procedures with arguments
// Use Array3Index more often
//
// Revision 1.5  2000/03/17 18:45:42  dav
// fixed a few more namespace problems
//
// Revision 1.4  2000/03/17 09:30:00  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
