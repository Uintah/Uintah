#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Ghost.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Malloc/Allocator.h>
#include <string>
#include <vector>

using std::vector;
using std::string;

namespace Uintah {

   class ProcessorContext;
   class VarLabel;
   class Patch;
   class TypeDescription;
   
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
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) = 0;
      };
      
      template<class T>
      class Action : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorContext*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&);
      public:
	 Action( T* ptr,
		 void (T::*pmf)(const ProcessorContext*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&) )
	    : ptr(ptr), pmf(pmf) {}
	 virtual ~Action() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorContext* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW);
	 }
      };
      template<class T, class Arg1>
      class Action1 : public ActionBase {
	 
	 T* ptr;
	 Arg1 arg1;
	 void (T::*pmf)(const ProcessorContext*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1);
      public:
	 Action1( T* ptr,
		 void (T::*pmf)(const ProcessorContext*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&,
				Arg1),
		  Arg1 arg1)
	    : ptr(ptr), pmf(pmf), arg1(arg1) {}
	 virtual ~Action1() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorContext* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1);
	 }
      };
      
      template<class T, class Arg1, class Arg2>
      class Action2 : public ActionBase {
	 
	 T* ptr;
	 Arg1 arg1;
	 Arg2 arg2;
	 void (T::*pmf)(const ProcessorContext*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1, Arg2 arg2);
      public:
	 Action2( T* ptr,
		 void (T::*pmf)(const ProcessorContext*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&,
				Arg1, Arg2),
		  Arg1 arg1, Arg2 arg2)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2) {}
	 virtual ~Action2() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorContext* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1, arg2);
	 }
      };

      template<class T, class Arg1, class Arg2, class Arg3>
      class Action3 : public ActionBase {
	 
	 T* ptr;
	 Arg1 arg1;
	 Arg2 arg2;
	 Arg3 arg3;
	 void (T::*pmf)(const ProcessorContext*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1, Arg2 arg2, Arg3 arg3);
      public:
	 Action3( T* ptr,
		 void (T::*pmf)(const ProcessorContext*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&,
				Arg1, Arg2, Arg3),
		  Arg1 arg1, Arg2 arg2, Arg3 arg3)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3) {}
	 virtual ~Action3() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorContext* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1, arg2, arg3);
	 }
      };

   public:
      template<class T>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorContext*,
			  const Patch*,
			  DataWarehouseP&,
			  DataWarehouseP&) )
	 : d_taskName( taskName ), 
	   d_patch( patch ),
	   d_action( scinew Action<T>(ptr, pmf) ),
	   d_fromDW( fromDW ),
	   d_toDW( toDW )
      {
	 d_completed = false;
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
      }
      
      template<class T>
      Task(const string&         taskName,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorContext*,
			  const Patch*,
			  DataWarehouseP&,
			  DataWarehouseP&) )
	 : d_taskName( taskName ), 
	   d_patch( 0 ),
	   d_action( scinew Action<T>(ptr, pmf) ),
	   d_fromDW( fromDW ),
	   d_toDW( toDW )
      {
	 d_completed = false;
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
      }
      
      template<class T, class Arg1>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorContext*,
			  const Patch*,
			  DataWarehouseP&,
			  DataWarehouseP&,
			  Arg1),
	   Arg1 arg1)
	 : d_taskName( taskName ), 
	   d_patch( patch ),
	   d_action( scinew Action1<T, Arg1>(ptr, pmf, arg1) ),
	   d_fromDW( fromDW ),
	   d_toDW( toDW )
      {
	 d_completed = false;
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
      }
      
      template<class T, class Arg1, class Arg2>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorContext*,
			  const Patch*,
			  DataWarehouseP&,
			  DataWarehouseP&,
			  Arg1, Arg2),
	   Arg1 arg1, Arg2 arg2)
	 : d_taskName( taskName ), 
	   d_patch( patch ),
	   d_action( scinew Action2<T, Arg1, Arg2>(ptr, pmf, arg1, arg2) ),
	   d_fromDW( fromDW ),
	   d_toDW( toDW )
      {
	 d_completed = false;
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
      }
      
      template<class T, class Arg1, class Arg2, class Arg3>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorContext*,
			  const Patch*,
			  DataWarehouseP&,
			  DataWarehouseP&,
			  Arg1, Arg2, Arg3),
	   Arg1 arg1, Arg2 arg2, Arg3 arg3)
	 : d_taskName( taskName ), 
	   d_patch( patch ),
	   d_action( scinew Action3<T, Arg1, Arg2, Arg3>(ptr, pmf, arg1, arg2, arg3) ),
	   d_fromDW( fromDW ),
	   d_toDW( toDW )
      {
	 d_completed = false;
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
      }
      
      ~Task();
      
      void usesMPI(bool state=true);
      void usesThreads(bool state);
      bool usesThreads() const {
	 return d_usesThreads;
      }
      
      //////////
      // Insert Documentation Here:
      void subpatchCapable(bool state=true);
      
      //////////
      // Insert Documentation Here:
      void requires(const DataWarehouseP& ds, const VarLabel*);
      
      //////////
      // Insert Documentation Here:
      void requires(const DataWarehouseP& ds, const VarLabel*, int matlIndex,
		    const Patch* patch, Ghost::GhostType gtype,
		    int numGhostCells = 0);
      
      //////////
      // Insert Documentation Here:
      void computes(const DataWarehouseP& ds, const VarLabel*);
      
      //////////
      // Insert Documentation Here:
      void computes(const DataWarehouseP& ds, const VarLabel*, int matlIndex,
		    const Patch* patch);

      //////////
      // Insert Documentation Here:
      void doit(const ProcessorContext* pc);
      const string& getName() const {
	 return d_taskName;
      }
      const Patch* getPatch() const {
	 return d_patch;
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
	 int		  d_matlIndex;
	 const Patch*    d_patch;
	 
	 Dependency(      Task*           task,
			  const DataWarehouseP& dw,
			  const VarLabel* d_var,
			  int matlIndex,
			  const Patch*);
	 
      }; // end struct Dependency
      
      //////////
      // Insert Documentation Here:
      void addComps(vector<Dependency*>&) const;

      //////////
      // Returns the list of variables that this Task computes.
      const vector<Dependency*>& getComputes() const;
      
      //////////
      // Insert Documentation Here:
      const vector<Dependency*>& getRequires() const;
      
   private:
      //////////
      // Insert Documentation Here:
      string              d_taskName;
      const Patch*       d_patch;
      ActionBase*         d_action;
      DataWarehouseP      d_fromDW;
      DataWarehouseP      d_toDW;
      bool                d_completed;
      vector<Dependency*> d_reqs;
      vector<Dependency*> d_comps;
      
      bool                d_usesMPI;
      bool                d_usesThreads;
      bool                d_subpatchCapable;
      
      Task(const Task&);
      Task& operator=(const Task&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.15  2000/05/30 20:19:35  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.14  2000/05/28 17:25:06  dav
// adding mpi stuff
//
// Revision 1.13  2000/05/15 19:39:50  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.12  2000/05/11 20:10:21  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.11  2000/05/10 20:03:03  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.10  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.9  2000/05/05 06:42:45  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.8  2000/04/26 06:49:00  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/20 18:56:31  sparker
// Updates to MPM
//
// Revision 1.6  2000/03/22 00:32:13  sparker
// Added Face-centered variable class
// Added Per-patch data class
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
