#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Ghost.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Malloc/Allocator.h>
#include <string>
#include <vector>
#include <iostream>

using std::vector;
using std::string;
using std::ostream;

namespace Uintah {
   class TaskGraph;
   class ProcessorGroup;
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
	 virtual void doit(const ProcessorGroup* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) = 0;
      };

      template<class T>
      class NPAction : public ActionBase {

         T* ptr;
         void (T::*pmf)(const ProcessorGroup*,
                        DataWarehouseP&,
                        DataWarehouseP&);
      public: // class NPAction
         NPAction( T* ptr,
                 void (T::*pmf)(const ProcessorGroup*,
                                DataWarehouseP&,
                                DataWarehouseP&) )
            : ptr(ptr), pmf(pmf) {}
         virtual ~NPAction() {}

         //////////
         // Insert Documentation Here:
         virtual void doit(const ProcessorGroup* pc,
                           const Patch*,
                           DataWarehouseP& fromDW,
                           DataWarehouseP& toDW) {
            (ptr->*pmf)(pc, fromDW, toDW);
         }
      }; // end class ActionBase

      template<class T, class Arg1>
      class NPAction1 : public ActionBase {

         T* ptr;
	 Arg1 arg1;
         void (T::*pmf)(const ProcessorGroup*,
                        DataWarehouseP&,
                        DataWarehouseP&,
			Arg1);
      public:  // class NPAction1
         NPAction1( T* ptr,
                 void (T::*pmf)(const ProcessorGroup*,
                                DataWarehouseP&,
                                DataWarehouseP&,
				Arg1),
		   Arg1 arg1)
            : ptr(ptr), pmf(pmf), arg1(arg1) {}
         virtual ~NPAction1() {}

         //////////
         // Insert Documentation Here:
         virtual void doit(const ProcessorGroup* pc,
                           const Patch*,
                           DataWarehouseP& fromDW,
                           DataWarehouseP& toDW) {
            (ptr->*pmf)(pc, fromDW, toDW, arg1);
         }
      }; // end class NPAction1
      
      template<class T>
      class Action : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&);
      public: // class Action
	 Action( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&) )
	    : ptr(ptr), pmf(pmf) {}
	 virtual ~Action() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW);
	 }
      }; // end class Action

      template<class T, class Arg1>
      class Action1 : public ActionBase {
	 
	 T* ptr;
	 Arg1 arg1;
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1);
      public: // class Action1
	 Action1( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&,
				Arg1),
		  Arg1 arg1)
	    : ptr(ptr), pmf(pmf), arg1(arg1) {}
	 virtual ~Action1() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1);
	 }
      }; // end class Action1
      
      template<class T, class Arg1, class Arg2>
      class Action2 : public ActionBase {
	 
	 T* ptr;
	 Arg1 arg1;
	 Arg2 arg2;
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1, Arg2 arg2);
      public: // class Action2
	 Action2( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&,
				Arg1, Arg2),
		  Arg1 arg1, Arg2 arg2)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2) {}
	 virtual ~Action2() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1, arg2);
	 }
      }; // end class Action2

      template<class T, class Arg1, class Arg2, class Arg3>
      class Action3 : public ActionBase {
	 
	 T* ptr;
	 Arg1 arg1;
	 Arg2 arg2;
	 Arg3 arg3;
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1, Arg2 arg2, Arg3 arg3);
      public: // class Action3
	 Action3( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&,
				Arg1, Arg2, Arg3),
		  Arg1 arg1, Arg2 arg2, Arg3 arg3)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3) {}
	 virtual ~Action3() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1, arg2, arg3);
	 }
      }; // end Action3

   public: // class Task

      enum TaskType {
	 Normal,
	 Reduction,
	 Scatter,
	 Gather
      };

      Task(const string&         taskName)
	 : d_taskName(taskName),
	   d_patch(0),
	   d_action(0),
	   d_fromDW(0),
	   d_toDW(0)
      {
	 d_completed = false;
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
	 d_tasktype = Reduction;
      }

      template<class T>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
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
	 d_tasktype = Normal;
      }

     template<class T>
      Task(const string&         taskName,
           DataWarehouseP&       fromDW,
           DataWarehouseP&       toDW,
           T*                    ptr,
           void (T::*pmf)(const ProcessorGroup*,
                          DataWarehouseP&,
                          DataWarehouseP&) )
         : d_taskName( taskName ),
           d_patch( 0 ),
           d_action( scinew NPAction<T>(ptr, pmf) ),
           d_fromDW( fromDW ),
           d_toDW( toDW )
      {
         d_completed = false;
         d_usesThreads = false;
         d_usesMPI = false;
         d_subpatchCapable = false;
	 d_tasktype = Normal;
      }

      
     template<class T, class Arg1>
      Task(const string&         taskName,
           DataWarehouseP&       fromDW,
           DataWarehouseP&       toDW,
           T*                    ptr,
           void (T::*pmf)(const ProcessorGroup*,
                          DataWarehouseP&,
                          DataWarehouseP&,
			  Arg1),
	   Arg1 arg1)
         : d_taskName( taskName ),
           d_patch( 0 ),
           d_action( scinew NPAction1<T, Arg1>(ptr, pmf, arg1) ),
           d_fromDW( fromDW ),
           d_toDW( toDW )
      {
         d_completed = false;
         d_usesThreads = false;
         d_usesMPI = false;
         d_subpatchCapable = false;
	 d_tasktype = Normal;
      }

      
      template<class T>
      Task(const string&         taskName,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
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
	 d_tasktype = Normal;
      }
      
      template<class T, class Arg1>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
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
	 d_tasktype = Normal;
      }
      
      template<class T, class Arg1, class Arg2>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
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
	 d_tasktype = Normal;
      }
      
      template<class T, class Arg1, class Arg2, class Arg3>
      Task(const string&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
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
	 d_tasktype = Normal;
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
      void doit(const ProcessorGroup* pc);
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
	 DataWarehouseP   d_dw;
	 const VarLabel*  d_var;
	 int		  d_matlIndex;
	 const Patch*     d_patch;
	 Task*		  d_task;
	 int		  d_serialNumber;
	 
	 Dependency(      const DataWarehouseP& dw,
			  const VarLabel* d_var,
			  int matlIndex,
			  const Patch*,
			  Task* task);
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

      bool isReductionTask() const {
	 return d_tasktype == Reduction;
      }

      void setType(TaskType tasktype) {
	 d_tasktype = tasktype;
      }
      TaskType getType() const {
	 return d_tasktype;
      }

      void assignResource(int idx) {
	 resourceIndex = idx;
      }
      int getAssignedResourceIndex() const {
	 return resourceIndex;
      }

      //////////
      // Prints out information about the task...
      void display( ostream & out ) const;

      //////////
      // Prints out all information about the task, including dependencies
      void displayAll( ostream & out ) const;

   protected: // class Task
      friend class TaskGraph;
      bool visited;
      bool sorted;
      int resourceIndex;
   private: // class Task
      //////////
      // Insert Documentation Here:
      string              d_taskName;
      const Patch*        d_patch;
      ActionBase*         d_action;
      DataWarehouseP      d_fromDW;
      DataWarehouseP      d_toDW;
      bool                d_completed;
      vector<Dependency*> d_reqs;
      vector<Dependency*> d_comps;
      
      bool                d_usesMPI;
      bool                d_usesThreads;
      bool                d_subpatchCapable;
      TaskType		  d_tasktype;

      Task(const Task&);
      Task& operator=(const Task&);
   };
   
} // end namespace Uintah

ostream & operator << ( ostream & out, const Uintah::Task & task );
ostream & operator << ( ostream & out, const Uintah::Task::TaskType & tt );
ostream & operator << ( ostream & out, const Uintah::Task::Dependency & dep );

//
// $Log$
// Revision 1.22  2000/09/25 16:24:17  sparker
// Added a displayAll method to Task
//
// Revision 1.21  2000/09/13 20:57:25  sparker
// Added ostream operator for dependencies
//
// Revision 1.20  2000/08/23 22:33:40  dav
// added an output operator for task
//
// Revision 1.19  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.18  2000/06/17 07:06:44  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.17  2000/06/03 05:29:45  sparker
// Changed reduction variable emit to require ostream instead of ofstream
// emit now only prints number without formatting
// Cleaned up a few extraneously included files
// Added task constructor for an non-patch-based action with 1 argument
// Allow for patches and actions to be null
// Removed back pointer to this from Task::Dependency
//
// Revision 1.16  2000/06/01 23:16:18  guilkey
// Added code to the ReductionVariable stuff to "emit" it's data.  Added
// NPAction tasks.  NP=NonPatch, this is for tasks that don't need the patch.
//
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
