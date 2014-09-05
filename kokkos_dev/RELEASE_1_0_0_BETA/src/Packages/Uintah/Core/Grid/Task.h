#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/fixedvector.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/SimpleString.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <string>
#include <vector>
#include <iostream>

namespace Uintah {

using std::vector;
using std::string;
using std::ostream;

using namespace SCIRun;
   
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
         void (T::*pmf)(const ProcessorGroup*,
                        DataWarehouseP&,
                        DataWarehouseP&,
			Arg1);
	 Arg1 arg1;
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
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1);
	 Arg1 arg1;
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
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1, Arg2 arg2);
	 Arg1 arg1;
	 Arg2 arg2;
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
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1, Arg2 arg2, Arg3 arg3);
	 Arg1 arg1;
	 Arg2 arg2;
	 Arg3 arg3;
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

      template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
      class Action4 : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorGroup*,
			const Patch*,
			DataWarehouseP&,
			DataWarehouseP&,
			Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4);
	 Arg1 arg1;
	 Arg2 arg2;
	 Arg3 arg3;
	 Arg4 arg4;
      public: // class Action4
	 Action4( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const Patch*, 
				DataWarehouseP&,
				DataWarehouseP&,
				Arg1, Arg2, Arg3, Arg4),
		  Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2),
	      arg3(arg3), arg4(arg4) {}
	 virtual ~Action4() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const Patch* patch,
			   DataWarehouseP& fromDW,
			   DataWarehouseP& toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1, arg2, arg3, arg4);
	 }
      }; // end Action4

   public: // class Task

      enum TaskType {
	 Normal,
	 Reduction,
	 Scatter,
	 Gather
      };

      Task(const SimpleString&         taskName)
	:  d_resourceIndex(-1),
	   d_taskName(taskName),
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
      Task(const SimpleString&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
			  const Patch*,
			  DataWarehouseP&,
			  DataWarehouseP&) )
	: d_resourceIndex( -1 ),
	  d_taskName( taskName ), 
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
      Task(const SimpleString&         taskName,
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
      Task(const SimpleString&         taskName,
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
      Task(const SimpleString&         taskName,
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
      Task(const SimpleString&         taskName,
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
      Task(const SimpleString&         taskName,
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
      Task(const SimpleString&         taskName,
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
      
      template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
      Task(const SimpleString&         taskName,
	   const Patch*         patch,
	   DataWarehouseP&       fromDW,
	   DataWarehouseP&       toDW,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
			  const Patch*,
			  DataWarehouseP&,
			  DataWarehouseP&,
			  Arg1, Arg2, Arg3, Arg4),
	   Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
	 : d_taskName( taskName ), 
	   d_patch( patch ),
	   d_action( scinew Action4<T, Arg1, Arg2, Arg3, Arg4>(ptr, pmf, arg1, arg2, arg3, arg4) ),
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
      void requires(const DataWarehouseP& ds, const VarLabel*, \
		    int matlIndex=-1);
      
      //////////
      // Insert Documentation Here:
      void requires(const DataWarehouseP& ds, const VarLabel*, int matlIndex,
		    const Patch* patch, Ghost::GhostType gtype,
		    int numGhostCells = 0);
      
      //////////
      // Insert Documentation Here:
      void computes(const DataWarehouseP& ds, const VarLabel*,
		    int matlIndex=-1);
      
      //////////
      // Insert Documentation Here:
      void computes(const DataWarehouseP& ds, const VarLabel*, int matlIndex,
		    const Patch* patch);

      //////////
      // Tells the task to actually execute the function assigned to it.
      void doit(const ProcessorGroup* pc);

      inline const char* getName() const {
	 return d_taskName;
      }
      inline const Patch* getPatch() const {
	 return d_patch;
      }
      inline bool isCompleted() const {
	 return d_completed;
      }

      struct Dependency {
	 DataWarehouse*   d_dw;
	 const VarLabel*  d_var;
	 const Patch*     d_patch;
	 Task*		  d_task;
	 int		  d_matlIndex;
	 int		  d_serialNumber;
	 IntVector	  d_lowIndex, d_highIndex;

	 inline Dependency() {}
	 Dependency(const Dependency&);
	 Dependency& operator=(const Dependency& copy) {
	    d_dw=copy.d_dw;
	    d_var=copy.d_var;
	    d_patch=copy.d_patch;
	    d_task=copy.d_task;
	    d_matlIndex=copy.d_matlIndex;
	    d_serialNumber=copy.d_serialNumber;
	    d_lowIndex=copy.d_lowIndex;
	    d_highIndex=copy.d_highIndex;
	    return *this;
	 }
	 
	 inline Dependency(DataWarehouse* dw,
			   const VarLabel* var,
			   int matlIndex,
			   const Patch* patch,
			   Task* task,
			   const IntVector& lowIndex,
			   const IntVector& highIndex)
	 : d_dw(dw),
	   d_var(var),
	   d_patch(patch),
	   d_task(task),
	   d_matlIndex(matlIndex),
	   d_serialNumber(-123),
	   d_lowIndex(lowIndex),
	   d_highIndex(highIndex)
	 {
	 }
      }; // end struct Dependency
      
      //////////
      // Insert Documentation Here:
      void addComps(vector<Dependency*>&) const;

      static const int MAX_COMPS = 16;
      static const int MAX_REQS = 64;

      typedef fixedvector<Dependency, MAX_COMPS> compType;
      typedef fixedvector<Dependency, MAX_REQS> reqType;

      //////////
      // Returns the list of variables that this Task computes.
      const compType& getComputes() const {
	 return d_comps;
      }
      
      //////////
      // Insert Documentation Here:
      const reqType& getRequires() const {
	 return d_reqs;
      }

      //////////
      // Returns the list of variables that this Task computes.
      compType& getComputes() {
	 return d_comps;
      }
      
      //////////
      // Insert Documentation Here:
      reqType& getRequires() {
	 return d_reqs;
      }

      bool isReductionTask() const {
	 return d_tasktype == Reduction;
      }

      void setType(TaskType tasktype) {
	 d_tasktype = tasktype;
      }
      TaskType getType() const {
	 return d_tasktype;
      }

      void assignResource( int idx ) {
	 d_resourceIndex = idx;
      }
      int getAssignedResourceIndex() const {
	 return d_resourceIndex;
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
      int  d_resourceIndex;
   private: // class Task
      //////////
      // Insert Documentation Here:
      SimpleString        d_taskName;
      const Patch*        d_patch;
      ActionBase*         d_action;
      DataWarehouseP      d_fromDW;
      DataWarehouseP      d_toDW;
      bool                d_completed;
      reqType		  d_reqs;
      compType		  d_comps;
      
      bool                d_usesMPI;
      bool                d_usesThreads;
      bool                d_subpatchCapable;
      TaskType		  d_tasktype;

      Task(const Task&);
      Task& operator=(const Task&);
   };

} // End namespace Uintah
   

ostream & operator << ( ostream & out, const Uintah::Task & task );
ostream & operator << ( ostream & out, const Uintah::Task::TaskType & tt );
ostream & operator << ( ostream & out, const Uintah::Task::Dependency & dep );


#endif
