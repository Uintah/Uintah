#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/fixedvector.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/SimpleString.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
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
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* fromDW,
			   DataWarehouse* toDW) = 0;
      };

      template<class T>
      class NPAction : public ActionBase {

         T* ptr;
         void (T::*pmf)(const ProcessorGroup*,
                        DataWarehouse*,
                        DataWarehouse*);
      public: // class NPAction
         NPAction( T* ptr,
                 void (T::*pmf)(const ProcessorGroup*,
                                DataWarehouse*,
                                DataWarehouse*) )
            : ptr(ptr), pmf(pmf) {}
         virtual ~NPAction() {}

         //////////
         // Insert Documentation Here:
         virtual void doit(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
                           DataWarehouse* fromDW,
                           DataWarehouse* toDW) {
            (ptr->*pmf)(pc, fromDW, toDW);
         }
      }; // end class ActionBase

      template<class T, class Arg1>
      class NPAction1 : public ActionBase {

         T* ptr;
         void (T::*pmf)(const ProcessorGroup*,
                        DataWarehouse*,
                        DataWarehouse*,
			Arg1);
	 Arg1 arg1;
      public:  // class NPAction1
         NPAction1( T* ptr,
                 void (T::*pmf)(const ProcessorGroup*,
                                DataWarehouse*,
                                DataWarehouse*,
				Arg1),
		   Arg1 arg1)
            : ptr(ptr), pmf(pmf), arg1(arg1) {}
         virtual ~NPAction1() {}

         //////////
         // Insert Documentation Here:
         virtual void doit(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
                           DataWarehouse* fromDW,
                           DataWarehouse* toDW) {
            (ptr->*pmf)(pc, fromDW, toDW, arg1);
         }
      }; // end class NPAction1
      
      template<class T>
      class Action : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse*,
			DataWarehouse*);
      public: // class Action
	 Action( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse*,
				DataWarehouse*) )
	    : ptr(ptr), pmf(pmf) {}
	 virtual ~Action() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* fromDW,
			   DataWarehouse* toDW) {
	    (ptr->*pmf)(pc, patches, matls, fromDW, toDW);
	 }
      }; // end class Action

      template<class T, class Arg1>
      class Action1 : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse*,
			DataWarehouse*,
			Arg1 arg1);
	 Arg1 arg1;
      public: // class Action1
	 Action1( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse*,
				DataWarehouse*,
				Arg1),
		  Arg1 arg1)
	    : ptr(ptr), pmf(pmf), arg1(arg1) {}
	 virtual ~Action1() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* fromDW,
			   DataWarehouse* toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1);
	 }
      }; // end class Action1
      
      template<class T, class Arg1, class Arg2>
      class Action2 : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse*,
			DataWarehouse*,
			Arg1 arg1, Arg2 arg2);
	 Arg1 arg1;
	 Arg2 arg2;
      public: // class Action2
	 Action2( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse*,
				DataWarehouse*,
				Arg1, Arg2),
		  Arg1 arg1, Arg2 arg2)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2) {}
	 virtual ~Action2() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* fromDW,
			   DataWarehouse* toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1, arg2);
	 }
      }; // end class Action2

      template<class T, class Arg1, class Arg2, class Arg3>
      class Action3 : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse*,
			DataWarehouse*,
			Arg1 arg1, Arg2 arg2, Arg3 arg3);
	 Arg1 arg1;
	 Arg2 arg2;
	 Arg3 arg3;
      public: // class Action3
	 Action3( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse*,
				DataWarehouse*,
				Arg1, Arg2, Arg3),
		  Arg1 arg1, Arg2 arg2, Arg3 arg3)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3) {}
	 virtual ~Action3() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* fromDW,
			   DataWarehouse* toDW) {
	    (ptr->*pmf)(pc, patches, matls, fromDW, toDW, arg1, arg2, arg3);
	 }
      }; // end Action3

      template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
      class Action4 : public ActionBase {
	 
	 T* ptr;
	 void (T::*pmf)(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse*,
			DataWarehouse*,
			Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4);
	 Arg1 arg1;
	 Arg2 arg2;
	 Arg3 arg3;
	 Arg4 arg4;
      public: // class Action4
	 Action4( T* ptr,
		 void (T::*pmf)(const ProcessorGroup*, 
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse*,
				DataWarehouse*,
				Arg1, Arg2, Arg3, Arg4),
		  Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
	    : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2),
	      arg3(arg3), arg4(arg4) {}
	 virtual ~Action4() {}
	 
	 //////////
	 // Insert Documentation Here:
	 virtual void doit(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* fromDW,
			   DataWarehouse* toDW) {
	    (ptr->*pmf)(pc, patch, fromDW, toDW, arg1, arg2, arg3, arg4);
	 }
      }; // end Action4

   public: // class Task

      enum WhichDW {
	OldDW, NewDW
      };
      
      enum TaskType {
	 Normal,
	 Reduction,
	 InitialSend,
      };

      Task(const SimpleString&         taskName, TaskType type)
	:  d_taskName(taskName),
	   d_action(0)
      {
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
	 d_tasktype = type;
	 initialize();
      }

      template<class T>
      Task(const SimpleString&         taskName,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse*,
			  DataWarehouse*) )
	: d_taskName( taskName ), 
	  d_action( scinew Action<T>(ptr, pmf) )
      {
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
	 d_tasktype = Normal;
	 initialize();
      }

     template<class T, class Arg1>
      Task(const SimpleString&         taskName,
           T*                    ptr,
           void (T::*pmf)(const ProcessorGroup*,
                          DataWarehouse*,
                          DataWarehouse*,
			  Arg1),
	   Arg1 arg1)
         : d_taskName( taskName ),
           d_action( scinew NPAction1<T, Arg1>(ptr, pmf, arg1) )
      {
         d_usesThreads = false;
         d_usesMPI = false;
         d_subpatchCapable = false;
	 d_tasktype = Normal;
	 initialize();
      }

      template<class T, class Arg1>
      Task(const SimpleString&         taskName,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse*,
			  DataWarehouse*,
			  Arg1),
	   Arg1 arg1)
	 : d_taskName( taskName ), 
	   d_action( scinew Action1<T, Arg1>(ptr, pmf, arg1) )
      {
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
	 d_tasktype = Normal;
	 initialize();
      }
      
      template<class T, class Arg1, class Arg2>
      Task(const SimpleString&         taskName,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse*,
			  DataWarehouse*,
			  Arg1, Arg2),
	   Arg1 arg1, Arg2 arg2)
	 : d_taskName( taskName ), 
	   d_action( scinew Action2<T, Arg1, Arg2>(ptr, pmf, arg1, arg2) )
      {
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
	 d_tasktype = Normal;
	 initialize();
      }
      
      template<class T, class Arg1, class Arg2, class Arg3>
      Task(const SimpleString&         taskName,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse*,
			  DataWarehouse*,
			  Arg1, Arg2, Arg3),
	   Arg1 arg1, Arg2 arg2, Arg3 arg3)
	 : d_taskName( taskName ), 
	   d_action( scinew Action3<T, Arg1, Arg2, Arg3>(ptr, pmf, arg1, arg2, arg3) )
      {
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
	 d_tasktype = Normal;
	 initialize();
      }
      
      template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
      Task(const SimpleString&         taskName,
	   T*                    ptr,
	   void (T::*pmf)(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse*,
			  DataWarehouse*,
			  Arg1, Arg2, Arg3, Arg4),
	   Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
	 : d_taskName( taskName ), 
	   d_action( scinew Action4<T, Arg1, Arg2, Arg3, Arg4>(ptr, pmf, arg1, arg2, arg3, arg4) )
      {
	 d_usesThreads = false;
	 d_usesMPI = false;
	 d_subpatchCapable = false;
	 d_tasktype = Normal;
	 initialize();
      }

     void initialize();

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
      void requires(WhichDW, const VarLabel*, const PatchSubset* patches = 0,
		    const MaterialSubset* matls = 0);
      
      //////////
      // Insert Documentation Here:
      void requires(WhichDW, const VarLabel*, const MaterialSubset* matls);
      
      //////////
      // Insert Documentation Here:
      void requires(WhichDW, const VarLabel*,
		    Ghost::GhostType gtype, int numGhostCells = 0);
      
      //////////
      // Insert Documentation Here:
      void requires(WhichDW, const VarLabel*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    Ghost::GhostType gtype, int numGhostCells = 0);
      
      //////////
      // Insert Documentation Here:
      void requires(WhichDW, const VarLabel*,
		    const PatchSubset* patches,
		    Ghost::GhostType gtype, int numGhostCells = 0);
      
      //////////
      // Insert Documentation Here:
      void requires(WhichDW, const VarLabel*,
		    const MaterialSubset* matls,
		    Ghost::GhostType gtype, int numGhostCells = 0);
      
      //////////
      // Insert Documentation Here:
      void computes(const VarLabel*, const PatchSubset* patches = 0,
		    const MaterialSubset* matls = 0);
      
      //////////
      // Insert Documentation Here:
      void computes(const VarLabel*, const MaterialSubset* matls);
      
      //////////
      // Tells the task to actually execute the function assigned to it.
      void doit(const ProcessorGroup* pc, const PatchSubset*,
		const MaterialSubset*, DataWarehouse* fromDW,
		DataWarehouse* toDW);

      inline const char* getName() const {
	 return d_taskName;
      }
     inline const PatchSet* getPatchSet() const {
       return patch_set;
     }

     inline const MaterialSet* getMaterialSet() const {
       return matl_set;
     }

     struct Edge;

      struct Dependency {
	Dependency* next;
	Task* task;
	const VarLabel*  var;
	const PatchSubset* patches;
	const MaterialSubset* matls;
	Edge* req_head;
	Edge* req_tail;
	Edge* comp_head;
	Edge* comp_tail;
	Ghost::GhostType gtype;
	WhichDW dw;
	int numGhostCells;

	Dependency(Task* task,
		   WhichDW dw,
		   const VarLabel* var,
		   const PatchSubset* patches,
		   const MaterialSubset* matls,
		   Ghost::GhostType gtype = Ghost::None,
		   int numGhostCells = 0);
	~Dependency();
	inline void addComp(Edge* edge);
	inline void addReq(Edge* edge);

      private:
	Dependency();
	Dependency& operator=(const Dependency& copy);
	Dependency(const Dependency&);
      }; // end struct Dependency


     struct Edge {
       const Dependency* comp;
       Edge* compNext;
       const Dependency* req;
       Edge* reqNext;
       inline Edge(const Dependency* comp, const Dependency * req)
	 : comp(comp), compNext(0), req(req), reqNext(0)
       {
       }
     };

     const Dependency* getComputes() const {
       return comp_head;
     }
     const Dependency* getRequires() const {
       return req_head;
     }

     Dependency* getComputes() {
       return comp_head;
     }
     Dependency* getRequires() {
       return req_head;
     }

#if 0
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
#endif

      bool isReductionTask() const {
	 return d_tasktype == Reduction;
      }

      void setType(TaskType tasktype) {
	 d_tasktype = tasktype;
      }
      TaskType getType() const {
	 return d_tasktype;
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
      void setSets(const PatchSet* patches, const MaterialSet* matls);

   private: // class Task
      //////////
      // Insert Documentation Here:
      SimpleString        d_taskName;
      ActionBase*         d_action;
#if 0
      reqType		  d_reqs;
      compType		  d_comps;
#endif
     Dependency* comp_head;
     Dependency* comp_tail;
     Dependency* req_head;
     Dependency* req_tail;

     const PatchSet* patch_set;
     const MaterialSet* matl_set;
      
      bool                d_usesMPI;
      bool                d_usesThreads;
      bool                d_subpatchCapable;
      TaskType		  d_tasktype;

      Task(const Task&);
      Task& operator=(const Task&);
   };

  inline void Task::Dependency::addComp(Edge* edge)
  {
    if(comp_tail)
      comp_tail->compNext=edge;
    else
      comp_head=edge;
    comp_tail=edge;
  }
  inline void Task::Dependency::addReq(Edge* edge)
  {
    if(req_tail)
      req_tail->reqNext=edge;
    else
      req_head=edge;
    req_tail=edge;
  }
} // End namespace Uintah
   

ostream & operator << ( ostream & out, const Uintah::Task & task );
ostream & operator << ( ostream & out, const Uintah::Task::TaskType & tt );
ostream & operator << ( ostream & out, const Uintah::Task::Dependency & dep );


#endif
