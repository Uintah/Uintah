/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Util/constHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>

#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>


#include <Core/Grid/uintahshare.h>
namespace Uintah {

  class Level;
  class DataWarehouse;
  class ProcessorGroup;

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

  class UINTAHSHARE Task {
    class UINTAHSHARE ActionBase {
    public:
      virtual ~ActionBase();
      virtual void doit(const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW) = 0;
    };
    
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
        (ptr->*pmf)(pc, patches, matls, fromDW, toDW, arg1);
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
        (ptr->*pmf)(pc, patches, matls, fromDW, toDW, arg1, arg2);
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
        (ptr->*pmf)(pc, patches, matls, fromDW, toDW, arg1, arg2, arg3, arg4);
      }
    }; // end Action4
    
    template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
    class Action5 : public ActionBase {
      
      T* ptr;
      void (T::*pmf)(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*,
                     DataWarehouse*,
                     Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5);
      Arg1 arg1;
      Arg2 arg2;
      Arg3 arg3;
      Arg4 arg4;
      Arg5 arg5;
    public: // class Action4
      Action5( T* ptr,
               void (T::*pmf)(const ProcessorGroup*, 
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse*,
                              DataWarehouse*,
                              Arg1, Arg2, Arg3, Arg4, Arg5),
               Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
        : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2),
          arg3(arg3), arg4(arg4), arg5(arg5) {}
      virtual ~Action5() {}
      
      //////////
      // Insert Documentation Here:
      virtual void doit(const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW) {
        (ptr->*pmf)(pc, patches, matls, fromDW, toDW, arg1, arg2, arg3, arg4, arg5);
      }
    }; // end Action5
    
  public: // class Task
    
    enum WhichDW {
      OldDW=0, NewDW=1, CoarseOldDW=2, CoarseNewDW=3, ParentOldDW=4, ParentNewDW=5, 
      TotalDWs=6
    };
    enum {
      NoDW = -1, InvalidDW = -2
    };
    
    enum TaskType {
      Normal,
      Reduction,
      InitialSend,
      OncePerProc, // make sure to pass a PerProcessorPatchSet to the addTask function
      Output
    };
    
    Task(const std::string& taskName, TaskType type)
      :  d_taskName(taskName),
         d_action(0)
    {
      d_tasktype = type;
      initialize();
    }
    
    template<class T>
    Task(const std::string&         taskName,
         T*                    ptr,
         void (T::*pmf)(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*,
                        DataWarehouse*) )
      : d_taskName( taskName ), 
        d_action( scinew Action<T>(ptr, pmf) )
    {
      d_tasktype = Normal;
      initialize();
    }
    
    template<class T, class Arg1>
    Task(const std::string&         taskName,
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
      d_tasktype = Normal;
      initialize();
    }
    
    template<class T, class Arg1, class Arg2>
    Task(const std::string&         taskName,
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
      d_tasktype = Normal;
      initialize();
    }
    
    template<class T, class Arg1, class Arg2, class Arg3>
    Task(const std::string&         taskName,
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
      d_tasktype = Normal;
      initialize();
    }
    
    template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
    Task(const std::string&         taskName,
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
      d_tasktype = Normal;
      initialize();
    }
    
    template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
    Task(const std::string&         taskName,
         T*                    ptr,
         void (T::*pmf)(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*,
                        DataWarehouse*,
                        Arg1, Arg2, Arg3, Arg4, Arg5),
         Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
      : d_taskName( taskName ), 
        d_action( scinew Action5<T, Arg1, Arg2, Arg3, Arg4, Arg5>(ptr, pmf, arg1, arg2, arg3, arg4, arg5) )
    {
      d_tasktype = Normal;
      initialize();
    }
    
    void initialize();
    
    ~Task();
    
    void hasSubScheduler(bool state = true);
    bool inline getHasSubScheduler() const { return d_hasSubScheduler; }
    void usesMPI(bool state=true);
    void usesThreads(bool state);
    bool usesThreads() const {
      return d_usesThreads;
    }
    
    //////////
    // Insert Documentation Here:
    void subpatchCapable(bool state=true);

    enum DomainSpec {
      NormalDomain,  // <- Normal/default setting
      OutOfDomain,   // <- Require things from another level or set of patches
      CoarseLevel,   // <- AMR :  The data on the coarse level under the range of the fine patches (including extra cells or boundary layers)
      FineLevel,     // <- AMR :  The data on the fine level under the range of the coarse patches (including extra cells or boundary layers)
      OtherGridDomain // for when we copy data to new grid after a regrid.
    };

    //////////
    // Most general case
    void requires(WhichDW, const VarLabel*,
                  const PatchSubset* patches, DomainSpec patches_dom,
                  const MaterialSubset* matls, DomainSpec matls_dom,
                  Ghost::GhostType gtype, int numGhostCells = 0, bool oldTG=false);
    
    //////////
    // Insert Documentation Here:
    void requires(WhichDW, const VarLabel*,
                  Ghost::GhostType gtype, int numGhostCells = 0, bool oldTG=false);
    
    //////////
    // Insert Documentation Here:
    void requires(WhichDW, const VarLabel*,
                  const PatchSubset* patches, const MaterialSubset* matls,
                  Ghost::GhostType gtype, int numGhostCells = 0, bool oldTG=false);
    
    //////////
    // Insert Documentation Here:
    void requires(WhichDW, const VarLabel*,
                  const PatchSubset* patches,
                  Ghost::GhostType gtype, int numGhostCells = 0, bool oldTG=false);
    
    //////////
    // Insert Documentation Here:
    void requires(WhichDW, const VarLabel*,
                  const MaterialSubset* matls,
                  Ghost::GhostType gtype, int numGhostCells = 0, bool oldTG=false);
                  
    //////////
    // Insert Documentation Here: 
    void requires(WhichDW, const VarLabel*,
                  const MaterialSubset* matls, DomainSpec matls_dom,
                  Ghost::GhostType gtype, int numGhostCells = 0, bool oldTG=false);
    
    //////////
    // Requires only for reduction variables
    void requires(WhichDW, const VarLabel*, const Level* level = 0,
                  const MaterialSubset* matls = 0, DomainSpec matls_dom = NormalDomain, bool oldTG=false);
    
    //////////
    // Requires for reduction variables or perpatch veriables
    void requires(WhichDW, const VarLabel*, const MaterialSubset* matls, bool oldTG=false);
    
    //////////
    // Requires only for perpatch variables
    void requires(WhichDW, const VarLabel*, const PatchSubset* patches,
                  const MaterialSubset* matls = 0);
    
    //////////
    // Most general case
    void computes(const VarLabel*,
                  const PatchSubset* patches, DomainSpec patches_domain, 
                  const MaterialSubset* matls, DomainSpec matls_domain);
    
    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*, const PatchSubset* patches = 0,
                  const MaterialSubset* matls = 0);
    
    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*, const MaterialSubset* matls);
    
    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*, const MaterialSubset* matls,
                  DomainSpec matls_domain);

    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*,
                  const PatchSubset* patches, DomainSpec patches_domain);

    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*, const Level* level,
                  const MaterialSubset* matls = 0, DomainSpec matls_domain = NormalDomain);

     //////////
    // Most general case
    void modifies(const VarLabel*,
                  const PatchSubset* patches, DomainSpec patches_domain, 
                  const MaterialSubset* matls, DomainSpec matls_domain, bool oldTG=false);
    
    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*, const PatchSubset* patches,
                  const MaterialSubset* matls, bool oldTG=false);
    
    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*, const MaterialSubset* matls, bool oldTG=false);
    
    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*, const MaterialSubset* matls,
                  DomainSpec matls_domain, bool oldTG=false);
   
    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*, bool oldTG=false);

    //////////
    // Modify reduction vars
    void modifies(const VarLabel*, const Level* level,
                  const MaterialSubset* matls = 0, DomainSpec matls_domain = NormalDomain, bool oldTG=false);
    
    //////////
    // Tells the task to actually execute the function assigned to it.
    void doit(const ProcessorGroup* pc, const PatchSubset*,
              const MaterialSubset*, vector<DataWarehouseP>& dws);

    inline const char* getName() const {
      return d_taskName.c_str();
    }
    inline const PatchSet* getPatchSet() const {
      return patch_set;
    }
    
    inline const MaterialSet* getMaterialSet() const {
      return matl_set;
    }
    
    struct Edge;
      
    int d_phase;  //synchronized phase id, for dynamic task scheduling
    int d_comm;  //task communicator id, for threaded task scheduling
    int maxGhostCells; //max ghost cells of this task
    std::set<Task*> childTasks;
    std::set<Task*> allChildTasks;
    
    enum DepType {
      Modifies, Computes, Requires
    };
    
    struct UINTAHSHARE Dependency {
      Dependency* next;
      DepType deptype;
      Task* task;
      const VarLabel*  var;
      bool lookInOldTG;
      const PatchSubset* patches;
      const MaterialSubset* matls;
      const Level* reductionLevel;
      Edge* req_head;   // Used in compiling the task graph.
      Edge* req_tail;
      Edge* comp_head;
      Edge* comp_tail;
      DomainSpec patches_dom;
      DomainSpec matls_dom;
      Ghost::GhostType gtype;
      WhichDW whichdw;  // Used only by Requires
      
      // in the multi-TG construct, this will signify that the required
      // var will be constructed by the old TG
      int numGhostCells;
      int mapDataWarehouse() const {
        return task->mapDataWarehouse(whichdw);
      }
      
      Dependency(DepType deptype, Task* task, WhichDW dw, const VarLabel* var,
                 bool oldtg, 
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DomainSpec patches_dom = NormalDomain,
                 DomainSpec matls_dom = NormalDomain,
                 Ghost::GhostType gtype = Ghost::None,
                 int numGhostCells = 0);
      Dependency(DepType deptype, Task* task, WhichDW dw, const VarLabel* var,
                 bool oldtg, 
                 const Level* reductionLevel,
                 const MaterialSubset* matls,
                 DomainSpec matls_dom = NormalDomain);
      ~Dependency();
      inline void addComp(Edge* edge);
      inline void addReq(Edge* edge);

      constHandle<PatchSubset>
      getPatchesUnderDomain(const PatchSubset* domainPatches) const;
      
      constHandle<MaterialSubset>
      getMaterialsUnderDomain(const MaterialSubset* domainMaterials) const;

    private:
      static constHandle< PatchSubset >
      getOtherLevelPatchSubset(DomainSpec dom,
                                 const PatchSubset* subset,
                                 const PatchSubset* domainSubset, int ngc);
     
      Dependency();
      Dependency& operator=(const Dependency& copy);
      Dependency(const Dependency&);
    }; // end struct Dependency
    
    
    struct UINTAHSHARE Edge {
      const Dependency* comp;
      Edge* compNext;
      const Dependency* req;
      Edge* reqNext;
      inline Edge(const Dependency* comp, const Dependency * req)
        : comp(comp), compNext(0), req(req), reqNext(0)
      {
      }
    };

    typedef std::multimap<const VarLabel*, Dependency*, VarLabel::Compare> DepMap;
    
    const Dependency* getComputes() const {
      return comp_head;
    }
    const Dependency* getRequires() const {
      return req_head;
    }
    const Dependency* getModifies() const {
      return mod_head;
    }
    
    Dependency* getComputes() {
      return comp_head;
    }
    Dependency* getRequires() {
      return req_head;
    }
    Dependency* getModifies() {
      return mod_head;
    }

    // finds if it computes or modifies var
    bool hasComputes(const VarLabel* var, int matlIndex,
                     const Patch* patch) const;

    // finds if it requires or modifies var
    bool hasRequires(const VarLabel* var, int matlIndex, const Patch* patch,
                     IntVector lowOffset, IntVector highOffset,
                     WhichDW dw) const;

    // finds if it modifies var
    bool hasModifies(const VarLabel* var, int matlIndex,
                     const Patch* patch) const;

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
    void display( std::ostream & out ) const;
    
    //////////
    // Prints out all information about the task, including dependencies
    void displayAll( std::ostream & out ) const;
    
    int mapDataWarehouse(WhichDW dw) const;
    DataWarehouse* mapDataWarehouse(WhichDW dw, vector<DataWarehouseP>& dws) const;

    int getSortedOrder() const {
      return sortedOrder;
    }

    void setSortedOrder(int order) {
      sortedOrder = order;
    }

    void setMapping(int dwmap[TotalDWs]);

    void setSets(const PatchSet* patches, const MaterialSet* matls);

    
  private: // class Task
    Dependency* isInDepMap(const DepMap& depMap, const VarLabel* var,
                           int matlIndex, const Patch* patch) const;
    
    //////////
    // Insert Documentation Here:
    std::string d_taskName;
    ActionBase* d_action;
    Dependency* comp_head;
    Dependency* comp_tail;
    Dependency* req_head;
    Dependency* req_tail;
    Dependency* mod_head;
    Dependency* mod_tail;

    DepMap d_requiresOldDW;
    DepMap d_computes; // also contains modifies
    DepMap d_requires; // also contains modifies
    DepMap d_modifies;
    
    const PatchSet* patch_set;
    const MaterialSet* matl_set;
    
    bool                d_usesMPI;
    bool                d_usesThreads;
    bool                d_subpatchCapable;
    bool                d_hasSubScheduler;
    TaskType            d_tasktype;
    
    Task(const Task&);
    Task& operator=(const Task&);

    static const MaterialSubset* getGlobalMatlSubset();
    static MaterialSubset* globalMatlSubset;

    int dwmap[TotalDWs];
    int sortedOrder;

    UINTAHSHARE friend std::ostream & operator << ( std::ostream & out, const Uintah::Task & task );
    UINTAHSHARE friend std::ostream & operator << ( std::ostream & out, const Uintah::Task::TaskType & tt );
    UINTAHSHARE friend std::ostream & operator << ( std::ostream & out, const Uintah::Task::Dependency & dep );

  }; // end class Task
  
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

// This mus tbe at the bottom
#include <CCA/Ports/DataWarehouse.h>

#endif
