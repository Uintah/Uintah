/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CORE_GRID_TASK_H
#define CORE_GRID_TASK_H

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Util/constHandle.h>
#include <Core/Util/TupleHelpers.hpp>

#include <set>
#include <vector>
#include <string>
#include <iostream>

namespace Uintah {

class Level;
class DataWarehouse;
class ProcessorGroup;
class Task;
class DetailedTask;

/**************************************

 CLASS
   Task

 GENERAL INFORMATION
   Task.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

 KEYWORDS
   Task

 DESCRIPTION

 ****************************************/

class Task {
 
public:

  enum CallBackEvent
  {
      CPU      // <- normal CPU task, happens when a GPU enabled task runs on CPU
    , preGPU   // <- pre GPU kernel callback, happens before CPU->GPU copy (reserved, not implemented yet... )
    , GPU      // <- GPU kernel callback, happens after dw: CPU->GPU copy, kernel launch should be queued in this callback
    , postGPU  // <- post GPU kernel callback, happens after dw: GPU->CPU copy but before MPI sends.
  };
 

protected: // class Task

  // base Action class
  class ActionBase {

    public:

      virtual ~ActionBase(){};

      virtual void doit(       DetailedTask   * task
                       ,       CallBackEvent    event
                       , const ProcessorGroup * pg
                       , const PatchSubset    * patches
                       , const MaterialSubset * matls
                       ,       DataWarehouse  * fromDW
                       ,       DataWarehouse  * toDW
                       ,       void           * oldTaskGpuDW
                       ,       void           * newTaskGpuDW
                       ,       void           * stream
                       ,       int              deviceID
                       ) = 0;
  };



private: // class Task

  // CPU Action constructor
  template<typename T, typename... Args>
  class Action : public ActionBase {

      T * ptr;
      void (T::*pmf)( const ProcessorGroup * pg
                    , const PatchSubset    * patches
                    , const MaterialSubset * m_matls
                    ,       DataWarehouse  * fromDW
                    ,       DataWarehouse  * toDW
                    ,       Args...          args
                    );
      std::tuple<Args...> m_args;


public: // class Task

    Action( T * ptr
          , void (T::*pmf)( const ProcessorGroup * pg
                          , const PatchSubset    * patches
                          , const MaterialSubset * m_matls
                          ,       DataWarehouse  * fromDW
                          ,       DataWarehouse  * toDW
                          ,       Args...       args
                          )
          , Args... args
          )
      : ptr(ptr)
      , pmf(pmf)
      , m_args(std::forward<Args>(args)...)
    {}

    virtual ~Action() {}

    //////////
    //
    virtual void doit(       DetailedTask   * task
                     ,       CallBackEvent    event
                     , const ProcessorGroup * pg
                     , const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       void           * oldTaskGpuDW
                     ,       void           * newTaskGpuDW
                     ,       void           * stream
                     ,       int              deviceID
                     )
    {
      doit_impl(pg, patches, matls, fromDW, toDW, typename Tuple::gens<sizeof...(Args)>::type());
    }

private: // class Task

    template<int... S>
    void doit_impl( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       Tuple::seq<S...>
                  )
    {
      (ptr->*pmf)(pg, patches, matls, fromDW, toDW, std::get<S>(m_args)...);
    }

  };  // end CPU Action class



  // GPU (device) Action constructor
  template<typename T, typename... Args>
  class ActionDevice : public ActionBase {

    T * ptr;
    void (T::*pmf)(       DetailedTask   * dtask
                  ,       CallBackEvent    event
                  , const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * m_matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       void           * oldTaskGpuDW
                  ,       void           * newTaskGpuDW
                  ,       void           * stream
                  ,       int              deviceID
                  ,       Args...          args
                  );
    std::tuple<Args...> m_args;


public: // class Task

    ActionDevice( T * ptr
                , void (T::*pmf)(       DetailedTask   * dtask
                                ,       CallBackEvent    event
                                , const ProcessorGroup * pg
                                , const PatchSubset    * patches
                                , const MaterialSubset * m_matls
                                ,       DataWarehouse  * fromDW
                                ,       DataWarehouse  * toDW
                                ,       void           * oldTaskGpuDW
                                ,       void           * newTaskGpuDW
                                ,       void           * stream
                                ,       int              deviceID
                                ,       Args...          args
                                )
               , Args... args
               )
      : ptr(ptr)
      , pmf(pmf)
      , m_args(std::forward<Args>(args)...)
    {}

    virtual ~ActionDevice() {}

    virtual void doit(       DetailedTask   * dtask
                     ,       CallBackEvent    event
                     , const ProcessorGroup * pg
                     , const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       void           * oldTaskGpuDW
                     ,       void           * newTaskGpuDW
                     ,       void           * stream
                     ,       int              deviceID
                     )
    {
      doit_impl(dtask, event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID, typename Tuple::gens<sizeof...(Args)>::type());
    }

private : // class Task

    template<int... S>
    void doit_impl(       DetailedTask   * dtask
                  ,       CallBackEvent    event
                  , const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       void           * oldTaskGpuDW
                  ,       void           * newTaskGpuDW
                  ,       void           * stream
                  ,       int              deviceID
                  ,       Tuple::seq<S...>
                  )
      {
        (ptr->*pmf)(dtask, event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID, std::get<S>(m_args)...);
      }

  };  // end GPU (device) Action constructor



public: // class Task

  enum WhichDW {
     OldDW       = 0
  ,  NewDW       = 1
  ,  CoarseOldDW = 2
  ,  CoarseNewDW = 3
  ,  ParentOldDW = 4
  ,  ParentNewDW = 5
  ,  TotalDWs    = 6
  };

  enum {
      NoDW      = -1
    , InvalidDW = -2
  };

  enum TaskType {
      Normal
    , Reduction
    , InitialSend
    , OncePerProc // make sure to pass a PerProcessor PatchSet to the addTask function
    , Output
    , Spatial    // e.g. Radiometer task (spatial scheduling); must call task->setType(Task::Spatial)
  };
    


  Task( const std::string & taskName, TaskType type )
    : m_task_name(taskName)
    , m_action(nullptr)
  {
    d_tasktype = type;
    initialize();
  }

  // CPU Task constructor
  template<typename T, typename... Args>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup *
                      , const PatchSubset    *
                      , const MaterialSubset *
                      ,       DataWarehouse  *
                      ,       DataWarehouse  *
                      ,       Args...
                      )
       , Args... args
      )
     : m_task_name(taskName)
     , m_action(new Action<T, Args...>(ptr, pmf, std::forward<Args>(args)...))
  {
    d_tasktype = Normal;
    initialize();
  }

  // Device (GPU) Task constructor
  template<typename T, typename... Args>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)(       DetailedTask   * m_task
                      ,       CallBackEvent    event
                      , const ProcessorGroup * pg
                      , const PatchSubset    * patches
                      , const MaterialSubset * m_matls
                      ,       DataWarehouse  * fromDW
                      ,       DataWarehouse  * toDW
                      ,       void           * old_TaskGpuDW
                      ,       void           * new_TaskGpuDW
                      ,       void           * stream
                      ,       int              deviceID
                      ,       Args...          args
                      )
      , Args... args
      )
      : m_task_name(taskName)
      , m_action(new ActionDevice<T, Args...>(ptr, pmf, std::forward<Args>(args)...))
  {
    initialize();
    d_tasktype = Normal;
  }

  void initialize();

  virtual ~Task();

  void hasSubScheduler(bool state = true);
  inline bool getHasSubScheduler() const { return m_has_subscheduler; }

         void usesMPI(bool state);
  inline bool usesMPI() const { return m_uses_mpi; }

         void usesThreads(bool state);
  inline bool usesThreads() const { return m_uses_threads; }

         void usesDevice(bool state);
  inline bool usesDevice() const { return m_uses_device; }


  void subpatchCapable(bool state = true);

  enum MaterialDomainSpec {
      NormalDomain  // <- Normal/default setting
    , OutOfDomain   // <- Require things from all material
  };

  enum PatchDomainSpec {
      ThisLevel        // <- Normal/default setting
    , CoarseLevel      // <- AMR :  The data on the coarse level under the range of the fine patches (including extra cells or boundary layers)
    , FineLevel        // <- AMR :  The data on the fine level under the range of the coarse patches (including extra cells or boundary layers)
    , OtherGridDomain  // for when we copy data to new grid after a regrid.
  };

  //////////
  // Most general case
  void requires( WhichDW
               , const VarLabel       *
               , const PatchSubset    * patches
               , PatchDomainSpec        patches_dom
               , int                    level_offset
               , const MaterialSubset * matls
               , MaterialDomainSpec     matls_dom
               , Ghost::GhostType       gtype
               , int                    numGhostCells = 0
               , bool                   oldTG         = false
               );

  //////////
  // Like general case, level_offset is not specified
  void requires(       WhichDW
               , const VarLabel           *
               , const PatchSubset        * patches
               ,       PatchDomainSpec      patches_dom
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_dom
               ,       Ghost::GhostType     gtype
               ,       int                  numGhostCells = 0
               ,       bool                 oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel           *
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel         *
               , const PatchSubset      * patches
               , const MaterialSubset   * matls
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires( WhichDW
               , const VarLabel         *
               , const PatchSubset      * patches
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel        *
               , const MaterialSubset  * matls
               ,      Ghost::GhostType   gtype
               ,      int                numGhostCells = 0
               ,      bool               oldTG         = false);

  //////////
  //
  void requires(       WhichDW
               , const VarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_dom
               ,       Ghost::GhostType     gtype
               ,       int                  numGhostCells = 0
               ,       bool                 oldTG         = false
               );

  //////////
  // Requires only for Reduction variables
  void requires(       WhichDW
               , const VarLabel           *
               , const Level              * level     = nullptr
               , const MaterialSubset     * matls     = nullptr
               ,       MaterialDomainSpec   matls_dom = NormalDomain
               ,       bool                 oldTG     = false
               );

  //////////
  // Requires for reduction variables or PerPatch variables
  void requires(       WhichDW
               , const VarLabel       *
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  // Requires only for PerPatch variables
  void requires(       WhichDW
               , const VarLabel       *
               , const PatchSubset    * patches
               , const MaterialSubset * matls = nullptr
               );

  //////////
  // Most general case
  void computes( const VarLabel           *
               , const PatchSubset        * patches
               ,       PatchDomainSpec      patches_domain
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               );

  //////////
  //
  void computes( const VarLabel       *
               , const PatchSubset    * patches = nullptr
               , const MaterialSubset * matls   = nullptr
               );

  //////////
  //
  void computes( const VarLabel       *
               , const MaterialSubset * matls
               );

  //////////
  //
  void computes( const VarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               );

  //////////
  //
  void computes( const VarLabel        *
               , const PatchSubset     * patches
               ,       PatchDomainSpec   patches_domain
               );

  //////////
  //
  void computes( const VarLabel           *
               , const Level              * level
               , const MaterialSubset     * matls        = nullptr
               ,       MaterialDomainSpec   matls_domain = NormalDomain);

  //////////
  /*! \brief Allows a task to do a computes and modify with ghost cell specification.
   *
   *  \warning Uintah was built around the assumption that one is NOT allowed
      to compute or modify ghost cells. Therefore, it is unlawful in the Uintah sense
      to add a computes/modifies with ghost cells. However, certain components such
      as Wasatch break that design assumption from the point of view that,
      if a task can fill-in ghost values, then by all means do that and
      avoid an extra communication in the process. This, for example, is the
      case when one extrapolates data from the interior (e.g. Dynamic Smagorinsky
      model). Be aware that the ghost-values computed/modified in one patch will
      NOT be reproduced/correspond to interior cells of the neighboring patch,
      and vice versa.

      Another component which breaks this assumption is working with GPU tasks.
      Here it is not efficient to attempt to enlarge and copy variables within the
      GPU to make room for requires ghost cells.  Instead it is better to simply
      provide that extra room early when it's declared as a compute.  Then when it
      becomes a requires, no costly enlarging step is necessary.
   */
  void modifiesWithScratchGhost( const VarLabel           *
                               , const PatchSubset        * patches
                               ,       PatchDomainSpec      patches_domain
                               , const MaterialSubset     * matls
                               ,       MaterialDomainSpec   matls_domain
                               ,       Ghost::GhostType     gtype
                               ,       int                  numGhostCells
                               ,       bool                 oldTG = false
                               );

  void computesWithScratchGhost( const VarLabel           *
                               , const MaterialSubset     * matls
                               ,       MaterialDomainSpec   matls_domain
                               ,       Ghost::GhostType     gtype
                               ,       int                  numGhostCells
                               ,       bool                 oldTG = false
                               );

  //////////
  // Most general case
  void modifies( const VarLabel           *
               , const PatchSubset        * patches
               ,       PatchDomainSpec      patches_domain
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               ,       bool                 oldTG = false
               );

  //////////
  //
  void modifies(const VarLabel*,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                bool oldTG = false);

  //////////
  //
  void modifies( const VarLabel       *
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  //
  void modifies( const VarLabel       *
               , const MaterialSubset * matls
               , MaterialDomainSpec     matls_domain
               , bool                   oldTG = false
               );

  //////////
  //
  void modifies( const VarLabel *
               ,       bool     oldTG = false
               );

  //////////
  // Modify reduction vars
  void modifies( const VarLabel*
               , const Level* level
               , const MaterialSubset* matls = nullptr
               , MaterialDomainSpec matls_domain = NormalDomain
               , bool oldTG = false);

  //////////
  // Tells the task to actually execute the function assigned to it.
  virtual void doit(       DetailedTask                * task
                   ,       CallBackEvent                 event
                   , const ProcessorGroup              * pg
                   , const PatchSubset                 *
                   , const MaterialSubset              *
                   ,       std::vector<DataWarehouseP> & dws
                   ,       void                        * oldTaskGpuDW
                   ,       void                        * newTaskGpuDW
                   ,       void                        * stream
                   ,       int                           deviceID
                   );

  inline const std::string & getName() const { return m_task_name; }

  inline const PatchSet* getPatchSet() const { return m_patch_set; }

  inline const MaterialSet * getMaterialSet() const { return m_matl_set; }

  int m_phase;                    // synchronized phase id, for dynamic task scheduling
  int m_comm;                     // task communicator id, for threaded task scheduling
  int m_max_ghost_cells;          // max ghost cells of this task
  int m_max_fine_ghost_cells;     // max ghost cells of this task
  int m_max_level_offset;         // max level offset of this task

  std::set<Task*> m_child_tasks;
  std::set<Task*> m_all_child_tasks;

  enum DepType {
      Modifies
    , Computes
    , Requires
  };

  struct Edge;

  struct Dependency {
      Dependency           * m_next;
      DepType                m_dep_type;
      Task                 * m_task;
      const VarLabel       * m_var;
      bool                   m_look_in_old_tg;
      const PatchSubset    * m_patches;
      const MaterialSubset * m_matls;
      const Level          * m_reduction_level;
      Edge                 * m_req_head;   // Used in compiling the task graph.
      Edge                 * m_req_tail;
      Edge                 * m_comp_head;
      Edge                 * m_comp_tail;
      PatchDomainSpec        m_patches_dom;
      MaterialDomainSpec     m_matls_dom;
      Ghost::GhostType       m_gtype;
      WhichDW                m_whichdw;  // Used only by Requires

      // in the multi-TG construct, this will signify that the required
      // m_var will be constructed by the old TG
      int m_num_ghost_cells;
      int m_level_offset;

      int mapDataWarehouse() const { return m_task->mapDataWarehouse(m_whichdw); }

      Dependency(       DepType              deptype
                ,       Task               * task
                ,       WhichDW              dw
                , const VarLabel           * var
                ,       bool                 oldtg
                , const PatchSubset        * patches
                , const MaterialSubset     * matls
                ,       PatchDomainSpec      patches_dom   = ThisLevel
                ,       MaterialDomainSpec   matls_dom     = NormalDomain
                ,       Ghost::GhostType     gtype         = Ghost::None
                ,       int                  numGhostCells = 0
                ,       int                  level_offset  = 0
                );

      Dependency(       DepType              deptype
                ,       Task               * task
                ,       WhichDW              dw
                , const VarLabel           * var
                ,       bool                 oldtg
                , const Level              * reductionLevel
                , const MaterialSubset     * matls
                ,       MaterialDomainSpec   matls_dom = NormalDomain
                );

      ~Dependency();

      inline void addComp( Edge * edge );

      inline void addReq( Edge * edge );

      constHandle<PatchSubset>
      getPatchesUnderDomain( const PatchSubset * domainPatches ) const;

      constHandle<MaterialSubset>
      getMaterialsUnderDomain( const MaterialSubset * domainMaterials ) const;


  private:  // struct Dependency

      // eliminate copy, assignment and move
      Dependency( const Dependency & )            = delete;
      Dependency& operator=( const Dependency & ) = delete;
      Dependency( Dependency && )                 = delete;
      Dependency& operator=( Dependency && )      = delete;

      static constHandle<PatchSubset>

      getOtherLevelPatchSubset(       PatchDomainSpec   dom
                              ,       int               level_offset
                              , const PatchSubset     * subset
                              , const PatchSubset     * domainSubset
                              ,       int               ngc
                              );
  };  // end struct Dependency

  struct Edge {
      const Dependency * m_comp;
      Edge             * m_comp_next;
      const Dependency * m_req;
      Edge             * m_req_next;

      inline Edge( const Dependency * comp , const Dependency * req )
        : m_comp(comp)
        , m_comp_next(nullptr)
        , m_req(req)
        , m_req_next(nullptr)
      {}
  };

  typedef std::multimap<const VarLabel*, Dependency*, VarLabel::Compare> DepMap;

  const Dependency* getComputes() const { return m_comp_head; }

  const Dependency* getRequires() const { return m_req_head; }

  const Dependency* getModifies() const { return m_mod_head; }

  Dependency* getComputes() { return m_comp_head; }

  Dependency* getRequires() { return m_req_head; }

  Dependency* getModifies() { return m_mod_head; }

  // finds if it computes or modifies var
  bool hasComputes( const VarLabel * var
                  ,       int        matlIndex
                  , const Patch    * patch
                  ) const;

  // finds if it requires or modifies var
  bool hasRequires( const VarLabel        * var
                  ,       int               matlIndex
                  , const Patch           * patch
                  ,       Uintah::IntVector lowOffset
                  ,       Uintah::IntVector highOffset
                  ,       WhichDW dw
                  ) const;

  // finds if it modifies var
  bool hasModifies(const VarLabel* var,
                   int matlIndex,
                   const Patch* patch) const;

  bool isReductionTask() const { return d_tasktype == Reduction; }

  void setType(TaskType tasktype) { d_tasktype = tasktype; }

  TaskType getType() const { return d_tasktype; }

  //////////
  // Prints out information about the task...
  void display( std::ostream & out ) const;

  //////////
  // Prints out all information about the task, including dependencies
  void displayAll( std::ostream & out ) const;

  int mapDataWarehouse( WhichDW dw ) const;

  DataWarehouse* mapDataWarehouse( WhichDW dw, std::vector<DataWarehouseP> & dws ) const;

  int getSortedOrder() const { return m_sorted_order; }

  void setSortedOrder(int order) { m_sorted_order = order; }

  void setMapping(int dwmap[TotalDWs]);

  void setSets( const PatchSet * patches, const MaterialSet * matls );


private: // class Task

    Dependency* isInDepMap( const DepMap   & depMap
                          , const VarLabel * var
                          ,       int        matlIndex
                          , const Patch    * patch
                          ) const;

    std::string m_task_name;


protected: // class Task

    ActionBase* m_action;


private: // class Task

  // eliminate copy, assignment and move
  Task( const Task & )            = delete;
  Task& operator=( const Task & ) = delete;
  Task( Task && )                 = delete;
  Task& operator=( Task && )      = delete;


  static const MaterialSubset* getGlobalMatlSubset();

  static MaterialSubset* globalMatlSubset;

  Dependency * m_comp_head;
  Dependency * m_comp_tail;
  Dependency * m_req_head;
  Dependency * m_req_tail;
  Dependency * m_mod_head;
  Dependency * m_mod_tail;

  DepMap       m_requires_old_dw;
  DepMap       m_computes;  // also contains modifies
  DepMap       m_requires;  // also contains modifies
  DepMap       m_modifies;

  const PatchSet    * m_patch_set;
  const MaterialSet * m_matl_set;

  bool m_uses_mpi;
  bool m_uses_threads;
  bool m_uses_device;
  bool m_subpatch_capable;
  bool m_has_subscheduler;

  TaskType d_tasktype;

  int m_dwmap[TotalDWs];
  int m_sorted_order;

  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task & task);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task::TaskType & tt);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task::Dependency & dep);

}; // end class Task


inline void Task::Dependency::addComp( Edge * edge )
{
  if (m_comp_tail) {
    m_comp_tail->m_comp_next = edge;
  }
  else {
    m_comp_head = edge;
  }
  m_comp_tail = edge;
}

inline void Task::Dependency::addReq( Edge * edge )
{
  if (m_req_tail) {
    m_req_tail->m_req_next = edge;
  }
  else {
    m_req_head = edge;
  }
  m_req_tail = edge;
}

}  // End namespace Uintah

// This must be at the bottom
#include <CCA/Ports/DataWarehouse.h>

#endif // CORE_GRID_TASK_H
