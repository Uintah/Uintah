/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef CORE_GRID_UNSTRUCTURED_TASK_H
#define CORE_GRID_UNSTRUCTURED_TASK_H

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/UnstructuredVarLabel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Ports/UnstructuredDataWarehouseP.h>
#include <Core/Util/constHandle.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/TupleHelpers.hpp>

#include <set>
#include <vector>
#include <string>
#include <iostream>

namespace Uintah {

class UnstructuredLevel;
class UnstructuredDataWarehouse;
class ProcessorGroup;
class UnstructuredTask;
class UnstructuredDetailedTask;

/**************************************

 CLASS
   UnstructuredTask

 GENERAL INFORMATION
   UnstructuredTask.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

 KEYWORDS
   UnstructuredTask

 DESCRIPTION

 ****************************************/

class UnstructuredTask {
 
public: // class UnstructuredTask

  enum CallBackEvent
  {
      CPU      // <- normal CPU task, happens when a GPU enabled task runs on CPU
    , preGPU   // <- pre GPU kernel callback, happens before CPU->GPU copy (reserved, not implemented yet... )
    , GPU      // <- GPU kernel callback, happens after dw: CPU->GPU copy, kernel launch should be queued in this callback
    , postGPU  // <- post GPU kernel callback, happens after dw: GPU->CPU copy but before MPI sends.
  };
 

protected: // class UnstructuredTask

  // base Action class
  class ActionBase {

    public:

      virtual ~ActionBase(){};

      virtual void doit(       UnstructuredDetailedTask   * task
                       ,       CallBackEvent    event
                       , const ProcessorGroup * pg
                       , const UnstructuredPatchSubset    * patches
                       , const MaterialSubset * matls
                       ,       UnstructuredDataWarehouse  * fromDW
                       ,       UnstructuredDataWarehouse  * toDW
                       ,       void           * oldUnstructuredTaskGpuDW
                       ,       void           * newUnstructuredTaskGpuDW
                       ,       void           * stream
                       ,       int              deviceID
                       ) = 0;
  };


private: // class UnstructuredTask

  // CPU Action constructor
  template<typename T, typename... Args>
  class Action : public ActionBase {

      T * ptr;
      void (T::*pmf)( const ProcessorGroup * pg
                    , const UnstructuredPatchSubset    * patches
                    , const MaterialSubset * m_matls
                    ,       UnstructuredDataWarehouse  * fromDW
                    ,       UnstructuredDataWarehouse  * toDW
                    ,       Args...          args
                    );
      std::tuple<Args...> m_args;


  public: // class Action

    Action( T * ptr
          , void (T::*pmf)( const ProcessorGroup * pg
                          , const UnstructuredPatchSubset    * patches
                          , const MaterialSubset * m_matls
                          ,       UnstructuredDataWarehouse  * fromDW
                          ,       UnstructuredDataWarehouse  * toDW
                          ,       Args...          args
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
    virtual void doit(       UnstructuredDetailedTask   * task
                     ,       CallBackEvent    event
                     , const ProcessorGroup * pg
                     , const UnstructuredPatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       UnstructuredDataWarehouse  * fromDW
                     ,       UnstructuredDataWarehouse  * toDW
                     ,       void           * oldUnstructuredTaskGpuDW
                     ,       void           * newUnstructuredTaskGpuDW
                     ,       void           * stream
                     ,       int              deviceID
                     )
    {
      doit_impl(pg, patches, matls, fromDW, toDW, typename Tuple::gens<sizeof...(Args)>::type());
    }


  private: // class Action

    template<int... S>
    void doit_impl( const ProcessorGroup * pg
                  , const UnstructuredPatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       UnstructuredDataWarehouse  * fromDW
                  ,       UnstructuredDataWarehouse  * toDW
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
    void (T::*pmf)(       UnstructuredDetailedTask   * dtask
                  ,       CallBackEvent    event
                  , const ProcessorGroup * pg
                  , const UnstructuredPatchSubset    * patches
                  , const MaterialSubset * m_matls
                  ,       UnstructuredDataWarehouse  * fromDW
                  ,       UnstructuredDataWarehouse  * toDW
                  ,       void           * oldUnstructuredTaskGpuDW
                  ,       void           * newUnstructuredTaskGpuDW
                  ,       void           * stream
                  ,       int              deviceID
                  ,       Args...          args
                  );
    std::tuple<Args...> m_args;


  public: // class ActionDevice

    ActionDevice( T * ptr
                , void (T::*pmf)(       UnstructuredDetailedTask   * dtask
                                ,       CallBackEvent    event
                                , const ProcessorGroup * pg
                                , const UnstructuredPatchSubset    * patches
                                , const MaterialSubset * m_matls
                                ,       UnstructuredDataWarehouse  * fromDW
                                ,       UnstructuredDataWarehouse  * toDW
                                ,       void           * oldUnstructuredTaskGpuDW
                                ,       void           * newUnstructuredTaskGpuDW
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

    virtual void doit(       UnstructuredDetailedTask   * dtask
                     ,       CallBackEvent    event
                     , const ProcessorGroup * pg
                     , const UnstructuredPatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       UnstructuredDataWarehouse  * fromDW
                     ,       UnstructuredDataWarehouse  * toDW
                     ,       void           * oldUnstructuredTaskGpuDW
                     ,       void           * newUnstructuredTaskGpuDW
                     ,       void           * stream
                     ,       int              deviceID
                     )
    {
      doit_impl(dtask, event, pg, patches, matls, fromDW, toDW, oldUnstructuredTaskGpuDW, newUnstructuredTaskGpuDW, stream, deviceID, typename Tuple::gens<sizeof...(Args)>::type());
    }

  private : // class ActionDevice

    template<int... S>
    void doit_impl(       UnstructuredDetailedTask   * dtask
                  ,       CallBackEvent    event
                  , const ProcessorGroup * pg
                  , const UnstructuredPatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       UnstructuredDataWarehouse  * fromDW
                  ,       UnstructuredDataWarehouse  * toDW
                  ,       void           * oldUnstructuredTaskGpuDW
                  ,       void           * newUnstructuredTaskGpuDW
                  ,       void           * stream
                  ,       int              deviceID
                  ,       Tuple::seq<S...>
                  )
      {
        (ptr->*pmf)(dtask, event, pg, patches, matls, fromDW, toDW, oldUnstructuredTaskGpuDW, newUnstructuredTaskGpuDW, stream, deviceID, std::get<S>(m_args)...);
      }

  };  // end GPU (device) Action constructor



public: // class UnstructuredTask

  enum WhichDW {
     None        = -1
  ,  OldDW       = 0
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

  enum UnstructuredTaskType {
      Normal
    , Reduction
    , InitialSend
    , OncePerProc // make sure to pass a PerProcessor UnstructuredPatchSet to the addUnstructuredTask function
    , Output
    , Spatial     // e.g. Radiometer task (spatial scheduling); must call task->setType(UnstructuredTask::Spatial)
    , Hypre       // previously identified as a OncePerProc
  };


  UnstructuredTask( const std::string & taskName, UnstructuredTaskType type )
    : m_task_name(taskName)
    , m_action(nullptr)
  {
    d_tasktype = type;
    initialize();
  }

  // CPU UnstructuredTask constructor
  template<typename T, typename... Args>
  UnstructuredTask( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup *
                      , const UnstructuredPatchSubset    *
                      , const MaterialSubset *
                      ,       UnstructuredDataWarehouse  *
                      ,       UnstructuredDataWarehouse  *
                      ,       Args...
                      )
       , Args... args
      )
     : m_task_name(taskName)
     , m_action(scinew Action<T, Args...>(ptr, pmf, std::forward<Args>(args)...))
  {
    d_tasktype = Normal;
    initialize();
  }

  // Device (GPU) UnstructuredTask constructor
  template<typename T, typename... Args>
  UnstructuredTask( const std::string & taskName
      , T * ptr
      , void (T::*pmf)(       UnstructuredDetailedTask   * m_task
                      ,       CallBackEvent    event
                      , const ProcessorGroup * pg
                      , const UnstructuredPatchSubset    * patches
                      , const MaterialSubset * m_matls
                      ,       UnstructuredDataWarehouse  * fromDW
                      ,       UnstructuredDataWarehouse  * toDW
                      ,       void           * old_UnstructuredTaskGpuDW
                      ,       void           * new_UnstructuredTaskGpuDW
                      ,       void           * stream
                      ,       int              deviceID
                      ,       Args...          args
                      )
      , Args... args
      )
      : m_task_name(taskName)
      , m_action(scinew ActionDevice<T, Args...>(ptr, pmf, std::forward<Args>(args)...))
  {
    initialize();
    d_tasktype = Normal;
  }

  void initialize();

  virtual ~UnstructuredTask();

         void hasSubScheduler(bool state = true);
  inline bool getHasSubScheduler() const { return m_has_subscheduler; }

         void usesMPI(bool state);
  inline bool usesMPI() const { return m_uses_mpi; }

         void usesThreads(bool state);
  inline bool usesThreads() const { return m_uses_threads; }

         void usesDevice(bool state, int maxStreamsPerUnstructuredTask = 1);
  inline bool usesDevice() const { return m_uses_device; }
  inline int  maxStreamsPerUnstructuredTask() const { return  m_max_streams_per_task; }

  enum MaterialDomainSpec {
      NormalDomain  // <- Normal/default setting
    , OutOfDomain   // <- Require things from all material
  };

  enum UnstructuredPatchDomainSpec {
      ThisLevel        // <- Normal/default setting
    , CoarseLevel      // <- AMR :  The data on the coarse level under the range of the fine patches (including extra cells or boundary layers)
    , FineLevel        // <- AMR :  The data on the fine level over the range of the coarse patches (including extra cells or boundary layers)
    , OtherGridDomain  // for when we copy data to new grid after a regrid.
  };

  //////////
  // Most general case
  void requires( WhichDW
               , const UnstructuredVarLabel       *
               , const UnstructuredPatchSubset    * patches
               , UnstructuredPatchDomainSpec        patches_dom
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
               , const UnstructuredVarLabel           *
               , const UnstructuredPatchSubset        * patches
               ,       UnstructuredPatchDomainSpec      patches_dom
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_dom
               ,       Ghost::GhostType     gtype
               ,       int                  numGhostCells = 0
               ,       bool                 oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const UnstructuredVarLabel         *
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const UnstructuredVarLabel         *
               , const UnstructuredPatchSubset      * patches
               , const MaterialSubset   * matls
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires( WhichDW
               , const UnstructuredVarLabel         *
               , const UnstructuredPatchSubset      * patches
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const UnstructuredVarLabel        *
               , const MaterialSubset  * matls
               ,      Ghost::GhostType   gtype
               ,      int                numGhostCells = 0
               ,      bool               oldTG         = false);

  //////////
  //
  void requires(       WhichDW
               , const UnstructuredVarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_dom
               ,       Ghost::GhostType     gtype
               ,       int                  numGhostCells = 0
               ,       bool                 oldTG         = false
               );

  //////////
  // Requires only for Reduction variables
  void requires(       WhichDW
               , const UnstructuredVarLabel           *
               , const UnstructuredLevel              * level     = nullptr
               , const MaterialSubset     * matls     = nullptr
               ,       MaterialDomainSpec   matls_dom = NormalDomain
               ,       bool                 oldTG     = false
               );

  //////////
  // Requires for reduction variables or PerPatch variables
  void requires(       WhichDW
               , const UnstructuredVarLabel       *
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  // Requires only for PerPatch variables
  void requires(       WhichDW
               , const UnstructuredVarLabel       *
               , const UnstructuredPatchSubset    * patches
               , const MaterialSubset * matls = nullptr
               );

  //////////
  // Most general case
  void computes( const UnstructuredVarLabel           *
               , const UnstructuredPatchSubset        * patches
               ,       UnstructuredPatchDomainSpec      patches_domain
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               );

  //////////
  //
  void computes( const UnstructuredVarLabel       *
               , const UnstructuredPatchSubset    * patches = nullptr
               , const MaterialSubset * matls   = nullptr
               );

  //////////
  //
  void computes( const UnstructuredVarLabel       *
               , const MaterialSubset * matls
               );

  //////////
  //
  void computes( const UnstructuredVarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               );

  //////////
  //
  void computes( const UnstructuredVarLabel        *
               , const UnstructuredPatchSubset     * patches
               ,       UnstructuredPatchDomainSpec   patches_domain
               );

  //////////
  //
  void computes( const UnstructuredVarLabel           *
               , const UnstructuredLevel              * level
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
  void modifiesWithScratchGhost( const UnstructuredVarLabel           *
                               , const UnstructuredPatchSubset        * patches
                               ,       UnstructuredPatchDomainSpec      patches_domain
                               , const MaterialSubset     * matls
                               ,       MaterialDomainSpec   matls_domain
                               ,       Ghost::GhostType     gtype
                               ,       int                  numGhostCells
                               ,       bool                 oldTG = false
                               );

  void computesWithScratchGhost( const UnstructuredVarLabel           *
                               , const MaterialSubset     * matls
                               ,       MaterialDomainSpec   matls_domain
                               ,       Ghost::GhostType     gtype
                               ,       int                  numGhostCells
                               ,       bool                 oldTG = false
                               );

  //////////
  // Most general case
  void modifies( const UnstructuredVarLabel           *
               , const UnstructuredPatchSubset        * patches
               ,       UnstructuredPatchDomainSpec      patches_domain
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               ,       bool                 oldTG = false
               );

  //////////
  //
  void modifies( const UnstructuredVarLabel       *
               , const UnstructuredPatchSubset    * patches
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  //
  void modifies( const UnstructuredVarLabel       *
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  //
  void modifies( const UnstructuredVarLabel       *
               , const MaterialSubset * matls
               , MaterialDomainSpec     matls_domain
               , bool                   oldTG = false
               );

  //////////
  //
  void modifies( const UnstructuredVarLabel *
               ,       bool     oldTG = false
               );

  //////////
  // Modify reduction vars
  void modifies( const UnstructuredVarLabel*
               , const UnstructuredLevel* level
               , const MaterialSubset* matls = nullptr
               , MaterialDomainSpec matls_domain = NormalDomain
               , bool oldTG = false);

  //////////
  // Tells the task to actually execute the function assigned to it.
  virtual void doit(       UnstructuredDetailedTask                * task
                   ,       CallBackEvent                 event
                   , const ProcessorGroup              * pg
                   , const UnstructuredPatchSubset                 *
                   , const MaterialSubset              *
                   ,       std::vector<UnstructuredDataWarehouseP> & dws
                   ,       void                        * oldUnstructuredTaskGpuDW
                   ,       void                        * newUnstructuredTaskGpuDW
                   ,       void                        * stream
                   ,       int                           deviceID
                   );

  inline const std::string & getName() const { return m_task_name; }

  inline const UnstructuredPatchSet * getPatchSet() const { return m_patch_set; }

  inline const MaterialSet * getMaterialSet() const { return m_matl_set; }

  int m_phase{-1};                        // synchronized phase id, for dynamic task scheduling
  int m_comm{-1};                         // task communicator id, for threaded task scheduling
  std::map<int,int> m_max_ghost_cells;    // max ghost cells of this task
  int m_max_level_offset{0};              // max level offset of this task

  enum DepType {
      Modifies
    , Computes
    , Requires
  };


  struct Dependency {

      Dependency           * m_next{nullptr};
      DepType                m_dep_type;
      UnstructuredTask                 * m_task{nullptr};
      const UnstructuredVarLabel       * m_var{nullptr};
      bool                   m_look_in_old_tg;
      const UnstructuredPatchSubset    * m_patches{nullptr};
      const MaterialSubset * m_matls{nullptr};
      const UnstructuredLevel          * m_reduction_level{nullptr};
      UnstructuredPatchDomainSpec        m_patches_dom{ThisLevel};
      MaterialDomainSpec     m_matls_dom;
      Ghost::GhostType       m_gtype{Ghost::None};
      WhichDW                m_whichdw;  // Used only by requires


      // in the multi-TG construct, this will signify that the required
      // m_var will be constructed by the old TG
      int m_num_ghost_cells{0};
      int m_level_offset{0};

      int mapDataWarehouse() const { return m_task->mapDataWarehouse(m_whichdw); }

      Dependency(       DepType              deptype
                ,       UnstructuredTask               * task
                ,       WhichDW              dw
                , const UnstructuredVarLabel           * var
                ,       bool                 oldtg
                , const UnstructuredPatchSubset        * patches
                , const MaterialSubset     * matls
                ,       UnstructuredPatchDomainSpec      patches_dom   = ThisLevel
                ,       MaterialDomainSpec   matls_dom     = NormalDomain
                ,       Ghost::GhostType     gtype         = Ghost::None
                ,       int                  numGhostCells = 0
                ,       int                  level_offset  = 0
                );

      Dependency(       DepType              deptype
                ,       UnstructuredTask               * task
                ,       WhichDW              dw
                , const UnstructuredVarLabel           * var
                ,       bool                 oldtg
                , const UnstructuredLevel              * reductionLevel
                , const MaterialSubset     * matls
                ,       MaterialDomainSpec   matls_dom = NormalDomain
                );

      ~Dependency();

      constHandle<UnstructuredPatchSubset>
      getPatchesUnderDomain( const UnstructuredPatchSubset * domainPatches ) const;

      constHandle<MaterialSubset>
      getMaterialsUnderDomain( const MaterialSubset * domainMaterials ) const;


  private:  // struct Dependency

      // eliminate copy, assignment and move
      Dependency( const Dependency & )            = delete;
      Dependency& operator=( const Dependency & ) = delete;
      Dependency( Dependency && )                 = delete;
      Dependency& operator=( Dependency && )      = delete;

      static constHandle<UnstructuredPatchSubset>

      getOtherLevelPatchSubset(       UnstructuredPatchDomainSpec   dom
                              ,       int               level_offset
                              , const UnstructuredPatchSubset     * subset
                              , const UnstructuredPatchSubset     * domainSubset
                              ,       int               ngc
                              );
  };  // end struct Dependency



  const Dependency* getComputes() const { return m_comp_head; }

  const Dependency* getRequires() const { return m_req_head; }

  const Dependency* getModifies() const { return m_mod_head; }

  Dependency* getComputes() { return m_comp_head; }

  Dependency* getRequires() { return m_req_head; }

  Dependency* getModifies() { return m_mod_head; }

  // finds if it computes or modifies var
  bool hasComputes( const UnstructuredVarLabel * var
                  ,       int        matlIndex
                  , const UnstructuredPatch    * patch
                  ) const;

  // finds if it requires or modifies var
  bool hasRequires( const UnstructuredVarLabel          * var
                  ,       int                 matlIndex
                  , const UnstructuredPatch             * patch
                  ,       Uintah::IntVector   lowOffset
                  ,       Uintah::IntVector   highOffset
                  ,       WhichDW             dw
                  ) const;

  // finds if it modifies var
  bool hasModifies( const UnstructuredVarLabel * var
                  ,       int        matlIndex
                  , const UnstructuredPatch    * patch
                  ) const;

  bool isReductionUnstructuredTask() const { return d_tasktype == Reduction; }

  void setType(UnstructuredTaskType tasktype) { d_tasktype = tasktype; }

  UnstructuredTaskType getType() const { return d_tasktype; }

  //////////
  // Prints out information about the task...
  void display( std::ostream & out ) const;

  //////////
  // Prints out all information about the task, including dependencies
  void displayAll_DOUT( Uintah::Dout& dbg) const;
  
  void displayAll( std::ostream & out ) const;

  int mapDataWarehouse( WhichDW dw ) const;

  UnstructuredDataWarehouse* mapDataWarehouse( WhichDW dw, std::vector<UnstructuredDataWarehouseP> & dws ) const;

  int getSortedOrder() const { return m_sorted_order; }

  void setSortedOrder(int order) { m_sorted_order = order; }

  void setMapping(int dwmap[TotalDWs]);

  void setSets( const UnstructuredPatchSet * patches, const MaterialSet * matls );


private: // class UnstructuredTask

  using DepMap = std::multimap<const UnstructuredVarLabel*, Dependency*, UnstructuredVarLabel::Compare>;

  Dependency* isInDepMap( const DepMap   & depMap
                        , const UnstructuredVarLabel * var
                        ,       int        matlIndex
                        , const UnstructuredPatch    * patch
                        ) const;

  std::string m_task_name;

protected: // class UnstructuredTask

  ActionBase * m_action{nullptr};

  // eliminate copy, assignment and move
  UnstructuredTask( const UnstructuredTask & )            = delete;
  UnstructuredTask& operator=( const UnstructuredTask & ) = delete;
  UnstructuredTask( UnstructuredTask && )                 = delete;
  UnstructuredTask& operator=( UnstructuredTask && )      = delete;


  static const MaterialSubset* getGlobalMatlSubset();

  static MaterialSubset* globalMatlSubset;

  Dependency * m_comp_head{nullptr};
  Dependency * m_comp_tail{nullptr};
  Dependency * m_req_head{nullptr};
  Dependency * m_req_tail{nullptr};
  Dependency * m_mod_head{nullptr};
  Dependency * m_mod_tail{nullptr};

  DepMap       m_requires_old_dw;
  DepMap       m_computes;  // also contains modifies
  DepMap       m_requires;  // also contains modifies
  DepMap       m_modifies;

  const UnstructuredPatchSet    * m_patch_set{nullptr};
  const MaterialSet * m_matl_set{nullptr};

  bool m_uses_mpi{false};
  bool m_uses_threads{false};
  bool m_uses_device{false};
  int  m_max_streams_per_task{1};
  bool m_subpatch_capable{false};
  bool m_has_subscheduler{false};

  UnstructuredTaskType d_tasktype;

  int m_dwmap[TotalDWs];
  int m_sorted_order{-1};

  friend std::ostream & operator <<(std::ostream & out, const Uintah::UnstructuredTask & task);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::UnstructuredTask::UnstructuredTaskType & tt);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::UnstructuredTask::Dependency & dep);

}; // end class UnstructuredTask

}  // End namespace Uintah

// This must be at the bottom
#include <CCA/Ports/UnstructuredDataWarehouse.h>

#endif // CORE_GRID_TASK_H
