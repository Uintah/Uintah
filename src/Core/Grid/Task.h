/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#include <CCA/Ports/DataWarehouseP.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ExecutionObject.h>
#include <Core/Parallel/LoopExecution.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/SpaceTypes.h>
#include <Core/Parallel/UintahParams.h>
#include <Core/Util/constHandle.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/TupleHelpers.hpp>

#include <sci_defs/gpu_defs.h>

#if defined(KOKKOS_USING_GPU)
  #include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#endif

#include <set>
#include <vector>
#include <string>
#include <iostream>

namespace {
  Uintah::MasterLock kokkosInstances_mutex{};
}

namespace Uintah {

class Level;
class DataWarehouse;
class OnDemandDataWarehouse;
class ProcessorGroup;
class Task;

enum GPUMemcpyKind { GPUMemcpyUnknown      = 0,
                     GPUMemcpyHostToDevice = 1,
                     GPUMemcpyDeviceToHost = 2,
};

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

public: // class Task


protected: // class Task

  // base Action class
  class ActionBase {

    public:

      ActionBase() {};
      ActionBase(Task *ptr) : taskPtr(ptr) {};
      virtual ~ActionBase() {};

      virtual void doit( const PatchSubset    * patches
                       , const MaterialSubset * matls
                       ,       DataWarehouse  * fromDW
                       ,       DataWarehouse  * toDW
                       ,       UintahParams   & uintahParams
                       ) = 0;

#if defined(KOKKOS_USING_GPU)
    virtual void assignDevicesAndInstances(intptr_t dTask) = 0;

    virtual void assignDevicesAndInstances(intptr_t dTask,
                                           unsigned int deviceNum) = 0;

    virtual bool haveKokkosInstanceForThisTask(intptr_t dTask,
                                               unsigned int deviceNum ) const = 0;
    virtual void clearKokkosInstancesForThisTask(intptr_t dTask) = 0;

    virtual bool checkKokkosInstanceDoneForThisTask(intptr_t dTask,
                                                    unsigned int deviceNum ) const = 0;

    virtual bool checkAllKokkosInstancesDoneForThisTask(intptr_t dTask) const = 0;

    virtual void doKokkosDeepCopy( intptr_t dTask, unsigned int deviceNum,
                                   void* dst, void* src,
                                   size_t count, GPUMemcpyKind kind) = 0;

    virtual void doKokkosMemcpyPeerAsync( intptr_t dTask, unsigned int deviceNum,
                                                void* dst, int dstDevice,
                                          const void* src, int srcDevice,
                                          size_t count ) = 0;

    virtual void copyGpuGhostCellsToGpuVars(intptr_t dTask,
                                            unsigned int deviceNum,
                                            GPUDataWarehouse *taskgpudw) = 0;

    virtual void syncTaskGpuDW(intptr_t dTask,
                               unsigned int deviceNum,
                               GPUDataWarehouse *taskgpudw) = 0;
#endif  // defined(KOKKOS_USING_GPU)

    protected:
      Task *taskPtr {nullptr};
  };

public:

  // CPU nonportable Action constructor
  class ActionNonPortableBase : public ActionBase {

  public:

      ActionNonPortableBase() {};
      ActionNonPortableBase(Task *ptr) : ActionBase(ptr) {};
      virtual ~ActionNonPortableBase() {};

#if defined(KOKKOS_USING_GPU)
    typedef          std::map<unsigned int, Kokkos::DefaultExecutionSpace> kokkosInstanceMap;
    typedef typename kokkosInstanceMap::const_iterator kokkosInstanceMapIter;

    virtual void assignDevicesAndInstances(intptr_t dTask);

    virtual void assignDevicesAndInstances(intptr_t dTask,
                                           unsigned int deviceNum);

    virtual void setKokkosInstanceForThisTask(intptr_t dTask,
                                              unsigned int deviceNum);

    virtual bool haveKokkosInstanceForThisTask(intptr_t dTask,
                                               unsigned int deviceNum ) const;

    virtual Kokkos::DefaultExecutionSpace
    getKokkosInstanceForThisTask(intptr_t dTask,
                                 unsigned int deviceNum ) const;

    virtual void clearKokkosInstancesForThisTask(intptr_t dTask);

    virtual bool checkKokkosInstanceDoneForThisTask(intptr_t dTask,
                                                    unsigned int deviceNum ) const;

    virtual bool checkAllKokkosInstancesDoneForThisTask(intptr_t dTask) const;

    virtual void doKokkosDeepCopy( intptr_t dTask, unsigned int deviceNum,
                                   void* dst, void* src,
                                   size_t count, GPUMemcpyKind kind);

    virtual void doKokkosMemcpyPeerAsync( intptr_t dTask, unsigned int deviceNum,
                                                void* dst, int dstDevice,
                                          const void* src, int srcDevice,
                                          size_t count );

    virtual void copyGpuGhostCellsToGpuVars(intptr_t dTask,
                                            unsigned int deviceNum,
                                            GPUDataWarehouse *taskgpudw);

    virtual void syncTaskGpuDW(intptr_t dTask,
                               unsigned int deviceNum,
                               GPUDataWarehouse *taskgpudw);
  protected:
    // The task is pointed to by multiple DetailedTasks. As such, the
    // task has to keep track of the device and instance on a DetailedTask
    // basis. The DetailedTask's pointer address is used as the key.
    std::map<intptr_t, kokkosInstanceMap> m_kokkosInstances;
#endif  // defined(KOKKOS_USING_GPU)
  };

  // CPU Action constructor
  template<typename T, typename... Args>
  class ActionNonPortable : public ActionNonPortableBase {

      T * ptr;
      void (T::*pmf)( const ProcessorGroup * pg
                    , const PatchSubset    * patches
                    , const MaterialSubset * m_matls
                    ,       DataWarehouse  * fromDW
                    ,       DataWarehouse  * toDW
                    ,       Args...          args
                    );
      std::tuple<Args...> m_args;


  public: // class Action

    ActionNonPortable( Task * taskPtr
          , T * ptr
          , void (T::*pmf)( const ProcessorGroup * pg
                          , const PatchSubset    * patches
                          , const MaterialSubset * m_matls
                          ,       DataWarehouse  * fromDW
                          ,       DataWarehouse  * toDW
                          ,       Args...          args
                          )
          , Args... args
          )
      : ActionNonPortableBase(taskPtr)
      , ptr(ptr)
      , pmf(pmf)
      , m_args(std::forward<Args>(args)...)
    {}

    virtual ~ActionNonPortable() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      doit_impl(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW, typename Tuple::gens<sizeof...(Args)>::type());
    }

  private: // class Action

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

  };  // end CPU ActionNonPortable class


  // Kokkos enabled task portable Action constructor
  template<typename ExecSpace, typename MemSpace>
  class ActionPortableBase : public ActionBase {

  public:

      ActionPortableBase() {};
      ActionPortableBase(Task *ptr) : ActionBase(ptr) {};
      virtual ~ActionPortableBase() {};

#if defined(KOKKOS_USING_GPU)
    typedef          std::map<unsigned int, ExecSpace> kokkosInstanceMap;
    typedef typename kokkosInstanceMap::const_iterator kokkosInstanceMapIter;

    virtual void assignDevicesAndInstances(intptr_t dTask);

    virtual void assignDevicesAndInstances(intptr_t dTask,
                                           unsigned int deviceNum);

    virtual void setKokkosInstanceForThisTask(intptr_t dTask,
                                              unsigned int deviceNum);

    virtual bool haveKokkosInstanceForThisTask(intptr_t dTask,
                                               unsigned int deviceNum ) const;

    virtual ExecSpace getKokkosInstanceForThisTask(intptr_t dTask,
                                                   unsigned int deviceNum ) const;

    virtual void clearKokkosInstancesForThisTask(intptr_t dTask);

    // Need to have a method that matches the pure virtual. It calls
    // the implementation method which is templated.
    virtual bool checkKokkosInstanceDoneForThisTask(intptr_t dTask,
                                                    unsigned int deviceNum ) const
    {
      return this->checkKokkosInstanceDoneForThisTask_impl(dTask, deviceNum );
    };

    // To use enable_if with a member function there needs to be a
    // dummy template argument that is defaulted to ExecSpace which
    // is used perform the SFINAE (Substitution failure is not an error).

    // The reason is becasue there is no substitution occurring when
    // instantiating the member function because the template
    // argument ExecSpace is already known at that time.

    // If UintahSpaces::CPU return true
    template<typename ES = ExecSpace>
    typename std::enable_if<std::is_same<ES, UintahSpaces::CPU>::value, bool>::type
    checkKokkosInstanceDoneForThisTask_impl(intptr_t dTask,
                                            unsigned int deviceNum) const;

    // // If NOT UintahSpaces::CPU issue a fence
    template<typename ES = ExecSpace>
    typename std::enable_if<!std::is_same<ES, UintahSpaces::CPU>::value, bool>::type
    checkKokkosInstanceDoneForThisTask_impl(intptr_t dTask,
                                            unsigned int deviceNum) const;

    virtual bool checkAllKokkosInstancesDoneForThisTask(intptr_t dTask) const;

    // Need to have a method that matches the pure virtual. It calls
    // the implementation method which is templated.
    virtual void doKokkosDeepCopy(intptr_t dTask, unsigned int deviceNum,
                                  void* dst, void* src,
                                  size_t count, GPUMemcpyKind kind)
    {
      this->doKokkosDeepCopy_impl(dTask, deviceNum, dst, src, count, kind);
    };

    // To use enable_if with a member function there needs to be a
    // dummy template argument that is defaulted to ExecSpace which
    // is used perform the SFINAE (Substitution failure is not an error).

    // The reason is becasue there is no substitution occurring when
    // instantiating the member function because the template
    // argument ExecSpace is already known at that time.

    // If UintahSpaces::CPU or Kokkos::OpenMP do nothing
    template<typename ES = ExecSpace>
    typename std::enable_if<std::is_same<ES, UintahSpaces::CPU>::value ||
                            std::is_same<ES, Kokkos::OpenMP   >::value, void>::type
    doKokkosDeepCopy_impl(intptr_t dTask, unsigned int deviceNum,
                          void* dst, void* src,
                          size_t count, GPUMemcpyKind kind);

    // If NOT UintahSpaces::CPU and NOT Kokkos::OpenMP do Kokkos::deep_copy
    template<typename ES = ExecSpace>
    typename std::enable_if<!std::is_same<ES, UintahSpaces::CPU>::value &&
                            !std::is_same<ES, Kokkos::OpenMP   >::value, void>::type
    doKokkosDeepCopy_impl(intptr_t dTask, unsigned int deviceNum,
                          void* dst, void* src,
                          size_t count, GPUMemcpyKind kind);

    virtual void doKokkosMemcpyPeerAsync( intptr_t dTask, unsigned int deviceNum,
                                                void* dst, int dstDevice,
                                          const void* src, int srcDevice,
                                          size_t count );

    virtual void copyGpuGhostCellsToGpuVars(intptr_t dTask,
                                            unsigned int deviceNum,
                                            GPUDataWarehouse *taskgpudw);

    virtual void syncTaskGpuDW(intptr_t dTask,
                               unsigned int deviceNum,
                               GPUDataWarehouse *taskgpudw);

  protected:
    // The task is pointed to by multiple DetailedTasks. As such, the
    // task has to keep track of the Kokkos intance on a DetailedTask
    // basis. The DetailedTask's pointer address is used as the key.
    std::map<intptr_t, kokkosInstanceMap> m_kokkosInstances;
#endif  // defined(KOKKOS_USING_GPU)
  };

  template<typename T, typename ExecSpace, typename MemSpace, typename... Args>
  class ActionPortable : public ActionPortableBase<ExecSpace, MemSpace> {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * m_matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Args...                                args
                  );
    std::tuple<Args...> m_args;

  public: // class ActionPortable

    ActionPortable( Task * taskPtr
                  , T * ptr
                  , void (T::*pmf)( const PatchSubset                          * patches
                                  , const MaterialSubset                       * m_matls
                                  ,       OnDemandDataWarehouse                * fromDW
                                  ,       OnDemandDataWarehouse                * toDW
                                  ,       UintahParams                         & uintahParams
                                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                  ,       Args...                                args
                                  )
                  , Args... args
                  )
      : ActionPortableBase<ExecSpace, MemSpace>(taskPtr)
      , ptr(ptr)
      , pmf(pmf)
      , m_args(std::forward<Args>(args)...)
    {}

    virtual ~ActionPortable() {}

    void doit( const PatchSubset    * patches
             , const MaterialSubset * matls
             ,       DataWarehouse  * fromDW
             ,       DataWarehouse  * toDW
             ,       UintahParams   & uintahParams
             )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

#if defined(KOKKOS_USING_GPU)
      const int nInstances = this->taskPtr->maxInstancesPerTask();
      for (int i = 0; i < nInstances; i++) {
        ExecSpace instance = this->getKokkosInstanceForThisTask(uintahParams.getTaskIntPtr(), i);
        execObj.setInstance(instance, 0);
      }
#endif

      doit_impl(patches, matls, fromDW, toDW, uintahParams, execObj, typename Tuple::gens<sizeof...(Args)>::type());
    }

  private : // class ActionPortable

    template<int... S>
    void doit_impl( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       DataWarehouse                        * fromDW
                  ,       DataWarehouse                        * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Tuple::seq<S...>
                  )
      {
        (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj, std::get<S>(m_args)...);
      }

  };  // end Kokkos enabled task ActionPortable constructor

public: // class Task

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

  enum TaskType {
      Normal
    , Reduction        // tasks with MPI reductions
    , InitialSend
    , OncePerProc      // make sure to pass a PerProcessor PatchSet to the
                       // addTask function
    , Output
    , OutputGlobalVars // task the outputs the reduction variables
    , Spatial          // e.g. Radiometer task (spatial scheduling); must
                       // call task->setType(Task::Spatial)
    , Hypre            // previously identified as a OncePerProc
  };


  // CPU ancillary task constructor. Currently used with a TaskType of
  // Reduction and InitialSend.
  Task( const std::string & taskName, TaskType type );

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
     , m_action(scinew ActionNonPortable<T, Args...>(this, ptr, pmf, std::forward<Args>(args)...))
  {
    d_tasktype = Normal;
    initialize();
  }

  // Portable Task constructor
  template<typename T, typename ExecSpace, typename MemSpace, typename... Args>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset    * patches
                      , const MaterialSubset * m_matls
                      ,       OnDemandDataWarehouse  * fromDW
                      ,       OnDemandDataWarehouse  * toDW
                      ,       UintahParams& uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace>& execObj
                      ,       Args...          args
                      )
      , Args... args
      )
      : m_task_name(taskName)
      , m_action(scinew ActionPortable<T, ExecSpace, MemSpace, Args...>(this, ptr, pmf, std::forward<Args>(args)...))
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

         void setExecutionAndMemorySpace( const TaskAssignedExecutionSpace & executionSpaceTypeName
                                        , const TaskAssignedMemorySpace    & memorySpaceTypeName
                                        );

         TaskAssignedExecutionSpace getExecutionSpace() const;
         TaskAssignedMemorySpace    getMemorySpace() const;

         void usesDevice(bool state, int maxInstancesPerTask = -1);
  inline bool usesDevice() const { return m_uses_device; }
  inline int  maxInstancesPerTask() const { return  m_max_instances_per_task; }

  inline void setDebugFlag( bool in ){m_debugFlag = in;}
  inline bool getDebugFlag()const {return m_debugFlag;}

         void usesSimVarPreloading(bool state);
  inline bool usesSimVarPreloading() const { return m_preload_sim_vars; }

  enum MaterialDomainSpec {
      NormalDomain  // <- Normal/default setting
    , OutOfDomain   // <- Require things from all material
  };

  enum PatchDomainSpec {
      ThisLevel        // <- Normal/default setting
    , CoarseLevel      // <- AMR :  The data on the coarse level under the range of the fine patches (including extra cells or boundary layers)
    , FineLevel        // <- AMR :  The data on the fine level over the range of the coarse patches (including extra cells or boundary layers)
    , OtherGridDomain  // for when we copy data to new grid after a regrid.
  };

  enum class SearchTG{
      OldTG           // <- Search the OldTG for the computes if they aren't found in NewTG
    , NewTG
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
               , SearchTG               whichTG = SearchTG::NewTG
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
               ,       SearchTG             whichTG = SearchTG::NewTG
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel         *
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       SearchTG           whichTG = SearchTG::NewTG
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel         *
               , const PatchSubset      * patches
               , const MaterialSubset   * matls
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       SearchTG           whichTG = SearchTG::NewTG
               );

  //////////
  //
  void requires( WhichDW
               , const VarLabel         *
               , const PatchSubset      * patches
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       SearchTG           whichTG = SearchTG::NewTG
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel        *
               , const MaterialSubset  * matls
               ,      Ghost::GhostType   gtype
               ,      int                numGhostCells = 0
               ,      SearchTG           whichTG = SearchTG::NewTG
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_dom
               ,       Ghost::GhostType     gtype
               ,       int                  numGhostCells = 0
               ,       SearchTG             whichTG = SearchTG::NewTG
               );

  //////////
  // Requires only for Reduction variables
  void requires(       WhichDW
               , const VarLabel           *
               , const Level              * level     = nullptr
               , const MaterialSubset     * matls     = nullptr
               ,       MaterialDomainSpec   matls_dom = NormalDomain
               ,       SearchTG             whichTG   = SearchTG::NewTG
               );

  //////////
  // Requires for reduction variables or PerPatch variables
  void requires(       WhichDW
               , const VarLabel       *
               , const MaterialSubset * matls
               ,       SearchTG        whichTG = SearchTG::NewTG
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
                               ,       SearchTG             whichTG = SearchTG::NewTG
                               );

  void computesWithScratchGhost( const VarLabel           *
                               , const MaterialSubset     * matls
                               ,       MaterialDomainSpec   matls_domain
                               ,       Ghost::GhostType     gtype
                               ,       int                  numGhostCells
                               ,       SearchTG             whichTG = SearchTG::NewTG
                               );

  //////////
  // Most general case
  void modifies( const VarLabel           *
               , const PatchSubset        * patches
               ,       PatchDomainSpec      patches_domain
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               ,       SearchTG             whichTG = SearchTG::NewTG
               );

  //////////
  //
  void modifies( const VarLabel       *
               , const PatchSubset    * patches
               , const MaterialSubset * matls
               ,       SearchTG         whichTG = SearchTG::NewTG
               );

  //////////
  //
  void modifies( const VarLabel       *
               , const MaterialSubset * matls
               ,       SearchTG         whichTG = SearchTG::NewTG
               );

  //////////
  //
  void modifies( const VarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               ,       SearchTG             whichTG = SearchTG::NewTG
               );

  //////////
  //
  void modifies( const VarLabel *
               ,       SearchTG         whichTG = SearchTG::NewTG
               );

  //////////
  // Modify reduction vars
  void modifies( const VarLabel         *
               , const Level            * level
               , const MaterialSubset   * matls = nullptr
               ,       MaterialDomainSpec matls_domain = NormalDomain
               ,       SearchTG           whichTG = SearchTG::NewTG
               );

  //////////
  // Tells the task to actually execute the function assigned to it.
  virtual void doit( const PatchSubset                 *
                   , const MaterialSubset              *
                   ,       std::vector<DataWarehouseP> & dws
                   ,       UintahParams                & uintahParams
                   );

  //////////
#if defined(KOKKOS_USING_GPU)
  typedef std::set<unsigned int>       deviceNumSet;
  typedef deviceNumSet::const_iterator deviceNumSetIter;

  // Device and Instance related calls
  void assignDevice(intptr_t dTask, unsigned int device);

  // Most tasks will only run on one device.

  // But some, such as the data archiver task or send_old_data could
  // run on multiple devices.

  // This is not a good idea.  A task should only run on one device.
  // But the capability for a task to run on multiple nodes exists.
  deviceNumSet getDeviceNums(intptr_t dTask);

  // Task instance pass through methods.
  virtual void assignDevicesAndInstances(intptr_t dTask);

  virtual void assignDevicesAndInstances(intptr_t dTask, unsigned int deviceNum);

  virtual void clearKokkosInstancesForThisTask(intptr_t dTask);

  virtual bool checkAllKokkosInstancesDoneForThisTask(intptr_t dTask) const;

  virtual void doKokkosDeepCopy( intptr_t dTask, unsigned int deviceNum,
                                 void* dst, void* src,
                                 size_t count, GPUMemcpyKind kind);

  virtual void doKokkosMemcpyPeerAsync( intptr_t dTask, unsigned int deviceNum,
                                            void* dst, int dstDevice,
                                      const void* src, int srcDevice,
                                      size_t count );

  virtual void copyGpuGhostCellsToGpuVars(intptr_t dTask,
                                          unsigned int deviceNum,
                                          GPUDataWarehouse *taskgpudw);

  virtual void syncTaskGpuDW(intptr_t dTask,
                             unsigned int deviceNum,
                             GPUDataWarehouse *taskgpudw);
#endif  // defined(KOKKOS_USING_GPU)

  inline const std::string & getName() const { return m_task_name; }

  inline const PatchSet * getPatchSet() const { return m_patch_set; }

  inline const MaterialSet * getMaterialSet() const { return m_matl_set; }

  bool hasDistalRequires() const;         // determines if this Task
                                          // has any "distal" ghost
                                          // cell requirements

  int m_phase{-1};                        // synchronized phase id,
                                          // for dynamic task
                                          // scheduling
  int m_comm{-1};                         // task communicator id, for
                                          // threaded task scheduling
  std::map<int,int> m_max_ghost_cells;    // max ghost cells of this task
  int m_max_level_offset{0};              // max level offset of this task

  // Used in compiling the task graph with topological sort.
  //   Kept around for historical and reproducability reasons - APH, 04/05/19
  std::set<Task*> m_child_tasks;
  std::set<Task*> m_all_child_tasks;

  enum DepType {
      Modifies
    , Computes
    , Requires
  };

  struct Edge;

  struct Dependency {

      Dependency           * m_next{nullptr};
      DepType                m_dep_type;
      Task                 * m_task{nullptr};
      const VarLabel       * m_var{nullptr};
      bool                   m_look_in_old_tg;
      const PatchSubset    * m_patches{nullptr};
      const MaterialSubset * m_matls{nullptr};
      const Level          * m_reduction_level{nullptr};
      PatchDomainSpec        m_patches_dom{ThisLevel};
      MaterialDomainSpec     m_matls_dom;
      Ghost::GhostType       m_gtype{Ghost::None};
      WhichDW                m_whichdw;  // Used only by requires

      // in the multi-TG construct, this will signify that the required
      // m_var will be constructed by the old TG
      int m_num_ghost_cells{0};
      int m_level_offset{0};

      int mapDataWarehouse() const { return m_task->mapDataWarehouse(m_whichdw); }

      Dependency(       DepType              deptype
                ,       Task               * task
                ,       WhichDW              dw
                , const VarLabel           * var
                ,       SearchTG             whichTG
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
                ,       SearchTG             whichTG
                , const Level              * reductionLevel
                , const MaterialSubset     * matls
                ,       MaterialDomainSpec   matls_dom = NormalDomain
                );

      ~Dependency();

      constHandle<PatchSubset>
      getPatchesUnderDomain( const PatchSubset * domainPatches ) const;

      constHandle<MaterialSubset>
      getMaterialsUnderDomain( const MaterialSubset * domainMaterials ) const;

      // Used in compiling the task graph with topological sort.
      //   Kept around for historical and reproducability reasons - APH, 04/05/19
      Edge                 * m_req_head{nullptr};
      Edge                 * m_req_tail{nullptr};
      Edge                 * m_comp_head{nullptr};
      Edge                 * m_comp_tail{nullptr};
      inline void addComp( Edge * edge );
      inline void addReq(  Edge * edge );


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


  // Used in compiling the task graph with topological sort.
  //   Kept around for historical and reproducability reasons - APH, 04/05/19
  struct Edge {

     const Dependency * m_comp{nullptr};
     Edge             * m_comp_next{nullptr};
     const Dependency * m_req{nullptr};
     Edge             * m_req_next{nullptr};

     inline Edge( const Dependency * comp , const Dependency * req )
       : m_comp(comp)
       , m_req(req)
     {}

  }; // struct Edge



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
  bool hasRequires( const VarLabel          * var
                  ,       int                 matlIndex
                  , const Patch             * patch
                  ,       Uintah::IntVector   lowOffset
                  ,       Uintah::IntVector   highOffset
                  ,       WhichDW             dw
                  ) const;

  // finds if it modifies var
  bool hasModifies( const VarLabel * var
                  ,       int        matlIndex
                  , const Patch    * patch
                  ) const;

  bool isReductionTask() const { return d_tasktype == Reduction; }

  void setType(TaskType tasktype) { d_tasktype = tasktype; }

  TaskType getType() const { return d_tasktype; }

  //////////
  // Prints out information about the task...
  void display( std::ostream & out ) const;

  //////////
  // Prints out all information about the task, including dependencies
  void displayAll_DOUT( Uintah::Dout& dbg) const;

  void displayAll( std::ostream & out ) const;

  int mapDataWarehouse( WhichDW dw ) const;

  DataWarehouse* mapDataWarehouse( WhichDW dw, std::vector<DataWarehouseP> & dws ) const;

  int getSortedOrder() const { return m_sorted_order; }

  void setSortedOrder(int order) { m_sorted_order = order; }

  void setMapping(int dwmap[TotalDWs]);

  void setSets( const PatchSet * patches, const MaterialSet * matls );

  static const MaterialSubset* getGlobalMatlSubset();

private: // class Task

  using DepMap = std::multimap<const VarLabel*, Dependency*, VarLabel::Compare>;

  Dependency* isInDepMap( const DepMap   & depMap
                        , const VarLabel * var
                        ,       int        matlIndex
                        , const Patch    * patch
                        ) const;

  std::string m_task_name;

#if defined(KOKKOS_USING_GPU)
  std::map<intptr_t, deviceNumSet>  m_deviceNums;
#endif

protected: // class Task

  ActionBase * m_action{nullptr};

  // eliminate copy, assignment and move
  Task( const Task & )            = delete;
  Task& operator=( const Task & ) = delete;
  Task( Task && )                 = delete;
  Task& operator=( Task && )      = delete;




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

  const PatchSet    * m_patch_set{nullptr};
  const MaterialSet * m_matl_set{nullptr};

  bool m_uses_mpi{false};
  bool m_uses_threads{false};
  TaskAssignedExecutionSpace m_execution_space{};
  TaskAssignedMemorySpace    m_memory_space{};
  bool m_uses_device{false};
  bool m_preload_sim_vars{false};
  int  m_max_instances_per_task{0};
  bool m_subpatch_capable{false};
  bool m_has_subscheduler{false};
  bool m_debugFlag{false};

  TaskType d_tasktype;

  int m_dwmap[TotalDWs];
  int m_sorted_order{-1};

  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task & task);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task::TaskType & tt);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task::Dependency & dep);

}; // end class Task


// Used in compiling the task graph with topological sort.
//   Kept around for historical and reproducability reasons - APH, 04/05/19
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

#if defined(KOKKOS_USING_GPU)
//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
void
Task::ActionPortableBase<ExecSpace, MemSpace>::
assignDevicesAndInstances(intptr_t dTask)
{
  for (int i = 0; i < this->taskPtr->maxInstancesPerTask(); i++) {
   this->assignDevicesAndInstances(dTask, i);
  }
}

template<typename ExecSpace, typename MemSpace>
void
Task::ActionPortableBase<ExecSpace, MemSpace>::
assignDevicesAndInstances(intptr_t dTask, unsigned int device_id)
{
  if (this->haveKokkosInstanceForThisTask(dTask, device_id) == false) {
    this->taskPtr->assignDevice(dTask, device_id);

    this->setKokkosInstanceForThisTask(dTask, device_id);
  }
}

template<typename ExecSpace, typename MemSpace>
bool
Task::ActionPortableBase<ExecSpace, MemSpace>::
haveKokkosInstanceForThisTask(intptr_t dTask, unsigned int device_id) const
{
  bool retVal = false;

  // As m_kokkosInstance can be touched by multiple threads a mutext is needed.
  kokkosInstances_mutex.lock();
  {
    auto iter = m_kokkosInstances.find(dTask); // Instances for this task.

    if(iter != m_kokkosInstances.end())
    {
      kokkosInstanceMap iMap = iter->second;
      kokkosInstanceMapIter it = iMap.find(device_id);

      retVal = (it != iMap.end());
    }
  }
  kokkosInstances_mutex.unlock();

  return retVal;
}

template<typename ExecSpace, typename MemSpace>
ExecSpace
Task::ActionPortableBase<ExecSpace, MemSpace>::
getKokkosInstanceForThisTask(intptr_t dTask, unsigned int device_id) const
{
  // As m_kokkosInstance can be touched by multiple threads a mutext is needed.
  kokkosInstances_mutex.lock();
  {
    auto iter = m_kokkosInstances.find(dTask); // Instances for this task.

    if(iter != m_kokkosInstances.end())
    {
      kokkosInstanceMap iMap = iter->second;
      kokkosInstanceMapIter it = iMap.find(device_id);

      if (it != iMap.end())
      {
        kokkosInstances_mutex.unlock();

        return it->second;
      }
    }
  }
  kokkosInstances_mutex.unlock();

  printf("ERROR! - Task::ActionPortableBase::getKokkosInstanceForThisTask() - "
           "This task %s does not have an instance assigned for device %d\n",
           this->taskPtr->getName().c_str(), device_id);
  SCI_THROW(InternalError("Detected Kokkos execution failure on task: " +
                          this->taskPtr->getName(), __FILE__, __LINE__));

  ExecSpace instance;

  return instance;
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
void
Task::ActionPortableBase<ExecSpace, MemSpace>::
setKokkosInstanceForThisTask(intptr_t dTask,
                             unsigned int device_id)
{
  // if (instance == nullptr) {
  //   printf("ERROR! - Task::ActionPortableBase::setKokkosInstanceForThisTask() - "
  //          "A request was made to assign an instance to a nullptr address "
  //          "for this task %s\n", this->taskPtr->getName().c_str());
  //   SCI_THROW(InternalError("A request was made to assign an instance to a "
  //                           "nullptr address for this task :" +
  //                           this->taskPtr->getName() , __FILE__, __LINE__));
  // } else
  if(this->haveKokkosInstanceForThisTask(dTask, device_id) == true) {
    printf("ERROR! - Task::ActionPortableBase::setKokkosInstanceForThisTask() - "
           "This task %s already has an instance assigned for device %d\n",
           this->taskPtr->getName().c_str(), device_id);
    SCI_THROW(InternalError("Detected Kokkos execution failure on task: " +
                            this->taskPtr->getName(), __FILE__, __LINE__));
  } else {
    // printf("Task::ActionPortableBase::setKokkosInstanceForThisTask() - "
    //        "This task %s (%d) now has an instance assigned for device %d\n",
    //        this->taskPtr->getName().c_str(), dTask, device_id);
    // As m_kokkosInstances can be touched by multiple threads a
    // mutext is needed.
    kokkosInstances_mutex.lock();
    {
      ExecSpace instance;
      m_kokkosInstances[dTask][device_id] = instance;
    }
    kokkosInstances_mutex.unlock();
  }
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
void
Task::ActionPortableBase<ExecSpace, MemSpace>::
clearKokkosInstancesForThisTask(intptr_t dTask)
{
  // As m_kokkosInstances can be touched by multiple threads a mutext is needed.
  kokkosInstances_mutex.lock();
  {
    if(m_kokkosInstances.find(dTask) != m_kokkosInstances.end())
    {
      m_kokkosInstances[dTask].clear();
      m_kokkosInstances.erase(dTask);
    }
  }
  kokkosInstances_mutex.unlock();

  // printf("Task::ActionPortableBase::clearKokkosInstancesForThisTask() - "
  //     "Clearing instances for task %s\n",
  //     this->taskPtr->getName().c_str());
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
template<typename ES>
typename std::enable_if<std::is_same<ES, UintahSpaces::CPU>::value, bool>::type
Task::ActionPortableBase<ExecSpace, MemSpace>::
checkKokkosInstanceDoneForThisTask_impl(intptr_t dTask, unsigned int device_id) const
{
  return true;
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
template<typename ES>
typename std::enable_if<!std::is_same<ES, UintahSpaces::CPU>::value, bool>::type
Task::ActionPortableBase<ExecSpace, MemSpace>::
checkKokkosInstanceDoneForThisTask_impl(intptr_t dTask, unsigned int device_id) const
{
  // ARS - FIX ME - For now use the Kokkos fence but perhaps the direct
  // checks should be performed. Also see Task::ActionNonPortableBase (Task.cc)
  if (device_id != 0) {
   printf("Error, Task::checkKokkosInstanceDoneForThisTask is %u\n", device_id);
   exit(-1);
  }

  ExecSpace instance = this->getKokkosInstanceForThisTask(dTask, device_id);

#if defined(USE_KOKKOS_FENCE)
  instance.fence();

#elif defined(KOKKOS_ENABLE_CUDA)
  cudaStream_t stream = instance.cuda_stream();

  cudaError_t retVal = cudaStreamQuery(stream);

  if (retVal == cudaSuccess) {
    return true;
  }
  else if (retVal == cudaErrorNotReady ) {
    return false;
  }
  else if (retVal == cudaErrorLaunchFailure) {
    printf("ERROR! - Task::ActionNonPortableBase::checkKokkosInstanceDoneForThisTask(%d) - "
           "CUDA kernel execution failure on Task: %s\n",
           device_id, this->taskPtr->getName().c_str());
    SCI_THROW(InternalError("Detected CUDA kernel execution failure on Task: " +
                            this->taskPtr->getName() , __FILE__, __LINE__));
    return false;
  } else { // other error
    printf("\nA CUDA error occurred with error code %d.\n\n"
           "Waiting for 60 seconds\n", retVal);

    int sleepTime = 60;

    struct timespec ts;
    ts.tv_sec = (int) sleepTime;
    ts.tv_nsec = (int)(1.e9 * (sleepTime - ts.tv_sec));

    nanosleep(&ts, &ts);

    return false;
  }
#elif defined(KOKKOS_ENABLE_HIP)
  hipStream_t stream = instance.hip_stream();

  hipError_t retVal = hipStreamQuery(stream);

  if (retVal == hipSuccess) {
    return true;
  }
  else if (retVal == hipErrorNotReady ) {
    return false;
  }
  else if (retVal ==  hipErrorLaunchFailure) {
    printf("ERROR! - Task::ActionNonPortableBase::checkKokkosInstanceDoneForThisTask(%d) - "
           "HIP kernel execution failure on Task: %s\n",
           device_id, this->taskPtr->getName().c_str());
    SCI_THROW(InternalError("Detected HIP kernel execution failure on Task: " +
                            this->taskPtr->getName() , __FILE__, __LINE__));
    return false;
  } else { // other error
    printf("\nA HIP error occurred with error code %d.\n\n"
           "Waiting for 60 seconds\n", retVal);

    int sleepTime = 60;

    struct timespec ts;
    ts.tv_sec = (int) sleepTime;
    ts.tv_nsec = (int)(1.e9 * (sleepTime - ts.tv_sec));

    nanosleep(&ts, &ts);

    return false;
  }

#elif defined(KOKKOS_ENABLE_SYCL)
  sycl::queue que = instance.sycl_queue();
  // Not yet available.
  //  return que.ext_oneapi_empty();
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)

#elif defined(KOKKOS_ENABLE_OPENACC)

#endif

  return true;
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
bool
Task::ActionPortableBase<ExecSpace, MemSpace>::
checkAllKokkosInstancesDoneForThisTask(intptr_t dTask) const
{
  // A task can have multiple instances (such as an output task
  // pulling from multiple GPUs).  Check all instacnes to see if they
  // are done.  If any one instance isn't done, return false.  If
  // nothing returned false, then they all must be good to go.
  bool retVal = true;

  // As m_kokkosInstances can be touched by multiple threads get a local
  // copy so not to lock everything.
  kokkosInstanceMap kokkosInstances;

  kokkosInstances_mutex.lock();
  {
    auto iter = m_kokkosInstances.find(dTask);
    if(iter != m_kokkosInstances.end()) {
      kokkosInstances = iter->second;
    } else {
      kokkosInstances_mutex.unlock();

      return retVal;
    }
  }
  kokkosInstances_mutex.unlock();

  for (auto & it : kokkosInstances)
  {
    retVal = this->checkKokkosInstanceDoneForThisTask(dTask, it.first);
    if (retVal == false)
      break;
  }

  return retVal;
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
template<typename ES>
typename std::enable_if<std::is_same<ES, UintahSpaces::CPU>::value ||
                        std::is_same<ES, Kokkos::OpenMP   >::value, void>::type
Task::ActionPortableBase<ExecSpace, MemSpace>::
doKokkosDeepCopy_impl(intptr_t dTask, unsigned int deviceNum,
                          void* dst, void* src,
                          size_t count, GPUMemcpyKind kind)
{
  // Do nothing as all of the data is on the host.
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
template<typename ES>
typename std::enable_if<!std::is_same<ES, UintahSpaces::CPU>::value &&
                        !std::is_same<ES, Kokkos::OpenMP   >::value, void>::type
Task::ActionPortableBase<ExecSpace, MemSpace>::
doKokkosDeepCopy_impl(intptr_t dTask, unsigned int deviceNum,
                          void* dst, void* src,
                          size_t count, GPUMemcpyKind kind)
{
  ExecSpace instance = this->getKokkosInstanceForThisTask(dTask, deviceNum);

  char * srcPtr = static_cast< char *>(src);
  char * dstPtr = static_cast< char *>(dst);

  if(kind == GPUMemcpyHostToDevice)
  {
    // Create an unmanage Kokkos view from the raw pointers.
    Kokkos::View<char*, Kokkos::HostSpace>   hostView(srcPtr, count);
    Kokkos::View<char*, ExecSpace        > deviceView(dstPtr, count);
    // Deep copy the host view to the device view.
    Kokkos::deep_copy(instance, deviceView, hostView);
  }
  else if(kind == GPUMemcpyDeviceToHost)
  {
    // Create an unmanage Kokkos view from the raw pointers.
    Kokkos::View<char*, Kokkos::HostSpace>   hostView(dstPtr, count);
    Kokkos::View<char*, ExecSpace        > deviceView(srcPtr, count);
    // Deep copy the device view to the host view.
    Kokkos::deep_copy(instance, hostView, deviceView);
  }
}

//_____________________________________________________________________________
//
template<typename ExecSpace, typename MemSpace>
void
Task::ActionPortableBase<ExecSpace, MemSpace>::
doKokkosMemcpyPeerAsync( intptr_t dTask,
                       unsigned int deviceNum,
                             void* dst, int  dstDevice,
                       const void* src, int  srcDevice,
                       size_t count )
{
  ExecSpace instance = this->getKokkosInstanceForThisTask(dTask, deviceNum);

  SCI_THROW(InternalError("Error - doKokkosMemcpyPeerAsync is not implemented. No Kokkos equivalent function.", __FILE__, __LINE__));
}

template<typename ExecSpace, typename MemSpace>
void
Task::ActionPortableBase<ExecSpace, MemSpace>::
copyGpuGhostCellsToGpuVars(intptr_t dTask,
                           unsigned int deviceNum,
                           GPUDataWarehouse *taskgpudw)
{
  ExecSpace instance = this->getKokkosInstanceForThisTask(dTask, deviceNum);

  taskgpudw->copyGpuGhostCellsToGpuVarsInvoker(instance);
}

template<typename ExecSpace, typename MemSpace>
void
Task::ActionPortableBase<ExecSpace, MemSpace>::
syncTaskGpuDW(intptr_t dTask,
                           unsigned int deviceNum,
                           GPUDataWarehouse *taskgpudw)
{
  ExecSpace instance = this->getKokkosInstanceForThisTask(dTask, deviceNum);

  taskgpudw->syncto_device(instance);
}
#endif  // defined(KOKKOS_USING_GPU)

}  // End namespace Uintah

// This must be at the bottom
#include <CCA/Ports/DataWarehouse.h>

#endif // CORE_GRID_TASK_H
