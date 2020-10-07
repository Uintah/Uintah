/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
#define UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <CCA/Components/Schedulers/DWDatabase.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouseP.h>
#include <CCA/Components/Schedulers/SendState.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Core/Grid/Variables/VarLabelMatl.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/UintahMPI.h>

#include <iosfwd>
#include <map>
#include <vector>


using Uintah::Max;
using Uintah::FastHashTable;


namespace Uintah {

inline const Patch* getRealDomain( const Patch * patch )
{
  return patch->getRealPatch();
}

inline const Level* getRealDomain( const Level * level )
{
  return level;
}


class BufferInfo;
class DependencyBatch;
class DetailedDep;
class DetailedTasks;
class LoadBalancer;
class Patch;
class ProcessorGroup;
class SendState;
class TypeDescription;


template <class T> class constNCVariable;


/**************************************

 CLASS

   OnDemandDataWarehouse


 GENERAL INFORMATION

   OnDemandDataWarehouse.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


 KEYWORDS
   OnDemandDataWarehouse


 ****************************************/

class OnDemandDataWarehouse : public DataWarehouse {

public:

  OnDemandDataWarehouse( const ProcessorGroup * myworld
                       ,       Scheduler      * scheduler
                       , const int              generation
                       , const GridP          & grid
                       , const bool             isInitializationDW = false
                       );

  virtual ~OnDemandDataWarehouse();

  virtual bool exists( const VarLabel * label
                     ,       int        matlIndex
                     , const Patch    * patch
                     ) const;

  virtual bool exists( const VarLabel * label
                     ,       int        matlIndex
                     , const Level    * level
                     ) const;

  virtual ReductionVariableBase* getReductionVariable( const VarLabel * label
                                                     ,       int        matlIndex
                                                     , const Level    * level
                                                     ) const;

  void copyKeyDB( KeyDatabase<Patch> & varkeyDB
                , KeyDatabase<Level> & levekeyDB
                );

  virtual void doReserve();

  // Returns a (const) pointer to the grid.  This pointer can then be
  // used to (for example) get the number of levels in the grid.
  virtual const Grid * getGrid()
  {
    return m_grid.get_rep();
  }

  // Generic put and allocate, passing Variable as a pointer rather than
  // by reference to avoid ambiguity with other put overloaded methods.
  virtual void put(       Variable * var
                  , const VarLabel * label
                  ,       int        matlIndex
                  , const Patch    * patch
                  );

  // Reduction Variables
  virtual void get(       ReductionVariableBase & var
                  , const VarLabel              * label
                  , const Level                 * level     = nullptr
                  ,       int                     matlIndex = -1
                  );

  virtual void put( const ReductionVariableBase & var
                  , const VarLabel              * label
                  , const Level                 * level     = nullptr
                  ,       int                     matlIndex = -1
                  );

  virtual void override( const ReductionVariableBase & var
                       , const VarLabel              * label
                       , const Level                 * level     = nullptr
                       ,       int                     matlIndex = -1
                       );

  virtual void print(       std::ostream & intout
                    , const VarLabel     * label
                    , const Level        * level
                    ,       int            matlIndex = -1
                    );


  //__________________________________
  // Sole Variables

  virtual bool exists( const VarLabel* ) const;

  virtual void get(       SoleVariableBase & var
                  , const VarLabel         * label
                  , const Level            * level     = nullptr
                  ,       int                matlIndex = -1
                  );

  virtual void put( const SoleVariableBase & var
                  , const VarLabel         * label
                  , const Level            * level     = nullptr
                  ,       int                matlIndex = -1
                  );

  virtual void override( const SoleVariableBase & var
                       , const VarLabel         * label
                       , const Level            * level     = nullptr
                       ,       int                matlIndex = -1
                       );


  //__________________________________
  // Particle Variables

  virtual ParticleSubset* createParticleSubset(       particleIndex   numParticles
                                              ,       int             matlIndex
                                              , const Patch         * patch
                                              ,       IntVector       low  = IntVector(0, 0, 0)
                                              ,       IntVector       high = IntVector(0, 0, 0)
                                              );

  virtual void saveParticleSubset(       ParticleSubset * psubset
                                 ,       int              matlIndex
                                 , const Patch          * patch
                                 ,       IntVector        low  = IntVector(0, 0, 0)
                                 ,       IntVector        high = IntVector(0, 0, 0)
                                 );

  virtual bool haveParticleSubset(       int         matlIndex
                                 , const Patch     * patch
                                 ,       IntVector   low  = IntVector(0, 0, 0)
                                 ,       IntVector   high = IntVector(0, 0, 0)
                                 ,       bool        exact = false
                                 );

  virtual ParticleSubset* getParticleSubset(       int         matlIndex
                                           , const Patch     * patch
                                           ,       IntVector   low
                                           ,       IntVector   high
                                           );

  virtual ParticleSubset* getParticleSubset(       int         matlIndex
                                           , const Patch     * patch
                                           ,       IntVector   low
                                           ,       IntVector   high
                                           , const VarLabel  * posvar
                                           );

  virtual ParticleSubset* getParticleSubset(       int     matlIndex
                                           , const Patch * patch
                                           );

  virtual ParticleSubset* getDeleteSubset(       int     matlIndex
                                         , const Patch * patch
                                         );

  virtual std::map<const VarLabel*, ParticleVariableBase*>* getNewParticleState(       int     matlIndex
                                                                               , const Patch * patch
                                                                               );

  virtual ParticleSubset* getParticleSubset(       int                matlIndex
                                           , const Patch            * patch
                                           ,       Ghost::GhostType   gtype
                                           ,       int                numGhostCells
                                           , const VarLabel         * posvar
                                           );

  // returns the particle subset in the range of low->high
  // relPatch is used as the key and should be the patch you are querying from
  // level is used if you are querying from an old level
  virtual ParticleSubset* getParticleSubset(       int         matlIndex
                                           ,       IntVector   low
                                           ,       IntVector   high
                                           , const Patch     * relPatch
                                           , const VarLabel  * posvar
                                           , const Level     * level = nullptr
                                           );

  virtual void allocateTemporary( ParticleVariableBase & var
                                , ParticleSubset       * pset
                                );

  virtual void allocateAndPut(       ParticleVariableBase & car
                             , const VarLabel             * label
                             ,       ParticleSubset       * pset
                             );

  virtual void get(constParticleVariableBase & constVar
                  , const VarLabel           * label
                  , ParticleSubset           * pset
                  );

  virtual void get(       constParticleVariableBase & constVar
                  , const VarLabel                  * label
                  ,       int                         matlIndex
                  , const Patch                     * patch
                  );

  virtual void getModifiable(       ParticleVariableBase & var
                            , const VarLabel             * label
                            ,       ParticleSubset       * pset
                            );

  virtual void put(       ParticleVariableBase & var
                  , const VarLabel             * label
                  ,       bool                   replace = false
                  );

  virtual ParticleVariableBase* getParticleVariable( const VarLabel       * label
                                                   ,       ParticleSubset * pset
                                                   );

  virtual ParticleVariableBase* getParticleVariable( const VarLabel * label
                                                   ,       int        matlIndex
                                                   , const Patch    * patch
                                                   );
  void printParticleSubsets();

  virtual void getCopy(       ParticleVariableBase & var
                      , const VarLabel             * label
                      ,       ParticleSubset       * pset
                      );

  virtual void copyOut(       ParticleVariableBase & var
                      , const VarLabel             * label
                      ,       ParticleSubset       * pset
                      );

  // Remove particles that are no longer relevant
  virtual void deleteParticles( ParticleSubset * delset );

  using particleAddSetType = std::map<const VarLabel*, ParticleVariableBase*>;
  virtual void addParticles( const Patch              * patch
                           ,       int                  matlIndex
                           ,       particleAddSetType * addedstate
                           );


  //__________________________________
  // Grid Variables

  virtual void print();

  virtual void clear();

  void get(       constGridVariableBase & constVar
          , const VarLabel              * label
          ,       int                     matlIndex
          , const Patch                 * patch
          ,       Ghost::GhostType        gtype
          ,       int                     numGhostCells
          );

  void getModifiable(       GridVariableBase & var
                    , const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype         = Ghost::None
                    ,       int                numGhostCells = 0
                    );

  void allocateTemporary(       GridVariableBase & var
                        , const Patch            * patch
                        ,       Ghost::GhostType   gtype
                        ,       int                numGhostCells
                        );

  void allocateAndPut(       GridVariableBase & var
                     , const VarLabel         * label
                     ,       int                matlIndex
                     , const Patch            * patch
                     ,       Ghost::GhostType   gtype         = Ghost::None
                     ,       int                numGhostCells = 0
                     );

  void put(       GridVariableBase & var
          , const VarLabel         * label
          ,       int                matlIndex
          , const Patch            * patch
          ,       bool               replace = false
          );

  // returns the constGridVariable for all patches on the level
  virtual void getLevel(       constGridVariableBase & constGridVar
                       , const VarLabel              * label
                       ,       int                     matlIndex
                       , const Level                 * level
                       );

  //meant for the UnifiedScheduler only.  Puts a contiguous level in the *level* database so that
  //it doesn't need to constantly make new deep copies each time the full level is requested.
  void putLevelDB(       GridVariableBase * gridVar
                 , const VarLabel         * label
                 , const Level            * level
                 ,       int                matlIndex = -1
                 );

  virtual void getRegion(       constGridVariableBase & constVar
                        , const VarLabel              * label
                        ,       int                     matlIndex
                        , const Level                 * level
                        , const IntVector             & low
                        , const IntVector             & high
                        ,       bool                    useBoundaryCells = true
                        );

  virtual void getRegionModifiable(       GridVariableBase & var
                                  , const VarLabel         * label
                                  ,       int                matlIndex
                                  , const Level            * level
                                  , const IntVector        & low
                                  , const IntVector        & high
                                  ,       bool               useBoundaryCells = true
                                  );

  virtual void copyOut(       GridVariableBase & var
                      , const VarLabel         * label
                      ,       int                matlIndex
                      , const Patch            * patch
                      ,       Ghost::GhostType   gtype         = Ghost::None
                      ,       int                numGhostCells = 0
                      );

  virtual void getCopy(       GridVariableBase & var
                      , const VarLabel         * label
                      ,       int                matlIndex
                      , const Patch            * patch
                      ,       Ghost::GhostType   gtype         = Ghost::None
                      ,        int               numGhostCells = 0
                      );

  // PerPatch Variables
  virtual void get(       PerPatchBase & var
                  , const VarLabel     * label
                  ,       int            matlIndex
                  , const Patch        * patch
                  );

  virtual void put(       PerPatchBase & var
                  , const VarLabel     * label
                  ,       int            matlIndex
                  , const Patch        * patch
                  ,       bool           replace = false
                  );

  virtual ScrubMode setScrubbing(ScrubMode);


  //__________________________________
  // For related datawarehouses

  virtual DataWarehouse* getOtherDataWarehouse( Task::WhichDW );

  virtual void transferFrom(       DataWarehouse  * from
                           , const VarLabel       * label
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           );

  virtual void transferFrom(       DataWarehouse  * from
                           , const VarLabel       * label
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           ,       bool             replace
                           );

  virtual void transferFrom(       DataWarehouse  * from
                           , const VarLabel       * label
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           ,       bool             replace
                           , const PatchSubset    * newPatches
                           );

  template <typename ExecSpace, typename MemSpace>
          void transferFrom(       DataWarehouse                        * from
                           , const VarLabel                             * label
                           , const PatchSubset                          * patches
                           , const MaterialSubset                       * matls
                           ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                           ,       bool                                   replace
                           , const PatchSubset                          * newPatches
                           );


  virtual bool isFinalized() const;

  virtual void finalize();


#ifdef HAVE_CUDA

  static int getNumDevices();
  static void uintahSetCudaDevice(int deviceNum);
  static size_t getTypeDescriptionSize(const TypeDescription::Type& type);
  static GPUGridVariableBase* createGPUGridVariable(const TypeDescription::Type& type);
  static GPUPerPatchBase* createGPUPerPatch(const TypeDescription::Type& type);
  static GPUReductionVariableBase* createGPUReductionVariable(const TypeDescription::Type& type);

#endif


  virtual void unfinalize();

  virtual void refinalize();

  virtual size_t emit(       OutputContext & oc
                     , const VarLabel      * label
                     ,       int             matlIndex
                     , const Patch         * patch
                     );


#if HAVE_PIDX
     void emitPIDX(       PIDXOutputContext & context
                  , const VarLabel          * label
                  ,       int                 matlIndex
                  , const Patch             * patch
                  ,       unsigned char     * pidx_buffer
                  ,       size_t              pidx_bufferSize
                  );
#endif


  void exchangeParticleQuantities(       DetailedTasks * dts
                                 ,       LoadBalancer  * lb
                                 , const VarLabel      * pos_var
                                 ,       int             iteration
                                 );

  void sendMPI(       DependencyBatch       * batch
              , const VarLabel              * pos_var
              ,       BufferInfo            & buffer
              ,       OnDemandDataWarehouse * old_dw
              , const DetailedDep           * dep
              ,       LoadBalancer          * lb
              );

  void recvMPI(       DependencyBatch       * batch
              ,       BufferInfo            & buffer
              ,       OnDemandDataWarehouse * old_dw
              , const DetailedDep           * dep
              ,       LoadBalancer          * lb
              );

  void reduceMPI( const VarLabel       * label
                , const Level          * level
                , const MaterialSubset * matls
                ,       int              nComm
                );

  // Scrub counter manipulator functions -- when the scrub count goes to zero, the data is deleted
  void setScrubCount( const VarLabel * label
                    ,       int        matlIndex
                    , const Patch    * patch
                    ,       int        count
                    );

  int decrementScrubCount( const VarLabel * label
                         ,       int        matlIndex
                         , const Patch    * patch
                         );

  void scrub( const VarLabel * label
            ,       int        matlIndex
            , const Patch    * patch
            );

  void initializeScrubs(       int                        dwid
                       , const FastHashTable<ScrubItem> * scrubcounts
                       ,       bool                       add
                       );

 // For time step abort/recompute
 virtual bool abortTimeStep();
 virtual bool recomputeTimeStep();

  struct ValidNeighbors {
          GridVariableBase   * validNeighbor {nullptr};
    const Patch              * neighborPatch {nullptr};
          IntVector            low  {};
          IntVector            high {};

  };
  void getNeighborPatches( const VarLabel             * label
                         , const Patch                * patch
                         ,       Ghost::GhostType       gtype
                         ,       int                    numGhostCells
                         , std::vector<const Patch *> & adjacentNeighbors
                         );

  // This method will retrieve those neighbors, and also the regions (indicated in low and high) which constitute the ghost cells.
  // Data is return in the ValidNeighbors vector. IgnoreMissingNeighbors is designed for the Unified Scheduler so that it can request what
  // neighbor patches *should* be, and those neighbor patches we hope are found in the host side DW (this one) or the GPU DW
  //
  // TODO, This method might create a reference to the neighbor, and so these references
  // need to be deleted afterward. (It's not pretty, but it seemed to be the best option.)
  void getValidNeighbors( const VarLabel                    * label
                        ,       int                           matlIndex
                        , const Patch                       * patch
                        ,       Ghost::GhostType              gtype
                        ,       int                           numGhostCells
                        ,       std::vector<ValidNeighbors> & validNeighbors
                        ,       bool                          ignoreMissingNeighbors = false
                        );

  void logMemoryUse(       std::ostream  & out
                   ,       unsigned long & total
                   , const std::string   & tag
                   );

  // must be called by the thread that will run the test
  void pushRunningTask( const Task                                * task
                      ,       std::vector<OnDemandDataWarehouseP> * dws
                      );

  void popRunningTask();

  // does a final check to see if gets/puts/etc. consistent with
  // requires/computes/modifies for the current task.
  void checkTasksAccesses( const PatchSubset    * patches
                         , const MaterialSubset * matls
                         );

  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, CCVariable<T> >::type
  getCCVariable( const VarLabel         * label
               ,       int                matlIndex
               , const Patch            * patch
               ,       Ghost::GhostType   gtype         = Ghost::None
               ,       int                numGhostCells = 0
               )
  {
    CCVariable<T> var;
    if ( matlIndex != -999 ) {
      // Assumption: Modifies if it exists; Computes otherwise
      if ( this->exists( label, matlIndex, patch ) ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      } else {
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
    return var;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, KokkosView3<T, Kokkos::HostSpace> >::type
  getCCVariable( const VarLabel         * label
               ,       int                matlIndex
               , const Patch            * patch
               ,       Ghost::GhostType   gtype         = Ghost::None
               ,       int                numGhostCells = 0
               )
  {
    CCVariable<T> var;
    if ( matlIndex != -999 ) {
      // Assumption: Modifies if it exists; Computes otherwise
      if ( this->exists( label, matlIndex, patch ) ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      } else {
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
    return var.getKokkosView();
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, KokkosView3<T, Kokkos::CudaSpace> >::type
  getCCVariable( const VarLabel         * label
               ,       int                matlIndex
               , const Patch            * patch
               ,       Ghost::GhostType   gtype         = Ghost::None
               ,       int                numGhostCells = 0
               )
  {
    if ( matlIndex != -999 ) {
      return this->getGPUDW()->getKokkosView<T>( label->getName().c_str(), patch->getID(),  matlIndex, patch->getLevel()->getID() );
    } else {
      return KokkosView3<T, Kokkos::CudaSpace>();
    }
  }
#endif

  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, constCCVariable<T> >::type
  getConstCCVariable( const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype
                    ,       int                numGhostCells
                    )
  {
    constCCVariable<T> constVar;
    if ( matlIndex != -999 ) {
      this->get( constVar, label, matlIndex, patch, gtype, numGhostCells );
    }
    return constVar;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, KokkosView3<const T, Kokkos::HostSpace> >::type
  getConstCCVariable( const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype
                    ,       int                numGhostCells
                    )
  {
    constCCVariable<T> constVar;
    if ( matlIndex != -999 ) {
      this->get( constVar, label, matlIndex, patch, gtype, numGhostCells );
    }
    return constVar.getKokkosView();
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, KokkosView3<const T, Kokkos::CudaSpace> >::type
  getConstCCVariable( const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype
                    ,       int                numGhostCells
                    )
  {
    if ( matlIndex != -999 ) {
      return this->getGPUDW()->getKokkosView<const T>( label->getName().c_str(), patch->getID(),  matlIndex, patch->getLevel()->getID() );
    } else {
      return KokkosView3<const T, Kokkos::CudaSpace>();
    }
  }
#endif

  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, NCVariable<T> >::type
  getNCVariable( const VarLabel         * label
               ,       int                matlIndex
               , const Patch            * patch
               ,       Ghost::GhostType   gtype         = Ghost::None
               ,       int                numGhostCells = 0
               )
  {
    NCVariable<T> var;
    if ( matlIndex != -999 ) {
      // Assumption: Modifies if it exists; Computes otherwise
      if ( this->exists( label, matlIndex, patch ) ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      } else {
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
    return var;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, KokkosView3<T, Kokkos::HostSpace> >::type
  getNCVariable( const VarLabel         * label
               ,       int                matlIndex
               , const Patch            * patch
               ,       Ghost::GhostType   gtype         = Ghost::None
               ,       int                numGhostCells = 0
               )
  {
    NCVariable<T> var;
    if ( matlIndex != -999 ) {
      // Assumption: Modifies if it exists; Computes otherwise
      if ( this->exists( label, matlIndex, patch ) ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      } else {
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
    return var.getKokkosView();
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, KokkosView3<T, Kokkos::CudaSpace> >::type
  getNCVariable( const VarLabel         * label
               ,       int                matlIndex
               , const Patch            * patch
               ,       Ghost::GhostType   gtype         = Ghost::None
               ,       int                numGhostCells = 0
               )
  {
    if ( matlIndex != -999 ) {
      return this->getGPUDW()->getKokkosView<T>( label->getName().c_str(), patch->getID(),  matlIndex, patch->getLevel()->getID() );
    } else {
      return KokkosView3<T, Kokkos::CudaSpace>();
    }
  }
#endif

  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, constNCVariable<T> >::type
  getConstNCVariable( const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype
                    ,       int                numGhostCells
                    )
  {
    constNCVariable<T> constVar;
    if ( matlIndex != -999 ) {
      this->get( constVar, label, matlIndex, patch, gtype, numGhostCells );
    }
    return constVar;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, KokkosView3<const T, Kokkos::HostSpace> >::type
  getConstNCVariable( const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype
                    ,       int                numGhostCells
                    )
  {
    constNCVariable<T> constVar;
    if ( matlIndex != -999 ) {
      this->get( constVar, label, matlIndex, patch, gtype, numGhostCells );
    }
    return constVar.getKokkosView();
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, KokkosView3<const T, Kokkos::CudaSpace> >::type
  getConstNCVariable( const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype
                    ,       int                numGhostCells
                    )
  {
    if ( matlIndex != -999 ) {
      return this->getGPUDW()->getKokkosView<const T>( label->getName().c_str(), patch->getID(),  matlIndex, patch->getLevel()->getID() );
    } else {
      return KokkosView3<const T, Kokkos::CudaSpace>();
    }
  }
#endif

  template <typename grid_T,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, grid_T >::type
  getGridVariable( const VarLabel         * label
                 ,       int                matlIndex
                 , const Patch            * patch
                 ,       Ghost::GhostType   gtype           = Ghost::None
                 ,       int                numGhostCells   = 0
                 ,       bool               l_getModifiable = false
                 )
  {
    grid_T var;
    if ( matlIndex != -999 ) {
      if ( l_getModifiable ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      }else{
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
    return var;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename grid_T,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, KokkosView3<T, Kokkos::HostSpace> >::type
  getGridVariable( const VarLabel         * label
                 ,       int                matlIndex
                 , const Patch            * patch
                 ,       Ghost::GhostType   gtype           = Ghost::None
                 ,       int                numGhostCells   = 0
                 ,       bool               l_getModifiable = false
                 )
  {
    grid_T var;
    if ( matlIndex != -999 ) {
      if ( l_getModifiable ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      }else{
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
    return var.getKokkosView();
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename grid_T,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, KokkosView3<T, Kokkos::CudaSpace> >::type
  getGridVariable( const VarLabel         * label
                 ,       int                matlIndex
                 , const Patch            * patch
                 ,       Ghost::GhostType   gtype           = Ghost::None
                 ,       int                numGhostCells   = 0
                 ,       bool               l_getModifiable = false
                 )
  {
    if ( matlIndex != -999 ) {
      return this->getGPUDW()->getKokkosView<T>( label->getName().c_str(), patch->getID(),  matlIndex, patch->getLevel()->getID() );
    } else {
      return KokkosView3<T, Kokkos::CudaSpace>();
    }
  }
#endif

  template <typename grid_CT,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, grid_CT >::type
  getConstGridVariable( const VarLabel         * label
                      ,       int                matlIndex
                      , const Patch            * patch
                      ,       Ghost::GhostType   gtype
                      ,       int                numGhostCells
                      )
  {
    grid_CT constVar;
    if ( matlIndex != -999 ) {
      this->get( constVar, label, matlIndex, patch, gtype, numGhostCells );
    }
    return constVar;
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename grid_CT,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, KokkosView3<const T, Kokkos::HostSpace> >::type
  getConstGridVariable( const VarLabel         * label
                      ,       int                matlIndex
                      , const Patch            * patch
                      ,       Ghost::GhostType   gtype
                      ,       int                numGhostCells
                      )
  {
    grid_CT constVar;
    if ( matlIndex != -999 ) {
      this->get( constVar, label, matlIndex, patch, gtype, numGhostCells );
    }
    return constVar.getKokkosView();
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename grid_CT,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, KokkosView3<const T, Kokkos::CudaSpace> >::type
  getConstGridVariable( const VarLabel         * label
                      ,       int                matlIndex
                      , const Patch            * patch
                      ,       Ghost::GhostType   gtype
                      ,       int                numGhostCells
                      )
  {
    if ( matlIndex != -999 ) {
      return this->getGPUDW()->getKokkosView<const T>( label->getName().c_str(), patch->getID(),  matlIndex, patch->getLevel()->getID() );
    } else {
      return KokkosView3<const T, Kokkos::CudaSpace>();
    }
  }
#endif

  template <typename grid_T,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, void>::type
  assignGridVariable(       grid_T           & var
                    , const VarLabel         * label
                    ,       int                matlIndex
                    , const Patch            * patch
                    ,       Ghost::GhostType   gtype           = Ghost::None
                    ,       int                numGhostCells   = 0
                    ,       bool               l_getModifiable = false
                    )
  {
    if ( matlIndex != -999 ) {
      if ( l_getModifiable ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      }else{
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename grid_T,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, void >::type
  assignGridVariable(       KokkosView3<T, MemSpace> & kvar
                    , const VarLabel                 * label
                    ,       int                        matlIndex
                    , const Patch                    * patch
                    ,       Ghost::GhostType           gtype           = Ghost::None
                    ,       int                        numGhostCells   = 0
                    ,       bool                       l_getModifiable = false
                    )
  {
    grid_T var;
    if ( matlIndex != -999 ) {
      if ( l_getModifiable ) {
        this->getModifiable( var, label, matlIndex, patch, gtype, numGhostCells );
      }else{
        this->allocateAndPut( var, label, matlIndex, patch, gtype, numGhostCells );
      }
    }
    kvar = var.getKokkosView();
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename grid_T,typename T, typename MemSpace>
  inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, void >::type
  assignGridVariable(       KokkosView3<T, MemSpace> & var
                    , const VarLabel                 * label
                    ,       int                        matlIndex
                    , const Patch                    * patch
                    ,       Ghost::GhostType           gtype           = Ghost::None
                    ,       int                        numGhostCells   = 0
                    ,       bool                       l_getModifiable = false
                    )
  {
    if ( matlIndex != -999 ) {
      var = this->getGPUDW()->getKokkosView<T>( label->getName().c_str(), patch->getID(),  matlIndex, patch->getLevel()->getID() );
    } else {
      var = KokkosView3<T, Kokkos::CudaSpace>();
    }
  }
#endif

  ScrubMode getScrubMode() const { return m_scrub_mode; }

  // The following is for support of regridding
  virtual void getVarLabelMatlLevelTriples( std::vector<VarLabelMatl<Level> > & vars ) const;

  static bool s_combine_memory;

  //DS: 01042020: fix for OnDemandDW race condition
  //bool compareAndSwapAllocateOnCPU(char const* label, const int patchID, const int matlIndx, const int levelIndx);
  bool compareAndSwapSetValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  bool compareAndSwapSetInvalidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  bool isValidOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  bool compareAndSwapCopyingIntoCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  bool compareAndSwapAwaitingGhostDataOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  bool isValidWithGhostsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  void setValidWithGhostsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);
  bool compareAndSwapSetInvalidWithGhostsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx);

  friend class SchedulerCommon;
  friend class KokkosScheduler;
  friend class UnifiedScheduler;

private:

  enum AccessType {
      NoAccess = 0
    , PutAccess
    , GetAccess
    , ModifyAccess
  };


  //__________________________________
  // AccessInfo
  struct AccessInfo {
    AccessInfo()
    { }

    AccessInfo(AccessType type)
      : accessType {type}
    { }

    void encompassOffsets( IntVector low
                         , IntVector high
                         )
    {
      lowOffset  = Uintah::Max( low,  lowOffset );
      highOffset = Uintah::Max( high, highOffset );
    }

    AccessType accessType {NoAccess};
    IntVector lowOffset   {0, 0, 0};  // ghost cell access
    IntVector highOffset  {0, 0, 0};
  };


  //__________________________________
  // RunningTaskInfo
  using VarAccessMap = std::map<VarLabelMatl<Patch>, AccessInfo>;
  struct RunningTaskInfo {
    RunningTaskInfo()
      { }

    RunningTaskInfo( const Task                                * task
                   ,       std::vector<OnDemandDataWarehouseP> * dws
                   )
      : m_task{task}
      , m_dws{dws}
    { }

    RunningTaskInfo( const RunningTaskInfo & copy )
      : m_task{copy.m_task}
      , m_dws{copy.m_dws}
      , m_accesses{copy.m_accesses}
    { }

    RunningTaskInfo& operator=( const RunningTaskInfo & copy )
    {
      m_task     = copy.m_task;
      m_dws      = copy.m_dws;
      m_accesses = copy.m_accesses;
      return *this;
    }

    const Task                                * m_task     {nullptr};
          std::vector<OnDemandDataWarehouseP> * m_dws      {nullptr};
          VarAccessMap                          m_accesses {};
  };


  virtual DataWarehouse* getOtherDataWarehouse( Task::WhichDW     dw
                                              , RunningTaskInfo * info
                                              );

  void getGridVar(       GridVariableBase & var
                 , const VarLabel         * label
                 ,       int                matlIndex
                 , const Patch            * patch
                 ,       Ghost::GhostType   gtype
                 ,       int                numGhostCells
                 ,       int                exactWindow=0 //reallocate even if existing window is larger than requested. Exactly match dimensions
                 );

  inline Task::WhichDW getWhichDW( RunningTaskInfo * info );

  // These will throw an exception if access is not allowed for the current task.
  inline void checkGetAccess( const VarLabel         * label
                            ,       int                matlIndex
                            , const Patch            * patch
                            ,       Ghost::GhostType   gtype         = Ghost::None
                            ,       int                numGhostCells = 0
                            );

  inline void checkPutAccess( const VarLabel * label
                            ,       int        matlIndex
                            , const Patch    * patch
                            ,       bool       replace
                            );

  inline void checkModifyAccess( const VarLabel * label
                               ,       int        matlIndex
                               , const Patch    * patch
                               );

  // These will return false if access is not allowed for the current task.
  inline bool hasGetAccess( const Task            * runningTask
                          , const VarLabel        * label
                          ,       int               matlIndex
                          , const Patch           * patch
                          ,       IntVector         lowOffset
                          ,       IntVector         highOffset
                          ,       RunningTaskInfo * info
                          );

  inline bool hasPutAccess( const Task     * runningTask
                          , const VarLabel * label
                          ,       int        matlIndex
                          , const Patch    * patch
                          );

  void checkAccesses(       RunningTaskInfo  * runningTaskInfo
                    , const Task::Dependency * dep
                    ,       AccessType         accessType
                    , const PatchSubset      * patches
                    , const MaterialSubset   * matls
                    );

  void printDebuggingPutInfo( const VarLabel * label
                            ,       int        matlIndex
                            , const Patch    * patch
                            ,       int        line
                            );
                                
  void printDebuggingPutInfo( const VarLabel * label
                            ,       int        matlIndex
                            , const Level    * level
                            ,       int        line
                            );


#ifdef HAVE_CUDA

  std::map<Patch*, bool> assignedPatches; // indicates where a given patch should be stored in an accelerator

#endif


  using psetDBType           = std::multimap<PSPatchMatlGhost, ParticleSubset*>;
  using psetAddDBType        = std::map<std::pair<int, const Patch*>, std::map<const VarLabel*, ParticleVariableBase*>*>;
  using particleQuantityType = std::map<std::pair<int, const Patch*>, int> ;

  ParticleSubset* queryPSetDB(       psetDBType & db
                             , const Patch      * patch
                             ,       int          matlIndex
                             ,       IntVector    low
                             ,       IntVector    high
                             , const VarLabel   * pos_var
                             ,       bool         exact = false
                             );

  void insertPSetRecord(       psetDBType     & subsetDB
                       , const Patch          * patch
                       ,       IntVector        low
                       ,       IntVector        high
                       ,       int              matlIndex
                       ,       ParticleSubset * psubset
                       );


  inline bool hasRunningTask();

  inline std::map<std::thread::id, OnDemandDataWarehouse::RunningTaskInfo>* getRunningTasksInfo();

  inline RunningTaskInfo* getCurrentTaskInfo();


  // holds info on the running task for each std::thread::id
  std::map<std::thread::id, RunningTaskInfo> m_running_tasks {};

  ScrubMode m_scrub_mode {DataWarehouse::ScrubNone};

  DWDatabase<Patch>     m_var_DB       {};
  DWDatabase<Level>     m_level_DB     {};
  KeyDatabase<Patch>    m_var_key_DB   {};
  KeyDatabase<Level>    m_level_key_DB {};

  psetDBType            m_pset_db     {};
  psetDBType            m_delset_DB   {};
  psetAddDBType         m_addset_DB   {};
  particleQuantityType  m_foreign_particle_quantities {};
  bool                  m_exchange_particle_quantities {true};

  // Keep track of when this DW sent some (and which) particle information to another processor
  SendState m_send_state {};

  bool  m_finalized {false};
  GridP m_grid      {nullptr};

  // Is this the first DW -- created by the initialization timestep?
  bool  m_is_initialization_DW {false};

  //using for D2H copies only as of now. ONLY values used as of now are:  COPYING_IN, VALID (and reset ~VALID for invalid).
  //other Operations are handled by OnDemandDataWarehouse. copied from GPUDataWH

  Uintah::MasterLock * varLock;

  struct labelPatchMatlLevel {
    std::string label;
    int         patchID;
    int         matlIndx;
    int         levelIndx;

    labelPatchMatlLevel(const char * label, int patchID, int matlIndx, int levelIndx) {
      this->label = label;
      this->patchID = patchID;
      this->matlIndx = matlIndx;
      this->levelIndx = levelIndx;
    }

    //This so it can be used in an STL map
    bool operator<(const labelPatchMatlLevel& right) const {
      if (this->label < right.label) {
        return true;
      } else if (this->label == right.label && (this->patchID < right.patchID)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx < right.matlIndx)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx == right.matlIndx) && (this->levelIndx < right.levelIndx)) {
        return true;
      } else {
        return false;
      }
    }
  };

  enum status { UNALLOCATED               = 0x00000000,
                 ALLOCATING                = 0x00000001,
                 ALLOCATED                 = 0x00000002,
                 COPYING_IN                = 0x00000004,
                 VALID                     = 0x00000008,     //For when a variable has its data, this excludes any knowledge of ghost cells.
                 AWAITING_GHOST_COPY       = 0x00000010,     //For when when we know a variable is awaiting ghost cell data
                                                             //It is possible for VALID bit set to 0 or 1 with this bit set,
                                                             //meaning we can know a variable is awaiting ghost copies but we
                                                             //don't know from this bit alone if the variable is valid yet.
                 VALID_WITH_GHOSTS         = 0x00000020,     //For when a variable has its data and it has its ghost cells
                                                             //Note: Change to just GHOST_VALID?  Meaning ghost cells could be valid but the
                                                             //non ghost part is unknown?
                 DEALLOCATING              = 0x00000040,     //TODO: REMOVE THIS WHEN YOU CAN, IT'S NOT OPTIMAL DESIGN.
                 FORMING_SUPERPATCH        = 0x00000080,     //As the name suggests, when a number of individual patches are being formed
                                                             //into a superpatch, there is a period of time which other threads
                                                             //should wait until all patches have been processed.
                 SUPERPATCH                = 0x00000100,     //Indicates this patch is allocated as part of a superpatch.
                                                             //At the moment superpatches is only implemented for entire domain
                                                             //levels.  But it seems to make the most sense to have another set of
                                                             //logic in level.cc which subdivides a level into superpatches.
                                                             //If this bit is set, you should find the lowest numbered patch ID
                                                             //first and start with concurrency reads/writes there.  (Doing this
                                                             //avoids the Dining Philosopher's problem.
                 UNKNOWN                   = 0x00000200};    //Remove this when you can, unknown can be dangerous.
                                                             //It's only here to help track some host variables

  typedef volatile int atomicDataStatus;

  std::map<labelPatchMatlLevel, atomicDataStatus>   atomicStatusInHostMemory;	//maintain status of the variable in the host memory

}; // end class OnDemandDataWarehouse

}  // end namespace Uintah

#endif // end #ifndef UINTAH_COMPONENTS_SCHEDULERS_ONDEMANDDATAWAREHOUSE_H
