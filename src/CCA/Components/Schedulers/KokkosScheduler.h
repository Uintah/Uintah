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

#ifndef CCA_COMPONENTS_SCHEDULERS_KOKKOSSCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_KOKKOSSCHEDULER_H

#include <CCA/Components/Schedulers/MPIScheduler.h>

#ifdef HAVE_CUDA
  #include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
  #include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>
  #include <CCA/Components/Schedulers/GPUMemoryPool.h>
#endif

#include <sci_defs/cuda_defs.h>

#include <map>
#include <string>
#include <vector>

namespace Uintah {

class Task;
class DetailedTask;

/**************************************

CLASS
   KokkosScheduler


GENERAL INFORMATION
   KokkosScheduler.h

   Alan Humphrey, John Holmen, Brad Peterson, Damodar Sahasrabudhe
   Scientific Computing and Imaging Institute
   University of Utah


KEYWORDS
   Task Scheduler, Multi-threaded MPI, Kokkos, OpenMP, CPU, GPU

DESCRIPTION
   A multi-threaded scheduler that uses a combination of MPI + OpenMP and offers support
   for Kokkos-enabled CPU and GPU tasks. This relies on the Kokkos OpenMP back-end to
   manage OpenMP partitioning. Each partition has a "master" thread which runs a given
   functor.

   OpenMP 4.0 is required for this scheduler.

   This scheduler is designed primarily to support scheduling and executing
   a mixture of Kokkos-enabled tasks across CPU and GPU.

WARNING
   This scheduler is EXPERIMENTAL and undergoing extensive development.
   Not all tasks/components are Kokkos-enabled and/or thread-safe

   Requires MPI_THREAD_MULTIPLE support.

****************************************/


class KokkosScheduler : public MPIScheduler  {

  public:

  KokkosScheduler( const ProcessorGroup * myworld,
      KokkosScheduler * parentScheduler = nullptr );

    virtual ~KokkosScheduler();
    
    static int verifyAnyGpuActive();  // used only to check if this Uintah build can communicate with a GPU.  This function exits the program
    
    virtual void problemSetup( const ProblemSpecP & prob_spec, const MaterialManagerP & materialManager );
      
    virtual SchedulerP createSubScheduler();
    
    virtual void execute( int tgnum = 0, int iteration = 0 );
    
    virtual bool useInternalDeps() { return !m_is_copy_data_timestep; }
    
    void runTask( DetailedTask * dtask , int iteration , int thread_id , CallBackEvent event );

    void runTasks( int thread_id );

    static const int bufferPadding = 128;  // 32 threads can write floats out in one coalesced access.  (32 * 4 bytes = 128 bytes).
                                           // TODO: Ideally, this number should be determined from the CUDA arch during the
                                           // CMAKE/configure step so that future programmers don't have to manually remember to
                                           // update this value if it ever changes.

    static std::string myRankThread();

  private:

    // eliminate copy, assignment and move
    KokkosScheduler( const KokkosScheduler & )            = delete;
    KokkosScheduler& operator=( const KokkosScheduler & ) = delete;
    KokkosScheduler( KokkosScheduler && )                 = delete;
    KokkosScheduler& operator=( KokkosScheduler && )      = delete;

    void markTaskConsumed( int & numTasksDone, int & currphase, int numPhases, DetailedTask * dtask );

    // thread shared data, needs lock protection when accessed
    std::vector<int>             m_phase_tasks;
    std::vector<int>             m_phase_tasks_done;
    std::vector<DetailedTask*>   m_phase_sync_task;
    std::vector<int>             m_histogram;
    DetailedTasks              * m_detailed_tasks{nullptr};

    QueueAlg m_task_queue_alg{MostMessages};
    int      m_curr_iteration{0};
    int      m_num_tasks_done{0};
    int      m_num_tasks{0};
    int      m_curr_phase{0};
    int      m_num_phases{0};
    bool     m_abort{false};
    int      m_abort_point{0};

    int      m_num_partitions{0};
    int      m_threads_per_partition{0};

#ifdef HAVE_CUDA

    using DeviceVarDest = GpuUtilities::DeviceVarDestination;

    void assignStatusFlagsToPrepareACpuTask( DetailedTask * dtask );

    void assignDevicesAndStreams( DetailedTask* dtask );

    void assignDevicesAndStreamsFromGhostVars( DetailedTask* dtask );

    void findIntAndExtGpuDependencies( DetailedTask* dtask, int iteration, int t_id );

    void prepareGpuDependencies(       DetailedTask          * dtask
                               ,       DependencyBatch       * batch
                               , const VarLabel              * pos_var
                               ,       OnDemandDataWarehouse * dw
                               ,       OnDemandDataWarehouse * old_dw
                               , const DetailedDep           * dep
                               ,       DeviceVarDest           des
                               );

    void createTaskGpuDWs( DetailedTask * dtask );

    void gpuInitialize( bool reset = false );

    void syncTaskGpuDWs( DetailedTask * dtask );

    void performInternalGhostCellCopies( DetailedTask * dtask );

    void copyAllGpuToGpuDependences( DetailedTask * dtask );

    void copyAllExtGpuDependenciesToHost( DetailedTask * dtask );

    void initiateH2DCopies( DetailedTask * dtask );

    void turnIntoASuperPatch(GPUDataWarehouse* const       gpudw, 
                             const Level* const            level, 
                             const IntVector&              low,
                             const IntVector&              high,
                             const VarLabel* const         label, 
                             const Patch * const           patch, 
                             const int                     matlIndx, 
                             const int                     levelID ); 

    void prepareDeviceVars( DetailedTask * dtask );

    void copyDelayedDeviceVars( DetailedTask * dtask );

    bool delayedDeviceVarsValid( DetailedTask * dtask );	//check if the main patch is valid, not ghost cells.

    void prepareTaskVarsIntoTaskDW( DetailedTask * dtask );

    void prepareGhostCellsIntoTaskDW( DetailedTask * dtask );

    void markDeviceRequiresAndModifiesDataAsValid( DetailedTask * dtask );

    void markHostAsInvalid( DetailedTask * dtask );

    void markDeviceGhostsAsValid( DetailedTask * dtask );

    void markHostComputesDataAsValid( DetailedTask * dtask );

    void markDeviceComputesDataAsValid( DetailedTask * dtask );

    void markDeviceModifiesGhostAsInvalid( DetailedTask * dtask );

    void markHostRequiresAndModifiesDataAsValid( DetailedTask * dtask );

    void markDeviceAsInvalidHostAsValid( DetailedTask * dtask );

    void initiateD2HForHugeGhostCells( DetailedTask * dtask );

    void initiateD2H( DetailedTask * dtask);

    bool ghostCellsProcessingReady( DetailedTask * dtask );

    bool allHostVarsProcessingReady( DetailedTask * dtask );

    bool allGPUVarsProcessingReady( DetailedTask * dtask );

    void reclaimCudaStreamsIntoPool( DetailedTask * dtask );

    void freeCudaStreamsFromPool();

    cudaStream_t* getCudaStreamFromPool( int device );

    cudaError_t freeDeviceRequiresMem();

    cudaError_t freeComputesMem();

    void assignDevice( DetailedTask * task );

    struct GPUGridVariableInfo {

      GPUGridVariableInfo( DetailedTask * dtask
                         , double       * ptr
                         , IntVector      size
                         , int            device
                         )
        : m_dtask{dtask}
        , m_ptr{ptr}
        , m_size{size}
        , m_device{device}
      {}

      DetailedTask * m_dtask;
      double       * m_ptr;
      IntVector      m_size;
      int            m_device;
    };

    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_device_requires_ptrs;
    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_device_computes_ptrs;
    std::map<std::string, GPUGridVariableInfo>          m_device_computes_temp_ptrs;
    std::vector<VarLabel*>                              m_tmp_var_labels;

    std::vector<GPUGridVariableInfo>                    m_device_requires_allocation_ptrs;
    std::vector<GPUGridVariableInfo>                    m_device_computes_allocation_ptrs;
    std::vector<double*>                                m_host_computes_allocation_ptrs;

    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_host_requires_ptrs;
    std::map<VarLabelMatl<Patch>, GPUGridVariableInfo>  m_host_computes_ptrs;
    std::vector<std::queue<cudaEvent_t*> >              m_idle_events;

    int  m_num_devices;
    int  m_current_device;

    std::vector< std::string > m_material_names;

    struct labelPatchMatlDependency {

        labelPatchMatlDependency( const char          * label
                                ,       int             patchID
                                ,       int             matlIndex
                                ,       Task::DepType   depType
                                )
          : m_label{label}
          , m_patchID{patchID}
          , m_matlIndex{matlIndex}
          , m_depType{depType}
        {}

        // this so it can be used in an STL map
        bool operator<(const labelPatchMatlDependency& right) const
        {
          if (m_label < right.m_label) {
            return true;
          }
          else if (m_label == right.m_label && (m_patchID < right.m_patchID)) {
            return true;
          }
          else if (m_label == right.m_label && (m_patchID == right.m_patchID) && (m_matlIndex < right.m_matlIndex)) {
            return true;
          }
          else if (m_label == right.m_label && (m_patchID == right.m_patchID) && (m_matlIndex == right.m_matlIndex) && (m_depType < right.m_depType)) {
            return true;
          }
          else {
            return false;
          }
        }

        std::string   m_label;
        int           m_patchID;
        int           m_matlIndex;
        Task::DepType m_depType;
    };

#endif
};

} // namespace Uintah
   
#endif // CCA_COMPONENTS_SCHEDULERS_KOKKOSSCHEDULER_H
