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



#include <CCA/Components/Solvers/HypreSolver.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/StringUtil.h>
#include <Core/Util/Timers/Timers.hpp>
#include <iomanip>
#include <Core/Parallel/Portability.h>

// hypre includes
#include <_hypre_struct_mv.h>
#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>
#include <krylov.h>

//#define PRINTSYSTEM

#ifndef HYPRE_TIMING
#ifndef hypre_ClearTiming
// This isn't in utilities.h for some reason...
#define hypre_ClearTiming()
#endif
#endif

using namespace std;
using namespace Uintah;

//HYPRE_USING_CUDA gets defined in HYPRE_config.h if hypre is configured with cuda.
//if HYPRE_USING_CUDA or HYPRE_USING_KOKKOS (with kokkos-cuda backend) is enabled, 
//copy all values to gpu memory buffer and then pass it to SetBoxValues. Rest everything remains same.
//This is a temporary solution to get the code working. TODO: 
//1. Analyze performance of hypre gpu solve without considering the copying time. (look at solve only time)
//2. Improve performance, if possible. 
//3. If gpu hypre performs faster than mpi only cpu hypre, convert this code into portable task
//   (similar to CharOx)
//4. If gpu hypre can not perform, possibly use cpu only version. (may be with thread as rank approach)

//-------------DS: 04262019: Added to run hypre task using hypre-cuda.----------------
#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
#define cudaErrorCheck(err) \
  if(err != cudaSuccess) { \
    printf("error in cuda call at %s: %d. %s: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
    exit(1); \
  }
#endif
//-----------------  end of hypre-cuda  -----------------

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "SOLVER_DOING_COUT:+"

static DebugStream cout_doing("SOLVER_DOING_COUT", false);

namespace Uintah {

  void swapbytes( Uintah::hypre_solver_structP& ) {
    SCI_THROW(InternalError("Swap bytes for hypre_solver_structP is not implemented", __FILE__, __LINE__));
  }

  //==============================================================================
  //
  // Class HypreStencil7
  //
  //==============================================================================
  template<class GridVarType>
  class HypreStencil7 : public RefCounted {
  public:
    HypreStencil7( const Level          * level_in
                 , const MaterialSet    * matlset_in
                 , const VarLabel       * A_in
                 ,       Task::WhichDW    which_A_dw_in
                 , const VarLabel       * x_in
                 ,       bool             modifies_X_in
                 , const VarLabel       * b_in
                 ,       Task::WhichDW    which_b_dw_in
                 , const VarLabel       * guess_in
                 ,       Task::WhichDW    which_guess_dw_in
                 , const HypreParams    * params_in
                 ,       bool             isFirstSolve_in
                 )
      : m_level(level_in)
      , m_matlset(matlset_in)
      , m_A_label(A_in)
      , m_which_A_dw(which_A_dw_in)
      , m_X_label(x_in)
      , m_modifies_X(modifies_X_in)
      , m_b_label(b_in)
      , m_which_b_dw(which_b_dw_in)
      , m_guess_label(guess_in)
      , m_which_guess_dw(which_guess_dw_in)
      , m_params(params_in)
      , m_isFirstSolve(isFirstSolve_in)
    {
      // Time Step
      m_timeStepLabel    = VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );
      m_hypre_solver_label = VarLabel::create("hypre_solver_label",
                                            SoleVariable<hypre_solver_structP>::getTypeDescription());
      m_firstPassThrough = true;
      m_movingAverage    = 0.0;

      const char* hypre_superpatch_str = std::getenv("HYPRE_SUPERPATCH"); //use diff env variable if it conflicts with OMP. but using same will be consistent.
      if(hypre_superpatch_str){
        m_superpatch = atoi(hypre_superpatch_str);
      }

    }

    //---------------------------------------------------------------------------------------------

    virtual ~HypreStencil7() {
      VarLabel::destroy(m_timeStepLabel);
      VarLabel::destroy(m_hypre_solver_label);

      //-------------DS: 04262019: Added to run hypre task using hypre-cuda.----------------
#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
      if(m_buff){
        cudaErrorCheck(cudaFree(m_buff));
      }
#else
      if(m_buff){
        free(m_buff);
      }
#endif
      //-----------------  end of hypre-cuda  -----------------

    }


    //---------------------------------------------------------------------------------------------
    void getPatchExtents( const Patch      * patch
                          ,     IntVector  & lo
                          ,     IntVector  & hi)
    {
      typedef typename GridVarType::double_type double_type;
      Patch::VariableBasis basis = Patch::translateTypeToBasis(double_type::getTypeDescription()->getType(), true);

      if( m_params->getSolveOnExtraCells()) {
        lo  = patch->getExtraLowIndex(  basis, IntVector(0,0,0) );
        hi  = patch->getExtraHighIndex( basis, IntVector(0,0,0) );
      } else {
        lo = patch->getLowIndex(  basis );
        hi = patch->getHighIndex( basis );
      }
    }

    //---------------------------------------------------------------------------------------------
    double * getBuffer( size_t buff_size )
    {
      if (m_buff_size < buff_size) {
        m_buff_size = buff_size;

#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
        if (m_buff) {
          cudaErrorCheck(cudaFree((void*)m_buff));
        }

        cudaErrorCheck(cudaMalloc((void**)&m_buff, buff_size));
#else
        if (m_buff) {
          free(m_buff);
        }

        m_buff = (double *)malloc(buff_size);
#endif
      }

      return m_buff; // although m_buff is a member of the class and can be accessed inside task, it can not be directly
                     // accessed inside parallel_for on device (even though its a device pointer, value is not passed by reference)
                     // So return explicitly to a local variable. The local variable gets passed by copy.
    }


    //---------------------------------------------------------------------------------------------
    //   Create and populate a Hypre struct vector,
    template <typename ExecSpace, typename MemSpace>
    HYPRE_StructVector
    createPopulateHypreVector(   const timeStep_vartype   timeStep
                               , const bool               recompute
                               , const bool               do_setup
                               , const ProcessorGroup   * pg
                               , const HYPRE_StructGrid & grid
                               , const PatchSubset      * patches
                               , const int                matl
                               , const VarLabel         * Q_label
                               , OnDemandDataWarehouse  * Q_dw
                               , HYPRE_StructVector     * HQ
                               , ExecutionObject<ExecSpace, MemSpace> & execObj)
    {
      //__________________________________
      // Create the vector
      if( do_setup ){
        HYPRE_StructVectorDestroy( *HQ );
      }

      if (timeStep == 1 || recompute || do_setup ) {
        HYPRE_StructVectorCreate( pg->getComm(), grid, HQ );
        HYPRE_StructVectorInitialize( *HQ );
      }

      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);

        //__________________________________
        // Get Q
        if( Q_label ){
          ostringstream msg;
          msg<< "HypreSolver:createPopulateHypreVector ("<< Q_label->getName() <<")\n";
          printTask( patches, patch, cout_doing, msg.str() );

          //Q type should be "typename GridVarType::const_double_type" on host and KokkosView3 based on const_double_type on device;
          auto Q = Q_dw->getConstGridVariable<typename GridVarType::const_double_type, double, MemSpace> (Q_label, matl, patch, Ghost::None, 0);

          // find box range
          IntVector lo;
          IntVector hi;
          getPatchExtents( patch, lo, hi );

          //-------------DS: 04262019: Added to run hypre task using hypre-cuda and used Uintah::parallel_for to copy values portably.----------------
          //existing invokes cuda kernel inside HYPRE_StructMatrixSetBoxValues Ny*Nz into times. Copying entire patch into the buffer and then
          //calling HYPRE_StructMatrixSetBoxValues will lead to only 2 kernel calls - 1 parallel_for to copy values into the buffer and
          //1 kernel call by HYPRE_StructMatrixSetBoxValues. Although new method needs extra buffer and no cache reuse, it still should be faster than existing code
          IntVector hh(hi.x()-1, hi.y()-1, hi.z()-1);
          Uintah::BlockRange range( lo, hi );
          unsigned long Nx = abs(hi.x()-lo.x()), Ny = abs(hi.y()-lo.y()), Nz = abs(hi.z()-lo.z());
          int start_offset = lo.x() + lo.y()*Nx + lo.z()*Nx*Ny; //ensure starting point is 0 while indexing d_buff
          size_t buff_size = Nx*Ny*Nz*sizeof(double);
          double * d_buff = getBuffer( buff_size );	//allocate / reallocate d_buff;

          Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
            int id = (i + j*Nx + k*Nx*Ny - start_offset);
            d_buff[id] = Q(i, j, k);
          });

          HYPRE_StructVectorSetBoxValues( *HQ,
                                         lo.get_pointer(), hh.get_pointer(),
                                         d_buff);
        }  // label exist?
      }  // patch loop

      if (timeStep == 1 || recompute || do_setup){
        HYPRE_StructVectorAssemble( *HQ );
      }

      return *HQ;
    }

    //---------------------------------------------------------------------------------------------
    template <typename ExecSpace, typename MemSpace>
    void solve( const PatchSubset                          * patches
              , const MaterialSubset                       * matls
              ,       OnDemandDataWarehouse                * old_dw
              ,       OnDemandDataWarehouse                * new_dw
              ,       UintahParams                         & uintahParams
              ,       ExecutionObject<ExecSpace, MemSpace> & execObj
              ,       Handle<HypreStencil7<GridVarType>>
              )
    {
      const ProcessorGroup * pg = uintahParams.getProcessorGroup();

      if(pg->myRank()==0){
#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
        bool hypre_cuda = true;
#else
        bool hypre_cuda = false;
#endif
        if(std::is_same<ExecSpace, Kokkos::Cuda>::value){
          if(hypre_cuda == false){
            printf("######  Error at file %s, line %d: ExecSpace of HypreSolver task in Uintah is cuda, but hypre is NOT configured with cuda. ######\n", __FILE__, __LINE__);
            exit(1);
          }
        }
        else{
          if(hypre_cuda == true){
            printf("######  Error at file %s, line %d: ExecSpace of HypreSolver task in Uintah is CPU, but hypre is configured with cuda. ######\n", __FILE__, __LINE__);
            exit(1);
          }
        }
      }

      //__________________________________
      //   timers
      m_tHypreAll = hypre_InitializeTiming("Total Hypre time");
      hypre_BeginTiming(m_tHypreAll);

      m_tMatVecSetup = hypre_InitializeTiming("Matrix + Vector setup");
      m_tSolveOnly   = hypre_InitializeTiming("Solve time");


       //________________________________________________________
      // get struct from data warehouse
      struct hypre_solver_struct* hypre_solver_s = 0;

      if ( new_dw->exists( m_hypre_solver_label ) ) {
        new_dw->get( m_hypre_solverP, m_hypre_solver_label );
      }
      else {
        old_dw->get( m_hypre_solverP, m_hypre_solver_label );
        new_dw->put( m_hypre_solverP, m_hypre_solver_label );
      }

      hypre_solver_s = m_hypre_solverP.get().get_rep();
      bool recompute = hypre_solver_s->isRecomputeTimeStep;

      //__________________________________
      // timestep can come from the old_dw or parentOldDW
      timeStep_vartype timeStep(0);

      Task::WhichDW myOldDW = m_params->getWhichOldDW();
      DataWarehouse* pOldDW = new_dw->getOtherDataWarehouse(myOldDW);

      pOldDW->get(timeStep, m_timeStepLabel);

      //________________________________________________________
      // Solve frequency
      //
      const int solvFreq = m_params->solveFrequency;
      // note - the first timeStep in hypre is timeStep 1
      if (solvFreq == 0 || timeStep % solvFreq ) {
        new_dw->transferFrom(old_dw, m_X_label, patches, matls, true);
        return;
      }

      //________________________________________________________
      // Matrix setup frequency - this will destroy and recreate a new Hypre matrix at the specified setupFrequency
      //
      int suFreq = m_params->getSetupFrequency();
      bool do_setup = false;
      if (suFreq != 0){
        do_setup = (timeStep % suFreq == 0);
      }

      //________________________________________________________
      // update coefficient frequency - This will ONLY UPDATE the matrix coefficients without destroying/recreating the Hypre Matrix
      //
      const int updateCoefFreq = m_params->getUpdateCoefFrequency();
     bool updateCoefs = true;
      if (updateCoefFreq != 0){
        updateCoefs = (timeStep % updateCoefFreq == 0);
      }

      // if it the first pass through ignore the flags
      if(timeStep == 1 || recompute){
        updateCoefs = false;
        do_setup    = false;
      }

      //std::cout << "      HypreSolve  timestep: " << timeStep << " recompute: " << recompute << " m_firstPassThrough: " << m_firstPassThrough <<  " m_isFirstSolve: " << m_isFirstSolve <<" do_setup: " << do_setup << " updateCoefs: " << updateCoefs << std::endl;

      OnDemandDataWarehouse* A_dw     = reinterpret_cast<OnDemandDataWarehouse *>(new_dw->getOtherDataWarehouse( m_which_A_dw ));
      OnDemandDataWarehouse* b_dw     = reinterpret_cast<OnDemandDataWarehouse *>(new_dw->getOtherDataWarehouse( m_which_b_dw ));
      OnDemandDataWarehouse* guess_dw = reinterpret_cast<OnDemandDataWarehouse *>(new_dw->getOtherDataWarehouse( m_which_guess_dw ));

      ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));

      Timers::Simple timer;
      timer.start();

      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);

        hypre_BeginTiming(m_tMatVecSetup);
        //__________________________________
        // Setup grid
        HYPRE_StructGrid grid;
        if (timeStep == 1 || do_setup || recompute) {
          HYPRE_StructGridCreate(pg->getComm(), 3, &grid);

          if(m_superpatch){ //if m_superpatch is set then pass patch(0).lo and patch(n-1).hi to HYPRE_StructGridSetExtents. Then hypre will treat the rank's subdomain as one giant superpatch

            IntVector  lo, hi, superlo, superhi;
            getPatchExtents( patches->get(0), superlo, hi );  //lo of 0th patch will be superlo
            getPatchExtents( patches->get(patches->size()-1), lo, superhi ); //hi of n-1 th patch will be superhi
            unsigned long supercells = (superhi[0] - superlo[0]) * (superhi[1] - superlo[1]) * (superhi[2] - superlo[2]); //num cells in super patch
            unsigned long totcells = 0;

            if(m_superpatch_bulletproof==false){
              //check whether all patches fall within superpatch boundary. Converse checked by comparing number of cells which should match.
              for(int p=0;p<patches->size();p++){
                const Patch* patch = patches->get(p);
                getPatchExtents( patch, lo, hi );

                if(superlo <= lo && hi <= superhi){  //patch falls within superpatch boundaries.
                  totcells += (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
                }
                else{//raise error if patch is outside superpatch boundaries
                  printf("*** Error: super patch can not be used for this domain decomposition.  ***\n");
                  printf("rank %d: superlo [%d %d %d], superhi [%d %d %d], lo [%d %d %d], hi [%d %d %d] for patch %d at %s %d\n",
                         pg->myRank(), superlo[0], superlo[1], superlo[2], superhi[0], superhi[1], superhi[2], lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], patch->getID(), __FILE__, __LINE__ );
                  fflush(stdout);
                  exit(1);
                }
              }
              if(supercells != totcells){
              printf("*** Error: super patch can not be used for this domain decomposition.  ***\n");
              printf("rank %d: superlo [%d %d %d], superhi [%d %d %d] super patch has extra cells than the subdomain assigned to this rank at %s %d\n",
                     pg->myRank(), superlo[0], superlo[1], superlo[2], superhi[0], superhi[1], superhi[2], __FILE__, __LINE__ );
              fflush(stdout);
              exit(1);
              }
              m_superpatch_bulletproof = true;
            }

            if(pg->myRank()==0){
              printf("Warning: Using an experimental superpatch for hypre\n");
            }
            superhi -= IntVector(1,1,1);
            HYPRE_StructGridSetExtents(grid, superlo.get_pointer(), superhi.get_pointer()); //pass super patch boundaries to hypre
          }
          else{//add individual patches as they are without merging into super patch. Existing code as it is
            for(int p=0;p<patches->size();p++){
              const Patch* patch = patches->get(p);

              IntVector lo;
              IntVector hi;
              getPatchExtents( patch, lo, hi );
              hi -= IntVector(1,1,1);

              HYPRE_StructGridSetExtents(grid, lo.get_pointer(), hi.get_pointer());
            }
          }
          // Periodic boundaries
          const Level* level = getLevel(patches);
          IntVector periodic_vector = level->getPeriodicBoundaries();

          IntVector low, high;
          level->findCellIndexRange(low, high);
          IntVector range = high-low;

          int periodic[3];
          periodic[0] = periodic_vector.x() * range.x();
          periodic[1] = periodic_vector.y() * range.y();
          periodic[2] = periodic_vector.z() * range.z();
          HYPRE_StructGridSetPeriodic(grid, periodic);

          // Assemble the grid
          HYPRE_StructGridAssemble(grid);
        }

        //__________________________________
        // Create the stencil
        HYPRE_StructStencil stencil;
        if ( timeStep == 1 || do_setup || recompute) {
          if( m_params->getSymmetric()){

            HYPRE_StructStencilCreate(3, 4, &stencil);
            int offsets[4][3] = {{0,0,0},
              {-1,0,0},
              {0,-1,0},
              {0,0,-1}};
            for(int i=0;i<4;i++) {
              HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
            }

          } else {

            HYPRE_StructStencilCreate(3, 7, &stencil);
            int offsets[7][3] = {{0,0,0},
              {1,0,0}, {-1,0,0},
              {0,1,0}, {0,-1,0},
              {0,0,1}, {0,0,-1}};

            for(int i=0;i<7;i++){
              HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
            }
          }
        }

        //__________________________________
        // Create the matrix
        HYPRE_StructMatrix* HA = hypre_solver_s->HA_p;

        if( do_setup ){
          HYPRE_StructMatrixDestroy( *HA );
        }

        if (timeStep == 1 || recompute || do_setup) {
          HYPRE_StructMatrixCreate( pg->getComm(), grid, stencil, HA );
          HYPRE_StructMatrixSetSymmetric( *HA, m_params->getSymmetric() );
          int ghost[] = {1,1,1,1,1,1};
          HYPRE_StructMatrixSetNumGhost( *HA, ghost );
          HYPRE_StructMatrixInitialize( *HA );
        }

        // setup the coefficient matrix ONLY on the first timeStep, if
        // we are doing a recompute, or if we set setupFrequency != 0,
        // or if UpdateCoefFrequency != 0
        if (timeStep == 1 || recompute || do_setup || updateCoefs) {
          for(int p=0;p<patches->size();p++) {
            const Patch* patch = patches->get(p);
            printTask( patches, patch, cout_doing, "HypreSolver:solve: Create Matrix" );

            IntVector l;
            IntVector h;
            getPatchExtents( patch, l, h );

            //-------------DS: 04262019: Added to run hypre task using hypre-cuda and used Uintah::parallel_for to copy values portably.----------------
            //existing invokes cuda kernel inside HYPRE_StructMatrixSetBoxValues Ny*Nz into times. Copying entire patch into the buffer and then
            //calling HYPRE_StructMatrixSetBoxValues will lead to only 2 kernel calls - 1 parallel_for to copy values into the buffer and
            //1 kernel call by HYPRE_StructMatrixSetBoxValues. Although new method needs extra buffer and no cache reuse, it still should be faster than existing code
            IntVector hh(h.x()-1, h.y()-1, h.z()-1);
            Uintah::BlockRange range( l, h );
            int stencil_point = ( m_params->getSymmetric()) ? 4 : 7;
            unsigned long Nx = abs(h.x()-l.x()), Ny = abs(h.y()-l.y()), Nz = abs(h.z()-l.z());
            int start_offset = l.x() + l.y()*Nx + l.z()*Nx*Ny; //ensure starting point is 0 while indexing d_buff
            size_t buff_size = Nx*Ny*Nz*sizeof(double)*stencil_point;
            double * d_buff = getBuffer( buff_size );	//allocate / reallocate d_buff;
            //-----------------  end of hypre-cuda  -----------------

            //__________________________________
            // Feed it to Hypre
            int stencil_indices[] = {0,1,2,3,4,5,6};

            if( m_params->getSymmetric()){

              // use stencil4 as coefficient matrix. NOTE: This should be templated
              // on the stencil type. This workaround is to get things moving
              // until we convince component developers to move to stencil4. You must
              // set m_params->setUseStencil4(true) when you setup your linear solver
              // if you want to use stencil4. You must also provide a matrix of type
              // stencil4 otherwise this will crash.
              if ( m_params->getUseStencil4()) {

                auto AStencil4 = (A_dw->getConstGridVariable<typename GridVarType::symmetric_matrix_type, Stencil4, MemSpace> (m_A_label, matl, patch, Ghost::None, 0));

                Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
                  int id = (i + j*Nx + k*Nx*Ny - start_offset)*stencil_point;
                  d_buff[id + 0] = AStencil4(i, j, k).p;
                  d_buff[id + 1] = AStencil4(i, j, k).w;
                  d_buff[id + 2] = AStencil4(i, j, k).s;
                  d_buff[id + 3] = AStencil4(i, j, k).b;
                });

              } else { // use stencil7

                auto A = (A_dw->getConstGridVariable<typename GridVarType::matrix_type, Stencil7, MemSpace> (m_A_label, matl, patch, Ghost::None, 0));

                Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
                  int id = (i + j*Nx + k*Nx*Ny - start_offset)*stencil_point;
                  d_buff[id + 0] = A(i, j, k).p;
                  d_buff[id + 1] = A(i, j, k).w;
                  d_buff[id + 2] = A(i, j, k).s;
                  d_buff[id + 3] = A(i, j, k).b;
                });

              }
            } else { // if( m_params->getSymmetric())

              auto A = (A_dw->getConstGridVariable<typename GridVarType::matrix_type, Stencil7, MemSpace> (m_A_label, matl, patch, Ghost::None, 0));

              Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
                int id = (i + j*Nx + k*Nx*Ny - start_offset)*stencil_point;
                d_buff[id + 0] = A(i, j, k).p;
                d_buff[id + 1] = A(i, j, k).e;
                d_buff[id + 2] = A(i, j, k).w;
                d_buff[id + 3] = A(i, j, k).n;
                d_buff[id + 4] = A(i, j, k).s;
                d_buff[id + 5] = A(i, j, k).t;
                d_buff[id + 6] = A(i, j, k).b;
              });
            }

            HYPRE_StructMatrixSetBoxValues(*HA,
                                           l.get_pointer(), hh.get_pointer(),
                                           stencil_point, stencil_indices,
                                           d_buff);
          }
          if (timeStep == 1 || recompute || do_setup){
            HYPRE_StructMatrixAssemble(*HA);
          }
        }

        //__________________________________
        // Create the RHS
        HYPRE_StructVector HB;
        HB = createPopulateHypreVector<ExecSpace, MemSpace>(  timeStep, recompute, do_setup, pg, grid, patches, matl, m_b_label, b_dw, hypre_solver_s->HB_p, execObj);

        //__________________________________
        // Create the solution vector
        HYPRE_StructVector HX;
        HX = createPopulateHypreVector<ExecSpace, MemSpace>(  timeStep, recompute, do_setup, pg, grid, patches, matl, m_guess_label, guess_dw, hypre_solver_s->HX_p, execObj);

        hypre_EndTiming( m_tMatVecSetup );

        //__________________________________
        Timers::Simple solve_timer;
        solve_timer.start();

        hypre_BeginTiming(m_tSolveOnly);

        int num_iterations;
        double final_res_norm;

        //______________________________________________________________________
        // Solve the system
        switch( hypre_solver_s->solver_type ){
        //__________________________________
        // use symmetric SMG
        case smg: {
          HYPRE_StructSolver * solver  = hypre_solver_s->solver_p;

          if ( do_setup ){
            HYPRE_StructSMGDestroy( *solver );
          }

          if (timeStep == 1 || recompute || do_setup) {

            HYPRE_StructSMGCreate         (pg->getComm(), solver);
            HYPRE_StructSMGSetMemoryUse   (*solver,  0);
            HYPRE_StructSMGSetMaxIter     (*solver,  m_params->maxiterations);
            HYPRE_StructSMGSetTol         (*solver,  m_params->tolerance);
            HYPRE_StructSMGSetRelChange   (*solver,  0);
            HYPRE_StructSMGSetNumPreRelax (*solver,  m_params->npre);
            HYPRE_StructSMGSetNumPostRelax(*solver,  m_params->npost);
            HYPRE_StructSMGSetLogging     (*solver,  m_params->logging);

            HYPRE_StructSMGSetup (*solver,  *HA, HB, HX);
          }

          HYPRE_StructSMGSolve(*solver, *HA, HB, HX);

          HYPRE_StructSMGGetNumIterations( *solver, &num_iterations );
          HYPRE_StructSMGGetFinalRelativeResidualNorm( *solver, &final_res_norm );
          break;
        }
        //______________________________________________________________________
        //
        case pfmg:{

          HYPRE_StructSolver* solver =  hypre_solver_s->solver_p;

          if ( do_setup ){
            HYPRE_StructPFMGDestroy( *solver );
          }

          if ( timeStep == 1 || recompute || do_setup ) {

            HYPRE_StructPFMGCreate        ( pg->getComm(), solver );
            HYPRE_StructPFMGSetMaxIter    (*solver,   m_params->maxiterations);
            HYPRE_StructPFMGSetTol        (*solver,   m_params->tolerance);
            HYPRE_StructPFMGSetRelChange  (*solver,   0);

            /* weighted Jacobi = 1; red-black GS = 2 */
            HYPRE_StructPFMGSetRelaxType   (*solver,  m_params->relax_type);
            HYPRE_StructPFMGSetNumPreRelax (*solver,  m_params->npre);
            HYPRE_StructPFMGSetNumPostRelax(*solver,  m_params->npost);
            HYPRE_StructPFMGSetSkipRelax   (*solver,  m_params->skip);
            HYPRE_StructPFMGSetLogging     (*solver,  m_params->logging);

            HYPRE_StructPFMGSetup          (*solver,  *HA, HB,  HX);
          }

          HYPRE_StructPFMGSolve(*solver, *HA, HB, HX);

          HYPRE_StructPFMGGetNumIterations(*solver, &num_iterations);
          HYPRE_StructPFMGGetFinalRelativeResidualNorm(*solver,
                                                       &final_res_norm);
          break;
        }
        //______________________________________________________________________
        //
        case sparsemsg:{

          HYPRE_StructSolver* solver = hypre_solver_s->solver_p;
          if ( do_setup ){
            HYPRE_StructSparseMSGDestroy(*solver);
          }

          if ( timeStep == 1 || recompute || do_setup ) {

            HYPRE_StructSparseMSGCreate      (pg->getComm(), solver);
            HYPRE_StructSparseMSGSetMaxIter  (*solver, m_params->maxiterations);
            HYPRE_StructSparseMSGSetJump     (*solver, m_params->jump);
            HYPRE_StructSparseMSGSetTol      (*solver, m_params->tolerance);
            HYPRE_StructSparseMSGSetRelChange(*solver, 0);

            /* weighted Jacobi = 1; red-black GS = 2 */
            HYPRE_StructSparseMSGSetRelaxType   (*solver,  m_params->relax_type);
            HYPRE_StructSparseMSGSetNumPreRelax (*solver,  m_params->npre);
            HYPRE_StructSparseMSGSetNumPostRelax(*solver,  m_params->npost);
            HYPRE_StructSparseMSGSetLogging     (*solver,  m_params->logging);

            HYPRE_StructSparseMSGSetup(*solver, *HA, HB,  HX);
          }

          HYPRE_StructSparseMSGSolve(*solver, *HA, HB, HX);

          HYPRE_StructSparseMSGGetNumIterations(*solver, &num_iterations);
          HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(*solver,
                                                            &final_res_norm);
          break;
        }
        //______________________________________________________________________
        //
        case pcg: {

          HYPRE_StructSolver * solver         = hypre_solver_s->solver_p;
          HYPRE_StructSolver * precond_solver = hypre_solver_s->precond_solver_p;

          if( do_setup ){
            destroyPrecond( hypre_solver_s, *precond_solver );
            HYPRE_StructPCGDestroy(*solver);
          }

          if (timeStep == 1 || recompute || do_setup) {
            HYPRE_StructPCGCreate(pg->getComm(),solver);

            HYPRE_PtrToStructSolverFcn precond;
            HYPRE_PtrToStructSolverFcn precond_setup;

            setupPrecond( pg, precond, precond_setup, hypre_solver_s, *precond_solver );
            HYPRE_StructPCGSetPrecond( *solver, precond, precond_setup, *precond_solver );

            HYPRE_StructPCGSetMaxIter   (*solver, m_params->maxiterations);
            HYPRE_StructPCGSetTol       (*solver, m_params->tolerance);
            HYPRE_StructPCGSetTwoNorm   (*solver,  1);
            HYPRE_StructPCGSetRelChange (*solver,  0);
            HYPRE_StructPCGSetLogging   (*solver,  m_params->logging);

            HYPRE_StructPCGSetup        (*solver, *HA,HB, HX);
          }

          HYPRE_StructPCGSolve(*solver, *HA, HB, HX);

          HYPRE_StructPCGGetNumIterations(*solver, &num_iterations);
          HYPRE_StructPCGGetFinalRelativeResidualNorm(*solver,&final_res_norm);
          break;
        }

        //______________________________________________________________________
        //
        case hybrid: {

          HYPRE_StructSolver * solver         = hypre_solver_s->solver_p;
          HYPRE_StructSolver * precond_solver = hypre_solver_s->precond_solver_p;

          if ( do_setup ){
            destroyPrecond( hypre_solver_s, *precond_solver );
            HYPRE_StructHybridDestroy( *solver );
          }

          if ( timeStep == 1 || recompute || do_setup ) {
            HYPRE_StructHybridCreate(pg->getComm(), solver);

            HYPRE_PtrToStructSolverFcn precond;
            HYPRE_PtrToStructSolverFcn precond_setup;

            setupPrecond( pg, precond, precond_setup, hypre_solver_s, *precond_solver );
            HYPRE_StructHybridSetPrecond( *solver, precond, precond_setup, *precond_solver );

            HYPRE_StructHybridSetDSCGMaxIter    (*solver, 100);
            HYPRE_StructHybridSetPCGMaxIter     (*solver, m_params->maxiterations);
            HYPRE_StructHybridSetTol            (*solver, m_params->tolerance);
            HYPRE_StructHybridSetConvergenceTol (*solver, 0.90);
            HYPRE_StructHybridSetTwoNorm        (*solver, 1);
            HYPRE_StructHybridSetRelChange      (*solver, 0);
            HYPRE_StructHybridSetLogging        (*solver, m_params->logging);

            HYPRE_StructHybridSetup             (*solver, *HA, HB, HX);
          }

          HYPRE_StructHybridSolve(*solver, *HA, HB, HX);

          HYPRE_StructHybridGetNumIterations( *solver,&num_iterations );
          HYPRE_StructHybridGetFinalRelativeResidualNorm( *solver, &final_res_norm );
          break;
        }
        //______________________________________________________________________
        //
        case gmres: {

          HYPRE_StructSolver * solver         = hypre_solver_s->solver_p;
          HYPRE_StructSolver * precond_solver = hypre_solver_s->precond_solver_p;

          if ( do_setup ){
            destroyPrecond( hypre_solver_s, *precond_solver );
            HYPRE_StructGMRESDestroy(*solver);
          }
          if (timeStep == 1 || recompute || do_setup ) {
            HYPRE_StructGMRESCreate(pg->getComm(),solver);

            HYPRE_PtrToStructSolverFcn precond;
            HYPRE_PtrToStructSolverFcn precond_setup;

            setupPrecond( pg, precond, precond_setup, hypre_solver_s, *precond_solver );
            HYPRE_StructGMRESSetPrecond  (*solver, precond, precond_setup, *precond_solver );

            HYPRE_StructGMRESSetMaxIter  (*solver, m_params->maxiterations);
            HYPRE_StructGMRESSetTol      (*solver, m_params->tolerance);
            HYPRE_GMRESSetRelChange      ( (HYPRE_Solver)solver, 0);
            HYPRE_StructGMRESSetLogging  (*solver, m_params->logging);

            HYPRE_StructGMRESSetup       (*solver,*HA,HB,HX);
          }

          HYPRE_StructGMRESSolve(*solver,*HA,HB,HX);

          HYPRE_StructGMRESGetNumIterations(*solver, &num_iterations);
          HYPRE_StructGMRESGetFinalRelativeResidualNorm(*solver, &final_res_norm);
          break;
        }
        default:
          throw InternalError("Unknown solver type: "+ m_params->solvertype, __FILE__, __LINE__);
        }

        //______________________________________________________________________
        //
#ifdef PRINTSYSTEM
        //__________________________________
        //   Debugging
        vector<string> fname;
        m_params->getOutputFileName(fname);
        HYPRE_StructMatrixPrint( fname[0].c_str(), *HA, 0 );
        HYPRE_StructVectorPrint( fname[1].c_str(), *HB, 0 );
        HYPRE_StructVectorPrint( fname[2].c_str(), HX, 0 );
#endif

        printTask( patches, patches->get(0), cout_doing, "HypreSolver:solve: testConvergence" );

        solve_timer.stop();
        hypre_EndTiming ( m_tSolveOnly );

        //__________________________________
        // Push the solution into Uintah data structure
        for(int p=0;p<patches->size();p++){
          const Patch* patch = patches->get(p);
          printTask( patches, patch, cout_doing, "HypreSolver:solve: copy solution" );

          IntVector l;
          IntVector h;
          getPatchExtents( patch, l, h );

          CellIterator iter(l, h);

          auto Xnew = new_dw->getGridVariable<typename GridVarType::double_type, double, MemSpace> (m_X_label, matl, patch, Ghost::None, 0, m_modifies_X);

          Uintah::BlockRange range( l, h );

          //-------------DS: 04262019: Added to run hypre task using hypre-cuda and used Uintah::parallel_for to copy values portably.----------------
          //existing invokes cuda kernel inside HYPRE_StructVectorGetBoxValues Ny*Nz into times. Copying entire patch into the buffer and then
          //calling HYPRE_StructMatrixSetBoxValues will lead to only 2 kernel calls - 1 parallel_for to copy values into the buffer and
          //1 kernel call by HYPRE_StructMatrixSetBoxValues. Although new method needs extra buffer and no cache reuse, it still should be faster than existing code
          IntVector hh(h.x()-1, h.y()-1, h.z()-1);
          unsigned long Nx = abs(h.x()-l.x()), Ny = abs(h.y()-l.y()), Nz = abs(h.z()-l.z());
          int start_offset = l.x() + l.y()*Nx + l.z()*Nx*Ny; //ensure starting point is 0 while indexing d_buff
          size_t buff_size = Nx*Ny*Nz*sizeof(double);
          double * d_buff = getBuffer( buff_size );	//allocate / reallocate d_buff;

          // Get the solution back from hypre
          HYPRE_StructVectorGetBoxValues(HX,
              l.get_pointer(), hh.get_pointer(),
              d_buff);

          Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
            int id = (i + j*Nx + k*Nx*Ny - start_offset);
            Xnew(i, j, k) = d_buff[id];
          });
        }
        //__________________________________
        // clean up
         m_firstPassThrough  = false;
         hypre_solver_s->isRecomputeTimeStep  = false;

        if ( timeStep == 1 || do_setup || recompute ) {
          HYPRE_StructStencilDestroy(stencil);
          HYPRE_StructGridDestroy(grid);
        }

        hypre_EndTiming (m_tHypreAll);

        hypre_PrintTiming   ("Hypre Timings:", pg->getComm());
        hypre_FinalizeTiming( m_tMatVecSetup );
        hypre_FinalizeTiming( m_tSolveOnly );
        hypre_FinalizeTiming( m_tHypreAll );
        hypre_ClearTiming();

        timer.stop();

        if(pg->myRank() == 0) {

          cout << "Solve of " << m_X_label->getName()
               << " on level " << m_level->getIndex()
               << " completed in " << timer().seconds()
               << " s (solve only: " << solve_timer().seconds() << " s, ";

          if (timeStep > 2) {
            // alpha = 2/(N+1)
            // averaging window is 10 timeSteps.
            double alpha   = 2.0/(std::min( int(timeStep) - 2, 10) + 1);
            m_movingAverage = alpha*solve_timer().seconds() + (1-alpha) * m_movingAverage;

            cout << "mean: " <<  m_movingAverage << " s, ";
          }

          cout << num_iterations << " iterations, residual = "
               << final_res_norm << ")." << std::endl;
        }

        timer.reset( true );
        
        //__________________________________
        // Test for convergence failure
        
        if( final_res_norm > m_params->tolerance || std::isfinite(final_res_norm) == 0 ){
          if( m_params->getRecomputeTimeStepOnFailure() ){
            proc0cout << "  WARNING:  HypreSolver not converged in " << num_iterations
                      << " iterations, final residual= " << final_res_norm
                      << ", requesting the time step be recomputed.\n";

            new_dw->put( bool_or_vartype(true), VarLabel::find(abortTimeStep_name));
            new_dw->put( bool_or_vartype(true), VarLabel::find(recomputeTimeStep_name));
          } else {
            throw ConvergenceFailure("HypreSolver variable: "+ m_X_label->getName()+", solver: "+ m_params->solvertype+", preconditioner: "+ m_params->precondtype,
                                     num_iterations, final_res_norm,
                                     m_params->tolerance,__FILE__,__LINE__);
          }
        }
      }
    }

    //---------------------------------------------------------------------------------------------
    void
    setupPrecond( const ProcessorGroup              * pg
                 ,       HYPRE_PtrToStructSolverFcn & precond
                 ,       HYPRE_PtrToStructSolverFcn & pcsetup
                 ,struct hypre_solver_struct        * hypre_solver_s
                 ,       HYPRE_StructSolver         & precond_solver
                 )
    {
      switch( hypre_solver_s->precond_solver_type ){
      //__________________________________
      // use symmetric SMG as preconditioner
      case smg:{

        HYPRE_StructSMGCreate         (pg->getComm(),    &precond_solver);
        HYPRE_StructSMGSetMemoryUse   (precond_solver,   0);
        HYPRE_StructSMGSetMaxIter     (precond_solver,   m_params->precond_maxiters);
        HYPRE_StructSMGSetTol         (precond_solver,   m_params->precond_tolerance);
        HYPRE_StructSMGSetZeroGuess   (precond_solver);
        HYPRE_StructSMGSetNumPreRelax (precond_solver,   m_params->npre);
        HYPRE_StructSMGSetNumPostRelax(precond_solver,   m_params->npost);
        HYPRE_StructSMGSetLogging     (precond_solver,   0);

        precond = HYPRE_StructSMGSolve;
        pcsetup = HYPRE_StructSMGSetup;
        break;

      }
      //__________________________________
      // use symmetric PFMG as preconditioner
      case pfmg:{

        HYPRE_StructPFMGCreate        (pg->getComm(),    &precond_solver);
        HYPRE_StructPFMGSetMaxIter    (precond_solver,   m_params->precond_maxiters);
        HYPRE_StructPFMGSetTol        (precond_solver,   m_params->precond_tolerance);
        HYPRE_StructPFMGSetZeroGuess  (precond_solver);

        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructPFMGSetRelaxType   (precond_solver,  m_params->relax_type);
        HYPRE_StructPFMGSetNumPreRelax (precond_solver,  m_params->npre);
        HYPRE_StructPFMGSetNumPostRelax(precond_solver,  m_params->npost);
        HYPRE_StructPFMGSetSkipRelax   (precond_solver,  m_params->skip);
        HYPRE_StructPFMGSetLogging     (precond_solver,  0);

#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
        //DS 10252019: added levels to be solved on GPU. coarser level will be executed on CPU, which is faster. 12 is determined by experiments.
        //Can be modified later or added into ups file.
        HYPRE_StructPFMGSetDeviceLevel(precond_solver, 12);
#endif

        precond = HYPRE_StructPFMGSolve;
        pcsetup = HYPRE_StructPFMGSetup;
        break;

      }
      //__________________________________
      //  use symmetric SparseMSG as preconditioner
      case sparsemsg:{

        HYPRE_StructSparseMSGCreate       (pg->getComm(),   &precond_solver);
        HYPRE_StructSparseMSGSetMaxIter   (precond_solver,  m_params->precond_maxiters);
        HYPRE_StructSparseMSGSetJump      (precond_solver,  m_params->jump);
        HYPRE_StructSparseMSGSetTol       (precond_solver,  m_params->precond_tolerance);
        HYPRE_StructSparseMSGSetZeroGuess (precond_solver);

        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructSparseMSGSetRelaxType   (precond_solver,  m_params->relax_type);
        HYPRE_StructSparseMSGSetNumPreRelax (precond_solver,  m_params->npre);
        HYPRE_StructSparseMSGSetNumPostRelax(precond_solver,  m_params->npost);
        HYPRE_StructSparseMSGSetLogging     (precond_solver,  0);

        precond = HYPRE_StructSparseMSGSolve;
        pcsetup = HYPRE_StructSparseMSGSetup;
        break;

      }
      //__________________________________
      //  use two-step Jacobi as preconditioner
      case jacobi:{

        HYPRE_StructJacobiCreate      (pg->getComm(),    &precond_solver);
        HYPRE_StructJacobiSetMaxIter  (precond_solver,   m_params->precond_maxiters);
        HYPRE_StructJacobiSetTol      (precond_solver,   m_params->precond_tolerance);
        HYPRE_StructJacobiSetZeroGuess(precond_solver);

        precond = HYPRE_StructJacobiSolve;
        pcsetup = HYPRE_StructJacobiSetup;
        break;

      }
      //__________________________________
      //  use diagonal scaling as preconditioner
      case diagonal:{

        precond_solver = NULL;
        precond = HYPRE_StructDiagScale;
        pcsetup = HYPRE_StructDiagScaleSetup;

        break;

      }
      default:
        // This should have been caught in readParameters...
        throw InternalError("Unknown preconditionertype: "+ m_params->precondtype, __FILE__, __LINE__);
      }
    }

    //---------------------------------------------------------------------------------------------
    void
    destroyPrecond( struct hypre_solver_struct * hypre_solver_s
                   ,       HYPRE_StructSolver  & precond_solver )
    {

      switch( hypre_solver_s->precond_solver_type ){

      case smg:{
        HYPRE_StructSMGDestroy( precond_solver );
        break;
      }
      case pfmg:{
        HYPRE_StructPFMGDestroy( precond_solver );
        break;
      }
      case sparsemsg:{
        HYPRE_StructSparseMSGDestroy( precond_solver );
        break;
      }
      case jacobi:{
        HYPRE_StructJacobiDestroy( precond_solver );
        break;
      }
      case diagonal:{
        // do nothing
        break;
      }
      default:
        // This should have been caught in readParameters...
        throw InternalError("Unknown preconditionertype: "+ m_params->precondtype, __FILE__, __LINE__);
      }
    }

    //---------------------------------------------------------------------------------------------

  private:

    const Level*       m_level;
    const MaterialSet* m_matlset;
    const VarLabel*    m_A_label;
    Task::WhichDW      m_which_A_dw;
    const VarLabel*    m_X_label;
    bool               m_modifies_X;
    const VarLabel*    m_b_label;
    Task::WhichDW      m_which_b_dw;
    const VarLabel*    m_guess_label;
    Task::WhichDW      m_which_guess_dw;
    const HypreParams* m_params;
    bool               m_isFirstSolve;
    mutable double *   m_buff{nullptr};
    mutable size_t	   m_buff_size{0};

    const VarLabel*    m_timeStepLabel;
    const VarLabel*    m_hypre_solver_label;
    SoleVariable<hypre_solver_structP> m_hypre_solverP;
    bool   m_firstPassThrough;
    double m_movingAverage;

    //set by the environment variable HYPRE_SUPERPATCH. Hypre will combine all patches into a superpatch if set to HYPRE_SUPERPATCH 1
    //superpatch works only if patch to rank assignment is aligned with the domain number of patches in x, y, and z dimensions.
    //will work only if the subdomain assigned to the rank is rectangular/cubical.
    bool   m_superpatch {false};
    bool   m_superpatch_bulletproof {false}; //ensure all patches assigned to rank fall within the super-patch boundaries and set it to true. No need to do it for every timestep

    // hypre timers - note that these variables do NOT store timings - rather, each corresponds to
    // a different timer index that is managed by Hypre. To enable the use and reporting of these
    // hypre timings, #define HYPRE_TIMING in HypreSolver.h
    int m_tHypreAll;    // Tracks overall time spent in Hypre = matrix/vector setup & assembly + solve time.
    int m_tSolveOnly;   // Tracks time taken by hypre to solve the system of equations
    int m_tMatVecSetup; // Tracks the time taken by uintah/hypre to allocate and set matrix and vector box vaules

  }; // class HypreStencil7

  //==============================================================================
  //
  // HypreSolver2 Implementation
  //
  //==============================================================================

  HypreSolver2::HypreSolver2(const ProcessorGroup* myworld)
  : SolverCommon(myworld)
  {
    //-------------DS: 04262019: Added to run hypre task using hypre-cuda.----------------
#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
    int argc = 0;
    //std::string
    HYPRE_Init(argc, NULL);
#endif
    //-----------------  end of hypre-cuda  -----------------

    // Time Step
    m_timeStepLabel = VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );

    hypre_solver_label = VarLabel::create("hypre_solver_label",
                                          SoleVariable<hypre_solver_structP>::getTypeDescription());

    m_params = scinew HypreParams();

  }

  //---------------------------------------------------------------------------------------------

  HypreSolver2::~HypreSolver2()
  {
    //-------------DS: 04262019: Added to run hypre task using hypre-cuda.----------------
#if defined(HYPRE_USING_CUDA) || (defined(HYPRE_USING_KOKKOS) && defined(KOKKOS_ENABLE_CUDA))
    HYPRE_Finalize();
#endif
    //-----------------  end of hypre-cuda  -----------------

    VarLabel::destroy(m_timeStepLabel);
    VarLabel::destroy(hypre_solver_label);
    delete m_params;
  }

  //---------------------------------------------------------------------------------------------

  void HypreSolver2::readParameters(       ProblemSpecP & params_ps
                                   , const string       & varname)
  {
    bool found=false;
    if(params_ps){
      for( ProblemSpecP param_ps = params_ps->findBlock("Parameters"); param_ps != nullptr; param_ps = param_ps->findNextBlock("Parameters")) {

        string variable;
        if( param_ps->getAttribute("variable", variable) && variable != varname ) {
          continue;
        }

        int sFreq;
        int coefFreq;
        string str_solver;
        string str_precond;

        param_ps->getWithDefault ("solver",          str_solver,     "smg");
        param_ps->getWithDefault ("preconditioner",  str_precond,    "diagonal");
        param_ps->getWithDefault ("tolerance",       m_params->tolerance,          1.e-10);
        param_ps->getWithDefault ("maxiterations",   m_params->maxiterations,      75);
        param_ps->getWithDefault ("precond_maxiters",m_params->precond_maxiters,   1);
        param_ps->getWithDefault ("precond_tolerance",m_params->precond_tolerance, 0);

        param_ps->getWithDefault ("npre",            m_params->npre,           1);
        param_ps->getWithDefault ("npost",           m_params->npost,          1);
        param_ps->getWithDefault ("skip",            m_params->skip,           0);
        param_ps->getWithDefault ("jump",            m_params->jump,           0);
        param_ps->getWithDefault ("logging",         m_params->logging,        0);
        param_ps->getWithDefault ("setupFrequency",  sFreq,             1);
        param_ps->getWithDefault ("updateCoefFrequency",  coefFreq,             1);
        param_ps->getWithDefault ("solveFrequency",  m_params->solveFrequency, 1);
        param_ps->getWithDefault ("relax_type",      m_params->relax_type,     1);

        // change to lowercase
        m_params->solvertype  = string_tolower( str_solver );
        m_params->precondtype = string_tolower( str_precond );

        m_params->setSetupFrequency(sFreq);
        m_params->setUpdateCoefFrequency(coefFreq);
        // Options from the HYPRE_ref_manual 2.8
        // npre:   Number of relaxation sweeps before coarse grid correction
        // npost:  Number of relaxation sweeps after coarse grid correction
        // skip:   Skip relaxation on certain grids for isotropic
        //         problems. This can greatly improve effciency by eliminating
        //         unnecessary relaxations when the underlying problem is isotropic.
        // jump:   not in manual
        //
        // relax_type
        // 0 : Jacobi
        // 1 : Weighted Jacobi (default)
        // 2 : Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation)
        // 3 : Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation)

        found=true;
      }
    }
    if(!found){
      m_params->solvertype    = "smg";
      m_params->precondtype   = "diagonal";
      m_params->tolerance     = 1.e-10;
      m_params->maxiterations = 75;
      m_params->npre    = 1;
      m_params->npost   = 1;
      m_params->skip    = 0;
      m_params->jump    = 0;
      m_params->logging = 0;
      m_params->setSetupFrequency(1);
      m_params->setUpdateCoefFrequency(1);
      m_params->solveFrequency = 1;
      m_params->relax_type = 1;
    }
  }

  //---------------------------------------------------------------------------------------------

  void HypreSolver2::scheduleInitialize( const LevelP      & level
                                       ,       SchedulerP  & sched
                                       , const MaterialSet * matls
                                       )
  {
    const bool isRestart = sched->isRestartInitTimestep();
    Task* task = scinew Task("HypreSolver2::initialize_hypre", this, &HypreSolver2::initialize, isRestart );

    task->setType(Task::OncePerProc);  // must run this task on every proc.  It's possible to have
                                       // no patches on this proc when scheduling

    task->computes(hypre_solver_label);
    
    LoadBalancer * lb = sched->getLoadBalancer();

    sched->addTask(task, lb->getPerProcessorPatchSet(level), matls);
  }

 //---------------------------------------------------------------------------------------------

  void HypreSolver2::scheduleRestartInitialize( const LevelP      & level
                                              ,       SchedulerP  & sched
                                              , const MaterialSet * matls
                                              )
  {
    const bool isRestart = sched->isRestartInitTimestep();
    Task* task = scinew Task("HypreSolver2::restartInitialize_hypre", this, &HypreSolver2::initialize, isRestart);

    task->setType(Task::OncePerProc);  // must run this task on every proc.  It's possible to have
                                       // no patches  on this proc when scheduling restarts with regridding

    task->computes(hypre_solver_label);

    LoadBalancer * lb = sched->getLoadBalancer();

    sched->addTask(task, lb->getPerProcessorPatchSet(level), matls);
  }

  //---------------------------------------------------------------------------------------------

  void HypreSolver2::allocateHypreMatrices( DataWarehouse * new_dw,
                                            const bool isRecomputeTimeStep_in )
  {
    SoleVariable<hypre_solver_structP> hypre_solverP;
    hypre_solver_struct* hypre_struct = scinew hypre_solver_struct;

    hypre_struct->solver_p         = scinew HYPRE_StructSolver( nullptr );
    hypre_struct->precond_solver_p = scinew HYPRE_StructSolver( nullptr );
    hypre_struct->HA_p             = scinew HYPRE_StructMatrix( nullptr );
    hypre_struct->HX_p             = scinew HYPRE_StructVector( nullptr );
    hypre_struct->HB_p             = scinew HYPRE_StructVector( nullptr );
    hypre_struct->solver_type         = stringToSolverType( m_params->solvertype );
    hypre_struct->precond_solver_type = stringToSolverType( m_params->precondtype );

    hypre_struct->isRecomputeTimeStep = isRecomputeTimeStep_in;

    hypre_solverP.setData( hypre_struct );
    new_dw->put( hypre_solverP, hypre_solver_label );
  }

  //---------------------------------------------------------------------------------------------

  void
  HypreSolver2::initialize( const ProcessorGroup *
                          , const PatchSubset    *
                          , const MaterialSubset *
                          ,       DataWarehouse  *
                          ,       DataWarehouse  * new_dw
                          , const bool  isRestart
                          )
  {
    allocateHypreMatrices( new_dw, isRestart );
  }

  //---------------------------------------------------------------------------------------------

  template<typename GridVarType, typename functor>
  void HypreSolver2::createPortableHypreSolverTasks( const LevelP        & level
                                                   ,       SchedulerP    & sched
                                                   , const PatchSet      * patches
                                                   , const MaterialSet   * matls
                                                   , const VarLabel      * A_label
                                                   ,       Task::WhichDW   which_A_dw
                                                   , const VarLabel      * x_label
                                                   ,       bool            modifies_X
                                                   , const VarLabel      * b_label
                                                   ,       Task::WhichDW   which_b_dw
                                                   , const VarLabel      * guess_label
                                                   ,       Task::WhichDW   which_guess_dw
                                                   ,       bool            isFirstSolve /* = true */
                                                   ,       functor         TaskDependencies
                                                   )
  {
    HypreStencil7<GridVarType>* that = scinew HypreStencil7<GridVarType>(level.get_rep(), matls, A_label, which_A_dw, x_label, modifies_X, b_label, which_b_dw, guess_label, which_guess_dw, m_params, isFirstSolve);
    Handle<HypreStencil7<GridVarType> > handle = that;

    create_portable_tasks(TaskDependencies, that,
                          "Hypre:Matrix solve (SFCX)",
                          &HypreStencil7<GridVarType>::template solve<UINTAH_CPU_TAG>,
                          &HypreStencil7<GridVarType>::template solve<KOKKOS_OPENMP_TAG>,
                          &HypreStencil7<GridVarType>::template solve<KOKKOS_CUDA_TAG>,
                          sched, patches, matls, TASKGRAPH::DEFAULT, handle);
  }

  void
  HypreSolver2::scheduleSolve( const LevelP           & level
                             ,       SchedulerP       & sched
                             , const MaterialSet      * matls
                             , const VarLabel         * A_label
                             ,       Task::WhichDW      which_A_dw
                             , const VarLabel         * x_label
                             ,       bool               modifies_X
                             , const VarLabel         * b_label
                             ,       Task::WhichDW      which_b_dw
                             , const VarLabel         * guess_label
                             ,       Task::WhichDW      which_guess_dw
                             ,       bool               isFirstSolve /* = true */
                             )
  {
    //__________________________________
    //  Computes and requires
    auto TaskDependencies = [&](Task* task) {

      // Matrix A
      task->requires(which_A_dw, A_label, Ghost::None, 0);

      // Solution X
      if(modifies_X){
        task->modifies( x_label );
      } else {
        task->computes( x_label );
      }

      // Initial Guess
      if(guess_label){
        task->requires(which_guess_dw, guess_label, Ghost::None, 0);
      }

      // RHS  B
      task->requires(which_b_dw, b_label, Ghost::None, 0);

      // timestep
      // it could come from old_dw or parentOldDw
      Task::WhichDW old_dw = m_params->getWhichOldDW();
      task->requires( old_dw, m_timeStepLabel );

      // solve struct
      if (isFirstSolve) {
        task->requires( Task::OldDW, hypre_solver_label);
        task->computes( hypre_solver_label);
      }  else {
        task->requires( Task::NewDW, hypre_solver_label);
      }

      sched->overrideVariableBehavior(hypre_solver_label->getName(),false,true,false,false,true);

      task->setType(Task::Hypre);

      if( m_params->getRecomputeTimeStepOnFailure() ){
        task->computes( VarLabel::find(abortTimeStep_name) );
        task->computes( VarLabel::find(recomputeTimeStep_name) );
      }
    };

    LoadBalancer * lb = sched->getLoadBalancer();
    const PatchSet* patches = lb->getPerProcessorPatchSet(level);

    printSchedule(level, cout_doing, "HypreSolver:scheduleSolve");

    // The extra handle arg ensures that the stencil7 object will get freed
    // when the task gets freed.  The downside is that the refcount gets
    // tweaked everytime solve is called.

    TypeDescription::Type domtype = A_label->typeDescription()->getType();
    ASSERTEQ(domtype, x_label->typeDescription()->getType());
    ASSERTEQ(domtype, b_label->typeDescription()->getType());

    //__________________________________
    // bulletproofing
    IntVector periodic = level->getPeriodicBoundaries();
    if(periodic != IntVector(0,0,0)){

      IntVector l,h;
      level->findCellIndexRange( l, h );
      IntVector range = (h - l ) * periodic;

      if( fmodf(range.x(),2) != 0  || fmodf(range.y(),2) != 0 || fmodf(range.z(),2) != 0 ) {
        ostringstream warn;
        warn << "\nINPUT FILE WARNING: hypre solver: \n"
             << "With periodic boundary conditions the resolution of your grid "<<range<<", in each periodic direction, must be as close to a power of 2 as possible (i.e. M x 2^n).\n";

        if (m_params->solvertype == "smg") {
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }
        else {
          proc0cout << warn.str();
        }
      }
    }

    switch(domtype){
    case TypeDescription::SFCXVariable:
      createPortableHypreSolverTasks<SFCXTypes>(level, sched, patches, matls, A_label, which_A_dw, x_label, modifies_X, b_label, which_b_dw, guess_label, which_guess_dw, isFirstSolve, TaskDependencies);
      break;
    case TypeDescription::SFCYVariable:
      createPortableHypreSolverTasks<SFCYTypes>(level, sched, patches, matls, A_label, which_A_dw, x_label, modifies_X, b_label, which_b_dw, guess_label, which_guess_dw, isFirstSolve, TaskDependencies);
      break;
    case TypeDescription::SFCZVariable:
      createPortableHypreSolverTasks<SFCZTypes>(level, sched, patches, matls, A_label, which_A_dw, x_label, modifies_X, b_label, which_b_dw, guess_label, which_guess_dw, isFirstSolve, TaskDependencies);
      break;
    case TypeDescription::CCVariable:
      createPortableHypreSolverTasks<CCTypes>(level, sched, patches, matls, A_label, which_A_dw, x_label, modifies_X, b_label, which_b_dw, guess_label, which_guess_dw, isFirstSolve, TaskDependencies);
      break;
    case TypeDescription::NCVariable:
      createPortableHypreSolverTasks<NCTypes>(level, sched, patches, matls, A_label, which_A_dw, x_label, modifies_X, b_label, which_b_dw, guess_label, which_guess_dw, isFirstSolve, TaskDependencies);
      break;
    default:
      throw InternalError("Unknown variable type in scheduleSolve", __FILE__, __LINE__);
    }

  }

  //---------------------------------------------------------------------------------------------

  string HypreSolver2::getName(){
    return "hypre";
  }

  //---------------------------------------------------------------------------------------------
  //  Return the solver or preconditioner type
  SolverType HypreSolver2::stringToSolverType( std::string str )
  {
    if( str == "smg" ){
      return smg;
    }
    else if ( str == "pfmg" ){
      return pfmg;
    }
    else if ( str == "sparsemsg" ){
      return sparsemsg;
    }
    else if ( str == "cg" || str == "pcg" ){
      return pcg;
    }
    else if ( str == "hybrid" ){
      return hybrid;
    }
    else if ( str == "gmres" ){
      return gmres;
    }
    else if ( str == "jacobi" ){
      return jacobi;
    }
    else if ( str == "diagonal" ){
      return diagonal;
    }
    else {
      throw InternalError("ERROR:  Unknown solver type: "+ str, __FILE__, __LINE__);
    }
  }
  //---------------------------------------------------------------------------------------------
} // end namespace Uintah
