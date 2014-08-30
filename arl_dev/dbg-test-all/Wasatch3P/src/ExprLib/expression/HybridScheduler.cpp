/**
 *  \file   HybridScheduler.cpp
 *  \date   Aug 23, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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
 *
 */

#include <expression/HybridScheduler.h>

//-- SpatialOps includes
#include <spatialops/structured/MemoryTypes.h>
#include <spatialops/structured/ExternalAllocators.h>

#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace Expr{


  //=================================================================

  /**
   * @class GPULoadBalancer
   *
   * @brief Used to maintain loading and assignment information with respect to
   *          available hardware resources and tasks assigned to them.
   */
  class GPULoadBalancer {

  public:
    enum Method { RoundRobin, MinimumLoading };

    int gpuDeviceCount_;
    std::vector<unsigned int> deviceMemorySize_;
    std::vector<unsigned int> deviceLoading_;
    std::map< unsigned int, std::list<Vertex> > coalescingChains_;

    GPULoadBalancer() : gpuDeviceCount_(0), method_(RoundRobin), nextRR_(0), nextCID_(0) {}

    Method get_assignment_strategy() const {
      return method_;
    }

    void set_assignment_strategy( const Method m ) {
      method_ = m;
    }

    unsigned int get_next_cid() {
      return nextCID_++;
    }

    unsigned int get_next_device(){
      switch(method_){
        case RoundRobin: {
          return ( ( ++nextRR_ ) % gpuDeviceCount_ );
        }
        case MinimumLoading: {
          unsigned int index = 0;
          for( unsigned int i = 0; i < deviceLoading_.size(); ++i ){
            index = ( deviceLoading_[index] < deviceLoading_[i] ) ? index : i;
          }
          return index;
        }
        default:
          throw("Unknown device loading type\n");
      }
    }
  private:
    Method method_;
    int nextRR_;
    int nextCID_;
  };

  //=================================================================


  /**
   * \brief Boost visitor structure for coalescing paths
   *
   * Greedy chaining algorithm. Attempts to create the longest single-path chains possible
   */
  struct LoadBalanceVisitor: public boost::default_bfs_visitor {

    LoadBalanceVisitor( GPULoadBalancer* const gpuLB )
    : gpuLB_( gpuLB )
    {}

    inline void examine_edge( const Edge e, const Graph& g ) const{
      const Vertex src = boost::source(e, g);
      const Vertex dest = boost::target(e, g);
      Graph& g2 = const_cast<Graph&>(g);

      VertexProperty& svp = g2[src];
      VertexProperty& dvp = g2[dest];

      if( IS_GPU_INDEX(svp.execTarget) ){
#       ifdef DEBUG_SCHED_ALL
        std::cout << "Source execution target is GPU\n";
#       endif
        if( svp.chainID_ == -1 ){ // Source vertex is not part of a coalescing chain. Make a new one
          svp.chainID_ = gpuLB_->get_next_cid();
          std::list<Vertex> temp;
          gpuLB_->coalescingChains_.insert( std::pair<unsigned int, std::list<Vertex> >( svp.chainID_, temp ) );
          std::list<Vertex>& chain = gpuLB_->coalescingChains_[ svp.chainID_ ];
          chain.push_back( src );
        }

        //If the destination node is already taken, look at absorbing it
        if( IS_GPU_INDEX(dvp.execTarget) ){
#         ifdef DEBUG_SCHED_ALL
          std::cout << "Destination execution target is GPU\n";
#         endif
          if( dvp.chainID_ == -1 && svp.chainTail_ ) {
            // Add destvp to the chain if it isn't already taken
            svp.chainTail_ = false;
            dvp.chainID_ = svp.chainID_;
#           ifdef DEBUG_SCHED_ALL
            std::cout << "Destination is not part of an existing chain pushing to source chain, ID: " << dvp.chainID_ << std::endl;
#           endif
            std::list<Vertex>& chain = gpuLB_->coalescingChains_[ dvp.chainID_ ];
            chain.push_back(dest);
          } else {
            std::list<Vertex>& dchain = gpuLB_->coalescingChains_[dvp.chainID_];

            if( dest == dchain.front() && svp.chainTail_ ) {
              svp.chainTail_ = false;
              unsigned int t = dvp.chainID_;
              //Dest is the head of another chain, attach it to our current chain
              std::list<Vertex>& schain = gpuLB_->coalescingChains_[svp.chainID_];
              while( !dchain.empty() ){
                Vertex& v = dchain.front();
                VertexProperty& vp = g2[v];
                vp.chainID_ = svp.chainID_;
                schain.push_back(v);
                dchain.pop_front();
              }
              gpuLB_->coalescingChains_.erase(t);
            }
          }
        }
      } else {
        //If the destination node is already taken, look at absorbing it
        if( IS_GPU_INDEX(dvp.execTarget) && dvp.chainID_ == -1 ){
          dvp.chainID_ = gpuLB_->get_next_cid();

          std::list<Vertex> temp;
          gpuLB_->coalescingChains_.insert( std::pair<unsigned int, std::list<Vertex> >(dvp.chainID_, temp) );
          std::list<Vertex>& chain = gpuLB_->coalescingChains_[dvp.chainID_];
          chain.push_back(dest);
        }
      }
    }

    GPULoadBalancer* const gpuLB_;
  };

  //=================================================================


  HybridScheduler::HybridScheduler( Graph& depGraph, Graph& execGraph )
  : Scheduler(execGraph,depGraph),
#   ifdef ENABLE_CUDA
    rungpu_(true),
#   else
    rungpu_(false),
#   endif
    flip_(false),
    runCount_(0),
    dummyRun_(4),
    deviceID_(0),
    coalescingChain_(true),
    gpuLoadBalancer_( new GPULoadBalancer() )
  {
#   ifdef ENABLE_CUDA
    //Grab GPU information
    ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();

    /** Debug! **/
#   ifdef DEBUG_SCHED_ALL
    CDI.print_device_info();
#   endif

    /** Determine how many GPUs we have **/
    gpuLoadBalancer_->gpuDeviceCount_ = CDI.get_device_count();

    /** Update memory information **/
    CDI.update_memory_statistics();

    ema::cuda::CUDAMemStats CMS;
    for( int device = 0; device < gpuLoadBalancer_->gpuDeviceCount_; device++ ){
      CDI.get_memory_statistics(CMS, device);
      gpuLoadBalancer_->deviceMemorySize_.push_back(CMS.t);
      gpuLoadBalancer_->deviceLoading_.push_back(0);
    }
#   endif // ENABLE_CUDA
  }

  //-----------------------------------------------------------------

void HybridScheduler::call( VertexProperty& target )
{

# ifdef DEBUG_SCHED_ALL
  std::cout << "Executing Expression : " << target.expr->get_tags()[0] << " , stream : " << target.expr->get_cuda_stream() << std::endl;
# endif

# ifdef ENABLE_CUDA
    // Check for the status of the dependency expression CudaStreams
  if( target.expr->get_cuda_stream() != NULL ) this->wait_on_cuda_stream( target );
# endif

  FieldManagerList* const fml = extract_field_manager_list( this->fmls_, target.fmlid );

  try{
    (target.expr)->base_bind_fields(*fml);
    FieldDeps& fd = *((*this->fdm_)[target.id]);

#   ifdef ENABLE_CUDA
    if(target.mm==MEM_STATIC_GPU) fd.set_active_field_location( *fml, target.execTarget );
#   endif

    target.execute_expression();

#   ifdef ENABLE_UINTAH
    // When interfacing to Uintah, we play some tricks in some cases to get
    // hybrid CPU/GPU graphs working properly.  This is one of those tricks.
#   ifdef ENABLE_CUDA
    if(target.mm==MEM_STATIC_GPU) fd.validate_field_location( *fml, CPU_INDEX );
#   endif
#   endif

    this->run_pollers( target, fml );
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << std::endl << "Error trapped while executing expression: "
        << target.expr->get_tags()[0] << std::endl
        << "details follow..." << std::endl
        << err.what() << std::endl;
    throw std::runtime_error( msg.str() );
  }
}

//-------------------------------------------------------------------

#ifdef ENABLE_CUDA

void HybridScheduler::set_device_index( int deviceIndex )
{
  deviceID_        = deviceIndex;

  // coalescing algorithm is turned off, when the device index is set from
  // external sources like Wasatch. This would prevent in assigning multiple
  // device indices within the same tree for a multi-GPU system.
  coalescingChain_ = false;
}

//-------------------------------------------------------------------

void HybridScheduler::setup_cuda_stream()
{
  if(IS_CPU_INDEX(deviceID_)) return;

  const std::pair<VertIter, VertIter> execGraphVertices = boost::vertices(execGraph_);
  VertIter iter;
  for( iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
    VertexProperty& vp = execGraph_[*iter];
    // create cuda streams for expressions
    vp.expr->create_cuda_stream(deviceID_);

    std::pair<OutEdgeIter, OutEdgeIter> edges = boost::out_edges(*iter,execGraph_);
    for( OutEdgeIter eit = edges.first; eit != edges.second; ++eit ){
      VertexProperty& cp = execGraph_[boost::target(*eit, execGraph_)];
      // records cuda stream for expession
      cp.consumerStreamList.push_back( vp.expr->get_cuda_stream() );
    }
  }
}

//-------------------------------------------------------------------

void HybridScheduler::wait_on_cuda_stream( VertexProperty& target )
{
  if(IS_CPU_INDEX(deviceID_)) return;

  //set device context
  cudaSetDevice(deviceID_);

  std::vector<cudaStream_t>::iterator streamstart  = target.consumerStreamList.begin();
  std::vector<cudaStream_t>::iterator streamend    = target.consumerStreamList.end();
  cudaError err, Err;

  for( ; streamstart!=streamend; ++streamstart ){
    err = cudaStreamQuery( *streamstart );
    if(err == cudaSuccess) return;
    else{
      if(err == cudaErrorNotReady){
        if(cudaSuccess != (Err = cudaStreamSynchronize( *streamstart ))){
          std::ostringstream msg;
          msg << "ERROR ! Failed to synchronize stream : " << *streamstart << ", at " << __FILE__ << " : " << __LINE__
                << std::endl;
          msg << "\t - " << cudaGetErrorString(Err);
          throw(std::runtime_error(msg.str()));
        }
      }
#     ifndef NDEBUG
      else if(err == cudaErrorLaunchFailure){
        std::ostringstream msg;
        msg << "ERROR ! Detected a kernel failure from previous expressions : " << *streamstart << ", at " << __FILE__ << " : " << __LINE__
            << std::endl;
        msg << "\t - " << cudaGetErrorString(err);
        throw(std::runtime_error(msg.str()));
      }
      else if(err == cudaErrorInvalidResourceHandle){
        std::ostringstream msg;
        msg << "ERROR ! Invalid resource handle (stream) ,  : " << *streamstart << ", might have been created in a different context \n"
            << " at " << __FILE__ << " : " << __LINE__
            << std::endl;
        msg << "\t - " << cudaGetErrorString(err);
        throw(std::runtime_error(msg.str()));
      }
#     endif
      else{
        std::ostringstream msg;
        msg << "ERROR ! Failed at cudaStreamQuery with stream : " << *streamstart << ", Error Code = " <<  err << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        msg << "\t - " << cudaGetErrorString(err);
        throw(std::runtime_error(msg.str()));
      }
    }
  }
}
#endif

//-------------------------------------------------------------------

void HybridScheduler::exec_callback_handler( void* expr_vertex )
{
  const Vertex v = (Vertex) expr_vertex;
  VertexProperty& vpJustFinished = execGraph_[v];

  // Notify the vertex that a consumer has finished, it returns true, it can be freed.
  std::vector<VertexProperty*>::iterator vpit  = vpJustFinished.ancestorList.begin();
  std::vector<VertexProperty*>::iterator vpend = vpJustFinished.ancestorList.end();
  for( ; vpit!=vpend; ++vpit ){
    VertexProperty* vp = *vpit;
    if ((*vpit)->consumer_finished()) {
      if ( this->fdm_ != NULL && this->fmls_.size()>0 ) {
        FieldDeps& fd = *((*this->fdm_)[vp->id]);
#       ifndef DEBUG_NO_FIELD_RELEASE
        FieldManagerList* const fml = extract_field_manager_list( this->fmls_, vp->fmlid );
        fd.release_fields(*fml);
#       endif
      }
    }
  }

  // Notify the vertex that an ancestor has finished, it returns true if it is ready.
  // If it is, we either toss it to the thread pool or run it.
  FieldDeps& fd = *((*this->fdm_)[vpJustFinished.id]);

  vpend = vpJustFinished.consumerList.end();
  for( vpit = vpJustFinished.consumerList.begin(); vpit!=vpend; ++vpit ) {
    VertexProperty& destvp = (**vpit);

    //Here, destvp will be a consumer of vpJustFinished, so vpJustFinished must be prepared to be consumed on whichever
    //device destvp is set to execute on.

    //Check to see if this field needs to have a CONSUMER_FIELD
    //Note: adding consumer fields is a thread safe operation
    // TODO : GPU peer-peer copy is not yet handled

    if( vpJustFinished.execTarget != destvp.execTarget ){ // CPU <-> GPU copy
        if( eid2didx_.count(vpJustFinished.id) == 0 ){
          // consumer is added for the first time
          eid2didx_.insert(std::make_pair<ExpressionID, short int>(vpJustFinished.id, destvp.execTarget));
#         ifdef DEBUG_SCHED_ALL
          std::cout << "Field requires preparation for consumption: " << vpJustFinished.expr->get_tags()[0] << std::endl
                    << "Allocating on " << ((IS_GPU_INDEX(destvp.execTarget)) ? "GPU MEMORY" : "LOCAL MEMORY" )
                    << ", deviceIndex : " << destvp.execTarget << std::endl;
#         endif
          FieldManagerList* fml = extract_field_manager_list( this->fmls_, destvp.fmlid );
          fd.prep_field_for_consuption( *fml, destvp.execTarget );
        }
        else{
          // consumer already exists but required on a device with different device Index
          int targetexecTarget;
          typedef std::multimap<ExpressionID, short int>::const_iterator multiMapIter;
          std::pair<multiMapIter, multiMapIter> pairIter = eid2didx_.equal_range( vpJustFinished.id );
          for( std::multimap<ExpressionID, short int>::const_iterator MapIter = pairIter.first; MapIter != pairIter.second; ++MapIter ){
            targetexecTarget = MapIter->second;
            if( targetexecTarget != destvp.execTarget ){
              eid2didx_.insert(std::make_pair<ExpressionID, short int>( vpJustFinished.id, destvp.execTarget ));
#             ifdef DEBUG_SCHED_ALL
              std::cout << "Field requires preparation for consumption: " << vpJustFinished.expr->get_tags()[0] << std::endl
                        << "Allocating on " << ((IS_GPU_INDEX(destvp.execTarget)) ? "GPU MEMORY" : "LOCAL MEMORY" )
                        << ", deviceIndex : " << destvp.execTarget << std::endl;
#             endif
              FieldManagerList* fml = extract_field_manager_list( this->fmls_, destvp.fmlid );
              fd.prep_field_for_consuption( *fml, destvp.execTarget );
            }
          } // for loop
        } // else
    }

    if( destvp.ancestor_finished() ){
#     ifdef ENABLE_THREADS
      this->pool_.schedule(
          boost::threadpool::prio_task_func( destvp.priority,
              boost::bind( &HybridScheduler::call, this, destvp ) ) );

#     else
      this->call(destvp);
#     endif
    }
  }
  dec_remaining();
}

//-------------------------------------------------------------------

void HybridScheduler::dec_remaining()
{
# ifdef ENABLE_THREADS
  ExecMutex<1001> lock;
# endif
  --nremaining_;
  if( nremaining_ == 0 ) eid2didx_.clear();

# ifdef ENABLE_THREADS
  if( nremaining_ == 0 ) {
    this->schedBarrier_.post();
  }
# endif
}

//-------------------------------------------------------------------

void HybridScheduler::setup( const bool hasRegisteredFields )
{
  /* Notes on whats going on here
   * - gptr is the execution graph
   * - tgptr is the consumer ( dependency graph )
   *
   * Step 1: Reset and reconnect all variables to place the graph into state which is execute ready.
   *
   * Step 2: Inspect the execution graph, flag edge nodes and set compute device.
   *
   * Step 3: Inspect the execution graph w/ hardware targets -- coalesce paths where possible
   *
   * Step 4: If our graph nodes are allocated, we update their field managers
   *
   * Step 5: rebuild our task graph indices.
   */

  // Quick return if we're already valid or if we're not fully setup yet.
  if( !invalid_ ) return;

  // Clear scheduling lists
  rootList_.clear();
  ID2VP execVertexMap;

  // Reset load balancer variables
  gpuLoadBalancer_->coalescingChains_.clear();
  gpuLoadBalancer_->deviceLoading_.clear();

  //Update execution counters.
  nelements_ = boost::num_vertices(execGraph_);
  nremaining_ = nelements_;

  const std::pair<VertIter, VertIter> execGraphVertices = boost::vertices(execGraph_);
  const std::pair<VertIter, VertIter> depGraphVertices  = boost::vertices(depGraph_ );

  // ------------------------------

  /*Step - 1 Reconnect all signals and reset execution counts
   *        Determine consumer and parent counts for all nodes in the graph
   */
  VertIter iter;
  for( iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
    execVertexMap.insert(std::make_pair(execGraph_[*iter].id, *iter));
    VertexProperty& vp = execGraph_[*iter];

    vp.self_      = (void*) (*iter);
    vp.nparents   = 0;
    vp.nconsumers = 0;
    vp.chainID_   = -1;
    vp.chainTail_ = true;
    vp.execSignalCallback.reset(new VertexProperty::Signal());
    vp.execSignalCallback->connect( boost::bind(&HybridScheduler::exec_callback_handler, this, vp.self_) );
    vp.ancestorList.clear();
    vp.consumerList.clear();
#   ifdef ENABLE_CUDA
    vp.consumerStreamList.clear();
#   endif
    vp.set_is_edge(false);
  }

  for( iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
    VertexProperty& vp = execGraph_[*iter];

    std::pair<OutEdgeIter, OutEdgeIter> edges = boost::out_edges(*iter, execGraph_);
    for( OutEdgeIter eit = edges.first; eit != edges.second; ++eit ){
      /*
       *                                       /````> cp1 ( Add 1 to parent count )
       * Idea: ( Add 1 to consumer count ) vp  -----> cp2 ( Add 1 to parent count )
       *                                       \____> cp3 ( Add 1 to parent count )
       *
       * Doing it like this we compute all consumer and parent counts at the same time.
       */
      VertexProperty& cp = execGraph_[boost::target(*eit, execGraph_)];

// jcs Pollers....!
      vp.consumerList.push_back(&cp);
      cp.ancestorList.push_back(&vp);

      (vp.nconsumers)++;
      (cp.nparents)++;
    }
  }

  /* Step 2 - Create Cuda Streams for expressions and record them for performing a sanity check on the
   *          dependency streams for completion status
   *
   */

# ifdef ENABLE_CUDA
  this->setup_cuda_stream();
# endif

  /* Step 3 - Build our root list, classify persistence
   *        - The root list is composed of nodes that do not have any parents
   *        ( topologic edge nodes )
   *
   *        - At present all edge nodes, including leaf nodes are defined to be
   *            persistent. This may change in the future.
   *
   *        ( Not yet implemented -- requires changes to field registration guarantees )
   *        - Do a local sanity check on memory to make sure the device we're assigning
   *            can support any single field + its dependencies
   *
   *        - This initial pass will try and set node hardware targets based on
   *            execution + data transfer times.
   *
   */
  for( VertIter iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
    VertexProperty& vp = execGraph_[*iter];

    //For the execution graph nodes at the bottom of the tree are roots and have no parents.
    //Since edge nodes cannot be 'dynamic' we flag these nodes as persistent

    //TODO
    //vp.nodeMemoryBound_ = vp.nparents * fd.get_field_size(*this->fml_);

    if (vp.nparents == 0) {
      vp.set_is_edge(true);
      rootList_.push_back(*iter);
    }

    if( vp.nconsumers == 0 )  vp.set_is_edge(true);
    if( vp.get_is_edge()   ) {
      FieldDeps& fd = *((*this->fdm_)[vp.id]);
      FieldManagerList* fml = extract_field_manager_list( this->fmls_, vp.fmlid );
      vp.set_is_persistent(fd.lock_fields( *fml ));
    }

    // set the node execution targets and memory managers
    if( rungpu_ && vp.expr->is_gpu_runnable() ){
      if( flip_ ){
        if( runCount_ == dummyRun_ ){
          vp.execTarget = ( vp.vtp.eTimeCPU_ < vp.vtp.eTimeGPU_ ) ? CPU_INDEX : deviceID_;
        }
        else{
          vp.execTarget = deviceID_;
        }
      }
      else{
        vp.execTarget = deviceID_;
        if( vp.get_is_persistent() ) vp.mm = MEM_STATIC_GPU;
        else                         vp.mm = MEM_DYNAMIC_GPU;
      }
#     ifndef NDEBUG
      if(!IS_GPU_INDEX(vp.execTarget)){
        std::ostringstream msg;
        msg << std::endl << "Error ! Invalid deviceID found while setting "
            << "hardware execution node targets : " << vp.execTarget << std::endl
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
#     endif
    }
    else{
      vp.execTarget = CPU_INDEX;
      if( vp.get_is_persistent() ) vp.mm = MEM_EXTERNAL;
      else                         vp.mm = MEM_DYNAMIC;
    }

    vp.nremaining  = vp.nparents;
    vp.ncremaining = vp.nconsumers;
  }

  // synchronize modified vertex properties back to depGraph
  for( VertIter vit = depGraphVertices.first; vit != depGraphVertices.second; ++vit ){
    VertexProperty& evp = execGraph_[ execVertexMap[depGraph_[*vit].id] ];
    VertexProperty& dvp = depGraph_[*vit];
    if     ( dvp.poller.get() ) evp.poller = dvp.poller;
    else if( evp.poller.get() ) dvp.poller = evp.poller;
    dvp = evp;
  }

  /* Step 4 - BFS from each root node, calling our load balancer as necessary.
   *
   *    Example:
   *
   *        (A)          If we suppose that each node in this graph will be run on GPU,
   *       /   \         then our search will construct the following chains:
   *     (B)   (C)
   *    /   \  /         { A->B->D, C->E }
   *  (D)    (E)
   *
   */
  if( coalescingChain_ ){
    for( std::vector<Vertex>::iterator iter = rootList_.begin(); iter != rootList_.end(); ++iter ){
      boost::breadth_first_search( depGraph_, *iter, boost::color_map( boost::get(&VertexProperty::color, depGraph_)).visitor(LoadBalanceVisitor(gpuLoadBalancer_)));
    }

    for( std::map<unsigned int, std::list<Vertex> >::iterator mit = gpuLoadBalancer_->coalescingChains_.begin();
         mit != gpuLoadBalancer_->coalescingChains_.end(); ++mit ){
      std::list<Vertex>& chain = mit->second;

      unsigned int dIndex = gpuLoadBalancer_->get_next_device();
      gpuLoadBalancer_->deviceLoading_[dIndex] += chain.size();
      for( std::list<Vertex>::iterator lit = chain.begin(); lit != chain.end(); ++lit ){
        VertexProperty& vp = execGraph_[*lit];
        vp.execTarget = dIndex;
#       ifdef DEBUG_SCHED_ALL
        std::cout << "Setting device index to " << dIndex << " for GPU device in chain " << mit->first << std::endl;
#       endif
      }
    }

    // synchronize back to exec graph vertex properties
    for( VertIter vit = depGraphVertices.first; vit != depGraphVertices.second; ++vit ){
      const Vertex& ev = execVertexMap[depGraph_[*vit].id];
      VertexProperty& evp = execGraph_[ev];
      const VertexProperty& dvp = depGraph_[*vit];
      evp.execTarget = dvp.execTarget;
    }
  }

  /* Step 5 - If fields are already registered, then we push any field changes to the Field managers
   *
   *   If we know the fields associated with this expression have already been registered
   *   then we need to update the field memory manager to reflect changes which may have
   *   occurred during this setup.
   */
  if( hasRegisteredFields ){
    for( VertIter iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
      const VertexProperty& vp = execGraph_[*iter];
      FieldDeps& fd = *((*this->fdm_)[vp.id]);
      FieldManagerList* const fml = extract_field_manager_list( this->fmls_, vp.fmlid );
      fd.set_memory_manager( *fml, vp.mm, vp.execTarget );
    }
  }

  invalid_ = false;
}

//-------------------------------------------------------------------

void HybridScheduler::run()
{
  if( flip_ ){
    if( runCount_ == dummyRun_/2 ) {
      invalid_ = true;
      flip_ = false;
      this->setup( !flip_ ); //Activates CPU only graph
    }else if( runCount_ == dummyRun_ ) {
      invalid_ = true;
      flip_ = true;
      this->setup( flip_ ); //Activates Hybrid CPU-GPU graph
    }
  }

  //Execute everything in the root list
  for( RootIter rit = rootList_.begin(); rit != rootList_.end(); ++rit ) {
    VertexProperty& vp = execGraph_[*rit];
#   ifdef ENABLE_THREADS
    this->pool_.schedule( boost::threadpool::prio_task_func( vp.priority,
            boost::bind( &HybridScheduler::call, this, vp ) ) );
#   else
    this->call(vp);
#   endif
  }

# ifdef ENABLE_THREADS
  this->pool_.schedule( boost::threadpool::prio_task_func( 1, boost::bind(&HybridScheduler::block_pollers,this) ) );
  this->schedBarrier_.wait();
# else
  this->block_pollers();
# endif

  finish();
  runCount_++;
}

//-------------------------------------------------------------------

void HybridScheduler::finish()
{
  this->nelements_ = boost::num_vertices(execGraph_);
  this->nremaining_ = this->nelements_;

  const std::pair<VertIter, VertIter> execGraphVertices = boost::vertices(execGraph_);

  //grab the root list, default remaining count to parent count
  for( VertIter iter = execGraphVertices.first; iter != execGraphVertices.second; ++iter ){
    VertexProperty& vp = execGraph_[*iter];
    vp.nremaining  = vp.nparents;
    vp.ncremaining = vp.nconsumers;
  }
}

//===================================================================

} // namespace Expr
