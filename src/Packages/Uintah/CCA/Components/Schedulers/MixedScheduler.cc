
#include <Packages/Uintah/CCA/Components/Schedulers/MixedScheduler.h>

#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <set>
#include <map>
#include <algorithm>


using namespace Uintah;
using namespace SCIRun;

using std::cerr;
using std::map;
using std::set;
using std::find;

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

vector<MPI_Request> recv_ids;
vector<MPI_Request> recv_ids_backup;


#define DAV_DEBUG 0

static DebugStream dbg("MixedScheduler", false);

#define PARTICLESET_TAG		0x1000000
#define RECV_BUFFER_SIZE_TAG	0x2000000

struct DestType {
   // A unique destination for sending particle sets
   const Patch* patch;
   int matlIndex;
   int dest;
   DestType(int matlIndex, const Patch* patch, int dest)
      : patch(patch), matlIndex(matlIndex), dest(dest)
   {
   }
   bool operator<(const DestType& c) const {
      if(patch < c.patch)
	 return true;
      else if(patch == c.patch){
	 if(matlIndex < c.matlIndex)
	    return true;
	 else if(matlIndex == c.matlIndex)
	    return dest < c.dest;
      }
      // Never reached, but SGI compile complaines?
      return false;
   }
};

struct VarDestType {
   // A unique destination for sending particle sets
   const VarLabel* var;
   const Patch* patch;
   int matlIndex;
   int dest;
   VarDestType(const VarLabel* var, int matlIndex, const Patch* patch, int dest)
      : var(var), patch(patch), matlIndex(matlIndex), dest(dest)
   {
   }
   bool operator<(const VarDestType& c) const {
      VarLabel::Compare comp;
      if(comp(var, c.var))
	 return true;
      else if(!(comp(c.var, var))){
	 if(patch < c.patch)
	    return true;
	 else if(patch == c.patch){
	    if(matlIndex < c.matlIndex)
	       return true;
	    else if(matlIndex == c.matlIndex)
	       return dest < c.dest;
	 }
      }
      return false;
   }
};

MixedScheduler::MixedScheduler(const ProcessorGroup* myworld, Output* oport)
   : SchedulerCommon(myworld, oport), log(myworld, oport)
{
  d_generation = 0;

  d_threadPool = scinew ThreadPool( Parallel::getMaxThreads() );
}

MixedScheduler::~MixedScheduler()
{
}

void
MixedScheduler::problemSetup( const ProblemSpecP& prob_spec )
{
  log.problemSetup( prob_spec );
}

void
MixedScheduler::sendParticleSets( vector<Task*> & tasks,
				int me )
{
#if DAV_DEBUG
  cerr << "Begin sendParticleSets\n";
#endif
#if 0

   set<DestType> sent;

   // Run through all the tasks...
   for(vector<Task*>::iterator iter = tasks.begin();
       iter != tasks.end(); iter++){
      Task* task = *iter;

      // If the task belongs to me, I don't need to send data...
      if(task->getAssignedResourceIndex() == me)
	 continue;
      // If it is not a Normal task, I don't need to send data...
      if(task->getType() != Task::Normal)
	 continue;

      const Task::reqType& reqs = task->getRequires();

      // Run through all the data that this task requires...
      for(Task::reqType::const_iterator dep = reqs.begin();
	  dep != reqs.end(); dep++){
	 // If this is a PraticleVariable
	 if( dep->d_dw->isFinalized() && dep->d_patch &&
	     dep->d_var->typeDescription()->getType() 
	                             == TypeDescription::ParticleVariable){

	    // Figure out who to send it to...
	    int dest = task->getAssignedResourceIndex();

	    // If I have this data, then send it...
	    if(dep->d_dw->haveParticleSubset(dep->d_matlIndex,
					     dep->d_patch)){
	       DestType ddest(dep->d_matlIndex, dep->d_patch, dest);

	       // If I have already sent this data out to this receiver
	       // don't send it again...
	       if(sent.find(ddest) == sent.end()){

		  ParticleSubset* pset = dep->d_dw->
                     getParticleSubset(dep->d_matlIndex, dep->d_patch);
		  int numParticles = pset->numParticles();
		  ASSERT(dep->d_messageTag >= 0);

#if DAV_DEBUG
		    cerr << "Send Particle Set Data: " << *dep << " from " 
			 << me << " to " << dest << "\n";
#endif
		  // Do the actual send...
		  MPI_Bsend(&numParticles, 1, MPI_INT, dest,
			    PARTICLESET_TAG|dep->d_messageTag,
			    d_myworld->getComm());
		  // And record that I have sent it...
		  sent.insert(ddest);
	       }
	    }
	 }
      }
   }
#else
   NOT_FINISHED("new task stuff");
#endif
#if DAV_DEBUG
  cerr << "End sendParticleSets\n";
#endif
} // end sendParticleSets()

void
MixedScheduler::recvParticleSets( vector<Task*> & tasks, int me )
{
#if DAV_DEBUG
  cerr << "Preparing to recv particle set data\n";
#endif

#if 0
  vector<Task*>::iterator iter;

  for( iter = tasks.begin(); iter != tasks.end(); iter++ ){

    Task* task = *iter;

#if DAV_DEBUG
    cerr << "recvParticleSets: Task: " << *task << "\n";
#endif

    // If the task needs data, but I am not responsible for that task
    // then don't receive data for it...
    if(task->getAssignedResourceIndex() != me)
      continue;
    if(task->getType() != Task::Normal)
      continue;

    const Task::reqType& reqs = task->getRequires();

    Task::reqType::const_iterator dep;
    // Run through all the data that this task requires...
    for( dep = reqs.begin(); dep != reqs.end(); dep++ ){
#if DAV_DEBUG
      cerr << "Considering: " << *dep << ", DW isfinalized: " 
	   << dep->d_dw->isFinalized() << ", var type: " 
	   << dep->d_var->typeDescription()->getType() << "\n";
#endif

      // If I need a ParticleVariable
      if(dep->d_dw->isFinalized() && dep->d_patch &&
	 dep->d_var->typeDescription()->getType() 
	                          == TypeDescription::ParticleVariable){
#if DAV_DEBUG
      cerr << "  have Particle Subset: " 
	   << dep->d_dw->haveParticleSubset(dep->d_matlIndex, dep->d_patch) 
	   << "\n";
#endif

	// If I do NOT already have it...
	if(!dep->d_dw->haveParticleSubset(dep->d_matlIndex, dep->d_patch)){
	  int numParticles;
	  MPI_Status status;
	  ASSERT(dep->d_messageTag >= 0);

#if DAV_DEBUG
	    cerr << "Preparing MPI_Recv's for task: " << *(dep->d_task)
		 << " and dep: " << *dep << "\n";
#endif

	  // The above MPI_Bsend's are non-blocking sends...
	  // The MPI_Recv is a blocking Receive... but this is ok
	  //   because all the data has already been "sent".

	  // Receive the data...
	  MPI_Recv(&numParticles, 1, MPI_INT, MPI_ANY_SOURCE,
		   PARTICLESET_TAG|dep->d_messageTag, 
		   d_myworld->getComm(), &status);
	  // and stick it in the correct DataWarehouse...
	  //   This actually only creates the particleSubset specification
	  //   information.  The actual particle data will be sent/received
	  //   later.
	  dep->d_dw->createParticleSubset(numParticles, dep->d_matlIndex,
					  dep->d_patch);
	}
      }
    }
  }
#if DAV_DEBUG
  cerr << "End recvParticleSets\n";
#endif
#else
  NOT_FINISHED("new task stuff");
#endif
} // end recvParticleSets()

#if 0
map< MPI_Request, DependData >                 reqToDep;
map< DependData, vector<Task *>, DependData >  depToTasks;
map< TaskData, vector<DependData>, TaskData >  taskToDeps;
#endif

// This var is used to insure that all cooperating MPI tasks call
// global reductions in the same order.
vector<const Task*> reductionTasks;

void
MixedScheduler::createDepencyList( DataWarehouseP & old_dw,
				   vector<Task*>  & tasks,
				   int              me )
{
#if 0
#if DAV_DEBUG
  cerrLock.lock();
  cerr << "Start createDepencyList\n";
  cerrLock.unlock();
#endif  

  vector<Task*>::iterator iter;
  
  for( iter = tasks.begin(); iter != tasks.end(); iter++ ){

    Task * task = *iter;
    const Task::reqType& reqs = task->getRequires();

#if DAV_DEBUG
    cerr << "Task is-> " << *task << "\n";
#endif
    // We need to keep track of the reductionTasks so that they can
    // be called inorder by each MPI process...
    if( task->isReductionTask() ) {
#if DAV_DEBUG
      cerr << "Adding ReductionTask: " << *task << " to list of rts\n";
#endif
      reductionTasks.push_back( task );
    }

    // Run through all the dependencies required by this task
    for(Task::reqType::const_iterator dep = reqs.begin();
	dep != reqs.end(); dep++){
      OnDemandDataWarehouse* dw =
	dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);

      // If this data is already in the current DataWarehouse,
      // then don't add it to the dependency list...
      if( !dw->exists( dep->d_var, dep->d_patch ) ){

	// Some of the depencencies from reduction/scatter/gather
	// tasks don't really count, as this processor is not really
	// computing them.  Ie: I only need to wait until I have
	// finished the deps that I am computing before I can run a
	// scatter/gather/reduction task.
	if( ( task->isReductionTask() ) ||
	    ( task->getType() == Task::Gather ) || 
	    ( task->getType() == Task::Scatter ) ) {
	   // MAY NEED TO WORRY ABOUT MULTIPLE COMPUTES FOR REDUCTION
	   // TASKS.
	  const Task::Dependency* cmp = d_graph.getComputesForRequires(dep);
	  if( cmp->d_task->getAssignedResourceIndex() != me ){
	    continue; 
	  }
	}
	bool zeroParticles = false;

#if 0
This stuff is wrong...
	if( dep->d_var->typeDescription()->getType() == 
	                               TypeDescription::ParticleVariable ){
	  // If there are no particles associated with a dependency,
	  // then it really isn't a dependency so don't add it to the
	  // list of dependencies...
	  try {
	    ParticleSubset* pset = old_dw->getParticleSubset( dep->d_matlIndex,
							      dep->d_patch );
	    if( pset->numParticles() == 0 ){
#if DAV_DEBUG
	      cerr << "0 particles found so not adding " << *dep 
		   << " to list of dependencies.\n";
#endif
	      zeroParticles = true;
	    }
	  } catch (UnknownVariable & unknown ) {
#if DAV_DEBUG
	    cerr << "UnknownVariable: " << *dep << "\n";
#endif
	  }
	}
#endif
	if( !zeroParticles ){
	  DependData depData( dep );
	  TaskData   taskData( task );

	  vector<Task*> & list_of_tasks = depToTasks[ depData ];
	
	  list_of_tasks.insert( list_of_tasks.begin(), task );
	  vector<DependData> & data = taskToDeps[ taskData ];
	  data.insert( data.begin(), depData );
	}
      } 
    } // end for( reqIter = all this task's requirement )
  } // end for( iter )

#if DAV_DEBUG
  cerrLock.lock();

  cerr << "depToTasks " << depToTasks.size() << ":\n\n";

  for( map< DependData, vector<Task*>, DependData>::iterator iter =
	 depToTasks.begin(); iter != depToTasks.end(); iter++ )
    {
      DependData depData = (*iter).first;
      vector< Task * > theTasks = (*iter).second;

      cerr << "\nDependency: " << *(depData.dep) << " needed by " 
	   << theTasks.size() << ":\n";
      for( int t = 0; t < (int)theTasks.size(); t++ ){
	cerr << "  " << *(theTasks[t]) << "\n";
      }
    }

#if 1
  cerr << "\n\ntaskToDeps " << taskToDeps.size() << ":\n\n";

  for( map<TaskData, vector<DependData>, TaskData>::iterator iter =
	 taskToDeps.begin(); iter != taskToDeps.end(); iter++ )
    {
      TaskData taskData = (*iter).first;
      vector< DependData > dependencies = (*iter).second;

      cerr << "\n";
      taskData.task->displayAll( cerr );

      cerr << "\n  Depends on these " << dependencies.size() << " deps:\n";
      for( int t = 0; t < (int)dependencies.size(); t++ ){
	cerr << "    " << *(dependencies[t].dep) << "\n";
      }
    }
#endif
  cerr << "\n\n";
  cerrLock.unlock();
#endif
  // Really need to sort Reduction Tasks so that they can't block each
  // other out...  for now I am just GOING WITH A BLIND HACK... sigh.
  int numReductionTasks = (int)reductionTasks.size();
  if( numReductionTasks > 1 ){
    const Task * tempTask = reductionTasks[ numReductionTasks - 1 ];
    reductionTasks[ numReductionTasks-1 ] = reductionTasks[ numReductionTasks-2 ];
    reductionTasks[ numReductionTasks-2 ] = tempTask;
  }

#if DAV_DEBUG
  cerrLock.lock();
  cerr << "End createDepencyList\n";
  cerrLock.unlock();
#endif  
#else
  NOT_FINISHED("new task stuff");
#endif
} // end createDepencyList()


// makeAllRecvRequests also creates the Dependency lists used
// to know which tasks can be run at which time.
void
MixedScheduler::makeAllRecvRequests( vector<Task*>       & tasks, 
				   int                   me,
				   DataWarehouseP      & old_dw,
				   DataWarehouseP      & /*new_dw */)
{
#if 0
   SendState ss;
  vector<DependData> invalidDependencies;

  // Must be called before I start making recv calls, as the recv
  // call "puts" the var into the datawarehouse, making me think
  // that is has been computed.
  createDepencyList( old_dw, tasks, me );

#if DAV_DEBUG
  cerr << "Begin makeAllRecvRequests\n";
#endif
  // Run through all the current tasks...
  map<TaskData, vector<DependData>, TaskData>::iterator iter;
  
  for( iter = taskToDeps.begin(); iter != taskToDeps.end(); iter++ ) {

    const Task * task = (*iter).first.task;

    // Only deal with tasks assigned to me...
    if( task->getAssignedResourceIndex() != me )
      continue;
    // Currently, Scatter/Gather/Reduction tasks do not make generic mpi recv
    // requests... they handle there own data transmission.
    if( ( task->getType() == Task::Gather ) ||
	( task->getType() == Task::Scatter ) ||
 	( task->isReductionTask() ) ) {
      continue;
    }

    // Get the list of what data this task requires...
    vector<DependData>           & needs = (*iter).second;
    vector<DependData>::iterator   needIter;

    // Run through each piece of data and make MPI requests for it...
    //   (Only make MPI request if another processor is generating it.)
    for( needIter = needs.begin(); needIter != needs.end(); needIter++){
      
      const Task::Dependency* need = (*needIter).dep;
      const Task::Dependency* cmp = d_graph.getComputesForRequires( need );

      if( cmp->d_task->getAssignedResourceIndex() != me ) {
	// If another task has not already requested this data
	//   then send the request...
	if( !need->d_dw->exists(need->d_var, need->d_matlIndex, need->d_patch) ){

	  MPI_Request requestid;

	  OnDemandDataWarehouse* dw =
		dynamic_cast<OnDemandDataWarehouse*>(need->d_dw);
#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << "Request to (eventually) recv MPI data for: " << *need << "\n";
	  cerrLock.unlock();
#endif
	  int size;
	  dw->recvMPI(ss, old_dw, need->d_var, need->d_matlIndex,
		      need->d_patch, d_myworld, need,
		      MPI_ANY_SOURCE,
		      need->d_messageTag, &size, &requestid);
	  if(size != -1){
	    log.logRecv(need, size);
	    recv_ids.push_back(requestid);
	    reqToDep[ requestid ] = *needIter;
	  } else {
	    invalidDependencies.push_back( *needIter );
	  }
	} // end if !dep->d_dw->exists()

	vector<DependData>::iterator loc = find(invalidDependencies.begin(),
						invalidDependencies.end(),
						*needIter );
	if( loc != invalidDependencies.end() ){
#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << "removing dependency: " << *((*loc).dep) << " from task: "
	       << *task << " because 0 particles sent\n";
	  cerrLock.unlock();
#endif
	  TaskData taskData( task );
	  taskToDeps[ taskData ].erase( needIter );
	}
      } // end if cmp computer not me
    } // end for( needIter )
  } // end for( taskToDeps.begin... )
#else
  NOT_FINISHED("new task stuff");
#endif
} // end makeAllRecvRequests()

void
MixedScheduler::verifyChecksum( vector<Task*> & tasks, int me )
{
#if 0
  // Compute a simple checksum to make sure that all processes
  // are trying to use the same graph.  We should do two
  // things in the future:
  //  - make a flag to turn this off
  //  - make the checksum more sophisticated
  int checksum = ntasks;
  int result_checksum;
  MPI_Allreduce(&checksum, &result_checksum, 1, MPI_INT, MPI_MIN,
		d_myworld->getComm());
  if(checksum != result_checksum){
    cerr << "Failed task checksum comparison!\n";
    cerr << "Processor: " << d_myworld->myrank() << " of "
	 << d_myworld->size() << ": has sum " << checksum
	 << " and global is " << result_checksum << '\n';
    MPI_Abort(d_myworld->getComm(), 1);
  }
#endif
}

void
MixedScheduler::compile(const ProcessorGroup * pc, bool init_timestep )
{
  NOT_FINISHED("MixedScheduler::compile");
}

void
MixedScheduler::execute(const ProcessorGroup * pc )
{
  ASSERT(dt != 0);
#if 0
  vector<Task*>       tasks;
  int                 me = pc->myrank();

  reductionTasks.clear();
  reqToDep.clear();
  depToTasks.clear();
  taskToDeps.clear();
  recv_ids.clear();
  recv_ids_backup.clear();

  // Must call topologicalSort before assignResources because
  // topologicalSort adds internal (Reduction) tasks to graph.
  d_graph.nullSort(tasks);

  d_graph.assignUniqueMessageTags();

  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  lb->assignResources(d_graph, d_myworld);
  releasePort("load balancer");

  verifyChecksum( tasks, me );

  // cerr << "TASKS:\n";
  // displayTaskGraph( tasks );


  // Pull Scatter/Gather tasks out of the task list... they must be
  // handled separately... after all other tasks have been run.
  vector<Task *>           scatterGatherTasks;
  vector<Task *>           remove;
  vector<Task *>::iterator iter;
  for( iter = tasks.begin(); iter != tasks.end(); iter++){
    Task * task = *iter;
    if( task->getType() == Task::Gather ||
	task->getType() == Task::Scatter ){
      remove.push_back( task );
    }
  }
  scatterGatherTasks = remove;
  for( int num = 0; num < (int)remove.size(); num++ ){
    // Remove the scatter/gather tasks from the task list...
    vector<Task *>::iterator loc = find(tasks.begin(), tasks.end(),
					remove[ num ] );
    tasks.erase( loc );
  }

  // Particle Sets are just "meta" information about the particle data... 
  // This info must be sent before the actual particle data is sent in
  // order that the receiving DW be able to handle the particle data.

  sendParticleSets( tasks, me );
  recvParticleSets( tasks, me );

  sendInitialData( tasks, me );
  recvInitialData( tasks, old_dw, me );

  int                 numRequests;
  int                 numRequestReceived = 0;

  makeAllRecvRequests( tasks, me, old_dw, new_dw );

  // Need this backup (recv_ids_backup) because the request id in
  // recv_ids is set to 0 when the request comes in, thus I can't get
  // the request id out of recv_ids in order to determine what
  // dependencies are satisfied by this request.
  recv_ids_backup = recv_ids;

#if DAV_DEBUG
  cerrLock.lock();
  cerr << "Number of MPI_Requests: " << recv_ids.size() 
       << " which should be equal to " << reqToDep.size() << "\n\n";
  cerrLock.unlock();
#endif

  numRequests = (int)recv_ids.size();

  int numTasks = (int)tasks.size();
  int numTasksDone = 0;

#if DAV_DEBUG
  cerr << "The Tasks are: (only showing my tasks of: " << numTasks << ")\n";
  for( int i = 0; i < numTasks; i++ )
    {
      if( tasks[i]->getAssignedResourceIndex() == me )
	cerr << i << ": " << *tasks[i] << "\n";
    }
#endif

  // debug only
  static int iteration = -1;

  while( numTasksDone < numTasks ) {

    // debug only
    iteration++;

    vector<Task *>::iterator iter;
    vector<Task *>           done;

#if DAV_DEBUG
    cerrLock.lock();
    cerr << ".";
    cerrLock.unlock();
#endif
    int numAvail = d_threadPool->available();
#if DAV_DEBUG
    cerrLock.lock();
    cerr << "# OF THREADS AVAILABLE = " << numAvail << ", TASKS = " 
	 << tasks.size() << ", numTasksDone = " << numTasksDone 
	 << ", numTasks = " << numTasks << "\n";
    cerrLock.unlock();
#endif

    // Determine if there are any tasks that can be kicked off...
    //   These tasks don't require any (more) incoming communication.
    //   (Probably depended on a previous task on this processor or
    //   have received appropriate communication previously.)
    for( iter = tasks.begin(); numAvail > 0 && iter != tasks.end(); iter++){

#if DAV_DEBUG
      cerrLock.lock();
      cerr << "# of threads available = " << numAvail 
	   << ", tasks = " << tasks.size() <<"\n";
      if( tasks.size() == 1 )
	{
	  cerr << "task awaiting is: " << *tasks[0] << "\n";
	}
      cerrLock.unlock();
#endif

      TaskData taskData;
      Task   * task = *iter;
      taskData.task = task;

      // If this is a task in the tasks list that is not assigned to 
      // me, then it needs to be removed from the list and we need
      // to continue to the next iteration of this loop.
      if( task->getAssignedResourceIndex() != me ) {
	done.push_back( task );
	numTasksDone++;
	continue;
      }
#if DAV_DEBUG
      vector<DependData> & data = taskToDeps[ taskData ];
      cerrLock.lock();
      //      if( tasks.size() == 1 ){

      cerr << iteration << ") Considering task: " << *task;
	if( data.size() > 0 ){
	  cerr << "... which needs (" << data.size() << ":\n";
	  for(int i = 0; i < data.size(); i++){
	    cerr << "   " << *(data[i].dep) << "\n";
	  }
	} else {
	  cerr << "... which is ready to run.\n";
	}
	
	cerr << "\n";
      cerrLock.unlock();
      //      }
#endif

      // If the task has no dependencies left, then run it.
      if( taskToDeps[ taskData ].size() == 0 ) {

	// (However, if it is a reductionTask, then we can only run
	// it if it is in the right order!)
	if( task->isReductionTask() ) {
	  vector<const Task *>::iterator iter = find( reductionTasks.begin(),
						      reductionTasks.end(),
						      task );
	  if( iter == reductionTasks.end() ){
	    throw InternalError("Task not in list of reductionTasks");
	  }
	  // If this task is the reductionTask at the front of the queue,
	  // then remove it from the queue and proceed to run it.
	  if( task == reductionTasks[ 0 ] ){
#if DAV_DEBUG
	    cerr << "Reduction Task " << *task
		 << " is at the front of the queue\n";
#endif
	    reductionTasks.erase( iter );
	  } else {
#if DAV_DEBUG
	    cerr << "Reduction Task " << *task
		 << " is NOT at the front of the queue\n";
	    for( int jj = 0; jj < reductionTasks.size(); jj++ ){
	      cerr << "    " << *(reductionTasks[jj]) << "\n";
	    }
#endif
	    // Don't run it yet...
	    continue;
	  }
	}

#if DAV_DEBUG
	cerr << "Going to handle task: " << *task << "\n";
#endif
	// Mark the task done so it can be removed (later) from
	// the "tasks" vector.  Can't remove it now as this would
	// screw up the iterator.
	done.push_back( task );

        switch(task->getType()){
        case Task::Reduction:
          {
	    vector<MPI_Request> send_ids;
#if DAV_DEBUG
            cerr << "Running reduction task: " << *task << "\n";
#endif
            const Task::compType& comps = task->getComputes();
            ASSERTEQ(comps.size(), 1);
            const Task::Dependency* dep = &comps[0];
            OnDemandDataWarehouse* dw = 
              dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);
            dw->reduceMPI(dep->d_var, dep->d_matlIndex, d_myworld);
#if DAV_DEBUG
            cerr << "reduceMPI finished\n";
#endif
            numTasksDone++;
            dependenciesSatisfied( comps, me, send_ids, false );
	    ASSERTEQ( send_ids.size(), 0 );
          }
        break;
        case Task::Normal:
	  //double taskstart = Time::currentSeconds();
	  d_threadPool->assignThread( task, pc );  // which does the "doit"
	  //GET RID OF THIS SLEEP AFTER DEBUGGING IS COMPLETE : WARNING
	  //usleep(100000);
	  numAvail--;
	  //double sendstart = Time::currentSeconds();
          break;
        default:
          throw InternalError("Unknown task type");
        } // end switch task->getType()
      } // end if( taskToDeps[ taskData ].size() == 0 ){
    } // end for all tasks

    // Remove the "assigned" tasks from the task list...
    for( int num = 0; num < (int)done.size(); num++ ){

      vector<Task *>::iterator loc = find(tasks.begin(), tasks.end(),
					  done[ num ] );
#if DAV_DEBUG
      cerrLock.lock();
      cerr << "Removing " << **loc << " from queue (size now is: " 
	   << tasks.size()-1 << ")\n";
      cerrLock.unlock();
#endif
      tasks.erase( loc );
    }

    vector<Task *> completedTasks;

    // Ask the thread pool about what threads have completed...
    d_threadPool->getFinishedTasks( completedTasks );

    vector<MPI_Request> send_ids;

#if DAV_DEBUG
    cerr << "Size of send_ids is: " << send_ids.size() << "\n";
#endif

    // Run through the completed tasks and take care of things that
    // depended on them.
    for( vector<Task *>::iterator taskIter = completedTasks.begin();
         taskIter != completedTasks.end(); taskIter++ ){

      numTasksDone++;
      Task * task = *taskIter;

      //if task generated data that other processors need, send data out,
      // record fact that this data has been generated so that other
      // tasks that this processor is responsible for can be run
      const Task::compType& comps = task->getComputes();

      dependenciesSatisfied( comps, me, send_ids );
    } // end for taskIter => All Completed Tasks.

    // Wait for all the sends to completed
    if( send_ids.size() > 0 ){
#if DAV_DEBUG
      cerr << "MPI_Waitall on " << send_ids.size() << " send messages\n";
#endif
      vector<MPI_Status> statii(send_ids.size());
      MPI_Waitall((int)send_ids.size(), &send_ids[0], &statii[0]);
#if DAV_DEBUG
      cerr << "finished MPI_Waitall\n";
#endif
    }

    // Since there are no other tasks that can currently be run,
    // check to see if any data has come in that may allow us to
    // run another task...
    if( numRequestReceived < numRequests ) {
      int completed_index, completed = false;
      MPI_Status status;

      // Is it legal to specify numRequests the same every time even though
      // requests have actually come in.  The recv_ids places "0" in the
      // location of the satisfied request, so it probably works...
      MPI_Testany( numRequests,
                   & recv_ids[0],
                   & completed_index,
                   & completed,
                   & status );

      if( completed ){

	vector<MPI_Request> send_ids;	

        // Mark the fact that this data has been received...
        DependData d = reqToDep[ recv_ids_backup[ completed_index ] ];
#if DAV_DEBUG
	cerr << "MPI: This, just in: " << *(d.dep) << "\n";
#endif
	dependencySatisfied( d.dep, me, send_ids, false );

	ASSERTEQ( send_ids.size(), 0 );

        numRequestReceived++;
      }
    } // End if numRequestReceived < numRequests

    // Stop the processor from spinning out of control.  This shouldn't
    // make much of a difference...
    // usleep( 1000 );  // Perhaps it does make a big difference?
  } // End while( tasks.size() > 0 )

  // Handle all Scatter tasks first...
  vector<Task *>::iterator sgIter;
  for( sgIter = scatterGatherTasks.begin(); sgIter != scatterGatherTasks.end(); sgIter++){
    Task * task = *sgIter;

    if( ( task->getAssignedResourceIndex() == me ) && ( task->getType() == Task::Scatter ) ){
#if DAV_DEBUG
      cerr << "Run: " << *task << "\n";
#endif
      const Task::compType& comps = task->getComputes();
      ASSERTEQ(comps.size(), 1);
      const Task::Dependency* cmp = &comps[0];
      vector<const Task::Dependency*> reqs;
      d_graph.getRequiresForComputes(cmp, reqs);
               
      sgargs.dest.resize(reqs.size());
      sgargs.tags.resize(reqs.size());
      for(int r=0;r<(int)reqs.size();r++){
	const Task::Dependency* dep = reqs[r];
	sgargs.dest[r] = dep->d_task->getAssignedResourceIndex();
	sgargs.tags[r] = dep->d_messageTag;
      }
      task->doit( pc );
#if DAV_DEBUG
      cerr << "Done running scatter task\n";
#endif
      vector<MPI_Request> send_ids;
      dependenciesSatisfied( task->getComputes(), me, send_ids, false );
      ASSERTEQ( send_ids.size(), 0 );
    } // end if( me && Scatter )
  } // for( sgIter... )

  // Handle Gather tasks second
  for( sgIter = scatterGatherTasks.begin(); sgIter != scatterGatherTasks.end(); sgIter++){
    Task * task = *sgIter;

    if( ( task->getAssignedResourceIndex() == me ) && ( task->getType() == Task::Gather ) ){
#if DAV_DEBUG
      cerr << "Run: " << *task << "\n";
#endif
      const Task::reqType& reqs = task->getRequires();
      sgargs.dest.resize(reqs.size());
      sgargs.tags.resize(reqs.size());
      for(int r=0;r<(int)reqs.size();r++){
	const Task::Dependency* req = &reqs[r];
	const Task::Dependency* cmp = d_graph.getComputesForRequires(req);
	sgargs.dest[r] = cmp->d_task->getAssignedResourceIndex();
	sgargs.tags[r] = req->d_messageTag;
      }
      task->doit( pc );
#if DAV_DEBUG
      cerr << "Done running gather task\n";
#endif
      vector<MPI_Request> send_ids;
      dependenciesSatisfied( task->getComputes(), me, send_ids, false );
      ASSERTEQ( send_ids.size(), 0 );
    } // end if( me && Gather )
  } // for( sgIter... )

  new_dw->finalize();
  finalizeNodes();

#if DAV_DEBUG
  cerr << "Done MixedScheduler::Execute\n";
  // usleep( 500000 );
#endif
#else
  NOT_FINISHED("new task stuff");
#endif
} // End execute()

void
MixedScheduler::sendInitialData( vector<Task*> & tasks,
                               int             me )
{
#if 0
   SendState ss; // SHOULD BE PASSED IN! - Steve
#if DAV_DEBUG
  cerr << "Initial DW Data Send\n";
#endif

  set<VarDestType> varsent;

  vector<MPI_Request>      send_ids;
  vector<Task*>::iterator  iter;

  for( iter = tasks.begin(); iter != tasks.end(); iter++ ){
    Task* task = *iter;
    if(task->getAssignedResourceIndex() == me)
      continue;
    if(task->getType() != Task::Normal)
      continue;

    const Task::reqType& reqs = task->getRequires();
    for(Task::reqType::const_iterator dep = reqs.begin();
        dep != reqs.end(); dep++){
      if(dep->d_dw->isFinalized() && dep->d_patch){
        if(dep->d_dw->exists(dep->d_var, dep->d_matlIndex, dep->d_patch)){
          VarDestType ddest(dep->d_var,
                            dep->d_matlIndex,
                            dep->d_patch,
                            task->getAssignedResourceIndex());
          if(varsent.find(ddest) == varsent.end()){
            OnDemandDataWarehouse* dw = 
                     dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);
            if(!dw)
              throw InternalError("Wrong Datawarehouse?");
            MPI_Request requestid;
            ASSERT(dep->d_messageTag >= 0);
#if DAV_DEBUG
            cerr << "sending initial " << *dep
                 << " serial " << dep->d_messageTag << ", to " 
                 << dep->d_task->getAssignedResourceIndex() << '\n';
#endif
	    int size;
            dw->sendMPI(ss, dep->d_var, dep->d_matlIndex,
                        dep->d_patch, d_myworld, dep,
                        dep->d_task->getAssignedResourceIndex(),
                        dep->d_messageTag, &size, &requestid);
	    if(size != -1){
	      log.logSend(dep, size);
	      send_ids.push_back(requestid);
	      varsent.insert(ddest);
	    }
          }
        }
      }
    }
  }
  //  vector<MPI_Status> statii(send_ids.size());
  //  MPI_Waitall((int)send_ids.size(), &send_ids[0], &statii[0]);

#if DAV_DEBUG
  cerr << "Done Sending Initial DW Data\n";
#endif
#else
  NOT_FINISHED("new task stuff");
#endif
} // end sendInitialData()

void
MixedScheduler::recvInitialData( vector<Task*>  & tasks,
                               DataWarehouseP & old_dw,
                               int              me )
{
#if 0
   SendState ss; // SHOULD BE PASSED IN! - Steve
#if DAV_DEBUG
  cerr << "start: recvInitialData\n";
#endif

   vector<MPI_Request> recv_ids;
   vector<Task*>::iterator iter;

   for(iter = tasks.begin(); iter != tasks.end(); iter++){

      Task* task = *iter;
      if(task->getAssignedResourceIndex() != me)
         continue;
      if(task->getType() != Task::Normal)
         continue;

      const Task::reqType& reqs = task->getRequires();
      for(Task::reqType::const_iterator dep = reqs.begin();
          dep != reqs.end(); dep++){
#if DAV_DEBUG
         cerr << "Looking at dep: " << *dep << '\n';
#endif
         if(dep->d_dw->isFinalized() && dep->d_patch){
#if DAV_DEBUG
            cerr << "dw finalized and patch exists\n";
#endif
            if(!dep->d_dw->exists(dep->d_var, dep->d_matlIndex, dep->d_patch)){
#if DAV_DEBUG
               cerr << "Variable does not exist\n";
#endif
               OnDemandDataWarehouse* dw = 
                     dynamic_cast<OnDemandDataWarehouse*>(dep->d_dw);
               if(!dw)
                  throw InternalError("Wrong Datawarehouse?");
               MPI_Request requestid;
               ASSERT(dep->d_messageTag >= 0);

#if DAV_DEBUG
               cerr << "MPI: Request to receive initial " << *dep << ": serial #: " 
		    << dep->d_messageTag << '\n';
#endif
	       int size;
               dw->recvMPI(ss, old_dw, dep->d_var, dep->d_matlIndex,
                           dep->d_patch, d_myworld, dep,
                           MPI_ANY_SOURCE,
                           dep->d_messageTag, &size, &requestid);
	       if(size != -1){
		 log.logRecv(dep, size);
		 recv_ids.push_back(requestid);
	       }
#if DAV_DEBUG
	       cerr << "Made receive request\n";
#endif
            }
         }
      } // End for requires
   } // End for all tasks
#if DAV_DEBUG
   cerr << "Waiting for Initial Data for " << recv_ids.size() << " vars\n";
#endif
   vector<MPI_Status> statii(recv_ids.size());
   MPI_Waitall((int)recv_ids.size(), &recv_ids[0], &statii[0]);
#if DAV_DEBUG
  cerr << "Done Receiving Initial DW Data\n";
#endif
#else
  NOT_FINISHED("new task stuff");
#endif
} // end recvInitialData()

void
MixedScheduler::scheduleParticleRelocation(
                  const LevelP& level,
                  const VarLabel* old_posLabel,
                  const vector<vector<const VarLabel*> >& old_labels,
                  const VarLabel* new_posLabel,
                  const vector<vector<const VarLabel*> >& new_labels,
                  const MaterialSet* matls)
{
  UintahParallelPort* lbp = getPort("load balancer");
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
  reloc.scheduleParticleRelocation(this, d_myworld, lb, level,
				   old_posLabel, old_labels,
				   new_posLabel, new_labels, matls);
  releasePort("load balancer");
}

void
MixedScheduler::displayTaskGraph( vector<Task*> & taskGraph )
{
#if 0
  if( Parallel::usingMPI() && 
      Parallel::getRootProcessorGroup()->myrank() == 0 ){
    cerr << "\n---------------------------\n";
    cerr << "Begin: Tasks in Task List:\n";
    for(vector<Task*>::iterator iter = taskGraph.begin();
	iter != taskGraph.end(); iter++){
      cerr << "\n" << **iter 
	   << "\n  REQUIRES:\n";
      const Task::reqType& reqs = (*iter)->getRequires();
      for(Task::reqType::const_iterator riter = reqs.begin();
	  riter != reqs.end(); riter++){
	cerr << "     " << *(riter->d_var) << "		P: ";
	if( riter->d_patch )
	  cerr << riter->d_patch->getID();
	else
	  cerr << "?";
	cerr << " MI: " << riter->d_matlIndex << "\n";
      }

      cerr << "\n  COMPUTES:\n";
      const Task::compType& comps = (*iter)->getComputes();
      
      for(Task::compType::const_iterator riter = comps.begin();
	  riter != comps.end(); riter++){
	cerr << "     " << *(riter->d_var) << "		P: ";
	if( riter->d_patch )
	  cerr << riter->d_patch->getID();
	else
	  cerr << "?";
	cerr << " MI: " << riter->d_matlIndex << "\n";
      }
    }
    cerr << "End: Tasks in Task List:\n";
  }
#else
  NOT_FINISHED("new task stuff");
#endif
} // end displayTaskGraph

const Task::Dependency *
findRequirement( const Task::Dependency * comp, Task * task )
{
#if 0
  const Task::reqType& reqs = task->getRequires();

  for(Task::reqType::const_iterator req = reqs.begin(); req != reqs.end();
      req++){
    if( ( req->d_dw == comp->d_dw ) &&
	( req->d_var->getName() == comp->d_var->getName() ) &&
	( req->d_matlIndex == comp->d_matlIndex ) &&
	( req->d_patch == comp->d_patch ) ) {
      return req;
    }
  }
  cerr << "Couldn't find a matching req for comp: " << *comp << "\n";
  task->displayAll( cerr );
  throw InternalError( "Should have found a matching requirement! ");
#else
  NOT_FINISHED("new task stuff");
  return 0;
#endif
}

#if 0
void
MixedScheduler::dependencySatisfied( const Task::Dependency * comp,
				   int me,
				   vector<MPI_Request> & send_ids,
				   bool sendData )
{
   SendState ss; // SHOULD BE PASSED IN! - Steve
    DependData		    d( comp );
    vector<Task *>	    tasks = depToTasks[ d ];
    vector<Task*>::iterator taskIter;

    // List of processor ids that data has been sent to.
    vector<int>		    sentTo;

#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << *comp << " has been computed!\n";
	  cerr << "Finished task is: " << *comp->d_task << ", computes: " 
	       << tasks.size() << " things\n";
	  cerrLock.unlock();
#endif
    for( taskIter = tasks.begin(); taskIter != tasks.end(); taskIter++){

      Task * task = *taskIter;
#if DAV_DEBUG
      cerrLock.lock();
      cerr << *task << " needs it!\n";
      cerrLock.unlock();
#endif
      // Determine if I should send this data to other DWs
      //   The task that I am sending to must not be me, and the task computing 
      //   the data must be me.
      if( task->getAssignedResourceIndex() != me && sendData ) {
#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << "Task is not mine\n";
	  cerrLock.unlock();
#endif
	// Scatter/Gather/Reduction Tasks handle there own sending of data...
	if( comp->d_task->getType() == Task::Normal ) {

#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << "Is normal so may need to send data to it!\n"; 
	  cerrLock.unlock();
#endif
	  const Task::Dependency * req = findRequirement( comp, task );

	  vector<int>::iterator procLoc = find( sentTo.begin(), sentTo.end(),
						req->d_task->getAssignedResourceIndex() );

#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << "Checking to see if data has been sent to: "
	       << req->d_task->getAssignedResourceIndex() << "\n";
	  cerr << "    Size of vector is " << sentTo.size() << "\n";
	  cerrLock.unlock();
#endif
	  if( procLoc != sentTo.end() ){
#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << "The data has already been sent to that processor!\n"; 
	  cerrLock.unlock();
#endif
	    continue;
	  }

	  OnDemandDataWarehouse* dw = 
		    dynamic_cast<OnDemandDataWarehouse*>(comp->d_dw);
	  if(!dw)
	    throw InternalError("Wrong Datawarehouse?");


	  MPI_Request requestid;
	  ASSERT(req->d_messageTag >= 0);

	  // Need to change sendMPI to a blocking send and get
	  // rid of MPI_Waitall... less clutter, same functionality
	  // HMMM... I AM NOT SURE THE ABOVE IS RIGHT...
	  // SHOULD BE ABLE TO SEND IT AND KEEP GOING...
#if DAV_DEBUG
	  cerrLock.lock();
	  cerr << "Considering MPI Sending " << *req << "\n";
	  cerrLock.unlock();
#endif
	  int size;
	  dw->sendMPI(ss, req->d_var, req->d_matlIndex,
		      req->d_patch, d_myworld, req,
		      req->d_task->getAssignedResourceIndex(),
		      req->d_messageTag, &size, &requestid);

	  if(size != -1){
#if DAV_DEBUG
	    cerrLock.lock();
	    cerr << "MPI Sent: " << *req << "\n";
	    cerrLock.unlock();
#endif
	    log.logSend(req, size);
	    send_ids.push_back( requestid );
	    sentTo.push_back( req->d_task->getAssignedResourceIndex() );
#if DAV_DEBUG
	    cerrLock.lock();
	    cerr << "sent to " << req->d_task->getAssignedResourceIndex() << "\n";
	    cerrLock.unlock();
#endif
	  }
	}
      }

      // Take care of house keeping...
#if DAV_DEBUG
      cerrLock.lock();
      cerr << "Removing " << *d.dep << " from task " << *task << "\n";
      cerrLock.unlock();
#endif

      TaskData taskData( task );

      vector<DependData> & depDatas = taskToDeps[ taskData ];

      vector<DependData>::iterator loc = 
	find(depDatas.begin(), depDatas.end(), d);

      if( loc == depDatas.end() ){
	cerr << "Task is: " << *task << "\n";
	cerr << "depDatas size is: " << depDatas.size() << "\n";
	for( int kk = 0; kk < (int)depDatas.size(); kk++ ){
	  cerr << "   " << kk << " dep is: " << *(depDatas[kk].dep) << "\n";
	}
	cerr << "Dependency: " << *(d.dep) << " not found in taskToDeps\n";
	throw InternalError( "This should not happen" );
      }
	   
      taskToDeps[ taskData ].erase( loc );
    } // end for( taskIter )
#if DAV_DEBUG
    cerr << "\n\n";
#endif
}  // end dependencySatisfied()
#endif

#if 0
void
MixedScheduler::dependenciesSatisfied( const Task::compType & comps,
				     int me,
				     vector<MPI_Request> & send_ids,
				     bool sendData )
{
#if DAV_DEBUG
  cerr << "There were " << comps.size() << " computed dependencies.\n";
#endif

  Task::compType::const_iterator cmp;
  for( cmp = comps.begin(); cmp != comps.end(); cmp++ ) {
    dependencySatisfied( cmp, me, send_ids, sendData );
  }
}
#endif
