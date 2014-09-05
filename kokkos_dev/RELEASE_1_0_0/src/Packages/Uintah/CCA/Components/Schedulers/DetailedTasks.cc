
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/DetailedTask.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;

static DebugStream dbg("TaskGraph", false);

DetailedTasks::DetailedTasks(const ProcessorGroup* pg)
{
  int nproc = pg->size();
  stasks.resize(nproc);
  tasks.resize(nproc);
  for(int i=0;i<nproc;i++) {
    stasks[i]=scinew Task("send old data", Task::InitialSend);
    tasks[i]=scinew DetailedTask(stasks[i], 0, 0);
    tasks[i]->assignResource(i);
  }
}

DetailedTasks::~DetailedTasks()
{
  for(int i=0;i<(int)batches.size();i++)
    delete batches[i];

  for(int i=0;i<(int)tasks.size();i++)
    delete tasks[i];

  for(int i=0;i<(int)stasks.size();i++)
    delete stasks[i];
}

DependencyBatch::~DependencyBatch()
{
  DetailedDep* dep = head;
  while(dep){
    DetailedDep* tmp = dep->next;
    delete dep;
    dep = tmp;
  }
}

void
DetailedTasks::assignMessageTags()
{
  int serial=1;
  for(int i=0;i<(int)batches.size();i++)
    batches[i]->messageTag = serial++;
  maxSerial=serial;
  dbg << "MAXSERIAL=" << maxSerial << '\n';
} // end assignMessageTags()

void
DetailedTasks::add(DetailedTask* task)
{
  tasks.push_back(task);
}

#if 0
vector<DetailedReq*>&
DetailedTasks::getInitialRequires()
{
#if 0
  for(DetailedReq* req = task->getRequires();
      req != 0; req = req->next){
    if(req->req->dw == Task::OldDW)
      initreqs.push_back(req);
  }
#else
  NOT_FINISHED("DetailedTasks::add");
#endif
  cerr << initreqs.size() << " initreqs\n";
  return initreqs;
}
#endif

void
DetailedTasks::computeLocalTasks(int me)
{
  ASSERT(localtasks.size() == 0);
  for(int i=0;i<(int)tasks.size();i++){
    DetailedTask* task = tasks[i];
    if(task->getAssignedResourceIndex() == me
       || task->getTask()->getType() == Task::Reduction)
      localtasks.push_back(task);
  }
}

DetailedTask::DetailedTask(Task* task, const PatchSubset* patches,
			   const MaterialSubset* matls)
  : task(task), patches(patches), matls(matls), req_head(0),
    comp_head(0), resourceIndex(-1)
{
  if(patches)
    patches->addReference();
  if(matls)
    matls->addReference();
}

DetailedTask::~DetailedTask()
{
  if(patches && patches->removeReference())
    delete patches;
  if(matls && matls->removeReference())
    delete matls;
}

void DetailedTask::doit(const ProcessorGroup* pg, DataWarehouse* old_dw,
			DataWarehouse* new_dw)
{
  task->doit(pg, patches, matls, old_dw, new_dw);
}

void DetailedTasks::possiblyCreateDependency(DetailedTask* from,
					     Task::Dependency* comp,
					     const Patch* fromPatch,
					     DetailedTask* to,
					     Task::Dependency* req,
					     const Patch *toPatch,
					     int matl,
					     const IntVector& low,
					     const IntVector& high)
{
  // TODO - maybe still create internal depencies for threaded scheduler?
  // TODO - perhaps move at least some of this to TaskGraph?
  ASSERT(from->getAssignedResourceIndex() != -1);
  ASSERT(to->getAssignedResourceIndex() != -1);
#if 1
  dbg << "Dependency from " << *from << " to " << *to << "\n";
  if(comp)
    dbg << "From comp " << *comp;
  else
    dbg << "From OldDW ";
  dbg << " to req " << *req << '\n';
#endif
  if(from->getAssignedResourceIndex() == to->getAssignedResourceIndex())
    return;
  int toresource = to->getAssignedResourceIndex();
  DependencyBatch* batch = from->getComputes();
  for(;batch != 0; batch = batch->comp_next){
    if(batch->to == toresource)
      break;
  }
  if(!batch){
    batch = scinew DependencyBatch(toresource, from, to);
    batches.push_back(batch);
    from->addComputes(batch);
    to->addRequires(batch);
    dbg << "NEW BATCH!\n";
  }
  DetailedDep* dep = batch->head;
  for(;dep != 0; dep = dep->next){
    if(fromPatch == dep->fromPatch && matl == dep->matl
       && (req == dep->req
	   || (req->var->equals(dep->req->var)
	       && req->dw == dep->req->dw)))
		 break;
  }
  if(!dep){
    dep = scinew DetailedDep(batch->head, comp, req, fromPatch, matl, low, high);
    batch->head = dep;
    dbg << "ADDED " << low << " " << high << ", fromPatch = " << fromPatch->getID() << '\n';
  } else {
    IntVector l = Min(low, dep->low);
    IntVector h = Max(high, dep->high);
    IntVector d1 = h-l;
    IntVector d2 = high-low;
    IntVector d3 = dep->high-dep->low;
    int v1 = d1.x()*d1.y()*d1.z();
    int v2 = d2.x()*d2.y()*d2.z();
    int v3 = d3.x()*d3.y()*d3.z();
    if(v1 > v2+v3){
      // If we get this, perhaps we should allow multiple deps so
      // that we do not communicate more of the patch than necessary
      cerr << "WARNING: Possible extra communication between patches!\n";
    }
    dep->low=l;
    dep->high=h;
    dbg << "EXTENDED from " << dep->low << " " << dep->high << " to " << l << " " << h << "\n";
    dbg << *req->var << '\n';
    dbg << *dep->req->var << '\n';
    if(comp)
      dbg << *comp->var << '\n';
    if(dep->comp)
      dbg << *dep->comp->var << '\n';
  }
}

DetailedTask* DetailedTasks::getOldDWSendTask(int proc)
{
  // These are the first N tasks
  return tasks[proc];
}

void DetailedTask::addComputes(DependencyBatch* comp)
{
  comp->comp_next=comp_head;
  comp_head=comp;
}

void DetailedTask::addRequires(DependencyBatch* req)
{
  req->req_next=req_head;
  req_head=req;
}

ostream& operator<<(ostream& out, const DetailedTask& task)
{
  out << task.getTask()->getName();
  const PatchSubset* patches = task.getPatches();
  if(patches){
    out << ", on patch";
    if(patches->size() > 1)
      out << "es";
    out << " ";
    for(int i=0;i<patches->size();i++){
      if(i>0)
	out << ",";
      out << patches->get(i)->getID();
    }
  }
  const MaterialSubset* matls = task.getMaterials();
  if(matls){
    out << ", on material";
    if(matls->size() > 1)
      out << "s";
    out << " ";
    for(int i=0;i<matls->size();i++){
      if(i>0)
	out << ",";
      out << matls->get(i);
    }
  }
  out << ", resource ";
  if(task.getAssignedResourceIndex() == -1)
    out << "unassigned";
  else
    out << task.getAssignedResourceIndex();
  return out;
}

ostream& operator<<(ostream& out, const DetailedDep& dep)
{
  out << dep.req->var->getName() << " on patch " << dep.fromPatch->getID() 
      << ", matl " << dep.matl << ", low=" << dep.low << ", high=" << dep.high;
  return out;
}

