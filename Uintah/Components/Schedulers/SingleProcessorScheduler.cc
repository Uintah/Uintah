
// $Id$

#include <Uintah/Components/Schedulers/SingleProcessorScheduler.h>
#include <Uintah/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Uintah/Interface/LoadBalancer.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ScatterGatherBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Util/DebugStream.h>
#include <SCICore/Util/FancyAssert.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;
using SCICore::Thread::Time;

static const TypeDescription* specialType;

static SCICore::Util::DebugStream dbg("SingleProcessorScheduler", false);

SingleProcessorScheduler::SingleProcessorScheduler(const ProcessorGroup* myworld,
    	    	    	    	    	    	   Output* oport)
   : UintahParallelComponent(myworld), Scheduler(oport)
{
  d_generation = 0;
  if(!specialType)
     specialType = scinew TypeDescription(TypeDescription::ScatterGatherVariable,
				       "DataWarehouse::specialInternalScatterGatherType", false, -1);
  scatterGatherVariable = scinew VarLabel("DataWarehouse::scatterGatherVariable",
				       specialType, VarLabel::Internal);

  reloc_old_posLabel = reloc_new_posLabel = 0;
}

SingleProcessorScheduler::~SingleProcessorScheduler()
{
  delete reloc_old_posLabel;
  delete reloc_new_posLabel;
  delete scatterGatherVariable;

}

void
SingleProcessorScheduler::initialize()
{
   graph.initialize();
}

void
SingleProcessorScheduler::execute(const ProcessorGroup * pc,
			             DataWarehouseP   &,
			             DataWarehouseP   & dw )
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   lb->assignResources(graph, d_myworld);
   releasePort("load balancer");

   vector<Task*> tasks;
   graph.topologicalSort(tasks);

   int ntasks = (int)tasks.size();
   if(ntasks == 0){
      cerr << "WARNING: Scheduler executed, but no tasks\n";
   }
   dbg << "Executing " << ntasks << " tasks\n";

   emitEdges(tasks);

   for(int i=0;i<ntasks;i++){
      time_t t = time(NULL);
      double start = Time::currentSeconds();
      tasks[i]->doit(pc);
      double dt = Time::currentSeconds()-start;
      dbg << "Completed task: " << tasks[i]->getName();
      if(tasks[i]->getPatch())
	 dbg << " on patch " << tasks[i]->getPatch()->getID();
      dbg << " (" << dt << " seconds)\n";

      emitNode(tasks[i], t, dt);
   }

   dw->finalize();
   finalizeNodes();
}

void
SingleProcessorScheduler::addTask(Task* task)
{
   graph.addTask(task);
}

DataWarehouseP
SingleProcessorScheduler::createDataWarehouse(DataWarehouseP& parent_dw)
{
  int generation = d_generation++;
  return scinew OnDemandDataWarehouse(d_myworld, generation, parent_dw);
}


void
SingleProcessorScheduler::scheduleParticleRelocation(const LevelP& level,
						     DataWarehouseP& old_dw,
						     DataWarehouseP& new_dw,
						     const VarLabel* old_posLabel,
						     const vector<vector<const VarLabel*> >& old_labels,
						     const VarLabel* new_posLabel,
						     const vector<vector<const VarLabel*> >& new_labels,
						     int numMatls)
{
   reloc_old_posLabel = old_posLabel;
   reloc_old_labels = old_labels;
   reloc_new_posLabel = new_posLabel;
   reloc_new_labels = new_labels;
   reloc_numMatls = numMatls;
   for (int m = 0; m< numMatls; m++)
     ASSERTEQ(reloc_new_labels[m].size(), reloc_old_labels[m].size());
   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;

      Task* t = scinew Task("SingleProcessorScheduler::scatterParticles",
			    patch, old_dw, new_dw,
			    this, &SingleProcessorScheduler::scatterParticles);
      for(int m=0;m < numMatls;m++){
	 t->requires( new_dw, old_posLabel, m, patch, Ghost::None);
	 for(int i=0;i<old_labels[m].size();i++)
	    t->requires( new_dw, old_labels[m][i], m, patch, Ghost::None);
      }
      t->computes(new_dw, scatterGatherVariable, 0, patch);
      addTask(t);

      Task* t2 = scinew Task("SingleProcessorScheduler::gatherParticles",
			     patch, old_dw, new_dw,
			     this, &SingleProcessorScheduler::gatherParticles);
      // Particles are only allowed to be one cell out
      IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
      IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
      std::vector<const Patch*> neighbors;
      level->selectPatches(l, h, neighbors);
      for(int i=0;i<neighbors.size();i++)
	 t2->requires(new_dw, scatterGatherVariable, 0, neighbors[i], Ghost::None);
      for(int m=0;m < numMatls;m++){
	 t2->computes( new_dw, new_posLabel, m, patch);
	 for(int i=0;i<new_labels[m].size();i++)
	    t2->computes(new_dw, new_labels[m][i], m, patch);
      }

      addTask(t2);
   }
}

namespace Uintah {
   struct ScatterMaterialRecord {
      ParticleSubset* relocset;
      vector<ParticleVariableBase*> vars;
   };

   struct ScatterRecord : public ScatterGatherBase {
      vector<ScatterMaterialRecord*> matls;
   };
}

void
SingleProcessorScheduler::scatterParticles(const ProcessorGroup*,
					   const Patch* patch,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
   const Level* level = patch->getLevel();

   // Particles are only allowed to be one cell out
   IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
   IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
   vector<const Patch*> neighbors;
   level->selectPatches(l, h, neighbors);

   vector<ScatterRecord*> sr(neighbors.size());
   for(int i=0;i<sr.size();i++)
      sr[i]=0;
   for(int m = 0; m < reloc_numMatls; m++){
      ParticleSubset* pset = old_dw->getParticleSubset(m, patch);
      ParticleVariable<Point> px;
      new_dw->get(px, reloc_old_posLabel, pset);

      ParticleSubset* relocset = scinew ParticleSubset(pset->getParticleSet(),
						    false, -1, 0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;
	 if(!patch->getBox().contains(px[idx])){
	    //cerr << "WARNING: Particle left patch: " << px[idx] << ", patch: " << patch << '\n';
	    relocset->addParticle(idx);
	 }
      }
      if(relocset->numParticles() > 0){
	 // Figure out where they went...
	 for(ParticleSubset::iterator iter = relocset->begin();
	     iter != relocset->end(); iter++){
	    particleIndex idx = *iter;
	    // This loop should change - linear searches are not good!
	    int i;
	    for(i=0;i<neighbors.size();i++){
	       if(neighbors[i]->getBox().contains(px[idx])){
		  break;
	       }
	    }
	    if(i == neighbors.size()){
	       // Make sure that the particle left the world
	       if(level->containsPoint(px[idx]))
		  throw InternalError("Particle fell through the cracks!");
	    } else {
	       if(!sr[i]){
		 sr[i] = scinew ScatterRecord();
		 sr[i]->matls.resize(reloc_numMatls);
		 for(int m=0;m<reloc_numMatls;m++){
		   sr[i]->matls[m]=0;
		 }
	       }
	       if(!sr[i]->matls[m]){
		  ScatterMaterialRecord* smr=scinew ScatterMaterialRecord();
		  sr[i]->matls[m]=smr;
		  smr->vars.push_back(new_dw->getParticleVariable(reloc_old_posLabel, pset));
		  for(int v=0;v<reloc_old_labels[m].size();v++)
		     smr->vars.push_back(new_dw->getParticleVariable(reloc_old_labels[m][v], pset));
		  smr->relocset = scinew ParticleSubset(pset->getParticleSet(),
						     false, -1, 0);
	       }
	       sr[i]->matls[m]->relocset->addParticle(idx);
	    }
	 }
      }
      delete relocset;
   }
   for(int i=0;i<sr.size();i++){
      new_dw->scatter(sr[i], patch, neighbors[i]);
   }

}

void
SingleProcessorScheduler::gatherParticles(const ProcessorGroup*,
					  const Patch* patch,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
   const Level* level = patch->getLevel();

   // Particles are only allowed to be one cell out
   IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
   IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
   vector<const Patch*> neighbors;
   level->selectPatches(l, h, neighbors);

   vector<ScatterRecord*> sr;
   for(int i=0;i<neighbors.size();i++){
      if(patch != neighbors[i]){
	 ScatterGatherBase* sgb = new_dw->gather(neighbors[i], patch);
	 if(sgb != 0){
	    ScatterRecord* srr = dynamic_cast<ScatterRecord*>(sgb);
	    ASSERT(srr != 0);
	    sr.push_back(srr);
	 }
      }
   }
   for(int m=0;m<reloc_numMatls;m++){
      // Compute the new particle subset
      vector<ParticleSubset*> subsets;
      vector<ParticleVariableBase*> posvars;

      // Get the local subset without the deleted particles...
      ParticleSubset* pset = old_dw->getParticleSubset(m, patch);
      ParticleVariable<Point> px;
      new_dw->get(px, reloc_old_posLabel, pset);

      ParticleSubset* keepset = scinew ParticleSubset(pset->getParticleSet(),
						   false, -1, 0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;
	 if(patch->getBox().contains(px[idx]))
	    keepset->addParticle(idx);
      }
      subsets.push_back(keepset);
      particleIndex totalParticles = keepset->numParticles();
      ParticleVariableBase* pos = new_dw->getParticleVariable(reloc_old_posLabel, pset);
      posvars.push_back(pos);

      // Get the subsets from the neighbors
      for(int i=0;i<sr.size();i++){
	 if(sr[i]->matls[m]){
	    subsets.push_back(sr[i]->matls[m]->relocset);
	    posvars.push_back(sr[i]->matls[m]->vars[0]);
	    totalParticles += sr[i]->matls[m]->relocset->numParticles();
	 }
      }
      ParticleVariableBase* newpos = pos->clone();
      ParticleSubset* newsubset = new_dw->createParticleSubset(totalParticles, m, patch);
      newpos->gather(newsubset, subsets, posvars);
      new_dw->put(*newpos, reloc_new_posLabel);
      delete newpos;

      for(int v=0;v<reloc_old_labels[m].size();v++){
	 vector<ParticleVariableBase*> gathervars;
	 ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], pset);

	 gathervars.push_back(var);
	 for(int i=0;i<sr.size();i++){
	    if(sr[i]->matls[m])
	       gathervars.push_back(sr[i]->matls[m]->vars[v+1]);
	 }
	 ParticleVariableBase* newvar = var->clone();
	 newvar->gather(newsubset, subsets, gathervars);
	 new_dw->put(*newvar, reloc_new_labels[m][v]);
	 delete newvar;
      }
      for(int i=0;i<subsets.size();i++)
	 delete subsets[i];
   }
   for(int i=0;i<sr.size();i++){
      for(int m=0;m<reloc_numMatls;m++)
	 if(sr[i]->matls[m])
	    delete sr[i]->matls[m];
      delete sr[i];
   }
}

//
// $Log$
// Revision 1.16  2000/08/28 17:48:39  sparker
// Fixed delete for multi-material problems
//
// Revision 1.15  2000/08/23 21:40:04  sparker
// Fixed slight memory leak when particles cross patch boundaries
//
// Revision 1.14  2000/08/22 20:54:49  sparker
// Fixed memory leaks
//
// Revision 1.13  2000/08/21 15:36:26  jas
// Removed the deletion of the varlabels in the destructor.
//
// Revision 1.12  2000/08/18 23:26:14  guilkey
// Removed a bunch of deletes in the SingleProcessorScheduler that were
// recently added and were killing the MPM code after ~90 timesteps.
//
// Revision 1.11  2000/08/08 01:32:45  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.10  2000/07/28 22:45:15  jas
// particle relocation now uses separate var labels for each material.
// Addd <iostream> for ReductionVariable.  Commented out protected: in
// Scheduler class that preceeded scheduleParticleRelocation.
//
// Revision 1.9  2000/07/28 03:08:57  rawat
// fixed some cvs conflicts
//
// Revision 1.8  2000/07/28 03:01:54  rawat
// modified createDatawarehouse and added getTop()
//
// Revision 1.7  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.6  2000/07/26 20:14:11  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.5  2000/07/25 20:59:28  jehall
// - Simplified taskgraph output implementation
// - Sort taskgraph edges; makes critical path algorithm eastier
//
// Revision 1.4  2000/07/19 21:47:59  jehall
// - Changed task graph output to XML format for future extensibility
// - Added statistical information about tasks to task graph output
//
// Revision 1.3  2000/06/17 07:04:55  sparker
// Implemented initial load balancer modules
// Use ProcessorGroup
// Implemented TaskGraph - to contain the common scheduling stuff
//
// Revision 1.2  2000/06/16 22:59:39  guilkey
// Expanded "cycle detected" print statement
//
// Revision 1.1  2000/06/15 23:14:07  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.20  2000/06/15 21:57:11  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.19  2000/06/14 23:43:25  jehall
// - Made "cycle detected" exception more informative
//
// Revision 1.18  2000/06/08 17:11:39  jehall
// - Added quotes around task names so names with spaces are parsable
//
// Revision 1.17  2000/06/03 05:27:23  sparker
// Fixed dependency analysis for reduction variables
// Removed warnings
// Now allow for task patch to be null
// Changed DataWarehouse emit code
//
// Revision 1.16  2000/05/30 20:19:22  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.15  2000/05/30 17:09:37  dav
// MPI stuff
//
// Revision 1.14  2000/05/21 20:10:48  sparker
// Fixed memory leak
// Added scinew to help trace down memory leak
// Commented out ghost cell logic to speed up code until the gc stuff
//    actually works
//
// Revision 1.13  2000/05/19 18:35:09  jehall
// - Added code to dump the task dependencies to a file, which can be made
//   into a pretty dependency graph.
//
// Revision 1.12  2000/05/11 20:10:19  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.11  2000/05/07 06:02:07  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.10  2000/05/05 06:42:42  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.9  2000/04/28 21:12:04  jas
// Added some includes to get it to compile on linux.
//
// Revision 1.8  2000/04/26 06:48:32  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/20 18:56:25  sparker
// Updates to MPM
//
// Revision 1.6  2000/04/19 21:20:02  dav
// more MPI stuff
//
// Revision 1.5  2000/04/19 05:26:10  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.4  2000/04/11 07:10:40  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/17 01:03:16  dav
// Added some cocoon stuff, fixed some namespace stuff, etc
//
//
