

#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/ScatterGatherBase.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static const TypeDescription* specialType;

static DebugStream dbg("SingleProcessorScheduler", false);

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

   makeTaskGraphDoc(tasks);

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
	 for(int i=0;i<(int)old_labels[m].size();i++)
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
      Level::selectType neighbors;
      level->selectPatches(l, h, neighbors);
      for(int i=0;i<(int)neighbors.size();i++)
	 t2->requires(new_dw, scatterGatherVariable, 0, neighbors[i], Ghost::None);
      for(int m=0;m < numMatls;m++){
	 t2->computes( new_dw, new_posLabel, m, patch);
	 for(int i=0;i<(int)new_labels[m].size();i++)
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
   Level::selectType neighbors;
   level->selectPatches(l, h, neighbors);

   vector<ScatterRecord*> sr(neighbors.size());
   for(int i=0;i<(int)sr.size();i++)
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
	    for(i=0;i<(int)neighbors.size();i++){
	       if(neighbors[i]->getBox().contains(px[idx])){
		  break;
	       }
	    }
	    if(i == (int)neighbors.size()){
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
		  for(int v=0;v<(int)reloc_old_labels[m].size();v++)
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
   for(int i=0;i<(int)sr.size();i++){
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
   Level::selectType neighbors;
   level->selectPatches(l, h, neighbors);

   vector<ScatterRecord*> sr;
   for(int i=0;i<(int)neighbors.size();i++){
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
      for(int i=0;i<(int)sr.size();i++){
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

      for(int v=0;v<(int)reloc_old_labels[m].size();v++){
	 vector<ParticleVariableBase*> gathervars;
	 ParticleVariableBase* var = new_dw->getParticleVariable(reloc_old_labels[m][v], pset);

	 gathervars.push_back(var);
	 for(int i=0;i<(int)sr.size();i++){
	    if(sr[i]->matls[m])
	       gathervars.push_back(sr[i]->matls[m]->vars[v+1]);
	 }
	 ParticleVariableBase* newvar = var->clone();
	 newvar->gather(newsubset, subsets, gathervars);
	 new_dw->put(*newvar, reloc_new_labels[m][v]);
	 delete newvar;
      }
      // Delete keepset and all relocsets
      for(int i=0;i<(int)subsets.size();i++)
	 delete subsets[i];
   }
   for(int i=0;i<(int)sr.size();i++){
      for(int m=0;m<reloc_numMatls;m++)
	 if(sr[i]->matls[m])
	    delete sr[i]->matls[m];
      delete sr[i];
   }
}

LoadBalancer*
SingleProcessorScheduler::getLoadBalancer()
{
   UintahParallelPort* lbp = getPort("load balancer");
   LoadBalancer* lb = dynamic_cast<LoadBalancer*>(lbp);
   return lb;
}

void
SingleProcessorScheduler::releaseLoadBalancer()
{
   releasePort("load balancer");
} // End namespace Uintah
