

#include <Packages/Uintah/CCA/Components/Schedulers/SimpleScheduler.h>
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

static DebugStream dbg("SimpleScheduler", false);

SimpleScheduler::SimpleScheduler(const ProcessorGroup* myworld,
				 Output* oport)
   : SchedulerCommon(myworld, oport)
{
  d_generation = 0;
  if(!specialType)
     specialType = scinew TypeDescription(TypeDescription::ScatterGatherVariable,
				       "DataWarehouse::specialInternalScatterGatherType", false, -1);
  reloc_old_posLabel = reloc_new_posLabel = 0;
  reloc_matls = 0;
}

SimpleScheduler::~SimpleScheduler()
{
  if(reloc_matls && reloc_matls->removeReference())
    delete reloc_matls;
}

void SimpleScheduler::compile(const ProcessorGroup* pc)
{
  graph.topologicalSort(tasks);
}

void
SimpleScheduler::execute(const ProcessorGroup * pc)
{
  int ntasks = (int)tasks.size();
  if(ntasks == 0){
    cerr << "WARNING: Scheduler executed, but no tasks\n";
  }
  dbg << "Executing " << ntasks << " tasks\n";
  
  for(int i=0;i<ntasks;i++){
    double start = Time::currentSeconds();
    Task* task = tasks[i];
    const PatchSet* patchset = task->getPatchSet();
    const MaterialSet* matlset = task->getMaterialSet();
    if(patchset && matlset){
      for(int p=0;p<patchset->size();p++){
	const PatchSubset* patch_subset = patchset->getSubset(p);
	for(int m=0;m<matlset->size();m++){
	  const MaterialSubset* matl_subset = matlset->getSubset(m);
	  task->doit(pc, patch_subset, matl_subset, dw[0], dw[1]);
	}
      }
    } else {
      task->doit(pc, 0, 0, dw[0], dw[1]);
    }
    double dt = Time::currentSeconds()-start;
    dbg << "Completed task: " << tasks[i]->getName()
	<< " (" << dt << " seconds)\n";
  }

  dw[1]->finalize();
  finalizeNodes();
}

void
SimpleScheduler::scheduleParticleRelocation(const LevelP& level,
					    const VarLabel* old_posLabel,
					    const vector<vector<const VarLabel*> >& old_labels,
					    const VarLabel* new_posLabel,
					    const vector<vector<const VarLabel*> >& new_labels,
					    const MaterialSet* matls)
{
  const PatchSet* patches = level->allPatches();
  reloc_old_posLabel = old_posLabel;
  reloc_old_labels = old_labels;
  reloc_new_posLabel = new_posLabel;
  reloc_new_labels = new_labels;
  if(reloc_matls && reloc_matls->removeReference())
    delete reloc_matls;
  reloc_matls = matls;
  reloc_matls->addReference();
  ASSERTEQ(reloc_old_labels.size(), reloc_new_labels.size());
  int numMatls = reloc_old_labels.size();
  ASSERTEQ(numMatls, matls->size());
  for (int m = 0; m< numMatls; m++)
    ASSERTEQ(reloc_old_labels[m].size(), reloc_new_labels[m].size());

  Task* t = scinew Task("SimpleScheduler::relocateParticles",
			this, &SimpleScheduler::relocateParticles);
  t->requires(Task::NewDW, old_posLabel, Ghost::AroundCells, 1);

  for(int m=0;m < numMatls;m++){
    for(int i=0;i<(int)old_labels[m].size();i++){
      MaterialSubset* thismatl = scinew MaterialSubset();
      thismatl->add(m);
      t->requires( Task::NewDW, old_labels[m][i], thismatl, Ghost::AroundCells, 1);
    }
  }
  for(int m=0;m < numMatls;m++){
    t->computes(new_posLabel);
    for(int i=0;i<(int)new_labels[m].size();i++)
      t->computes(new_labels[m][i]);
  }

  addTask(t, patches, matls);
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
SimpleScheduler::relocateParticles(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw)
{
  cerr << "RELOCATE PARTICLES!!!!!!!!\n";
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();

    // Particles are only allowed to be one cell out
    IntVector l = patch->getCellLowIndex()-IntVector(1,1,1);
    IntVector h = patch->getCellHighIndex()+IntVector(1,1,1);
    Level::selectType neighbors;
    level->selectPatches(l, h, neighbors);

    vector<ScatterRecord*> sr(neighbors.size());
    for(int i=0;i<(int)sr.size();i++)
      sr[i]=0;
    int reloc_numMatls = reloc_new_labels.size();
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

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
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
    int reloc_numMatls = reloc_new_labels.size();
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
      cerr << "Creating particle set with " << totalParticles << " particles, on patch " << patch << ", material " << m << '\n';
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
}
