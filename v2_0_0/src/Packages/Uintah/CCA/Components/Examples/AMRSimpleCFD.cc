// TODO
// Don't compute error estimate on every timestep?
// Enable/disable flags for error estimate and output to avoid extra recompiles?
// Conservative advection...
// Coarsen after initialize???
// Is pressure coarse/fine right???
// Pressure strange with initial zvelocity=1
// Fill bctypes correctly for Coarse boundaries
// Set extraCells for level > 1 if necessary (instead of ups file)
// ExtraCells per variable instead of per level?
// Implement boundary-only variables
// Change applybc to work with coarse-fine boundaries
// refineInterface should not require old coarse if step == nsteps -1;
#include <Packages/Uintah/CCA/Components/Examples/AMRSimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;
using namespace std;
static DebugStream dbg("SimpleCFD", false);

template<class ArrayType>
static void print(const ArrayType& d, const char* name, int pre=0)
{
  cerr << name << ":\n";
  IntVector l(d.getLowIndex());
  IntVector h(d.getHighIndex());
  pre+=l.x()*13;
  for(int k=l.z();k<h.z();k++){
    cerr << "k=" << k << '\n';
    for(int j=h.y()-1;j>=l.y();j--){
      for(int i=0;i<pre;i++)
	cerr << ' ';
      for(int i=l.x();i<h.x();i++){
	cerr << setw(13) << d[IntVector(i,j,k)];
      }
      cerr << '\n';
    }
  }
}

AMRSimpleCFD::AMRSimpleCFD(const ProcessorGroup* myworld)
  : SimpleCFD(myworld)
{
}

AMRSimpleCFD::~AMRSimpleCFD()
{
}

void AMRSimpleCFD::problemSetup(const ProblemSpecP& params, GridP& grid,
				SimulationStateP& sharedState)
{
  SimpleCFD::problemSetup(params, grid, sharedState);
  ProblemSpecP cfd = params->findBlock("SimpleCFD");
  if(!cfd->get("err_density_grad", err_density_grad))
    err_density_grad=0;
  if(!cfd->get("err_temperature_grad", err_temperature_grad))
    err_temperature_grad=0;
  if(!cfd->get("err_pressure_grad", err_pressure_grad))
    err_pressure_grad=0;
  if(!cfd->get("err_vorticity_mag", err_vorticity_mag))
    err_vorticity_mag=0;
}

void AMRSimpleCFD::scheduleInitialize(const LevelP& level,
				      SchedulerP& sched)
{
  dbg << "AMRSimpleCFD::scheduleInitialize on level " << level->getIndex() << '\n';
  SimpleCFD::scheduleInitialize(level, sched);
  if(keep_pressure){
    Task* task = scinew Task("initialize",
			     this, &AMRSimpleCFD::initialize);
    task->computes(lb_->pressure2);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  }
}

void AMRSimpleCFD::initialize(const ProcessorGroup*,
			      const PatchSubset* patches, const MaterialSubset* matls,
			      DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      if(keep_pressure){
	CCVariable<double> pressure;
	new_dw->allocateAndPut(pressure, lb_->pressure2, matl, patch);
	pressure.initialize(0);
      }
    }
  }
}

void AMRSimpleCFD::scheduleRefineInterface(const LevelP& fineLevel,
				       SchedulerP& sched,
				       int step, int nsteps)
{
}

template<class ArrayType, class constArrayType>
void refineFaces(const Patch* patch, const Level* level,
		 const Level* coarseLevel, const IntVector& dir,
		 Patch::FaceType lowFace, Patch::FaceType highFace,
		 ArrayType& xvel, const VarLabel* label,
		 double factor, int matl, DataWarehouse* coarse_old_dw,
		 DataWarehouse* coarse_new_dw, Patch::VariableBasis basis)
{
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    if(patch->getBCType(face) != Patch::Coarse)
      continue;

    {

      IntVector l,h;
      patch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), l, h);
      if(face == highFace){
	l += dir;
	h += dir;
      } else if(face != lowFace){
	h += dir;
      }
      switch(face){
      case Patch::xminus:
      case Patch::xplus:
	l-=IntVector(0,1,1);
	h+=IntVector(0,1,1);
	break;
      case Patch::yminus:
      case Patch::yplus:
	l-=IntVector(1,0,1);
	h+=IntVector(1,0,1);
	break;
      case Patch::zminus:
      case Patch::zplus:
	l-=IntVector(1,1,0);
	h+=IntVector(1,1,0);
	break;
      default:
	break;
      }
      if(face != Patch::xminus && face != Patch::xplus && patch->getBCType(Patch::xminus) == Patch::None){
	l+=IntVector(1,0,0);
      }
      if(face != Patch::xminus && face != Patch::xplus && patch->getBCType(Patch::xplus) == Patch::None){
	h-=IntVector(1,0,0);
      }
      if(face != Patch::yminus && face != Patch::yplus && patch->getBCType(Patch::yminus) == Patch::None){
	l+=IntVector(0,1,0);
      }
      if(face != Patch::yminus && face != Patch::yplus && patch->getBCType(Patch::yplus) == Patch::None){
	h-=IntVector(0,1,0);
      }
      if(face != Patch::zminus && face != Patch::zplus && patch->getBCType(Patch::zminus) == Patch::None){
	l+=IntVector(0,0,1);
      }
      if(face != Patch::zminus && face != Patch::zplus && patch->getBCType(Patch::zplus) == Patch::None){
	h-=IntVector(0,0,1);
      }
      IntVector coarseLow = level->mapCellToCoarser(l);
      IntVector coarseHigh = level->mapCellToCoarser(h+level->getRefinementRatio()-IntVector(1,1,1));
      switch(face){
      case Patch::xminus:
	coarseHigh+=IntVector(1,0,0);
	break;
      case Patch::xplus:
	if(basis == Patch::XFaceBased)
	  coarseHigh += IntVector(1,0,0);
	else
	  coarseLow -= IntVector(1,0,0);
	break;
      case Patch::yminus:
	coarseHigh+=IntVector(0,1,0);
	break;
      case Patch::yplus:
	if(basis == Patch::YFaceBased)
	  coarseHigh += IntVector(0,1,0);
	else
	  coarseLow -= IntVector(0,1,0);
	break;
      case Patch::zminus:
	coarseHigh+=IntVector(0,0,1);
	break;
      case Patch::zplus:
	if(basis == Patch::ZFaceBased)
	  coarseHigh += IntVector(0,0,1);
	else
	  coarseLow -= IntVector(0,0,1);
	break;
      }
      l = Max(l, xvel.getLowIndex());
      h = Min(h, xvel.getHighIndex());
      if(factor < 1.e-10){
	constArrayType xvel0;
	coarse_old_dw->getRegion(xvel0, label, matl, coarseLevel,
				 coarseLow, coarseHigh);
	for(CellIterator iter(l,h); !iter.done(); iter++){
	  IntVector idx = *iter;

	  Vector w;
	  IntVector cidx = level->mapToCoarser(idx, dir, w);
	  if(cidx.x()+1 >= coarseHigh.x()){
	    cidx.x(cidx.x()-1);
	    w.x(1);
	  }
	  if(cidx.y()+1 >= coarseHigh.y()){
	    cidx.y(cidx.y()-1);
	    w.y(1);
	  }
	  if(cidx.z()+1 >= coarseHigh.z()){
	    cidx.z(cidx.z()-1);
	    w.z(1);
	  }
	  switch(face){
	  case Patch::xminus:
	    w.x(0);
	    break;
	  case Patch::xplus:
	    w.x(1);
	    break;
	  case Patch::yminus:
	    w.y(0);
	    break;
	  case Patch::yplus:
	    w.y(1);
	    break;
	  case Patch::zminus:
	    w.z(0);
	    break;
	  case Patch::zplus:
	    w.z(1);
	    break;
	  }
	  double x0 = xvel0[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
	    + xvel0[cidx+IntVector(1,0,0)]*w.x()*(1-w.y())*(1-w.z())
	    + xvel0[cidx+IntVector(0,1,0)]*(1-w.x())*w.y()*(1-w.z())
	    + xvel0[cidx+IntVector(1,1,0)]*w.x()*w.y()*(1-w.z())
	    + xvel0[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*w.z()
	    + xvel0[cidx+IntVector(1,0,1)]*w.x()*(1-w.y())*w.z()
	    + xvel0[cidx+IntVector(0,1,1)]*(1-w.x())*w.y()*w.z()
	    + xvel0[cidx+IntVector(1,1,1)]*w.x()*w.y()*w.z();
	  xvel[idx] = x0;
	}
      } else if(factor > 1-1.e-10){
	constArrayType xvel1;
	coarse_new_dw->getRegion(xvel1, label, matl, coarseLevel,
				 coarseLow, coarseHigh);
	for(CellIterator iter(l,h); !iter.done(); iter++){
	  IntVector idx = *iter;

	  Vector w;
	  IntVector cidx = level->mapToCoarser(idx, dir, w);
	  if(cidx.x()+1 >= coarseHigh.x()){
	    cidx.x(cidx.x()-1);
	    w.x(1);
	  }
	  if(cidx.y()+1 >= coarseHigh.y()){
	    cidx.y(cidx.y()-1);
	    w.y(1);
	  }
	  if(cidx.z()+1 >= coarseHigh.z()){
	    cidx.z(cidx.z()-1);
	    w.z(1);
	  }
	  switch(face){
	  case Patch::xminus:
	    w.x(0);
	    break;
	  case Patch::xplus:
	    w.x(1);
	    break;
	  case Patch::yminus:
	    w.y(0);
	    break;
	  case Patch::yplus:
	    w.y(1);
	    break;
	  case Patch::zminus:
	    w.z(0);
	    break;
	  case Patch::zplus:
	    w.z(1);
	    break;
	  }
	  double x1 = xvel1[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(1,0,0)]*w.x()*(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(0,1,0)]*(1-w.x())*w.y()*(1-w.z())
	    + xvel1[cidx+IntVector(1,1,0)]*w.x()*w.y()*(1-w.z())
	    + xvel1[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*w.z()
	    + xvel1[cidx+IntVector(1,0,1)]*w.x()*(1-w.y())*w.z()
	    + xvel1[cidx+IntVector(0,1,1)]*(1-w.x())*w.y()*w.z()
	    + xvel1[cidx+IntVector(1,1,1)]*w.x()*w.y()*w.z();
	  xvel[idx] = x1;
	}
      } else {
	constArrayType xvel0;
	coarse_old_dw->getRegion(xvel0, label, matl, coarseLevel,
				 coarseLow, coarseHigh);
	constArrayType xvel1;
	coarse_new_dw->getRegion(xvel1, label, matl, coarseLevel,
				 coarseLow, coarseHigh);

	for(CellIterator iter(l,h); !iter.done(); iter++){
	  IntVector idx = *iter;

	  Vector w;
	  IntVector cidx = level->mapToCoarser(idx, dir, w);
	  if(cidx.x()+1 >= coarseHigh.x()){
	    cidx.x(cidx.x()-1);
	    w.x(1);
	  }
	  if(cidx.y()+1 >= coarseHigh.y()){
	    cidx.y(cidx.y()-1);
	    w.y(1);
	  }
	  if(cidx.z()+1 >= coarseHigh.z()){
	    cidx.z(cidx.z()-1);
	    w.z(1);
	  }
	  switch(face){
	  case Patch::xminus:
	    w.x(0);
	    break;
	  case Patch::xplus:
	    w.x(1);
	    break;
	  case Patch::yminus:
	    w.y(0);
	    break;
	  case Patch::yplus:
	    w.y(1);
	    break;
	  case Patch::zminus:
	    w.z(0);
	    break;
	  case Patch::zplus:
	    w.z(1);
	    break;
	  }
	  double x0 = xvel0[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())	    + xvel0[cidx+IntVector(1,0,0)]*w.x()*(1-w.y())*(1-w.z())
	    + xvel0[cidx+IntVector(0,1,0)]*(1-w.x())*w.y()*(1-w.z())
	    + xvel0[cidx+IntVector(1,1,0)]*w.x()*w.y()*(1-w.z())
	    + xvel0[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*w.z()
	    + xvel0[cidx+IntVector(1,0,1)]*w.x()*(1-w.y())*w.z()
	    + xvel0[cidx+IntVector(0,1,1)]*(1-w.x())*w.y()*w.z()
	    + xvel0[cidx+IntVector(1,1,1)]*w.x()*w.y()*w.z();
	  double x1 = xvel1[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(1,0,0)]*w.x()*(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(0,1,0)]*(1-w.x())*w.y()*(1-w.z())
	    + xvel1[cidx+IntVector(1,1,0)]*w.x()*w.y()*(1-w.z())
	    + xvel1[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*w.z()
	    + xvel1[cidx+IntVector(1,0,1)]*w.x()*(1-w.y())*w.z()
	    + xvel1[cidx+IntVector(0,1,1)]*(1-w.x())*w.y()*w.z()
	    + xvel1[cidx+IntVector(1,1,1)]*w.x()*w.y()*w.z();
	  double x = (1-factor)*x0+factor*x1;
	  xvel[idx] = x;
	}
      }
    }
  }
}

void AMRSimpleCFD::addRefineDependencies(Task* task, const VarLabel* var,
					 int step, int nsteps)
{
  ASSERTRANGE(step, 0, nsteps+1);
  if(step != nsteps)
    task->requires(Task::CoarseOldDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain,
		   Ghost::AroundCells, 1);
  if(step != 0)
    task->requires(Task::CoarseNewDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain,
		   Ghost::AroundCells, 1);
}

void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				    CCVariable<double>& val,
				    DataWarehouse* new_dw,
				    const VarLabel* label,
				    int matl,
				    double factor)
{
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  refineFaces<CCVariable<double>, constCCVariable<double> >
    (patch, level, coarseLevel, IntVector(0,0,0), Patch::invalidFace,
     Patch::invalidFace, val, label, factor, matl,
     coarse_old_dw, coarse_new_dw, Patch::CellBased);
}

void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				    SFCXVariable<double>& val,
				    DataWarehouse* new_dw,
				    const VarLabel* label,
				    int matl,
				    double factor)
{
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  refineFaces<SFCXVariable<double>, constSFCXVariable<double> >
    (patch, level, coarseLevel, IntVector(1,0,0), Patch::xminus,
     Patch::xplus, val, label, factor, matl,
     coarse_old_dw, coarse_new_dw, Patch::XFaceBased);
}

void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				    SFCYVariable<double>& val,
				    DataWarehouse* new_dw,
				    const VarLabel* label,
				    int matl,
				    double factor)
{
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  refineFaces<SFCYVariable<double>, constSFCYVariable<double> >
    (patch, level, coarseLevel, IntVector(0,1,0), Patch::yminus,
     Patch::yplus, val, label, factor, matl,
     coarse_old_dw, coarse_new_dw, Patch::YFaceBased);
}

void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				    SFCZVariable<double>& val,
				    DataWarehouse* new_dw,
				    const VarLabel* label,
				    int matl,
				    double factor)
{
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  refineFaces<SFCZVariable<double>, constSFCZVariable<double> >
    (patch, level, coarseLevel, IntVector(0,0,1), Patch::zminus,
     Patch::zplus, val, label, factor, matl,
     coarse_old_dw, coarse_new_dw, Patch::ZFaceBased);
}

void AMRSimpleCFD::scheduleCoarsen(const LevelP& coarseLevel,
				   SchedulerP& sched)
{
  dbg << "AMRSimpleCFD::scheduleCoarsen on level " << coarseLevel->getIndex() << '\n';
  Task* task = scinew Task("coarsen",
			   this, &AMRSimpleCFD::coarsen);
  task->requires(Task::NewDW, lb_->density,
		 0, Task::FineLevel, 0, Task::NormalDomain,
		 Ghost::None, 0);
  task->modifies(lb_->density);

  if(do_thermal){
    task->requires(Task::NewDW, lb_->temperature,
		   0, Task::FineLevel, 0, Task::NormalDomain,
		   Ghost::None, 0);
    task->modifies(lb_->temperature);
  }
  task->requires(Task::NewDW, lb_->xvelocity,
		 0, Task::FineLevel, 0, Task::NormalDomain,
		 Ghost::None, 0);
  task->modifies(lb_->xvelocity);

  task->requires(Task::NewDW, lb_->yvelocity,
		 0, Task::FineLevel, 0, Task::NormalDomain,
		 Ghost::None, 0);
  task->modifies(lb_->yvelocity);

  task->requires(Task::NewDW, lb_->zvelocity,
		 0, Task::FineLevel, 0, Task::NormalDomain,
		 Ghost::None, 0);
  task->modifies(lb_->zvelocity);
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());

  // Re-solve/apply the pressure, using the pressure2 variable
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver)
    throw InternalError("SimpleCFD needs a solver component to work");
  schedulePressureSolve(coarseLevel, sched, solver, lb_->pressure2,
			lb_->pressure2_matrix, lb_->pressure2_rhs);
  releasePort("solver");
}

void AMRSimpleCFD::scheduleErrorEstimate(const LevelP& coarseLevel,
					 SchedulerP& sched)
{
  // Estimate error - this should probably be in it's own schedule,
  // and the simulation controller should not schedule it every time step
  Task* task = scinew Task("errorEstimate", this, &AMRSimpleCFD::errorEstimate);
  task->requires(Task::NewDW, lb_->density, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->temperature, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->pressure, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->density, Ghost::AroundCells, 1);
  task->modifies(sharedState_->get_refineFlag_label());
  task->computes(lb_->density_gradient_mag);
  task->computes(lb_->temperature_gradient_mag);
  task->computes(lb_->pressure_gradient_mag);
  task->computes(lb_->ccvorticitymag);
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}

void AMRSimpleCFD::coarsen(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw)
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  IntVector rr(fineLevel->getRefinementRatio());
  double ratio = 1./(rr.x()*rr.y()*rr.z());
  for(int p=0;p<patches->size();p++){
    const Patch* coarsePatch = patches->get(p);
    // Find the overlapping regions...
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> density;
      new_dw->getModifiable(density, lb_->density, matl, coarsePatch);
      //print(density, "before coarsen density");
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
	constCCVariable<double> fine_den;
	new_dw->get(fine_den, lb_->density, matl, finePatch,
		    Ghost::None, 0);
	IntVector fl(finePatch->getCellLowIndex());
	IntVector fh(finePatch->getCellHighIndex());
	IntVector l(fineLevel->mapCellToCoarser(fl));
	IntVector h(fineLevel->mapCellToCoarser(fh));
	l = Max(l, coarsePatch->getCellLowIndex());
	h = Min(h, coarsePatch->getCellHighIndex());
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  double d=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
	      !inside.done(); inside++){
	    d+=fine_den[fineStart+*inside];
	  }
	  density[*iter]=d*ratio;
	}
      }

      //print(density, "coarsened density");
      if(do_thermal){
	CCVariable<double> temp;
	new_dw->getModifiable(temp, lb_->temperature, matl, coarsePatch);
	for(int i=0;i<finePatches.size();i++){
	  const Patch* finePatch = finePatches[i];
	  constCCVariable<double> fine_temp;
	  new_dw->get(fine_temp, lb_->temperature, matl, finePatch,
		      Ghost::None, 0);
	  IntVector fl(finePatch->getCellLowIndex());
	  IntVector fh(finePatch->getCellHighIndex());
	  IntVector l(fineLevel->mapCellToCoarser(fl));
	  IntVector h(fineLevel->mapCellToCoarser(fh));
	  l = Max(l, coarsePatch->getCellLowIndex());
	  h = Min(h, coarsePatch->getCellHighIndex());
	  for(CellIterator iter(l, h); !iter.done(); iter++){
	    double d=0;
	    IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	    for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
		!inside.done(); inside++){
	      d+=fine_temp[fineStart+*inside];
	    }
	    temp[*iter]=d*ratio;
	  }
	}
	//print(temp, "coarsened temperature");
      }
      SFCXVariable<double> xvel;
      new_dw->getModifiable(xvel, lb_->xvelocity, matl, coarsePatch);
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
	SFCXVariable<double> fine_xvel;
	new_dw->getCopy(fine_xvel, lb_->xvelocity, matl, finePatch,
			Ghost::AroundFacesX, 1);
	refineFaces<SFCXVariable<double>, constSFCXVariable<double> >
	  (finePatch, fineLevel, coarseLevel, IntVector(1,0,0),
	   Patch::xminus, Patch::xplus, fine_xvel,
	   lb_->xvelocity, 1.0, matl, old_dw, new_dw, Patch::XFaceBased);
	IntVector fl(finePatch->getSFCXLowIndex());
	IntVector fh(finePatch->getSFCXHighIndex());
	IntVector l(fineLevel->mapCellToCoarser(fl));
	IntVector h(fineLevel->mapCellToCoarser(fh+IntVector(rr.x()-1, 0, 0)));
	l = Max(l, coarsePatch->getSFCXLowIndex());
	h = Min(h, coarsePatch->getSFCXHighIndex());
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  double d=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(-1,0,0), IntVector(0,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    d+=fine_xvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(1,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    d+=fine_xvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(1,0,0), IntVector(2,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    d+=fine_xvel[fineStart+*inside]*0.5;
	  }
	  xvel[*iter]=d*ratio;
	}
      }
      //print(xvel, "coarsened xvel");
      SFCYVariable<double> yvel;
      new_dw->getModifiable(yvel, lb_->yvelocity, matl, coarsePatch);
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
	SFCYVariable<double> fine_yvel;
	new_dw->getCopy(fine_yvel, lb_->yvelocity, matl, finePatch,
			Ghost::AroundFacesY, 1);
	refineFaces<SFCYVariable<double>, constSFCYVariable<double> >
	  (finePatch, fineLevel, coarseLevel, IntVector(0,1,0),
	   Patch::yminus, Patch::yplus, fine_yvel,
	   lb_->yvelocity, 1.0, matl, old_dw, new_dw, Patch::YFaceBased);
	IntVector fl(finePatch->getSFCYLowIndex());
	IntVector fh(finePatch->getSFCYHighIndex());
	IntVector l(fineLevel->mapCellToCoarser(fl));
	IntVector h(fineLevel->mapCellToCoarser(fh+IntVector(0, rr.y()-1, 0)));
	l = Max(l, coarsePatch->getSFCYLowIndex());
	h = Min(h, coarsePatch->getSFCYHighIndex());
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  double d=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(0,-1,0), IntVector(rr.x(),0,rr.z()));
	      !inside.done(); inside++){
	    d+=fine_yvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(rr.x(),1,rr.z()));
	      !inside.done(); inside++){
	    d+=fine_yvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(0,1,0), IntVector(rr.x(),2,rr.z()));
	      !inside.done(); inside++){
	    d+=fine_yvel[fineStart+*inside]*0.5;
	  }
	  yvel[*iter]=d*ratio;
	}
      }
      //print(yvel, "coarsened yvel");
      SFCZVariable<double> zvel;
      new_dw->getModifiable(zvel, lb_->zvelocity, matl, coarsePatch);
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
	SFCZVariable<double> fine_zvel;
	new_dw->getCopy(fine_zvel, lb_->zvelocity, matl, finePatch,
			Ghost::AroundFacesZ, 1);
	refineFaces<SFCZVariable<double>, constSFCZVariable<double> >
	  (finePatch, fineLevel, coarseLevel, IntVector(0,0,1),
	   Patch::zminus, Patch::zplus, fine_zvel,
	   lb_->zvelocity, 1.0, matl, old_dw, new_dw, Patch::ZFaceBased);
	IntVector fl(finePatch->getSFCZLowIndex());
	IntVector fh(finePatch->getSFCZHighIndex());
	IntVector l(fineLevel->mapCellToCoarser(fl));
	IntVector h(fineLevel->mapCellToCoarser(fh+IntVector(0, 0, rr.z()-1)));
	l = Max(l, coarsePatch->getSFCZLowIndex());
	h = Min(h, coarsePatch->getSFCZHighIndex());
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  double d=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(0,0,-1), IntVector(rr.x(),rr.y(),0));
	      !inside.done(); inside++){
	    d+=fine_zvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(rr.x(),rr.y(),1));
	      !inside.done(); inside++){
	    d+=fine_zvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(0,0,1), IntVector(rr.x(),rr.y(),2));
	      !inside.done(); inside++){
	    d+=fine_zvel[fineStart+*inside]*0.5;
	  }
	  zvel[*iter]=d*ratio;
	}
      }
      //print(zvel, "coarsened zvel");      
    }
  }
}

void AMRSimpleCFD::errorEstimate(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse*,
				 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      Vector dx(patch->dCell());
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());

      CCVariable<int> refineFlag;
      new_dw->getModifiable(refineFlag, sharedState_->get_refineFlag_label(),
			    matl, patch);

      IntVector l(patch->getCellLowIndex());
      IntVector h(patch->getCellHighIndex());

      constCCVariable<double> density;
      new_dw->get(density, lb_->density, matl, patch, Ghost::AroundCells, 1);
      CCVariable<double> density_gradient_mag;
      new_dw->allocateAndPut(density_gradient_mag, lb_->density_gradient_mag,
			     matl, patch);
      double err_density_grad_inv = 1./err_density_grad;
      for(CellIterator iter(l, h); !iter.done(); iter++){
	IntVector idx(*iter);
	double gx, gy, gz;
	if(idx.x() == l.x()){
	  gx = (density[idx+IntVector(1,0,0)]-density[idx])*inv_dx.x();
	} else if(idx.x() == h.x()-1){
	  gx = (density[idx]-density[idx-IntVector(1,0,0)])*inv_dx.x();
	} else {
	  gx = (density[idx+IntVector(1,0,0)]-density[idx-IntVector(1,0,0)])*0.5*inv_dx.x();
	}
	if(idx.y() == l.y()){
	  gy = (density[idx+IntVector(0,1,0)]-density[idx])*inv_dx.y();
	} else if(idx.y() == h.y()-1){
	  gy = (density[idx]-density[idx-IntVector(0,1,0)])*inv_dx.y();
	} else {
	  gy = (density[idx+IntVector(0,1,0)]-density[idx-IntVector(0,1,0)])*0.5*inv_dx.y();
	}
	if(idx.z() == l.z()){
	  gz = (density[idx+IntVector(0,0,1)]-density[idx])*inv_dx.z();
	} else if(idx.z() == h.z()-1){
	  gz = (density[idx]-density[idx-IntVector(0,0,1)])*inv_dx.z();
	} else {
	  gz = (density[idx+IntVector(0,0,1)]-density[idx-IntVector(0,0,1)])*0.5*inv_dx.z();
	}
	Vector grad(gx, gy, gz);
	density_gradient_mag[idx]=grad.length();
	if(density_gradient_mag[idx] > err_density_grad)
	  refineFlag[idx]=true;
	density_gradient_mag[idx] *= err_density_grad_inv;
      }

      // Temperature gradient
      if(err_temperature_grad > 0){
	CCVariable<double> temperature_gradient_mag;
	new_dw->allocateAndPut(temperature_gradient_mag,
			       lb_->temperature_gradient_mag,
			       matl, patch);
	constCCVariable<double> temperature;
	new_dw->get(temperature, lb_->temperature, matl, patch, Ghost::AroundCells, 1);
	double inv_err_temperature_grad = 1./err_temperature_grad;
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  IntVector idx(*iter);
	  double gx, gy, gz;
	  if(idx.x() == l.x()){
	    gx = (temperature[idx+IntVector(1,0,0)]-temperature[idx])*inv_dx.x();
	  } else if(idx.x() == h.x()-1){
	    gx = (temperature[idx]-temperature[idx-IntVector(1,0,0)])*inv_dx.x();
	  } else {
	    gx = (temperature[idx+IntVector(1,0,0)]-temperature[idx-IntVector(1,0,0)])*0.5*inv_dx.x();
	  }
	  if(idx.y() == l.y()){
	    gy = (temperature[idx+IntVector(0,1,0)]-temperature[idx])*inv_dx.y();
	  } else if(idx.y() == h.y()-1){
	    gy = (temperature[idx]-temperature[idx-IntVector(0,1,0)])*inv_dx.y();
	  } else {
	    gy = (temperature[idx+IntVector(0,1,0)]-temperature[idx-IntVector(0,1,0)])*0.5*inv_dx.y();
	  }
	  if(idx.z() == l.z()){
	    gz = (temperature[idx+IntVector(0,0,1)]-temperature[idx])*inv_dx.z();
	  } else if(idx.z() == h.z()-1){
	    gz = (temperature[idx]-temperature[idx-IntVector(0,0,1)])*inv_dx.z();
	  } else {
	    gz = (temperature[idx+IntVector(0,0,1)]-temperature[idx-IntVector(0,0,1)])*0.5*inv_dx.z();
	  }
	  Vector grad(gx, gy, gz);
	  temperature_gradient_mag[idx]=grad.length();
	  if(temperature_gradient_mag[idx] > err_temperature_grad)
	    refineFlag[idx]=true;
	  temperature_gradient_mag[idx] *= inv_err_temperature_grad;
	}
      }

      if(err_pressure_grad > 0){
	// Pressure gradient
	CCVariable<double> pressure_gradient_mag;
	new_dw->allocateAndPut(pressure_gradient_mag, lb_->pressure_gradient_mag,
			       matl, patch);
	constCCVariable<double> pressure;
	new_dw->get(pressure, lb_->pressure, matl, patch, Ghost::AroundCells, 1);
	double inv_err_pressure_grad = 1./err_pressure_grad;
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  IntVector idx(*iter);
	  double gx, gy, gz;
	  if(idx.x() == l.x()){
	    gx = (pressure[idx+IntVector(1,0,0)]-pressure[idx])*inv_dx.x();
	  } else if(idx.x() == h.x()-1){
	    gx = (pressure[idx]-pressure[idx-IntVector(1,0,0)])*inv_dx.x();
	  } else {
	    gx = (pressure[idx+IntVector(1,0,0)]-pressure[idx-IntVector(1,0,0)])*0.5*inv_dx.x();
	  }
	  if(idx.y() == l.y()){
	    gy = (pressure[idx+IntVector(0,1,0)]-pressure[idx])*inv_dx.y();
	  } else if(idx.y() == h.y()-1){
	    gy = (pressure[idx]-pressure[idx-IntVector(0,1,0)])*inv_dx.y();
	  } else {
	    gy = (pressure[idx+IntVector(0,1,0)]-pressure[idx-IntVector(0,1,0)])*0.5*inv_dx.y();
	  }
	  if(idx.z() == l.z()){
	    gz = (pressure[idx+IntVector(0,0,1)]-pressure[idx])*inv_dx.z();
	  } else if(idx.z() == h.z()-1){
	    gz = (pressure[idx]-pressure[idx-IntVector(0,0,1)])*inv_dx.z();
	  } else {
	    gz = (pressure[idx+IntVector(0,0,1)]-pressure[idx-IntVector(0,0,1)])*0.5*inv_dx.z();
	  }
	  Vector grad(gx, gy, gz);
	  pressure_gradient_mag[idx]=grad.length();
	  if(pressure_gradient_mag[idx] > err_pressure_grad)
	    refineFlag[idx]=true;
	  pressure_gradient_mag[idx] *= inv_err_pressure_grad;
	}
      }

      if(err_vorticity_mag > 0){
	// Vorticity
	CCVariable<double> ccvorticitymag;
	new_dw->allocateAndPut(ccvorticitymag, lb_->ccvorticitymag,
			       matl, patch);
	constCCVariable<Vector> vel;
	new_dw->get(vel, lb_->ccvelocity, matl, patch, Ghost::AroundCells, 1);
	double inv_err_vorticity_mag = 1./err_vorticity_mag;
	for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  Vector gx, gy, gz;
	  if(idx.x() == l.x()){
	    gx = (vel[idx+IntVector(1,0,0)]-vel[idx])*inv_dx.x();
	  } else if(idx.x() == h.x()-1){
	    gx = (vel[idx]-vel[idx-IntVector(1,0,0)])*inv_dx.x();
	  } else {
	    gx = (vel[idx+IntVector(1,0,0)]-vel[idx-IntVector(1,0,0)])*0.5*inv_dx.x();
	  }
	  if(idx.y() == l.y()){
	    gy = (vel[idx+IntVector(0,1,0)]-vel[idx])*inv_dx.y();
	  } else if(idx.y() == h.y()-1){
	    gy = (vel[idx]-vel[idx-IntVector(0,1,0)])*inv_dx.y();
	  } else {
	    gy = (vel[idx+IntVector(0,1,0)]-vel[idx-IntVector(0,1,0)])*0.5*inv_dx.y();
	  }
	  if(idx.z() == l.z()){
	    gz = (vel[idx+IntVector(0,0,1)]-vel[idx])*inv_dx.z();
	  } else if(idx.z() == h.z()-1){
	    gz = (vel[idx]-vel[idx-IntVector(0,0,1)])*inv_dx.z();
	  } else {
	    gz = (vel[idx+IntVector(0,0,1)]-vel[idx-IntVector(0,0,1)])*0.5*inv_dx.z();
	  }
	  Vector w(gy.z()-gz.y(), gz.x()-gx.z(), gx.y()-gy.x());
	  ccvorticitymag[idx]=w.length();
	  if(ccvorticitymag[idx] > err_vorticity_mag)
	    refineFlag[idx]=true;
	  ccvorticitymag[idx] *= inv_err_vorticity_mag;
	}
      }
    }
  }
}
