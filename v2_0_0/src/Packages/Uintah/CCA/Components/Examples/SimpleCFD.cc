// TODO
// make bcs be seperate per face/cell
// overlapping nodes in corners
// Do something about saving pressure2 on bottom level
// Why doesn't it complain if initial temperature not specified?
// Replace pressure pin with regular pressure BC
// Random initial velocities messed??? (pattern)
// Max timestep increase/decrease
// Vorticity confinement
// Periodic boundaries
// Periodic version of Rayleigh-Taylor
// Don't compute CC velocities if not saving and not needed otherwise
// Turn vorticity on/off too!
// - buoyancy - should we have an ambient temperature that changes?
// - Turbulence model?
// - velcor on ICs/internal boundaries?
// - 3D
// - function parser for ics/bcs
// Correct face bcs for advection...
// If viscosity or diffusion are zero, use an alternate method to apply sources, avoiding solve
// More efficient multi-scalar advection
// Look at neighbor stuff for projectVelocity
#include <Packages/Uintah/CCA/Components/Examples/SimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <values.h>
#include <iomanip>

using namespace Uintah;
using namespace std;

static DebugStream dbg("SimpleCFD", false);

#if defined(__sgi) && !defined(__GNUC__)
#  define BREAK 
#else
#  define BREAK break
#endif

#if 1
template<class ArrayType>
static void print(const ArrayType& d, const char* name, int pre=0)
{
  IntVector l(d.getLowIndex());
  IntVector h(d.getHighIndex());
  cerr << name << ": " << l << "-" << h << "\n";
  pre+=l.x()*13;
  for(int k=h.z()-1;k>=l.z();k--){
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
#endif

#if 1
template<class ArrayType>
static void printhex(const ArrayType& d, const char* name, int pre=0)
{
  IntVector l(d.getLowIndex());
  IntVector h(d.getHighIndex());
  cerr << name << ": " << l << "-" << h << "\n";
  pre+=l.x()*13;
  for(int k=h.z()-1;k>=l.z();k--){
    cerr << "k=" << k << '\n';
    for(int j=h.y()-1;j>=l.y();j--){
      for(int i=0;i<pre;i++)
	cerr << ' ';
      for(int i=l.x();i<h.x();i++){
	cerr << setw(13) << setbase(16) << d[IntVector(i,j,k)];
      }
      cerr << '\n';
    }
  }
  cerr << setbase(10);
}
#endif

#if 0
static void printMat(const Array3<Stencil7>& A, const char* name)
{
  IntVector l(A.getLowIndex());
  IntVector h(A.getHighIndex());
  Array3<double> tmp(l,h);
  for(CellIterator iter(l,h); !iter.done(); iter++)
    tmp[*iter]=A[*iter].p;
  cerr << "Ap ";
  print(tmp, name,6);
  for(CellIterator iter(l,h); !iter.done(); iter++)
    tmp[*iter]=A[*iter].n;
  cerr << "An ";
  print(tmp, name,6);
  for(CellIterator iter(l,h); !iter.done(); iter++)
    tmp[*iter]=A[*iter].s;
  cerr << "As ";
  print(tmp,  name,6);
  for(CellIterator iter(l,h); !iter.done(); iter++)
    tmp[*iter]=A[*iter].w;
  cerr << "Aw ";
  print(tmp, name,0);
  for(CellIterator iter(l,h); !iter.done(); iter++)
    tmp[*iter]=A[*iter].e;
  cerr << "Ae ";
  print(tmp, name,13);
}
#endif

#if 1
static void print(const Array3<double>& xvel, const Array3<double>& yvel,
		  const Array3<double>& zvel, const char* name)
{
  cerr << "X ";
  print(xvel, name);
  cerr << "Y ";
  print(yvel, name, 6);
  cerr << "Z ";
  print(zvel, name, 6);
}
#endif

SimpleCFD::SimpleCFD(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
  diffusion_params_ = 0;
  viscosity_params_ = 0;
  pressure_params_ = 0;
  keep_pressure=true; // True converges faster, false uses less memory
  old_initial_guess=true; // True converges faster, false uses less memory
  do_thermal=false;
}

SimpleCFD::~SimpleCFD()
{
  delete lb_;
  delete diffusion_params_;
  delete viscosity_params_;
  delete pressure_params_;
}

void SimpleCFD::problemSetup(const ProblemSpecP& params, GridP& grid,
			 SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP cfd = params->findBlock("SimpleCFD");
  cfd->require("delt_multiplier", delt_multiplier_);
  cfd->require("viscosity", viscosity_);
  cfd->require("diffusion", density_diffusion_);
  cfd->require("dissipation", density_dissipation_);
  cfd->require("maxadvect", maxadvect_);
  cfd->require("pressure_pin", pressure_pin_);
  cfd->get("keep_pressure", keep_pressure);
  cfd->get("old_initial_guess", old_initial_guess);
  cfd->get("do_thermal", do_thermal);
  if(!grid->getLevel(0)->containsPoint(pressure_pin_))
    throw ProblemSetupException("velocity pressure pin point is not with the domain");

  cfd->require("advection_tolerance", advection_tolerance_);
  cfd->require("buoyancy", buoyancy_);
  cfd->require("thermal_conduction", thermal_conduction_);
  if(!cfd->get("vorticity_confinement_scale", vorticity_confinement_scale_))
    vorticity_confinement_scale_=0;
  if(!cfd->get("random_initial_velocities", random_initial_velocities))
    random_initial_velocities=Vector(0,0,0);
  if(buoyancy_ > 0 || thermal_conduction_ > 0)
    do_thermal=true;
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);

  regiondb.problemSetup(cfd, grid);
  ics.setupCondition<double>("density", IntVector(0,0,0));
  if(do_thermal)
    ics.setupCondition<double>("temperature", IntVector(0,0,0));
  ics.setupCondition<double>("xvelocity", IntVector(1,0,0));
  ics.setupCondition<double>("yvelocity", IntVector(0,1,0));
  ics.setupCondition<double>("zvelocity", IntVector(0,0,1));
  ics.problemSetup(cfd, regiondb);

  bcs.setupCondition<double>("density", IntVector(0,0,0));
  if(do_thermal)
    bcs.setupCondition<double>("temperature", IntVector(0,0,0));
  bcs.setupCondition<double>("xvelocity", IntVector(1,0,0));
  bcs.setupCondition<double>("yvelocity", IntVector(0,1,0));
  bcs.setupCondition<double>("zvelocity", IntVector(0,0,1));
  bcs.setupCondition<double>("pressure", IntVector(0,0,0));
  bcs.problemSetup(cfd, regiondb);

  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver)
    throw InternalError("SimpleCFD needs a solver component to work");

  ProblemSpecP solverinfo = cfd->findBlock("Solver");
  diffusion_params_ = solver->readParameters(solverinfo, "diffusion");
  viscosity_params_ = solver->readParameters(solverinfo, "viscosity");
  pressure_params_ = solver->readParameters(solverinfo, "pressure");
  conduction_params_ = solver->readParameters(solverinfo, "conduction");

  releasePort("solver");
}
 
void SimpleCFD::scheduleInitialize(const LevelP& level,
			       SchedulerP& sched)
{
  dbg << "SimpleCFD::scheduleInitialize on level " << level->getIndex() << '\n';
  Task* task = scinew Task("initialize",
			   this, &SimpleCFD::initialize);
  task->computes(lb_->bctype);
  task->computes(lb_->density);
  task->computes(lb_->xvelocity);
  task->computes(lb_->yvelocity);
  task->computes(lb_->zvelocity);
  if(keep_pressure)
    task->computes(lb_->pressure);
  if(do_thermal)
    task->computes(lb_->temperature);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
#if 0
  task = scinew Task("hackbcs", this, &SimpleCFD::hackbcs);
  sched->addTask(task, level->allPatches(), sharedState_->allMaterials());
#endif
  task = scinew Task("interpolateVelocities", this, &SimpleCFD::interpolateVelocities);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::AroundCells, 1);
  task->computes(lb_->ccvelocity);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}
 
void SimpleCFD::scheduleComputeStableTimestep(const LevelP& level,
					      SchedulerP& sched)
{
  dbg << "SimpleCFD::scheduleComputeStableTimestep on level " << level->getIndex() << '\n';
  Task* task = scinew Task("computeStableTimestep",
			   this, &SimpleCFD::computeStableTimestep);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::None, 0);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void
SimpleCFD::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched,
				int step, int nsteps )
{
  dbg << "SimpleCFD::scheduleTimeAdvance on level " << level->getIndex() << '\n';
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver)
    throw InternalError("SimpleCFD needs a solver component to work");

  Task* task;
  task = scinew Task("advectVelocity", this, &SimpleCFD::advectVelocity,
		     step, nsteps);
#if 0
  task->requires(Task::OldDW, sharedState_->get_delt_label());
#endif
  task->requires(Task::OldDW, lb_->bctype, Ghost::AroundNodes, 2);
  task->requires(Task::OldDW, lb_->xvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::OldDW, lb_->yvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::OldDW, lb_->zvelocity, Ghost::AroundCells, maxadvect_+1);
  task->computes(lb_->xvelocity);
  task->computes(lb_->yvelocity);
  task->computes(lb_->zvelocity);
  if(level->getIndex()>0){
    addRefineDependencies(task, lb_->xvelocity, step, nsteps);
    addRefineDependencies(task, lb_->yvelocity, step, nsteps);
    addRefineDependencies(task, lb_->zvelocity, step, nsteps);
  }
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  task = scinew Task("applyForces", this, &SimpleCFD::applyForces);
  if(sharedState_->getGravity().length() > 0 || buoyancy_ > 0
     || vorticity_confinement_scale_ != 0){
#if 0
    task->requires(Task::OldDW, sharedState_->get_delt_label());
#endif
    task->requires(Task::OldDW, lb_->density, Ghost::AroundFaces, 1);
    task->modifies(lb_->xvelocity);
    task->modifies(lb_->yvelocity);
    task->modifies(lb_->zvelocity);
    if(do_thermal)
      task->requires(Task::OldDW, lb_->temperature, Ghost::AroundFaces, 1);
    if(vorticity_confinement_scale_ != 0){
      task->requires(Task::OldDW, lb_->ccvelocity, Ghost::None, 0);
      task->computes(lb_->ccvorticity);
      task->computes(lb_->ccvorticitymag);
      task->computes(lb_->vcforce);
      task->computes(lb_->NN);
    }
  }
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  for(int dir=0;dir<3;dir++){
    task = scinew Task("applyViscosity", this, &SimpleCFD::applyViscosity, dir,
		       step, nsteps);
#if 0
    task->requires(Task::OldDW, sharedState_->get_delt_label());
#endif
    task->requires(Task::OldDW, lb_->bctype, Ghost::AroundFaces, 1);
    if(dir == 0){
      task->requires(Task::NewDW, lb_->xvelocity, Ghost::None, 0);
      task->computes(lb_->xvelocity_matrix);
      task->computes(lb_->xvelocity_rhs);
      if(level->getIndex()>0)
	addRefineDependencies(task, lb_->xvelocity, step+1, nsteps);
    } else if(dir == 1){
      task->requires(Task::NewDW, lb_->yvelocity, Ghost::None, 0);
      task->computes(lb_->yvelocity_matrix);
      task->computes(lb_->yvelocity_rhs);
      if(level->getIndex()>0)
	addRefineDependencies(task, lb_->yvelocity, step+1, nsteps);
    } else if(dir ==2){
      task->requires(Task::NewDW, lb_->zvelocity, Ghost::None, 0);
      task->computes(lb_->zvelocity_matrix);
      task->computes(lb_->zvelocity_rhs);
      if(level->getIndex()>0)
	addRefineDependencies(task, lb_->zvelocity, step+1, nsteps);
    }
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
    if(dir == 0){
      solver->scheduleSolve(level, sched, sharedState_->allMaterials(),
			    lb_->xvelocity_matrix, lb_->xvelocity, true,
			    lb_->xvelocity_rhs, lb_->xvelocity,
			    old_initial_guess?Task::OldDW:Task::NewDW,
			    viscosity_params_);
    } else if(dir == 1){
      solver->scheduleSolve(level, sched, sharedState_->allMaterials(),
			    lb_->yvelocity_matrix, lb_->yvelocity, true,
			    lb_->yvelocity_rhs, lb_->yvelocity,
			    old_initial_guess?Task::OldDW:Task::NewDW,
			    viscosity_params_);
    } else if(dir == 2){
      solver->scheduleSolve(level, sched, sharedState_->allMaterials(),
			    lb_->zvelocity_matrix, lb_->zvelocity, true,
			    lb_->zvelocity_rhs, lb_->zvelocity,
			    old_initial_guess?Task::OldDW:Task::NewDW,
			    viscosity_params_);
    }
  }

  schedulePressureSolve(level, sched, solver,
			lb_->pressure, lb_->pressure_matrix, lb_->pressure_rhs);

  task = scinew Task("advectScalars", this, &SimpleCFD::advectScalars, step, nsteps);
#if 0
  task->requires(Task::OldDW, sharedState_->get_delt_label());
#endif
  task->requires(Task::OldDW, lb_->bctype, Ghost::AroundNodes, 2);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::OldDW, lb_->density, Ghost::AroundCells, maxadvect_);
  task->computes(lb_->density);
  if(level->getIndex()>0){
    addRefineDependencies(task, lb_->xvelocity, step+1, nsteps);
    addRefineDependencies(task, lb_->yvelocity, step+1, nsteps);
    addRefineDependencies(task, lb_->zvelocity, step+1, nsteps);
    addRefineDependencies(task, lb_->density, step, nsteps);
  }
  if(do_thermal){
    task->requires(Task::OldDW, lb_->temperature, Ghost::AroundCells, maxadvect_);
    task->computes(lb_->temperature);
    if(level->getIndex()>0)
      addRefineDependencies(task, lb_->temperature, step, nsteps);
  }
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  scheduleDiffuseScalar(sched, level, "density",
			lb_->density, lb_->density_matrix,
			lb_->density_rhs, density_diffusion_,
			solver, diffusion_params_, step, nsteps);
  if(do_thermal){
    scheduleDiffuseScalar(sched, level, "temperature", lb_->temperature,
			  lb_->temperature_matrix, lb_->temperature_rhs,
			  thermal_conduction_, solver, conduction_params_,
			  step, nsteps);
  }

  if(density_dissipation_ > 0) {
    task = scinew Task("dissipateScalars", this, &SimpleCFD::dissipateScalars);
#if 0
    task->requires(Task::OldDW, sharedState_->get_delt_label());
#endif
    task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
    task->requires(Task::NewDW, lb_->density, Ghost::None, 0);
    task->modifies(lb_->density);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  }

  task = scinew Task("updatebcs", this, &SimpleCFD::updatebcs);
  task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
  task->computes(lb_->bctype);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  task = scinew Task("interpolateVelocities", this, &SimpleCFD::interpolateVelocities);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::AroundCells, 1);
  task->computes(lb_->ccvelocity);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  releasePort("solver");
}

void SimpleCFD::scheduleDiffuseScalar(SchedulerP& sched, const LevelP& level,
				      const string& name,
				      const VarLabel* scalar,
				      const VarLabel* scalar_matrix,
				      const VarLabel* scalar_rhs,
				      double rate,
				      SolverInterface* solver,
				      const SolverParameters* solverparams,
				      int step, int nsteps)
{
  string taskname = "diffuseScalar: "+name;
  Task* task = scinew Task(taskname, this, &SimpleCFD::diffuseScalar,
			   DiffuseInfo(name, scalar, scalar_matrix,
				       scalar_rhs, rate, step, nsteps));
#if 0
  task->requires(Task::OldDW, sharedState_->get_delt_label());
#endif
  task->requires(Task::OldDW, lb_->bctype, Ghost::AroundFaces, 1);
  task->requires(Task::NewDW, scalar, Ghost::None, 0);
  task->computes(scalar_matrix);
  task->computes(scalar_rhs);
  if(level->getIndex()>0)
    addRefineDependencies(task, scalar, step+1, nsteps);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  solver->scheduleSolve(level, sched, sharedState_->allMaterials(),
			scalar_matrix, scalar, true,
			scalar_rhs, scalar,
			old_initial_guess?Task::OldDW:Task::NewDW,
			solverparams);
}

void SimpleCFD::schedulePressureSolve(const LevelP& level, SchedulerP& sched,
				      SolverInterface* solver,
				      const VarLabel* pressure,
				      const VarLabel* pressure_matrix,
				      const VarLabel* pressure_rhs)
{
  Task* task;

  task = scinew Task("projectVelocity", this, &SimpleCFD::projectVelocity,
		     pressure, pressure_matrix, pressure_rhs);
  task->requires(Task::OldDW, lb_->bctype, Ghost::AroundNodes, 1);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::AroundCells, 1);
  task->computes(pressure_matrix);
  task->computes(pressure_rhs);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  solver->scheduleSolve(level, sched, sharedState_->allMaterials(),
			pressure_matrix, pressure, false,
			pressure_rhs, keep_pressure?pressure:0,
			Task::OldDW, pressure_params_);

  task = scinew Task("applyProjection", this, &SimpleCFD::applyProjection,
		     pressure);
  task->requires(Task::OldDW, lb_->bctype, Ghost::AroundNodes, 1);
  task->requires(Task::NewDW, pressure, Ghost::AroundFaces, 1);
  task->modifies(lb_->xvelocity);
  task->modifies(lb_->yvelocity);
  task->modifies(lb_->zvelocity);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void SimpleCFD::computeStableTimestep(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse*,
				      DataWarehouse* new_dw)
{
  double delt=MAXDOUBLE;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      constSFCXVariable<double> xvel;
      new_dw->get(xvel, lb_->xvelocity, matl, patch, Ghost::None, 0);
      double maxx=0;
      for(CellIterator iter = patch->getSFCXIterator(); !iter.done(); iter++)
	maxx = Max(maxx, Abs(xvel[*iter]));
      constSFCYVariable<double> yvel;
      new_dw->get(yvel, lb_->yvelocity, matl, patch, Ghost::None, 0);
      double maxy=0;
      for(CellIterator iter = patch->getSFCYIterator(); !iter.done(); iter++)
	maxy = Max(maxy, Abs(yvel[*iter]));
      constSFCZVariable<double> zvel;
      new_dw->get(zvel, lb_->zvelocity, matl, patch, Ghost::None, 0);
      double maxz=0;
      for(CellIterator iter = patch->getSFCZIterator(); !iter.done(); iter++)
	maxz = Max(maxz, Abs(zvel[*iter]));
      Vector t_inv(Vector(maxx, maxy, maxz)/patch->dCell());
      if(t_inv.maxComponent() > 0)
	delt=Min(delt, 1./t_inv.maxComponent());
    }
  }
  if(delt != MAXDOUBLE){
    delt *= delt_multiplier_;
    const Level* level = getLevel(patches);
    GridP grid = level->getGrid();
    for(int i=1;i<=level->getIndex();i++)
      delt *= grid->getLevel(i)->timeRefinementRatio();
  }
  new_dw->put(delt_vartype(delt), sharedState_->get_delt_label());
}

void SimpleCFD::initialize(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      SFCXVariable<double> xvel;
      new_dw->allocateAndPut(xvel, lb_->xvelocity, matl, patch);
      SFCYVariable<double> yvel;
      new_dw->allocateAndPut(yvel, lb_->yvelocity, matl, patch);
      SFCZVariable<double> zvel;
      new_dw->allocateAndPut(zvel, lb_->zvelocity, matl, patch);

      ics.getCondition<double>("xvelocity")->apply(xvel, patch);
      ics.getCondition<double>("yvelocity")->apply(yvel, patch);
      ics.getCondition<double>("zvelocity")->apply(zvel, patch);

      CCVariable<double> den;
      new_dw->allocateAndPut(den, lb_->density, matl, patch);
      ics.getCondition<double>("density")->apply(den, patch);
      if(do_thermal){
	CCVariable<double> temp;
	new_dw->allocateAndPut(temp, lb_->temperature, matl, patch);
	ics.getCondition<double>("temperature")->apply(temp, patch);
      }
	
      // Create bcs....
      NCVariable<int> bctype;
      new_dw->allocateAndPut(bctype, lb_->bctype, matl, patch);
      bcs.set(bctype, patch);

      if(keep_pressure){
	CCVariable<double> pressure;
	new_dw->allocateAndPut(pressure, lb_->pressure, matl, patch);
	pressure.initialize(0);
      }
      if(random_initial_velocities.x()>0){
	for(CellIterator iter(patch->getSFCXIterator(1)); !iter.done(); iter++){
	  IntVector idx(*iter);
	  xvel[idx] += 2*random_initial_velocities.x()*(drand48()-0.5);
	}
      }
      if(random_initial_velocities.y()>0){
	for(CellIterator iter(patch->getSFCYIterator(1)); !iter.done(); iter++){
	  IntVector idx(*iter);
	  yvel[idx] += 2*random_initial_velocities.y()*(drand48()-0.5);
	}
      }
      if(random_initial_velocities.z()>0){
	for(CellIterator iter(patch->getSFCZIterator(1)); !iter.done(); iter++){
	  IntVector idx(*iter);
	  zvel[idx] += 2*random_initial_velocities.z()*(drand48()-0.5);
	}
      }
#if 1
      if(dbg.active()){
	printhex(bctype, "bctype");
	print(xvel, yvel, zvel, "Initial velocity");
	print(den, "Initial density");
      }
#endif
    }
  }
}

static inline bool inside(const IntVector& i, const IntVector& l,
			  const IntVector& h)
{
  return i.x() >= l.x() && i.y() >= l.y() && i.z() >= l.z()
    && i.x() < h.x() && i.y() < h.y() && i.z() < h.z();
}

template<class ArrayType>
void contribute(const ArrayType& field, constNCVariable<int>& bctype,
		Condition<double>* bc, const IntVector& idx,
		double weight, double& w,
		typename ArrayType::value_type& value)
{
  if(::inside(idx, field.getLowIndex(), field.getHighIndex())){
    BCRegion<double>* bc2 = bc->get(bctype[idx]);
    switch(bc2->getType()){
    case BC::FreeFlow:
    case BC::FixedRate:
    case BC::FixedValue:
    case BC::CoarseGrid:
    case BC::FixedFlux:
      value += field[idx]*weight;
      w+=weight;
      break;
    case BC::Exterior:
      break;
    }
  }
}

template<class ArrayType>
bool Interpolate(typename ArrayType::value_type& value, const Vector& v,
		 const Vector& offset, const Vector& inv_dx,
		 const ArrayType& field, Condition<double>* bcs,
		 constNCVariable<int>& bctype)
{
  Vector pos(v*inv_dx-offset);
  int ix = RoundDown(pos.x());
  int iy = RoundDown(pos.y());
  int iz = RoundDown(pos.z());
  double fx = pos.x()-ix;
  double fy = pos.y()-iy;
  double fz = pos.z()-iz;
  IntVector l(field.getLowIndex());
  IntVector h(field.getHighIndex());

  double w = 0;
  value = 0;
  contribute(field, bctype, bcs, IntVector(ix, iy, iz), (1-fx)*(1-fy)*(1-fz),
	     w, value);
  contribute(field, bctype, bcs, IntVector(ix+1, iy, iz), fx*(1-fy)*(1-fz),
	     w, value);
  contribute(field, bctype, bcs, IntVector(ix, iy+1, iz), (1-fx)*fy*(1-fz),
	     w, value);
  contribute(field, bctype, bcs, IntVector(ix+1, iy+1, iz), fx*fy*(1-fz),
	     w, value);
  contribute(field, bctype, bcs, IntVector(ix, iy, iz+1), (1-fx)*(1-fy)*fz,
	     w, value);
  contribute(field, bctype, bcs, IntVector(ix+1, iy, iz+1), fx*(1-fy)*fz,
	     w, value);
  contribute(field, bctype, bcs, IntVector(ix, iy+1, iz+1), (1-fx)*fy*fz,
	     w, value);
  contribute(field, bctype, bcs, IntVector(ix+1, iy+1, iz+1), fx*fy*fz,
	     w, value);
  if(w < 1.e-8)
    return false;
  value *= 1./w;
  return true;
}

static bool particleTrace(Vector& p, int nsteps, double delt,
			  const Vector& inv_dx,
			  constSFCXVariable<double>& xvel,
			  constSFCYVariable<double>& yvel,
			  constSFCZVariable<double>& zvel,
			  Condition<double>* xbc,
			  Condition<double>* ybc,
			  Condition<double>* zbc,
			  constNCVariable<int>& bctype)
{
  for(int i=0;i<nsteps;i++){
    double x;
    if(!Interpolate(x, p, Vector(0, 0.5, 0.5), inv_dx, xvel, xbc, bctype))
      return false;
    double y;
    if(!Interpolate(y, p, Vector(0.5, 0, 0.5), inv_dx, yvel, ybc, bctype))
      return false;
    double z;
    if(!Interpolate(z, p, Vector(0.5, 0.5, 0), inv_dx, zvel, zbc, bctype))
      return false;
    Vector v(x,y,z);
    p-=v*delt;
  }
  return true;
}

static bool particleTrace(Vector& p, double delt, const Vector& inv_dx,
			  constSFCXVariable<double>& xvel,
			  constSFCYVariable<double>& yvel,
			  constSFCZVariable<double>& zvel,
			  double tolerance,
			  Condition<double>* xbc,
			  Condition<double>* ybc,
			  Condition<double>* zbc,
			  constNCVariable<int>& bctype)
{
  int nsteps=1;
  Vector p1 = p;
  bool success1 = particleTrace(p1, nsteps, delt/nsteps, inv_dx,
				xvel, yvel, zvel, xbc, ybc, zbc, bctype);
  double tolerance2 = tolerance*tolerance;
  int maxsteps = 128;
  for(;;) {
    nsteps<<=1;
    if(nsteps > maxsteps)
      break;
    Vector p2 = p;
    bool success2 = particleTrace(p2, nsteps, delt/nsteps, inv_dx, xvel, yvel, zvel, xbc, ybc, zbc, bctype);
    if(success1 && success2){
      if((p2-p1).length2() < tolerance2)
	break;
      p1=p2;
    } else if(success2){
      // This one was successful but the last one was not
      success1=true;
      p1=p2;
    } else if(success1){
      // The last one was successful, so ignore this one
    } else {
      // Neither were succesful, keep trying
    }
  }
  p=p1;
  return success1;
}

void SimpleCFD::advect(Array3<double>& q, const Array3<double>& qold,
		       CellIterator iter,
		       const Patch* patch, double delt, const Vector& offset,
		       constSFCXVariable<double>& xvel,
		       constSFCYVariable<double>& yvel,
		       constSFCZVariable<double>& zvel,
		       constNCVariable<int>& bctype,
		       Condition<double>* cbc,
		       Condition<double>* xbc,
		       Condition<double>* ybc,
		       Condition<double>* zbc)
{
  Vector dx(patch->dCell());
  Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
  for(; !iter.done(); iter++){
    IntVector idx(*iter);
    BCRegion<double>* b = cbc->get(bctype[idx]);
    switch(b->getType()){
    case BC::FreeFlow:
    case BC::FixedRate:
      {
	Vector v = (idx.asVector()+offset)*dx;
	if(particleTrace(v, delt, inv_dx, xvel, yvel, zvel, advection_tolerance_, xbc, ybc, zbc, bctype)){
	  double value;
	  if(Interpolate(value, v, offset, inv_dx, qold, cbc, bctype)){
	    q[idx] = value;
	  } else {
	    //cerr << "WARNING: outside: " << v << '\n';
	    q[idx]=qold[idx];
	  }
	} else {
	  //Vector v2 = (idx.asVector()+offset)*dx;
	  //cerr << "WARNING: trace failed: " << v << ", original v=" << v2 << '\n';
	  q[idx]=qold[idx];
	}
      }
      break;
    case BC::FixedValue:
    case BC::FixedFlux:
      q[idx] = b->getValue();
      break;
    case BC::Exterior:
    case BC::CoarseGrid:
      cerr << "bc at idx: " << idx << "=" << b->getType() << '\n';
      cerr << "bctype=" << setbase(16) << bctype[idx] << '\n';
      
      SCI_THROW(InternalError("Don't know what to do here"));
    }
  }
}

void SimpleCFD::advectVelocity(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw,
			       int step, int nsteps)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, sharedState_->get_delt_label(), level);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constNCVariable<int> bctype;
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundNodes, 2);

      constSFCXVariable<double> xvel_old;
      old_dw->get(xvel_old, lb_->xvelocity, matl, patch,
		  Ghost::AroundCells, maxadvect_+1);
      SFCXVariable<double> xvel;
      new_dw->allocateAndPut(xvel, lb_->xvelocity, matl, patch);

      constSFCYVariable<double> yvel_old;
      old_dw->get(yvel_old, lb_->yvelocity, matl, patch,
		  Ghost::AroundCells, maxadvect_+1);
      SFCYVariable<double> yvel;
      new_dw->allocateAndPut(yvel, lb_->yvelocity, matl, patch);

      constSFCZVariable<double> zvel_old;
      old_dw->get(zvel_old, lb_->zvelocity, matl, patch,
		  Ghost::AroundCells, maxadvect_+1);
      SFCZVariable<double> zvel;
      new_dw->allocateAndPut(zvel, lb_->zvelocity, matl, patch);

      if(level->getIndex() > 0){
	refineBoundaries(patch, xvel_old.castOffConst(),
			 new_dw, lb_->xvelocity, matl,
			 double(step)/double(nsteps));
	refineBoundaries(patch, yvel_old.castOffConst(),
			 new_dw, lb_->yvelocity, matl,
			 double(step)/double(nsteps));
	refineBoundaries(patch, zvel_old.castOffConst(),
			 new_dw, lb_->zvelocity, matl,
			 double(step)/double(nsteps));
      }
      advect(xvel, xvel_old, patch->getSFCXIterator(), patch, delT,
	     Vector(0, 0.5, 0.5), xvel_old, yvel_old, zvel_old, bctype,
	     bcs.getCondition<double>("xvelocity", Patch::XFaceBased),
	     bcs.getCondition<double>("xvelocity", Patch::XFaceBased),
	     bcs.getCondition<double>("yvelocity", Patch::YFaceBased),
	     bcs.getCondition<double>("zvelocity", Patch::ZFaceBased));
      advect(yvel, yvel_old, patch->getSFCYIterator(), patch, delT,
	     Vector(0.5, 0, 0.5), xvel_old, yvel_old, zvel_old, bctype,
	     bcs.getCondition<double>("yvelocity", Patch::YFaceBased),
	     bcs.getCondition<double>("xvelocity", Patch::XFaceBased),
	     bcs.getCondition<double>("yvelocity", Patch::YFaceBased),
	     bcs.getCondition<double>("zvelocity", Patch::ZFaceBased));
      advect(zvel, zvel_old, patch->getSFCZIterator(), patch, delT,
	     Vector(0.5, 0.5, 0), xvel_old, yvel_old, zvel_old, bctype,
	     bcs.getCondition<double>("zvelocity", Patch::ZFaceBased),
	     bcs.getCondition<double>("xvelocity", Patch::XFaceBased),
	     bcs.getCondition<double>("yvelocity", Patch::YFaceBased),
	     bcs.getCondition<double>("zvelocity", Patch::ZFaceBased));
    }
  }
}

void SimpleCFD::applyForces(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label(), level);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      SFCXVariable<double> xvel;
      new_dw->getModifiable(xvel, lb_->xvelocity, matl, patch);

      SFCYVariable<double> yvel;
      new_dw->getModifiable(yvel, lb_->yvelocity, matl, patch);

      SFCZVariable<double> zvel;
      new_dw->getModifiable(zvel, lb_->zvelocity, matl, patch);
      
      Vector grav = sharedState_->getGravity();
      if(vorticity_confinement_scale_ != 0){
	constCCVariable<Vector> vel;
	old_dw->get(vel, lb_->ccvelocity, matl, patch, Ghost::None, 0);
	CCVariable<Vector> ccvorticity;
	CCVariable<double> ccvorticitymag;
	new_dw->allocateAndPut(ccvorticity, lb_->ccvorticity, matl, patch);
	new_dw->allocateAndPut(ccvorticitymag, lb_->ccvorticitymag, matl, patch);
	Vector dx(patch->dCell());
	Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
	IntVector l(patch->getCellLowIndex());
	IntVector h(patch->getCellHighIndex());
	if(patch->getBCType(Patch::xminus) == Patch::Neighbor)
	  l -= IntVector(1,0,0);
	if(patch->getBCType(Patch::xplus) == Patch::Neighbor)
	  h += IntVector(1,0,0);
	if(patch->getBCType(Patch::yminus) == Patch::Neighbor)
	  l -= IntVector(0,1,0);
	if(patch->getBCType(Patch::yplus) == Patch::Neighbor)
	  h += IntVector(0,1,0);
	if(patch->getBCType(Patch::zminus) == Patch::Neighbor)
	  l -= IntVector(0,0,1);
	if(patch->getBCType(Patch::zplus) == Patch::Neighbor)
	  h += IntVector(0,0,1);
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
	  ccvorticity[idx]=w;
	  ccvorticitymag[idx]=w.length();
	}
	CCVariable<Vector> vcforce;
	CCVariable<Vector> NN;
	new_dw->allocateAndPut(vcforce, lb_->vcforce, matl, patch);
	new_dw->allocateAndPut(NN, lb_->NN, matl, patch);
	Vector maxforce(0,0,0);
	for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  double gx, gy, gz;
	  if(idx.x() == l.x()){
	    gx = (ccvorticitymag[idx+IntVector(1,0,0)]-ccvorticitymag[idx])*inv_dx.x();
	  } else if(idx.x() == h.x()-1){
	    gx = (ccvorticitymag[idx]-ccvorticitymag[idx-IntVector(1,0,0)])*inv_dx.x();
	  } else {
	    gx = (ccvorticitymag[idx+IntVector(1,0,0)]-ccvorticitymag[idx-IntVector(1,0,0)])*0.5*inv_dx.x();
	  }
	  if(idx.y() == l.y()){
	    gy = (ccvorticitymag[idx+IntVector(0,1,0)]-ccvorticitymag[idx])*inv_dx.y();
	  } else if(idx.y() == h.y()-1){
	    gy = (ccvorticitymag[idx]-ccvorticitymag[idx-IntVector(0,1,0)])*inv_dx.y();
	  } else {
	    gy = (ccvorticitymag[idx+IntVector(0,1,0)]-ccvorticitymag[idx-IntVector(0,1,0)])*0.5*inv_dx.y();
	  }
	  if(idx.z() == l.z()){
	    gz = (ccvorticitymag[idx+IntVector(0,0,1)]-ccvorticitymag[idx])*inv_dx.z();
	  } else if(idx.z() == h.z()-1){
	    gz = (ccvorticitymag[idx]-ccvorticitymag[idx-IntVector(0,0,1)])*inv_dx.z();
	  } else {
	    gz = (ccvorticitymag[idx+IntVector(0,0,1)]-ccvorticitymag[idx-IntVector(0,0,1)])*0.5*inv_dx.z();
	  }
	  Vector N(gx, gy, gz);
	  if(N.length2()>0.0)
	    N.normalize();
	  NN[idx]=N;
	  vcforce[idx] = Cross(N, ccvorticity[idx])*dx*vorticity_confinement_scale_;
	  maxforce=Max(maxforce, Vector(vcforce[idx].length()));
	}
	cerr << "maxforce=" << maxforce << '\n';
	double constant = 0.5*delT;
	for(CellIterator iter(patch->getSFCXIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  double fx;
	  if(idx.x() == l.x())
	    fx = vcforce[idx].x();
	  else if(idx.x() == h.x())
	    fx = vcforce[idx+IntVector(-1,0,0)].x();
	  else
	    fx = (vcforce[idx].x()+vcforce[idx+IntVector(-1,0,0)].x());
	  xvel[idx] += fx*constant;
	}
	for(CellIterator iter(patch->getSFCYIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  double fy;
	  if(idx.y() == l.y())
	    fy = vcforce[idx].y();
	  else if(idx.y() == h.y())
	    fy = vcforce[idx+IntVector(0,-1,0)].y();
	  else
	    fy = (vcforce[idx].y()+vcforce[idx+IntVector(0,-1,0)].y());
	  yvel[idx] += fy*constant;
	}
	for(CellIterator iter(patch->getSFCZIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  double fz;
	  if(idx.z() == l.z())
	    fz = vcforce[idx].z();
	  else if(idx.z() == h.z())
	    fz = vcforce[idx+IntVector(0,0,-1)].z();
	  else
	    fz = (vcforce[idx].z()+vcforce[idx+IntVector(0,0,-1)].z());
	  zvel[idx] += fz*constant;
	}
      }
      if(grav.length() > 0) {
	constCCVariable<double> den;
	old_dw->get(den, lb_->density, matl, patch, Ghost::AroundFaces, 1);
	
	constCCVariable<double> temp;
	if(do_thermal && buoyancy_>0)
	  old_dw->get(temp, lb_->temperature, matl, patch, Ghost::AroundFaces, 1);

	if(grav.x() != 0){
	  double gconstant = grav.x()*delT;
	  if(do_thermal && buoyancy_>0){
	    double bconstant = buoyancy_*grav.x()/grav.length()*delT;
	    for(CellIterator iter(patch->getSFCXIterator(1)); !iter.done(); iter++){
	      IntVector idx(*iter);
	      double t = (temp[idx]+temp[idx+IntVector(-1,0,0)])*0.5;
	      double d = (den[idx]+den[idx+IntVector(-1,0,0)])*0.5;
	      double fx = d*gconstant-t*bconstant;
	      xvel[idx] += fx;
	    }
	  } else {
	    for(CellIterator iter(patch->getSFCXIterator(1)); !iter.done(); iter++){
	      IntVector idx(*iter);
	      double d = (den[idx]+den[idx+IntVector(-1,0,0)])*0.5;
	      double fx = d*gconstant;
	      xvel[idx] += fx;
	    }
	  }
	}
	if(grav.y() != 0) {
	  double gconstant = grav.y()*delT;
	  if(do_thermal && buoyancy_>0){
	    double bconstant = buoyancy_*grav.y()/grav.length()*delT;
	    for(CellIterator iter(patch->getSFCYIterator(1)); !iter.done(); iter++){
	      IntVector idx(*iter);
	      double d = (den[idx]+den[idx+IntVector(0,-1,0)])*0.5;
	      double t = (temp[idx]+temp[idx+IntVector(0,-1,0)])*0.5;
	      double fy = d*gconstant-t*bconstant;
	      yvel[idx] += fy;
	    }
	    
	  } else {
	    for(CellIterator iter(patch->getSFCYIterator(1)); !iter.done(); iter++){
	      IntVector idx(*iter);
	      double d = (den[idx]+den[idx+IntVector(0,-1,0)])*0.5;
	      double fy = d*gconstant;
	      yvel[idx] += fy;
	    }
	  }
	}
	if(grav.z() != 0) {
	  double gconstant = grav.z()*delT;
	  if(do_thermal && buoyancy_>0){
	    double bconstant = buoyancy_*grav.z()/grav.length()*delT;
	    for(CellIterator iter(patch->getSFCZIterator(1)); !iter.done(); iter++){
	      IntVector idx(*iter);
	      double d = (den[idx]+den[idx+IntVector(0,0,-1)])*0.5;
	      double t = (temp[idx]+temp[idx+IntVector(0,0,-1)])*0.5;
	      double fz = d*gconstant-t*bconstant;
	      zvel[idx] += fz;
	    }
	  } else {
	    for(CellIterator iter(patch->getSFCZIterator(1)); !iter.done(); iter++){
	      IntVector idx(*iter);
	      double d = (den[idx]+den[idx+IntVector(0,0,-1)])*0.5;
	      double fz = d*gconstant;
	      zvel[idx] += fz;
	    }
	  }
	}
      }
    }
  }
}

void SimpleCFD::applybc(const IntVector& idx, const IntVector&,
			const IntVector&, const IntVector& h2,
			const Array3<double>& field, double delt,
			const Vector& inv_dx2, double diff,
			constNCVariable<int>& bctype,
			Condition<double>* scalar_bc,
			Condition<double>* xface_bc,
			Condition<double>* yface_bc,
			Condition<double>* zface_bc,
			const IntVector& FW, const IntVector& FE,
			const IntVector& FS, const IntVector& FN,
			const IntVector& FB, const IntVector& FT,
			Array3<Stencil7>& A, Array3<double>& rhs)
{
  BCRegion<double>* bc = scalar_bc->get(bctype[idx]);
  IntVector W(-1,0,0);
  IntVector E(1,0,0);
  IntVector S(0,-1,0);
  IntVector N(0,1,0);
  IntVector B(0,0,-1);
  IntVector T(0,0,1);
  switch(bc->getType()){
  case BC::FreeFlow:
  case BC::FixedRate:
    A[idx].p=1;
    rhs[idx]=field[idx];
    if(bc->getType() == BC::FixedRate)
      rhs[idx] += bc->getValue()*delt;
    {
      BCRegion<double>* bc1 = xface_bc->get(bctype[idx+FW]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	// fall through
      case BC::FreeFlow:
	{
	  BCRegion<double>* bc2 = scalar_bc->get(bctype[idx+W]);
	  switch(bc2->getType()){
	  case BC::FreeFlow:
	  case BC::FixedRate:
	    A[idx].p+=diff*delt*inv_dx2.x();
	    A[idx].w=-diff*delt*inv_dx2.x();
	    break;
	  case BC::FixedValue:
	    A[idx].p+=diff*delt*inv_dx2.x();
	    A[idx].w=0;
	    rhs[idx]+=bc2->getValue()*(diff*delt*inv_dx2.x());
	    break;
	  case BC::CoarseGrid:
	    A[idx].p+=diff*delt*inv_dx2.x();
	    A[idx].w=0;
	    rhs[idx]+=field[idx+W]*(diff*delt*inv_dx2.x());
	    break;	    
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	  case BC::Exterior:
	    break;
	  }
	}
	break;
      case BC::FixedValue:
	A[idx].p+=2*diff*delt*inv_dx2.x();
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	A[idx].w = 0;
	break;
      case BC::FixedFlux:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	A[idx].w = 0;
	break;
      case BC::Exterior:
      case BC::CoarseGrid:
	throw InternalError("unknown BC");
	BREAK;
      }
    }
    {
      BCRegion<double>* bc1 = xface_bc->get(bctype[idx+FE]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	// fall through
      case BC::FreeFlow:
	{
	  BCRegion<double>* bc2 = scalar_bc->get(bctype[idx+E]);
	  switch(bc2->getType()){
	  case BC::FreeFlow:
	  case BC::FixedRate:
	    A[idx].p+=diff*delt*inv_dx2.x();
	    A[idx].e=-diff*delt*inv_dx2.x();
	    break;
	  case BC::FixedValue:
	    A[idx].p+=diff*delt*inv_dx2.x();
	    A[idx].e=0;
	    rhs[idx]+=bc2->getValue()*(diff*delt*inv_dx2.x());
	    break;
	  case BC::CoarseGrid:
	    A[idx].p+=diff*delt*inv_dx2.x();
	    A[idx].e=0;
	    rhs[idx]+=field[idx+E]*(diff*delt*inv_dx2.x());
	    break;	    
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	  case BC::Exterior:
	    break;
	  }
	}
	break;
      case BC::FixedValue:
	A[idx].p+=2*diff*delt*inv_dx2.x();
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	A[idx].e = 0;
	break;
      case BC::FixedFlux:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	A[idx].e = 0;
	break;
      case BC::Exterior:
      case BC::CoarseGrid:
	throw InternalError("unknown BC");
	BREAK;
      }
    }
    {
      BCRegion<double>* bc1 = yface_bc->get(bctype[idx+FS]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	// fall through
      case BC::FreeFlow:
	{
	  BCRegion<double>* bc2 = scalar_bc->get(bctype[idx+S]);
	  switch(bc2->getType()){
	  case BC::FreeFlow:
	  case BC::FixedRate:
	    A[idx].p+=diff*delt*inv_dx2.y();
	    A[idx].s=-diff*delt*inv_dx2.y();
	    break;
	  case BC::FixedValue:
	    A[idx].p+=diff*delt*inv_dx2.y();
	    A[idx].s=0;
	    rhs[idx]+=bc2->getValue()*(diff*delt*inv_dx2.y());
	    break;
	  case BC::CoarseGrid:
	    A[idx].p+=diff*delt*inv_dx2.y();
	    A[idx].s=0;
	    rhs[idx]+=field[idx+S]*(diff*delt*inv_dx2.y());
	    break;	    
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	  case BC::Exterior:
	    break;
	  }
	}
	break;
      case BC::FixedValue:
	A[idx].p+=2*diff*delt*inv_dx2.x();
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	A[idx].s = 0;
	break;
      case BC::FixedFlux:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	A[idx].s = 0;
	break;
      case BC::Exterior:
      case BC::CoarseGrid:
	throw InternalError("unknown BC");
	BREAK;
      }
    }
    {
      BCRegion<double>* bc1 = yface_bc->get(bctype[idx+FN]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	// fall through
      case BC::FreeFlow:
	{
	  BCRegion<double>* bc2 = scalar_bc->get(bctype[idx+N]);
	  switch(bc2->getType()){
	  case BC::FreeFlow:
	  case BC::FixedRate:
	    A[idx].p+=diff*delt*inv_dx2.y();
	    A[idx].n=-diff*delt*inv_dx2.y();
	    break;
	  case BC::FixedValue:
	    A[idx].p+=diff*delt*inv_dx2.y();
	    A[idx].n=0;
	    rhs[idx]+=bc2->getValue()*(diff*delt*inv_dx2.y());
	    break;
	  case BC::CoarseGrid:
	    A[idx].p+=diff*delt*inv_dx2.y();
	    A[idx].n=0;
	    rhs[idx]+=field[idx+N]*(diff*delt*inv_dx2.y());
	    break;	    
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	  case BC::Exterior:
	    break;
	  }
	}
	break;
      case BC::FixedValue:
	A[idx].p+=2*diff*delt*inv_dx2.y();
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	A[idx].n = 0;
	break;
      case BC::FixedFlux:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	A[idx].n = 0;
	break;
      case BC::Exterior:
      case BC::CoarseGrid:
	throw InternalError("unknown BC");
	BREAK;
      }
    }
    {
      BCRegion<double>* bc1 = zface_bc->get(bctype[idx+FB]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	// fall through
      case BC::FreeFlow:
	{
	  BCRegion<double>* bc2 = scalar_bc->get(bctype[idx+B]);
	  switch(bc2->getType()){
	  case BC::FreeFlow:
	  case BC::FixedRate:
	    A[idx].p+=diff*delt*inv_dx2.z();
	    A[idx].b=-diff*delt*inv_dx2.z();
	    break;
	  case BC::FixedValue:
	    A[idx].p+=diff*delt*inv_dx2.z();
	    A[idx].b=0;
	    rhs[idx]+=bc2->getValue()*(diff*delt*inv_dx2.z());
	    break;
	  case BC::CoarseGrid:
	    A[idx].p+=diff*delt*inv_dx2.z();
	    A[idx].b=0;
	    rhs[idx]+=field[idx+B]*(diff*delt*inv_dx2.z());
	    break;	    
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	  case BC::Exterior:
	    break;
	  }
	}
	break;
      case BC::FixedValue:
	A[idx].p+=2*diff*delt*inv_dx2.z();
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	A[idx].b = 0;
	break;
      case BC::FixedFlux:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	A[idx].b = 0;
	break;
      case BC::Exterior:
      case BC::CoarseGrid:
	throw InternalError("unknown BC");
	BREAK;
      }
    }
    {
      BCRegion<double>* bc1 = zface_bc->get(bctype[idx+FT]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	// fall through
      case BC::FreeFlow:
	{
	  BCRegion<double>* bc2 = scalar_bc->get(bctype[idx+T]);
	  switch(bc2->getType()){
	  case BC::FreeFlow:
	  case BC::FixedRate:
	    A[idx].p+=diff*delt*inv_dx2.z();
	    A[idx].t=-diff*delt*inv_dx2.z();
	    break;
	  case BC::FixedValue:
	    A[idx].p+=diff*delt*inv_dx2.z();
	    A[idx].t=0;
	    rhs[idx]+=bc2->getValue()*(diff*delt*inv_dx2.z());
	    break;
	  case BC::CoarseGrid:
	    A[idx].p+=diff*delt*inv_dx2.z();
	    A[idx].t=0;
	    rhs[idx]+=field[idx+T]*(diff*delt*inv_dx2.z());
	    break;	    
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	  case BC::Exterior:
	    break;
	  }
	}
	break;
      case BC::FixedValue:
	A[idx].p+=2*diff*delt*inv_dx2.z();
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	A[idx].t = 0;
	break;
      case BC::FixedFlux:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	A[idx].t = 0;
	break;
      case BC::Exterior:
      case BC::CoarseGrid:
	throw InternalError("unknown BC");
	BREAK;
      }
    }
    break;
  case BC::FixedValue:
    A[idx].p = 1;
    A[idx].n = A[idx].s=A[idx].w=A[idx].e=A[idx].t=A[idx].b=0;
    rhs[idx] = bc->getValue();
    break;
  case BC::FixedFlux:
  case BC::Exterior:
  case BC::CoarseGrid:
    throw InternalError("unknown BC");
    BREAK;
  }
}

void SimpleCFD::applyViscosity(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw,
			       int dir, int step, int nsteps)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label(), level);
  double delt=delT;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      Vector dx(patch->dCell());
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
      Vector inv_dx2(inv_dx*inv_dx);

      constNCVariable<int> bctype;
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundFaces, 1);

      IntVector h2=patch->getCellHighIndex()+IntVector(1,1,1);
      if(dir == 0){
	constSFCXVariable<double> xvel;
	new_dw->get(xvel, lb_->xvelocity, matl, patch, Ghost::None, 0);
	if(level->getIndex()>0)
	  refineBoundaries(patch, xvel.castOffConst(), new_dw,
			   lb_->xvelocity, matl,
			   double(step+1)/double(nsteps));

	SFCXVariable<Stencil7> A_xvel;
	new_dw->allocateAndPut(A_xvel,  lb_->xvelocity_matrix, matl, patch);
	SFCXVariable<double> rhs_xvel;
	new_dw->allocateAndPut(rhs_xvel, lb_->xvelocity_rhs, matl, patch);

	// Viscosity
	IntVector l=patch->getGhostCellLowIndex(1);
	IntVector h=patch->getGhostCellHighIndex(1);
	if(patch->getBCType(Patch::xplus) != Patch::Neighbor)
	  h += IntVector(1,0,0);
	Condition<double>* xvel_bc = bcs.getCondition<double>("xvelocity", Patch::XFaceBased);
	Condition<double>* xvel_bc_yface = bcs.getCondition<double>("xvelocity", Patch::YFaceBased);
	Condition<double>* xvel_bc_zface = bcs.getCondition<double>("xvelocity", Patch::ZFaceBased);
	Condition<double>* xvel_bc_cc = bcs.getCondition<double>("xvelocity", Patch::CellBased);

	IntVector FW=IntVector(-1,0,0);
	IntVector FE=IntVector(0,0,0);
	IntVector FS=IntVector(0,0,0);
	IntVector FN=IntVector(0,1,0);
	IntVector FB=IntVector(0,0,0);
	IntVector FT=IntVector(0,0,1);
	for(CellIterator iter(patch->getSFCXIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  applybc(idx, l, h, h2, xvel, delt, inv_dx2, viscosity_,
		  bctype, xvel_bc, xvel_bc_cc, xvel_bc_yface, xvel_bc_zface,
		  FW, FE, FS, FN, FB, FT, A_xvel, rhs_xvel);
	}
      } else if(dir == 1){
	constSFCYVariable<double> yvel;
	new_dw->get(yvel, lb_->yvelocity, matl, patch, Ghost::None, 0);
	if(level->getIndex()>0)
	  refineBoundaries(patch, yvel.castOffConst(), new_dw,
			   lb_->yvelocity, matl,
			   double(step+1)/double(nsteps));

	SFCYVariable<Stencil7> A_yvel;
	new_dw->allocateAndPut(A_yvel,  lb_->yvelocity_matrix, matl, patch);
	SFCYVariable<double> rhs_yvel;
	new_dw->allocateAndPut(rhs_yvel, lb_->yvelocity_rhs, matl, patch);

	IntVector l=patch->getGhostCellLowIndex(1);
	IntVector h=patch->getGhostCellHighIndex(1);
	if(patch->getBCType(Patch::yplus) != Patch::Neighbor)
	  h += IntVector(0,1,0);
	Condition<double>* yvel_bc = bcs.getCondition<double>("yvelocity", Patch::YFaceBased);
	Condition<double>* yvel_bc_xface = bcs.getCondition<double>("yvelocity", Patch::XFaceBased);
	Condition<double>* yvel_bc_zface = bcs.getCondition<double>("yvelocity", Patch::ZFaceBased);
	Condition<double>* yvel_bc_cc = bcs.getCondition<double>("yvelocity", Patch::CellBased);

	IntVector FW=IntVector(0,0,0);
	IntVector FE=IntVector(1,0,0);
	IntVector FS=IntVector(0,-1,0);
	IntVector FN=IntVector(0,0,0);
	IntVector FB=IntVector(0,0,0);
	IntVector FT=IntVector(0,0,1);
	for(CellIterator iter(patch->getSFCYIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  applybc(idx, l, h, h2, yvel, delt, inv_dx2, viscosity_,
		  bctype, yvel_bc, yvel_bc_xface, yvel_bc_cc, yvel_bc_zface,
		  FW, FE, FS, FN, FB, FT, A_yvel, rhs_yvel);
	}
      } else if(dir == 2){
	constSFCZVariable<double> zvel;
	new_dw->get(zvel, lb_->zvelocity, matl, patch, Ghost::None, 0);
	if(level->getIndex()>0)
	  refineBoundaries(patch, zvel.castOffConst(), new_dw,
			   lb_->zvelocity, matl,
			   double(step+1)/double(nsteps));
      
	SFCZVariable<Stencil7> A_zvel;
	new_dw->allocateAndPut(A_zvel,  lb_->zvelocity_matrix, matl, patch);
	SFCZVariable<double> rhs_zvel;
	new_dw->allocateAndPut(rhs_zvel, lb_->zvelocity_rhs, matl, patch);

	IntVector l=patch->getGhostCellLowIndex(1);
	IntVector h=patch->getGhostCellHighIndex(1);
	if(patch->getBCType(Patch::zplus) != Patch::Neighbor)
	  h += IntVector(0,0,1);
	Condition<double>* zvel_bc = bcs.getCondition<double>("zvelocity", Patch::ZFaceBased);
	Condition<double>* zvel_bc_xface = bcs.getCondition<double>("zvelocity", Patch::XFaceBased);
	Condition<double>* zvel_bc_yface = bcs.getCondition<double>("zvelocity", Patch::YFaceBased);
	Condition<double>* zvel_bc_cc = bcs.getCondition<double>("zvelocity", Patch::CellBased);

	IntVector FW=IntVector(0,0,0);
	IntVector FE=IntVector(1,0,0);
	IntVector FS=IntVector(0,0,0);
	IntVector FN=IntVector(0,1,0);
	IntVector FB=IntVector(0,0,-1);
	IntVector FT=IntVector(0,0,0);
	for(CellIterator iter(patch->getSFCZIterator()); !iter.done(); iter++){
	  IntVector idx(*iter);
	  applybc(idx, l, h, h2, zvel, delt, inv_dx2, viscosity_,
		  bctype, zvel_bc, zvel_bc_xface, zvel_bc_yface, zvel_bc_cc,
		  FW, FE, FS, FN, FB, FT, A_zvel, rhs_zvel);
	}
      }
    }
  }
}

void SimpleCFD::projectVelocity(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw, DataWarehouse* new_dw,
				const VarLabel*,
				const VarLabel* pressure_matrix,
				const VarLabel* pressure_rhs)
{
  IntVector FW(0,0,0);
  IntVector FE(1,0,0);
  IntVector FS(0,0,0);
  IntVector FN(0,1,0);
  IntVector FB(0,0,0);
  IntVector FT(0,0,1);
  IntVector W(-1,0,0);
  IntVector E(1,0,0);
  IntVector S(0,-1,0);
  IntVector N(0,1,0);
  IntVector B(0,0,-1);
  IntVector T(0,0,1);
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, sharedState_->get_delt_label(), level);
  double delt = delT;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constNCVariable<int> bctype;
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundNodes, 1);

      constSFCXVariable<double> xvel;
      new_dw->get(xvel, lb_->xvelocity, matl, patch, Ghost::AroundCells, 1);

      constSFCYVariable<double> yvel;
      new_dw->get(yvel, lb_->yvelocity, matl, patch, Ghost::AroundCells, 1);

      constSFCZVariable<double> zvel;
      new_dw->get(zvel, lb_->zvelocity, matl, patch, Ghost::AroundCells, 1);

      CCVariable<Stencil7> A;
      new_dw->allocateAndPut(A, pressure_matrix, matl, patch);
      CCVariable<double> rhs;
      new_dw->allocateAndPut(rhs, pressure_rhs, matl, patch);

      // Velocity correction...
      IntVector l(patch->getCellLowIndex());
      IntVector h(patch->getCellHighIndex());
      l -= IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor?1:0,
		     patch->getBCType(Patch::yminus) == Patch::Neighbor?1:0,
		     patch->getBCType(Patch::zminus) == Patch::Neighbor?1:0);

      h += IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor?1:0,
		     patch->getBCType(Patch::yplus) == Patch::Neighbor?1:0,
		     patch->getBCType(Patch::zplus) == Patch::Neighbor?1:0);
      Condition<double>* xbc = bcs.getCondition<double>("xvelocity", Patch::XFaceBased);
      Condition<double>* ybc = bcs.getCondition<double>("yvelocity", Patch::YFaceBased);
      Condition<double>* zbc = bcs.getCondition<double>("zvelocity", Patch::ZFaceBased);
      Condition<double>* pbc = bcs.getCondition<double>("pressure", Patch::CellBased);
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	BCRegion<double>* bc = pbc->get(bctype[idx]);
	switch(bc->getType()){
	case BC::FreeFlow:
	case BC::FixedRate:
	  A[idx].p=0;
	  rhs[idx]=0;
	  if(bc->getType() == BC::FixedRate)
	    rhs[idx] += bc->getValue()*delt;
	  {
	    BCRegion<double>* bc1 = xbc->get(bctype[idx+FW]);
	    switch(bc1->getType()){
	    case BC::FixedValue:
	    case BC::FixedRate:
	    case BC::FreeFlow:
	      {
		BCRegion<double>* bc2 = pbc->get(bctype[idx+W]);
		switch(bc2->getType()){
		case BC::FreeFlow:
		case BC::FixedRate:
		  A[idx].p += 1;
		  A[idx].w = -1;
		  rhs[idx] += xvel[idx+FW];
		  break;
		case BC::FixedValue:
		  A[idx].p += 1;
		  A[idx].w = 0;
		  rhs[idx] += bc2->getValue();
		  break;
		case BC::CoarseGrid:
		  A[idx].p += 1;
		  A[idx].w = 0;
		  rhs[idx] += xvel[idx+FW];
		  //rhs[idx] += boundarypressure[idx+W];
		  break;
		case BC::Exterior:
		  rhs[idx] += xvel[idx+FW];
		  A[idx].w = 0;
		  break;
		case BC::FixedFlux:
		  throw InternalError("unknown pressure bc");
		}
	      }
	      break;
	    case BC::CoarseGrid:
	    case BC::FixedFlux:
	    case BC::Exterior:
	      throw InternalError("Unknown pressureBC");
	    }

	    bc1 = xbc->get(bctype[idx+FE]);
	    switch(bc1->getType()){
	    case BC::FixedValue:
	    case BC::FixedRate:
	    case BC::FreeFlow:
	      {
		BCRegion<double>* bc2 = pbc->get(bctype[idx+E]);
		switch(bc2->getType()){
		case BC::FreeFlow:
		case BC::FixedRate:
		  A[idx].p += 1;
		  A[idx].e = -1;
		  rhs[idx] -= xvel[idx+FE];
		  break;
		case BC::FixedValue:
		  A[idx].p += 1;
		  A[idx].e = 0;
		  rhs[idx] += bc2->getValue();
		  break;
		case BC::CoarseGrid:
		  A[idx].p += 1;
		  A[idx].e = 0;
		  rhs[idx] -= xvel[idx+FE];
		  //rhs[idx] += boundarypressure[idx+E];
		  break;
		case BC::Exterior:
		  rhs[idx] -= xvel[idx+FE];
		  A[idx].e = 0;
		  break;
		case BC::FixedFlux:
		  throw InternalError("unknown pressure bc");
		}
	      }
	      break;
	    case BC::CoarseGrid:
	    case BC::FixedFlux:
	    case BC::Exterior:
	      throw InternalError("Unknown pressureBC");
	    }

	    bc1 = ybc->get(bctype[idx+FS]);
	    switch(bc1->getType()){
	    case BC::FixedValue:
	    case BC::FixedRate:
	    case BC::FreeFlow:
	      {
		BCRegion<double>* bc2 = pbc->get(bctype[idx+S]);
		switch(bc2->getType()){
		case BC::FreeFlow:
		case BC::FixedRate:
		  A[idx].p += 1;
		  A[idx].s = -1;
		  rhs[idx] += yvel[idx+FS];
		  break;
		case BC::FixedValue:
		  A[idx].p += 1;
		  A[idx].s = 0;
		  rhs[idx] += bc2->getValue();
		  break;
		case BC::CoarseGrid:
		  A[idx].p += 1;
		  A[idx].s = 0;
		  rhs[idx] += yvel[idx+FS];
		  //rhs[idx] += boundarypressure[idx+S];
		  break;
		case BC::Exterior:
		  rhs[idx] += yvel[idx+FS];
		  A[idx].s = 0;
		  break;
		case BC::FixedFlux:
		  throw InternalError("unknown pressure bc");
		}
	      }
	      break;
	    case BC::CoarseGrid:
	    case BC::FixedFlux:
	    case BC::Exterior:
	      throw InternalError("Unknown pressureBC");
	    }

	    bc1 = ybc->get(bctype[idx+FN]);
	    switch(bc1->getType()){
	    case BC::FixedValue:
	    case BC::FixedRate:
	    case BC::FreeFlow:
	      {
		BCRegion<double>* bc2 = pbc->get(bctype[idx+N]);
		switch(bc2->getType()){
		case BC::FreeFlow:
		case BC::FixedRate:
		  A[idx].p += 1;
		  A[idx].n = -1;
		  rhs[idx] -= yvel[idx+FN];
		  break;
		case BC::FixedValue:
		  A[idx].p += 1;
		  A[idx].n = 0;
		  rhs[idx] += bc2->getValue();
		  break;
		case BC::CoarseGrid:
		  A[idx].p += 1;
		  A[idx].n = 0;
		  rhs[idx] -= yvel[idx+FN];
		  //rhs[idx] += boundarypressure[idx+N];
		  break;
		case BC::Exterior:
		  rhs[idx] -= yvel[idx+FN];
		  A[idx].n = 0;
		  break;
		case BC::FixedFlux:
		  throw InternalError("unknown pressure bc");
		}
	      }
	      break;
	    case BC::CoarseGrid:
	    case BC::FixedFlux:
	    case BC::Exterior:
	      throw InternalError("Unknown pressureBC");
	    }

	    bc1 = zbc->get(bctype[idx+FB]);
	    switch(bc1->getType()){
	    case BC::FixedValue:
	    case BC::FixedRate:
	    case BC::FreeFlow:
	      {
		BCRegion<double>* bc2 = pbc->get(bctype[idx+B]);
		switch(bc2->getType()){
		case BC::FreeFlow:
		case BC::FixedRate:
		  A[idx].p += 1;
		  A[idx].b = -1;
		  rhs[idx] += zvel[idx+FB];
		  break;
		case BC::FixedValue:
		  A[idx].p += 1;
		  A[idx].b = 0;
		  rhs[idx] += bc2->getValue();
		  break;
		case BC::CoarseGrid:
		  A[idx].p += 1;
		  A[idx].b = 0;
		  rhs[idx] += zvel[idx+FB];
		  //rhs[idx] += boundarypressure[idx+B];
		  break;
		case BC::Exterior:
		  rhs[idx] += zvel[idx+FB];
		  A[idx].b = 0;
		  break;
		case BC::FixedFlux:
		  throw InternalError("unknown pressure bc");
		}
	      }
	      break;
	    case BC::CoarseGrid:
	    case BC::FixedFlux:
	    case BC::Exterior:
	      throw InternalError("Unknown pressureBC");
	    }

	    bc1 = zbc->get(bctype[idx+FT]);
	    switch(bc1->getType()){
	    case BC::FixedValue:
	    case BC::FixedRate:
	    case BC::FreeFlow:
	      {
		BCRegion<double>* bc2 = pbc->get(bctype[idx+T]);
		switch(bc2->getType()){
		case BC::FreeFlow:
		case BC::FixedRate:
		  A[idx].p += 1;
		  A[idx].t = -1;
		  rhs[idx] -= zvel[idx+FT];
		  break;
		case BC::FixedValue:
		  A[idx].p += 1;
		  A[idx].t = 0;
		  rhs[idx] += bc2->getValue();
		  break;
		case BC::CoarseGrid:
		  A[idx].p += 1;
		  A[idx].t = 0;
		  rhs[idx] -= zvel[idx+FT];
		  //rhs[idx] += boundarypressure[idx+T];
		  break;
		case BC::Exterior:
		  rhs[idx] -= zvel[idx+FT];
		  A[idx].t = 0;
		  break;
		case BC::FixedFlux:
		  throw InternalError("unknown pressure bc");
		}
	      }
	      break;
	    case BC::CoarseGrid:
	    case BC::FixedFlux:
	    case BC::Exterior:
	      throw InternalError("Unknown pressureBC");
	    }

	  }
	  break;
	case BC::FixedValue:
	  A[idx].p = 1;
	  A[idx].n = A[idx].s=A[idx].w=A[idx].e=A[idx].t=A[idx].b=0;
	  rhs[idx] = bc->getValue();
	  break;
	case BC::CoarseGrid:
	case BC::Exterior:
	case BC::FixedFlux:
	  throw InternalError("Unknown pressure BC");
          BREAK;
	}
      }

      if(level->getIndex() == 0){
	IntVector pin=level->getCellIndex(pressure_pin_);
	if(patch->containsCell(pin)){
	  rhs[pin] = 0;
	  A[pin].p=1;
	  A[pin].w=0;
	  A[pin].e=0;
	  A[pin].s=0;
	  A[pin].n=0;
	  A[pin].b=0;
	  A[pin].t=0;
	}
	if(patch->containsCell(pin+IntVector(1,0,0)))
	  A[pin+IntVector(1,0,0)].w=0;
	if(patch->containsCell(pin+IntVector(-1,0,0)))
	  A[pin+IntVector(-1,0,0)].e=0;
	if(patch->containsCell(pin+IntVector(0,1,0)))
	  A[pin+IntVector(0,1,0)].s=0;
	if(patch->containsCell(pin+IntVector(0,-1,0)))
	  A[pin+IntVector(0,-1,0)].n=0;
	if(patch->containsCell(pin+IntVector(0,0,1)))
	  A[pin+IntVector(0,0,1)].b=0;
	if(patch->containsCell(pin+IntVector(0,0,-1)))
	  A[pin+IntVector(0,0,-1)].t=0;
      }
    }
  }
}

void SimpleCFD::applyProjection(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw, DataWarehouse* new_dw,
				const VarLabel* pressure)
{
  //const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constNCVariable<int> bctype;
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundNodes, 1);

      SFCXVariable<double> xvel;
      new_dw->getModifiable(xvel, lb_->xvelocity, matl, patch);

      SFCYVariable<double> yvel;
      new_dw->getModifiable(yvel, lb_->yvelocity, matl, patch);

      SFCZVariable<double> zvel;
      new_dw->getModifiable(zvel, lb_->zvelocity, matl, patch);
      
      constCCVariable<double> sol;
      new_dw->get(sol, pressure, matl, patch, Ghost::AroundFaces, 1);

      Condition<double>* xbc = bcs.getCondition<double>("xvelocity", Patch::XFaceBased);
      Condition<double>* ybc = bcs.getCondition<double>("yvelocity", Patch::YFaceBased);
      Condition<double>* zbc = bcs.getCondition<double>("zvelocity", Patch::ZFaceBased);
      Condition<double>* pbc = bcs.getCondition<double>("pressure", Patch::CellBased);

      for(CellIterator iter(patch->getSFCXIterator(0)); !iter.done(); iter++){
	IntVector idx(*iter);
	BCRegion<double>* bc = xbc->get(bctype[idx]);
	switch(bc->getType()){
	case BC::FreeFlow:
	case BC::FixedRate:
	case BC::FixedFlux:
	  {
	    BCRegion<double>* bcr = pbc->get(bctype[idx]);
	    double pr;
	    switch(bcr->getType()){
	    case BC::FreeFlow:
	    case BC::FixedRate:
	    case BC::FixedValue:
	      pr = sol[idx];
	      break;
	    case BC::CoarseGrid:
	      pr = 0;//boundarypressure[idx];
	      break;
	    case BC::Exterior:
	      pr = sol[idx+IntVector(-1,0,0)];
	      break;
	    case BC::FixedFlux:
	      throw InternalError("Unknown bc");
              BREAK;
	    }
	    BCRegion<double>* bcl = pbc->get(bctype[idx+IntVector(-1,0,0)]);
	    double pl;
	    switch(bcl->getType()){
	    case BC::FreeFlow:
	    case BC::FixedRate:
	    case BC::FixedValue:
	      pl = sol[idx+IntVector(-1,0,0)];
	      break;
	    case BC::CoarseGrid:
	      pl = 0;//boundarypressure[idx+IntVector(-1,0,0)];
	      break;
	    case BC::Exterior:
	      pl = sol[idx];
	      break;
	    case BC::FixedFlux:
	      throw InternalError("Unknown bc");
	      BREAK;
	    }
	    double gx = pr-pl;
	    xvel[idx] -= gx;
	  }
	  break;
	case BC::FixedValue:
	  break;
	case BC::CoarseGrid:
	case BC::Exterior:
	  throw InternalError("unknown bc");
	  BREAK;
	}
      }
      for(CellIterator iter(patch->getSFCYIterator(0)); !iter.done(); iter++){
	IntVector idx(*iter);
	BCRegion<double>* bc = ybc->get(bctype[idx]);
	switch(bc->getType()){
	case BC::FreeFlow:
	case BC::FixedRate:
	case BC::FixedFlux:
	  {
	    BCRegion<double>* bcr = pbc->get(bctype[idx]);
	    double pr;
	    switch(bcr->getType()){
	    case BC::FreeFlow:
	    case BC::FixedRate:
	    case BC::FixedValue:
	      pr = sol[idx];
	      break;
	    case BC::CoarseGrid:
	      pr = 0;//boundarypressure[idx];
	      break;
	    case BC::Exterior:
	      pr = sol[idx+IntVector(0,-1,0)];
	      break;
	    case BC::FixedFlux:
	      throw InternalError("Unknown bc");
	      BREAK;
	    }
	    BCRegion<double>* bcl = pbc->get(bctype[idx+IntVector(0,-1,0)]);
	    double pl;
	    switch(bcl->getType()){
	    case BC::FreeFlow:
	    case BC::FixedRate:
	    case BC::FixedValue:
	      pl = sol[idx+IntVector(0,-1,0)];
	      break;
	    case BC::CoarseGrid:
	      pl = 0;//boundarypressure[idx+IntVector(0,-1,0)];
	      break;
	    case BC::Exterior:
	      pl = sol[idx];
	      break;
	    case BC::FixedFlux:
	      throw InternalError("Unknown bc");
	      BREAK;
	    }
	    double gy = pr-pl;
	    yvel[idx] -= gy;
	  }
	  break;
	case BC::FixedValue:
	  break;
	case BC::CoarseGrid:
	case BC::Exterior:
	  throw InternalError("unknown bc");
	  BREAK;
	}
      }
      for(CellIterator iter(patch->getSFCZIterator(0)); !iter.done(); iter++){
	IntVector idx(*iter);
	BCRegion<double>* bc = zbc->get(bctype[idx]);
	switch(bc->getType()){
	case BC::FreeFlow:
	case BC::FixedRate:
	case BC::FixedFlux:
	  {
	    BCRegion<double>* bcr = pbc->get(bctype[idx]);
	    double pr;
	    switch(bcr->getType()){
	    case BC::FreeFlow:
	    case BC::FixedRate:
	    case BC::FixedValue:
	      pr = sol[idx];
	      break;
	    case BC::CoarseGrid:
	      pr = 0;//boundarypressure[idx];
	      break;
	    case BC::Exterior:
	      pr = sol[idx+IntVector(0,0,-1)];
	      break;
	    case BC::FixedFlux:
	      throw InternalError("Unknown bc");
	      BREAK;
	    }
	    BCRegion<double>* bcl = pbc->get(bctype[idx+IntVector(0,0,-1)]);
	    double pl;
	    switch(bcl->getType()){
	    case BC::FreeFlow:
	    case BC::FixedRate:
	    case BC::FixedValue:
	      pl = sol[idx+IntVector(0,0,-1)];
	      break;
	    case BC::CoarseGrid:
	      pl = 0;//boundarypressure[idx+IntVector(0,0,-1)];
	      break;
	    case BC::Exterior:
	      pl = sol[idx];
	      break;
	    case BC::FixedFlux:
	      throw InternalError("Unknown bc");
	      BREAK;
	    }
	    double gz = pr-pl;
	    zvel[idx] -= gz;
	  }
	  break;
	case BC::FixedValue:
	  break;
	case BC::CoarseGrid:
	case BC::Exterior:
	  throw InternalError("unknown bc");
	  BREAK;
	}
      }
#if 0
      print(xvel, yvel, zvel, "Corrected velocity");
#endif
    }
  }
}

void SimpleCFD::advectScalars(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw, DataWarehouse* new_dw,
			      int step, int nsteps)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label(), level);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constNCVariable<int> bctype;
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundNodes, 2);

      constSFCXVariable<double> xvel;
      new_dw->get(xvel, lb_->xvelocity, matl, patch,
		  Ghost::AroundCells, maxadvect_+1);

      constSFCYVariable<double> yvel;
      new_dw->get(yvel, lb_->yvelocity, matl, patch,
		  Ghost::AroundCells, maxadvect_+1);

      constSFCZVariable<double> zvel;
      new_dw->get(zvel, lb_->zvelocity, matl, patch,
		  Ghost::AroundCells, maxadvect_+1);

      {
	constCCVariable<double> den_old;
	old_dw->get(den_old, lb_->density, matl, patch,
		    Ghost::AroundCells, maxadvect_);
	CCVariable<double> den;
	new_dw->allocateAndPut(den, lb_->density, matl, patch);

	if(level->getIndex() > 0){
	  refineBoundaries(patch, xvel.castOffConst(),
			   new_dw, lb_->xvelocity, matl,
			   double(step+1)/double(nsteps));
	  refineBoundaries(patch, yvel.castOffConst(),
			   new_dw, lb_->yvelocity, matl,
			   double(step+1)/double(nsteps));
	  refineBoundaries(patch, zvel.castOffConst(),
			   new_dw, lb_->zvelocity, matl,
			   double(step+1)/double(nsteps));
	  refineBoundaries(patch, den_old.castOffConst(),
			   new_dw, lb_->density, matl,
			   double(step)/double(nsteps));
	}
	advect(den, den_old, patch->getCellIterator(), patch, delT,
	       Vector(0.5, 0.5, 0.5), xvel, yvel, zvel, bctype,
	       bcs.getCondition<double>("density", Patch::CellBased),
	       bcs.getCondition<double>("xvelocity", Patch::XFaceBased),
	       bcs.getCondition<double>("yvelocity", Patch::YFaceBased),
	       bcs.getCondition<double>("zvelocity", Patch::ZFaceBased));
      }
      if(do_thermal){
	constCCVariable<double> temp_old;
	old_dw->get(temp_old, lb_->temperature, matl, patch,
		    Ghost::AroundCells, maxadvect_);
	CCVariable<double> temp;
	new_dw->allocateAndPut(temp, lb_->temperature, matl, patch);

	if(level->getIndex() > 0)
	  refineBoundaries(patch, temp_old.castOffConst(),
			   new_dw, lb_->temperature, matl,
			   double(step)/double(nsteps));
	
	advect(temp, temp_old, patch->getCellIterator(), patch, delT,
	       Vector(0.5, 0.5, 0.5), xvel, yvel, zvel, bctype,
	       bcs.getCondition<double>("temperature", Patch::CellBased),
	       bcs.getCondition<double>("xvelocity", Patch::XFaceBased),
	       bcs.getCondition<double>("yvelocity", Patch::YFaceBased),
	       bcs.getCondition<double>("zvelocity", Patch::ZFaceBased));
      }
    }
  }
}

void SimpleCFD::diffuseScalar(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw, DataWarehouse* new_dw,
			      DiffuseInfo di)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label(), level);
  double delt=delT;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      Vector dx(patch->dCell());
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
      Vector inv_dx2(inv_dx*inv_dx);

      constNCVariable<int> bctype;
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundFaces, 1);

      CCVariable<double> s;
      new_dw->getModifiable(s, di.scalar, matl, patch);

      if(level->getIndex() > 0)
	refineBoundaries(patch, s, new_dw, di.scalar, matl,
			 double(di.step+1)/double(di.nsteps));

      CCVariable<Stencil7> A;
      new_dw->allocateAndPut(A, di.scalar_matrix, matl, patch);
      CCVariable<double> rhs;
      new_dw->allocateAndPut(rhs, di.scalar_rhs, matl, patch);

      // Diffusion
      IntVector l=patch->getGhostCellLowIndex(1);
      IntVector h=patch->getGhostCellHighIndex(1);
      IntVector h2=patch->getCellHighIndex()+IntVector(1,1,1);
      Condition<double>* scalar_bc=bcs.getCondition<double>(di.varname,
							    Patch::CellBased);
      Condition<double>* xflux_bc=bcs.getCondition<double>(di.varname,
							   Patch::XFaceBased);
      Condition<double>* yflux_bc=bcs.getCondition<double>(di.varname,
							   Patch::YFaceBased);
      Condition<double>* zflux_bc=bcs.getCondition<double>(di.varname,
							   Patch::ZFaceBased);
      IntVector FW(0,0,0);
      IntVector FE(1,0,0);
      IntVector FS(0,0,0);
      IntVector FN(0,1,0);
      IntVector FB(0,0,0);
      IntVector FT(0,0,1);
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	applybc(idx, l, h, h2, s, delt, inv_dx2, di.rate,
		bctype, scalar_bc, xflux_bc, yflux_bc, zflux_bc,
		FW, FE, FS, FN, FB, FT, A, rhs);
      }
    }
  }
}

void SimpleCFD::dissipateScalars(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label(), level);
  double delt=delT;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constNCVariable<int> bctype;
      // This should probably be aroundNodes?
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::None, 0);

      CCVariable<double> den;
      new_dw->getModifiable(den, lb_->density, matl, patch);

      double factor = 1./(1+delt*density_dissipation_);
      Condition<double>* den_bc = bcs.getCondition<double>("density", Patch::CellBased);
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	BCRegion<double>* bc = den_bc->get(bctype[idx]);
	switch(bc->getType()){
	case BC::FreeFlow:
	case BC::FixedRate:
	  den[idx]*=factor;	
	  break;
	case BC::FixedValue:
	case BC::FixedFlux:
	  break;
	case BC::CoarseGrid:
	case BC::Exterior:
	  // Shouldn't happen
	  throw InternalError("illegalBC");
	  BREAK;
	}
      }
#if 0
      print(den, "dissipated density");
#endif
    }
  }
}

void SimpleCFD::updatebcs(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw, DataWarehouse* new_dw)
{
#if 1
  if(dbg.active()){
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
  
	constCCVariable<double> den;
	new_dw->get(den, lb_->density, matl, patch, Ghost::None, 0);
	cerr << "Level: " << getLevel(patches)->getIndex() << '\n';
	print(den, "density");
	if(do_thermal){
	  constCCVariable<double> temp;
	  new_dw->get(temp, lb_->temperature, matl, patch, Ghost::None, 0);
	  cerr << "Level: " << getLevel(patches)->getIndex() << '\n';
	  print(temp, "temperature");
	}
	constCCVariable<double> pressure;
	new_dw->get(pressure, lb_->pressure, matl, patch, Ghost::None, 0);
	print(pressure, "pressure");
	constSFCXVariable<double> xvel;
	new_dw->get(xvel, lb_->xvelocity, matl, patch, Ghost::None, 0);
	print(xvel, "xvelocity");
	constSFCYVariable<double> yvel;
	new_dw->get(yvel, lb_->yvelocity, matl, patch, Ghost::None, 0);
	print(yvel, "yvelocity");
	constSFCZVariable<double> zvel;
	new_dw->get(zvel, lb_->zvelocity, matl, patch, Ghost::None, 0);
	print(zvel, "zvelocity");
      }
    }
  }
#endif
#if 0
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
  
      constCCVariable<double> den;
      new_dw->get(den, lb_->density, matl, patch, Ghost::None, 0);
      double max=0;
      double sum=0;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	sum+=den[*iter];
	max = Max(den[*iter], max);
      }
      cerr << "Level: " << getLevel(patches)->getIndex() << ", max=" << max << ", sum=" << sum << '\n';
    }
  }
#endif
  new_dw->transferFrom(old_dw, lb_->bctype, patches, matls);
}

void SimpleCFD::interpolateVelocities(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse*,
				      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constSFCXVariable<double> xvel;
      new_dw->get(xvel, lb_->xvelocity, matl, patch, Ghost::AroundCells, 1);

      constSFCYVariable<double> yvel;
      new_dw->get(yvel, lb_->yvelocity, matl, patch, Ghost::AroundCells, 1);

      constSFCZVariable<double> zvel;
      new_dw->get(zvel, lb_->zvelocity, matl, patch, Ghost::AroundCells, 1);
      
      CCVariable<Vector> vel;
      new_dw->allocateAndPut(vel, lb_->ccvelocity, matl, patch);

      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	Vector v(xvel[idx]+xvel[idx+IntVector(1,0,0)],
		 yvel[idx]+yvel[idx+IntVector(0,1,0)],
		 zvel[idx]+zvel[idx+IntVector(0,0,1)]);
	vel[idx]=v*0.5;
      }
    }
  }
}

void
SimpleCFD::addRefineDependencies( Task* /*task*/, 
				  const VarLabel* /*label*/,
				  int /*step*/, int /*nsteps*/ )
{
}

void
SimpleCFD::refineBoundaries(const Patch* /*patch*/,
			    CCVariable<double>& /*val*/,
			    DataWarehouse* /*new_dw*/,
			    const VarLabel* /*label*/,
			    int /*matl*/,
			    double /*factor*/)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}

void
SimpleCFD::refineBoundaries(const Patch* /*patch*/,
			    SFCXVariable<double>& /*val*/,
			    DataWarehouse* /*new_dw*/,
			    const VarLabel* /*label*/,
			    int /*matl*/,
			    double /*factor*/)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}

void
SimpleCFD::refineBoundaries(const Patch* /*patch*/,
			    SFCYVariable<double>& /*val*/,
			    DataWarehouse* /*new_dw*/,
			    const VarLabel* /*label*/,
			    int /*matl*/,
			    double /*factor*/)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}

void
SimpleCFD::refineBoundaries(const Patch* /*patch*/,
			    SFCZVariable<double>& /*val*/,
			    DataWarehouse* /*new_dw*/,
			    const VarLabel* /*label*/,
			    int /*matl*/,
			    double /*factor*/)
{
  throw InternalError("trying to do AMR iwth the non-AMR component!");
}



void
SimpleCFD::hackbcs(const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset* matls,
		   DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      NCVariable<int> bctype;
      new_dw->getModifiable(bctype, lb_->bctype, matl, patch);

      for(int p2=0;p2<patches->size();p2++){
	const Patch* patch2 = patches->get(p2);
	if(patch2 == patch)
	  continue;
	NCVariable<int> bctype2;
	new_dw->getModifiable(bctype2, lb_->bctype, matl, patch2);
	IntVector l = Max(bctype.getLowIndex(), bctype2.getLowIndex());
	IntVector h = Min(bctype.getHighIndex(), bctype2.getHighIndex());
	IntVector diff = h-l;
	int total = diff.x()*diff.y()*diff.z();
	if(diff.x() > 0 && diff.y() > 0 && diff.z() > 0)
	  cerr << "Hacking " << total << " bcs on patch " << *patch << '\n';
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  bcs.merge(bctype[*iter], bctype2[*iter]);
	}
      }
    }
  }
}
