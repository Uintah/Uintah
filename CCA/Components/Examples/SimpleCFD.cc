// TODO
// Max timestep increase/decrease
// Precompute 1./diagonal in CG???
// My malloc is misaligned???
// Vorticity confinement
// Flop counting for CGSolver
// Periodic boundaries
// Periodic version of Rayleigh-Taylor
// Inital timestep in ups file
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
#include <values.h>
#include <iomanip>

using namespace Uintah;
using namespace std;

#if 0
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
  Point pin_continuous;
  cfd->require("pressure_pin", pin_continuous);
  cfd->get("keep_pressure", keep_pressure);
  cfd->get("old_initial_guess", old_initial_guess);
  cfd->get("do_thermal", do_thermal);
  if(!grid->getLevel(0)->containsPoint(pin_continuous))
    throw ProblemSetupException("velocity pressure pin point is not with the domain");
  pin_=grid->getLevel(0)->getCellIndex(pin_continuous);

  cfd->require("advection_tolerance", advection_tolerance_);
  cfd->require("buoyancy", buoyancy_);
  cfd->require("thermal_conduction", thermal_conduction_);
  if(!cfd->get("vorticity_confinement_scale", vorticity_confinement_scale))
    vorticity_confinement_scale=0;
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
}
 
void SimpleCFD::scheduleComputeStableTimestep(const LevelP& level,
					  SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
			   this, &SimpleCFD::computeStableTimestep);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::None, 0);
  task->computes(sharedState_->get_delt_label());
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
}

void SimpleCFD::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched)
{
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver)
    throw InternalError("SimpleCFD needs a solver component to work");

  Task* task;
  task = scinew Task("advectVelocity", this, &SimpleCFD::advectVelocity);
  task->requires(Task::OldDW, sharedState_->get_delt_label());
  task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
  task->requires(Task::OldDW, lb_->xvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::OldDW, lb_->yvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::OldDW, lb_->zvelocity, Ghost::AroundCells, maxadvect_+1);
  task->computes(lb_->xvelocity);
  task->computes(lb_->yvelocity);
  task->computes(lb_->zvelocity);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  task = scinew Task("applyForces", this, &SimpleCFD::applyForces);
  if(sharedState_->getGravity().length() > 0 || buoyancy_ > 0){
    task->requires(Task::OldDW, sharedState_->get_delt_label());
    task->requires(Task::OldDW, lb_->density, Ghost::AroundFaces, 1);
    task->modifies(lb_->xvelocity);
    task->modifies(lb_->yvelocity);
    task->modifies(lb_->zvelocity);
    if(do_thermal)
      task->requires(Task::OldDW, lb_->temperature, Ghost::AroundFaces, 1);
  }
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  for(int dir=0;dir<3;dir++){
    task = scinew Task("applyViscosity", this, &SimpleCFD::applyViscosity, dir);
    task->requires(Task::OldDW, sharedState_->get_delt_label());
    task->requires(Task::OldDW, lb_->bctype, Ghost::AroundFaces, 1);
    if(dir == 0){
      task->requires(Task::NewDW, lb_->xvelocity, Ghost::None, 0);
      task->computes(lb_->xvelocity_matrix);
      task->computes(lb_->xvelocity_rhs);
    } else if(dir == 1){
      task->requires(Task::NewDW, lb_->yvelocity, Ghost::None, 0);
      task->computes(lb_->yvelocity_matrix);
      task->computes(lb_->yvelocity_rhs);
    } else if(dir ==2){
      task->requires(Task::NewDW, lb_->zvelocity, Ghost::None, 0);
      task->computes(lb_->zvelocity_matrix);
      task->computes(lb_->zvelocity_rhs);
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

  task = scinew Task("projectVelocity", this, &SimpleCFD::projectVelocity);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::AroundCells, 1);
  task->computes(lb_->pressure_matrix);
  task->computes(lb_->pressure_rhs);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  solver->scheduleSolve(level, sched, sharedState_->allMaterials(),
			lb_->pressure_matrix, lb_->pressure, false,
			lb_->pressure_rhs, keep_pressure?lb_->pressure:0,
			Task::OldDW, pressure_params_);

  task = scinew Task("applyProjection", this, &SimpleCFD::applyProjection);
  task->requires(Task::NewDW, lb_->pressure, Ghost::AroundFaces, 1);
  task->modifies(lb_->xvelocity);
  task->modifies(lb_->yvelocity);
  task->modifies(lb_->zvelocity);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  task = scinew Task("advectScalars", this, &SimpleCFD::advectScalars);
  task->requires(Task::OldDW, sharedState_->get_delt_label());
  task->requires(Task::OldDW, lb_->bctype, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::AroundCells, maxadvect_+1);
  task->requires(Task::OldDW, lb_->density, Ghost::AroundCells, maxadvect_);
  task->computes(lb_->density);
  if(do_thermal){
    task->requires(Task::OldDW, lb_->temperature, Ghost::AroundCells, maxadvect_);
    task->computes(lb_->temperature);
  }
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  scheduleDiffuseScalar(sched, level, "density",
			lb_->density, lb_->density_matrix,
			lb_->density_rhs, density_diffusion_,
			solver, diffusion_params_);
  if(do_thermal){
    scheduleDiffuseScalar(sched, level, "temperature", lb_->temperature,
			  lb_->temperature_matrix, lb_->temperature_rhs,
			  thermal_conduction_, solver, conduction_params_);
  }

  if(density_dissipation_ > 0) {
    task = scinew Task("dissipateScalars", this, &SimpleCFD::dissipateScalars);
    task->requires(Task::OldDW, sharedState_->get_delt_label());
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
				      const SolverParameters* solverparams)
{
  string taskname = "diffuseScalar: "+name;
  Task* task = scinew Task(taskname, this, &SimpleCFD::diffuseScalar,
			   name, scalar, scalar_matrix, scalar_rhs, rate);
  task->requires(Task::OldDW, sharedState_->get_delt_label());
  task->requires(Task::OldDW, lb_->bctype, Ghost::AroundFaces, 1);
  task->requires(Task::NewDW, scalar, Ghost::None, 0);
  task->computes(scalar_matrix);
  task->computes(scalar_rhs);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  solver->scheduleSolve(level, sched, sharedState_->allMaterials(),
			scalar_matrix, scalar, true,
			scalar_rhs, scalar,
			old_initial_guess?Task::OldDW:Task::NewDW,
			solverparams);
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
	maxx = Max(maxx, xvel[*iter]);
      constSFCYVariable<double> yvel;
      new_dw->get(yvel, lb_->yvelocity, matl, patch, Ghost::None, 0);
      double maxy=0;
      for(CellIterator iter = patch->getSFCYIterator(); !iter.done(); iter++)
	maxy = Max(maxy, yvel[*iter]);
      constSFCZVariable<double> zvel;
      new_dw->get(zvel, lb_->zvelocity, matl, patch, Ghost::None, 0);
      double maxz=0;
      for(CellIterator iter = patch->getSFCZIterator(); !iter.done(); iter++)
	maxz = Max(maxz, zvel[*iter]);
      Vector t_inv(Vector(maxx, maxy, maxz)/patch->dCell());
      if(t_inv.maxComponent() > 0)
	delt=Min(delt, 1./t_inv.maxComponent());
    }
  }
  if(delt != MAXDOUBLE)
    delt *= delt_multiplier_;
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
#if 0
      print(xvel, yvel, zvel, "Initial velocity");
      print(den, "Initial density");
      print(bctype, "bctype");
#endif
    }
  }
}

template<class ArrayType>
bool Interpolate(typename ArrayType::value_type& value, const Vector& v,
		 const Vector& offset, const Vector& inv_dx,
		 const ArrayType& field)
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
  if(ix == l.x()-1 && fx >= 0.5){
    ix=l.x();
    fx=0;
  } else if(ix+1 == h.x() && fx <= 0.5){
    ix=h.x()-2;
    fx=1;
  }
  if(iy == l.y()-1 && fy >= 0.5){
    iy=l.y();
    fy=0;
  } else if(iy+1 == h.y() && fy <= 0.5){
    iy=h.y()-2;
    fy=1;
  }
  if(iz == l.z()-1 && fz >= 0.5){
    iz=l.z();
    fz=0;
  } else if(iz+1 == h.z() && fz <= 0.5){
    iz=h.z()-2;
    fz=1;
  }
  if(ix >= l.x() && iy >= l.y() && iz >= l.z()
     && ix+1 < h.x() && iy+1 < h.y() && iz+1 < h.z()){
    // Center
    value = field[IntVector(ix, iy, iz)]*(1-fx)*(1-fy)*(1-fz)
      + field[IntVector(ix+1, iy, iz)]*fx*(1-fy)*(1-fz)
      + field[IntVector(ix, iy+1, iz)]*(1-fx)*fy*(1-fz)
      + field[IntVector(ix+1, iy+1, iz)]*fx*fy*(1-fz)
      + field[IntVector(ix, iy, iz+1)]*(1-fx)*(1-fy)*fz
      + field[IntVector(ix+1, iy, iz+1)]*fx*(1-fy)*fz
      + field[IntVector(ix, iy+1, iz+1)]*(1-fx)*fy*fz
      + field[IntVector(ix+1, iy+1, iz+1)]*fx*fy*fz;
    return true;
  } else {
    // Outside
    return false;
  }
}

static bool particleTrace(Vector& p, int nsteps, double delt,
			  const Vector& inv_dx,
			  constSFCXVariable<double>& xvel,
			  constSFCYVariable<double>& yvel,
			  constSFCZVariable<double>& zvel)
{
  for(int i=0;i<nsteps;i++){
    double x;
    if(!Interpolate(x, p, Vector(0, 0.5, 0.5), inv_dx, xvel))
      return false;
    double y;
    if(!Interpolate(y, p, Vector(0.5, 0, 0.5), inv_dx, yvel))
      return false;
    double z;
    if(!Interpolate(z, p, Vector(0.5, 0.5, 0), inv_dx, zvel))
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
			  double tolerance)
{
  int nsteps=1;
  Vector p1 = p;
  bool success1 = particleTrace(p1, nsteps, delt/nsteps, inv_dx,
				xvel, yvel, zvel);
  double tolerance2 = tolerance*tolerance;
  int maxsteps = 128;
  for(;;) {
    nsteps<<=1;
    if(nsteps > maxsteps)
      break;
    Vector p2 = p;
    bool success2 = particleTrace(p2, nsteps, delt/nsteps, inv_dx, xvel, yvel, zvel);
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
		       Condition<double>* cbc)
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
	if(particleTrace(v, delt, inv_dx, xvel, yvel, zvel, advection_tolerance_)){
	  double value;
	  if(Interpolate(value, v, offset, inv_dx, qold)){
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
    }
  }
}

void SimpleCFD::advectVelocity(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label() );
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constNCVariable<int> bctype;
      // This should probably be aroundNodes?
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundCells, 1);

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

      advect(xvel, xvel_old, patch->getSFCXIterator(), patch, delT,
	     Vector(0, 0.5, 0.5), xvel_old, yvel_old, zvel_old, bctype,
	     bcs.getCondition<double>("xvelocity", Patch::XFaceBased));
      advect(yvel, yvel_old, patch->getSFCYIterator(), patch, delT,
	     Vector(0.5, 0, 0.5), xvel_old, yvel_old, zvel_old, bctype,
	     bcs.getCondition<double>("yvelocity", Patch::YFaceBased));
      advect(zvel, zvel_old, patch->getSFCZIterator(), patch, delT,
	     Vector(0.5, 0.5, 0), xvel_old, yvel_old, zvel_old, bctype,
	     bcs.getCondition<double>("zvelocity", Patch::ZFaceBased));
    }
  }
}

void SimpleCFD::applyForces(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw, DataWarehouse* new_dw)
{
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
      
      delt_vartype delT;
      old_dw->get(delT, sharedState_->get_delt_label() );
      Vector grav = sharedState_->getGravity();
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


static inline bool inside(const IntVector& i, const IntVector& l,
			  const IntVector& h)
{
  return i.x() >= l.x() && i.y() >= l.y() && i.z() >= l.z()
    && i.x() < h.x() && i.y() < h.y() && i.z() < h.z();
}

void SimpleCFD::applybc(const IntVector& idx, const IntVector& l,
			const IntVector& h, const IntVector& h2,
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
    if(inside(idx+FW, l,h2)){
      BCRegion<double>* bc1 = xface_bc->get(bctype[idx+FW]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	// fall through
      case BC::FreeFlow:
	if(inside(idx+W, l, h)){
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
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	    //break;
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
      }
    }
    if(inside(idx+FE, l,h2)){
      BCRegion<double>* bc1 = xface_bc->get(bctype[idx+FE]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.x());
	// fall through
      case BC::FreeFlow:
	if(inside(idx+E, l, h)){
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
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	    //break;
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
      }
    }
    if(inside(idx+FS, l,h2)){
      BCRegion<double>* bc1 = yface_bc->get(bctype[idx+FS]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	// fall through
      case BC::FreeFlow:
	if(inside(idx+S, l, h)){
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
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	    //break;
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
      }
    }
    if(inside(idx+FN, l,h2)){
      BCRegion<double>* bc1 = yface_bc->get(bctype[idx+FN]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	// fall through
      case BC::FreeFlow:
	if(inside(idx+N, l, h)){
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
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	    //break;
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
      }
    }
    if(inside(idx+FB, l,h2)){
      BCRegion<double>* bc1 = zface_bc->get(bctype[idx+FB]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	// fall through
      case BC::FreeFlow:
	if(inside(idx+B, l, h)){
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
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	    //break;
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
      }
    }
    if(inside(idx+FT, l,h2)){
      BCRegion<double>* bc1 = zface_bc->get(bctype[idx+FT]);
      switch(bc1->getType()){
      case BC::FixedRate:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.z());
	// fall through
      case BC::FreeFlow:
	if(inside(idx+T, l, h)){
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
	  case BC::FixedFlux:
	    throw InternalError("unknown BC");
	    //break;
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
      }
    }
    break;
  case BC::FixedValue:
    A[idx].p = 1;
    A[idx].n = A[idx].s=A[idx].w=A[idx].e=A[idx].t=A[idx].b=0;
    rhs[idx] = bc->getValue();
    break;
  case BC::FixedFlux:
    throw InternalError("unknown BC");
    //break;
  }
}

void SimpleCFD::applyViscosity(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw,
			       int dir)
{
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label() );
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
				DataWarehouse* old_dw, DataWarehouse* new_dw)
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
      
      CCVariable<Stencil7> A;
      new_dw->allocateAndPut(A, lb_->pressure_matrix, matl, patch);
      CCVariable<double> rhs;
      new_dw->allocateAndPut(rhs, lb_->pressure_rhs, matl, patch);

      // Velocity correction...
      IntVector l(patch->getCellLowIndex());
      IntVector h(patch->getCellHighIndex());
      if(patch->getBCType(Patch::xminus) == Patch::Neighbor)
	l -= IntVector(1,0,0);
      if(patch->getBCType(Patch::xplus) == Patch::Neighbor)
	h += IntVector(1,0,0);
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	A[idx].p = 6;
	A[idx].w = -1;
	A[idx].e = -1;
	A[idx].s = -1;
	A[idx].n = -1;
	A[idx].b = -1;
	A[idx].t = -1;
#define OUT -9999999
	if(idx.x() == l.x()) {
	  A[idx].p -=1;
	} else if(idx.x() == h.x()-1){
	  A[idx].p -=1;
	}
	double gx = xvel[idx+IntVector(1,0,0)]-xvel[idx];
	if(idx.y() == l.y()) {
	  A[idx].p -=1;
	} else if(idx.y() == h.y()-1){
	  A[idx].p -=1;
	}
	double gy = yvel[idx+IntVector(0,1,0)]-yvel[idx];
	if(idx.z() == l.z()) {
	  A[idx].p -=1;
	} else if(idx.z() == h.z()-1){
	  A[idx].p -=1;
	}
	double gz = zvel[idx+IntVector(0,0,1)]-zvel[idx];
	rhs[idx] = -(gx+gy+gz);
      }
      if(patch->containsCell(pin_)){
	rhs[pin_] = 0;
	A[pin_].p=1;
	A[pin_].w=0;
	A[pin_].e=0;
	A[pin_].s=0;
	A[pin_].n=0;
	A[pin_].b=0;
	A[pin_].t=0;
      }
      if(patch->containsCell(pin_+IntVector(1,0,0)))
	A[pin_+IntVector(1,0,0)].w=0;
      if(patch->containsCell(pin_+IntVector(-1,0,0)))
	A[pin_+IntVector(-1,0,0)].e=0;
      if(patch->containsCell(pin_+IntVector(0,1,0)))
	A[pin_+IntVector(0,1,0)].s=0;
      if(patch->containsCell(pin_+IntVector(0,-1,0)))
	A[pin_+IntVector(0,-1,0)].n=0;
      if(patch->containsCell(pin_+IntVector(0,0,1)))
	A[pin_+IntVector(0,0,1)].s=0;
      if(patch->containsCell(pin_+IntVector(0,0,-1)))
	A[pin_+IntVector(0,0,-1)].n=0;
    }
  }
}

void SimpleCFD::applyProjection(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* old_dw, DataWarehouse* new_dw)
{
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
      
      constCCVariable<double> sol;
      new_dw->get(sol, lb_->pressure, matl, patch, Ghost::AroundFaces, 1);

      for(CellIterator iter(patch->getSFCXIterator(1)); !iter.done(); iter++){
	IntVector idx(*iter);
	double gx = sol[idx]-sol[idx+IntVector(-1,0,0)];
	xvel[idx] -= gx;
      }
      for(CellIterator iter(patch->getSFCYIterator(1)); !iter.done(); iter++){
	IntVector idx(*iter);
	double gy = sol[idx]-sol[idx+IntVector(0,-1,0)];
	yvel[idx] -= gy;
      }
      for(CellIterator iter(patch->getSFCZIterator(1)); !iter.done(); iter++){
	IntVector idx(*iter);
	double gz = sol[idx]-sol[idx+IntVector(0,0,-1)];
	zvel[idx] -= gz;
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
			      DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label() );
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      constNCVariable<int> bctype;
      // This should probably be aroundNodes?
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundCells, 1);

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

	advect(den, den_old, patch->getCellIterator(), patch, delT,
	       Vector(0.5, 0.5, 0.5), xvel, yvel, zvel, bctype,
	       bcs.getCondition<double>("density", Patch::CellBased));
      }
      if(do_thermal){
	constCCVariable<double> temp_old;
	old_dw->get(temp_old, lb_->temperature, matl, patch,
		    Ghost::AroundCells, maxadvect_);
	CCVariable<double> temp;
	new_dw->allocateAndPut(temp, lb_->temperature, matl, patch);
	
	advect(temp, temp_old, patch->getCellIterator(), patch, delT,
	       Vector(0.5, 0.5, 0.5), xvel, yvel, zvel, bctype,
	       bcs.getCondition<double>("temperature", Patch::CellBased));
      }
    }
  }
}

void SimpleCFD::diffuseScalar(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw, DataWarehouse* new_dw,
			      string varname, const VarLabel* scalar,
			      const VarLabel* scalar_matrix,
			      const VarLabel* scalar_rhs,
			      double rate)
{
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label() );
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
      new_dw->getModifiable(s, scalar, matl, patch);

      CCVariable<Stencil7> A;
      new_dw->allocateAndPut(A, scalar_matrix, matl, patch);
      CCVariable<double> rhs;
      new_dw->allocateAndPut(rhs, scalar_rhs, matl, patch);

      // Diffusion
      IntVector l=patch->getGhostCellLowIndex(1);
      IntVector h=patch->getGhostCellHighIndex(1);
      IntVector h2=patch->getCellHighIndex()+IntVector(1,1,1);
      Condition<double>* scalar_bc = bcs.getCondition<double>(varname, Patch::CellBased);
      Condition<double>* xflux_bc = bcs.getCondition<double>(varname, Patch::XFaceBased);
      Condition<double>* yflux_bc = bcs.getCondition<double>(varname, Patch::YFaceBased);
      Condition<double>* zflux_bc = bcs.getCondition<double>(varname, Patch::ZFaceBased);
      IntVector FW(0,0,0);
      IntVector FE(1,0,0);
      IntVector FS(0,0,0);
      IntVector FN(0,1,0);
      IntVector FB(0,0,0);
      IntVector FT(0,0,1);
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	applybc(idx, l, h, h2, s, delt, inv_dx2, rate,
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
  delt_vartype delT;
  old_dw->get(delT, sharedState_->get_delt_label() );
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
  new_dw->transferFrom(old_dw, lb_->bctype, patches, matls);
}

void SimpleCFD::interpolateVelocities(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw,
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
