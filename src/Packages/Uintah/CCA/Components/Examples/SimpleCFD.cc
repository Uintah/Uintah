// TODO
// - Parallel
// - driven cavity
// - Turbulence model?
// - velcor on ICs/internal boundaries?
// - Separate solver
// - 3D
// Correct face bcs for advection...
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
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <values.h>
#include <iomanip>

using namespace Uintah;
using namespace std;

template<class T>
static void print(const Array3<T>& d, const char* name, int pre=0)
{
  cerr << name << ":\n";
  IntVector l(d.getLowIndex());
  IntVector h(d.getHighIndex());
  for(int j=h.y()-1;j>=l.y();j--){
    for(int i=0;i<pre;i++)
      cerr << ' ';
    for(int i=l.x();i<h.x();i++){
      cerr << setw(13) << d[IntVector(i,j,0)];
    }
    cerr << '\n';
  }
}

static void print(const Array3<Stencil7>& A, const char* name)
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
}

SimpleCFD::SimpleCFD(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  lb_ = scinew ExamplesLabel();
}

SimpleCFD::~SimpleCFD()
{
  delete lb_;
}

void SimpleCFD::problemSetup(const ProblemSpecP& params, GridP& grid,
			 SimulationStateP& sharedState)
{
  sharedState_ = sharedState;
  ProblemSpecP cfd = params->findBlock("SimpleCFD");
  cfd->require("delt_multiplier", delt_multiplier_);
  cfd->require("viscosity", viscosity_);
  cfd->require("viscosity_tolerance", viscosity_tolerance_);
  cfd->require("diffusion", density_diffusion_);
  cfd->require("diffusion_tolerance", density_diffusion_tolerance_);
  cfd->require("dissipation", density_dissipation_);
  Point pin_continuous;
  cfd->require("correction_pin", pin_continuous);
  pin_=grid->getLevel(0)->getCellIndex(pin_continuous);
  cfd->require("correction_tolerance", correction_tolerance_);
  cfd->require("advection_tolerance", advection_tolerance_);
  mymat_ = new SimpleMaterial();
  sharedState->registerSimpleMaterial(mymat_);

  regiondb.problemSetup(cfd, grid);
  ics.setupCondition<double>("density", IntVector(0,0,0));
  ics.setupCondition<double>("xvelocity", IntVector(1,0,0));
  ics.setupCondition<double>("yvelocity", IntVector(0,1,0));
  ics.setupCondition<double>("zvelocity", IntVector(0,0,1));
  ics.problemSetup(cfd, regiondb);

  bcs.setupCondition<double>("density", IntVector(0,0,0));
  bcs.setupCondition<double>("xvelocity", IntVector(1,0,0));
  bcs.setupCondition<double>("yvelocity", IntVector(0,1,0));
  bcs.setupCondition<double>("zvelocity", IntVector(0,0,1));
  bcs.problemSetup(cfd, regiondb);
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
  Task* task;
  task = scinew Task("advectVelocity", this, &SimpleCFD::advectVelocity);
  task->requires(Task::OldDW, sharedState_->get_delt_label());
  task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
  task->requires(Task::OldDW, lb_->xvelocity, Ghost::None, 0);
  task->requires(Task::OldDW, lb_->yvelocity, Ghost::None, 0);
  task->requires(Task::OldDW, lb_->zvelocity, Ghost::None, 0);
  task->computes(lb_->xvelocity);
  task->computes(lb_->yvelocity);
  task->computes(lb_->zvelocity);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  if(viscosity_ > 0){
    task = scinew Task("applyViscosity", this, &SimpleCFD::applyViscosity);
    task->requires(Task::OldDW, sharedState_->get_delt_label());
    task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
    task->requires(Task::NewDW, lb_->xvelocity, Ghost::None, 0);
    task->requires(Task::NewDW, lb_->yvelocity, Ghost::None, 0);
    task->requires(Task::NewDW, lb_->zvelocity, Ghost::None, 0);
    task->modifies(lb_->xvelocity);
    task->modifies(lb_->yvelocity);
    task->modifies(lb_->zvelocity);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  }

  task = scinew Task("projectVelocity", this, &SimpleCFD::projectVelocity);
  task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::None, 0);
  task->modifies(lb_->xvelocity);
  task->modifies(lb_->yvelocity);
  task->modifies(lb_->zvelocity);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  task = scinew Task("advectScalars", this, &SimpleCFD::advectScalars);
  task->requires(Task::OldDW, sharedState_->get_delt_label());
  task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->xvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->yvelocity, Ghost::None, 0);
  task->requires(Task::NewDW, lb_->zvelocity, Ghost::None, 0);
  task->requires(Task::OldDW, lb_->density, Ghost::None, 0);
  task->computes(lb_->density);
  sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());

  if(density_diffusion_ > 0){
    task = scinew Task("diffuseScalars", this, &SimpleCFD::diffuseScalars);
    task->requires(Task::OldDW, sharedState_->get_delt_label());
    task->requires(Task::OldDW, lb_->bctype, Ghost::None, 0);
    task->requires(Task::NewDW, lb_->density, Ghost::None, 0);
    task->modifies(lb_->density);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
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
	delt=Min(delt, t_inv.maxComponent());
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

      // Create bcs....
      NCVariable<int> bctype;
      new_dw->allocateAndPut(bctype, lb_->bctype, matl, patch);
      bcs.set(bctype, patch);
      print(xvel, yvel, zvel, "Initial velocity");
      print(den, "Initial density");
      print(bctype, "bctype");
    }
  }
}

template<class ArrayType>
bool Interpolate(typename ArrayType::value_type& value, const Vector& v,
		 const Vector& offset, const Vector& inv_dx,
		 ArrayType& field)
{
  Vector pos(v*inv_dx-offset);
  int ix = RoundDown(pos.x());
  int iy = RoundDown(pos.y());
  double fx = pos.x()-ix;
  double fy = pos.y()-iy;
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
  if(ix >= l.x() && iy >= l.y() && ix+1 < h.x() && iy+1 < h.y()){
    // Center
    value = field[IntVector(ix, iy, 0)]*(1-fx)*(1-fy)
      + field[IntVector(ix+1, iy, 0)]*fx*(1-fy)
      + field[IntVector(ix, iy+1, 0)]*(1-fx)*fy
      + field[IntVector(ix+1, iy+1, 0)]*fx*fy;
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
      q[idx] = b->getValue();
      break;
    case BC::FixedFlux:
      throw InternalError("Do not know how to do this...");
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
		  Ghost::AroundCells, 1);
      SFCXVariable<double> xvel;
      new_dw->allocateAndPut(xvel, lb_->xvelocity, matl, patch);

      constSFCYVariable<double> yvel_old;
      old_dw->get(yvel_old, lb_->yvelocity, matl, patch,
		  Ghost::AroundCells, 1);
      SFCYVariable<double> yvel;
      new_dw->allocateAndPut(yvel, lb_->yvelocity, matl, patch);

      constSFCZVariable<double> zvel_old;
      old_dw->get(zvel_old, lb_->zvelocity, matl, patch,
		  Ghost::AroundCells, 1);
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

static inline bool inside(const IntVector& i, const IntVector& l,
			  const IntVector& h)
{
  return i.x() >= l.x() && i.y() >= l.y() && i.z() >= l.z()
    && i.x() < h.x() && i.y() < h.y() && i.z() < h.z();
}

void SimpleCFD::applybc(const IntVector& idx, const IntVector& l,
			const IntVector& h, const IntVector& h2,
			Array3<double>& field, double delt,
			const Vector& inv_dx2, double diff,
			constNCVariable<int>& bctype,
			Condition<double>* scalar_bc,
			Condition<double>* xface_bc,
			Condition<double>* yface_bc,
			Condition<double>* zface_bc,
			const IntVector& FN, const IntVector& FS,
			const IntVector& FW, const IntVector& FE,
			Array3<Stencil7>& A, Array3<double>& rhs)
{
  BCRegion<double>* bc = scalar_bc->get(bctype[idx]);
  IntVector N(0,1,0);
  IntVector S(0,-1,0);
  IntVector W(-1,0,0);
  IntVector E(1,0,0);
  switch(bc->getType()){
  case BC::FreeFlow:
  case BC::FixedRate:
    A[idx].p=1;
    rhs[idx]=field[idx];
    if(bc->getType() == BC::FixedRate)
      rhs[idx] += bc->getValue()*delt;
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
	A[idx].p+=2*diff*delt*inv_dx2.x();
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	A[idx].n = 0;
	break;
      case BC::FixedFlux:
	rhs[idx] += 2*bc1->getValue()*(diff*delt*inv_dx2.y());
	A[idx].n = 0;
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
    break;
  case BC::FixedValue:
    A[idx].p = 1;
    A[idx].n = A[idx].s=A[idx].w=A[idx].e=0;
    rhs[idx] = bc->getValue();
    break;
  case BC::FixedFlux:
    throw InternalError("unknown BC");
    //break;
  }
}

static void Mult(Array3<double>& B, const Array3<Stencil7>& A,
	  const Array3<double>& X,
	  CellIterator iter)
{
  IntVector l(iter.begin());
  IntVector h1(iter.end()-IntVector(1,1,1));
  // Center
  for(; !iter.done(); iter++){
    IntVector idx = *iter;
    double result = A[idx].p*X[idx];
    if(idx.x() > l.x())
      result += A[idx].w*X[idx+IntVector(-1,0,0)];
    if(idx.x() < h1.x())
      result += A[idx].e*X[idx+IntVector(1,0,0)];
    if(idx.y() > l.y())
      result += A[idx].s*X[idx+IntVector(0,-1,0)];
    if(idx.y() < h1.y())
      result += A[idx].n*X[idx+IntVector(0,1,0)];
    B[idx] = result;
  }
}

static void Sub(Array3<double>& r, const Array3<double>& a, const Array3<double>& b,
		CellIterator iter)
{
  for(; !iter.done(); iter++)
    r[*iter] = a[*iter]-b[*iter];
}

static void DivDiagonal(Array3<double>& r, const Array3<double>& a, const Array3<Stencil7>& A,
		CellIterator iter)
{
  for(; !iter.done(); iter++)
    r[*iter] = a[*iter]/A[*iter].p;
}

static double Dot(const Array3<double>& a, const Array3<double>& b,
		CellIterator iter)
{
  double sum=0;
  for(; !iter.done(); iter++)
    sum += a[*iter]*b[*iter];
  return sum;
}

static void ScMult_Add(Array3<double>& r, double s, const Array3<double>& a, const Array3<double>& b,
		CellIterator iter)
{
  for(; !iter.done(); iter++)
    r[*iter] = s*a[*iter]+b[*iter];
}

static int cg(const CellIterator& iter,
	      const Array3<Stencil7>& A, Array3<double>& X,
	      const Array3<double>& B, Array3<double>& tmp_R,
	      Array3<double>& tmp_Q,
	      Array3<double>& tmp_D, double tolerance)
{
  // R = A*X
  Mult(tmp_R, A, X, iter);
  // R = B-R
  Sub(tmp_R, B, tmp_R, iter);
  // D = R/Ap
  DivDiagonal(tmp_D, tmp_R, A, iter);

  double dnew=Dot(tmp_R, tmp_D, iter);
  double d0=dnew;
  if(d0 < -tolerance*tolerance){
    cerr << "Not positive definite???\n";
  }
  if(d0 < tolerance*tolerance)
    return 0;

  int niter=0;
  int toomany=0;
  IntVector diff(iter.end()-iter.begin());
  int size = diff.x()*diff.y()*diff.z();
  if(toomany == 0)
    toomany=2*size;

  while(niter < toomany){
    if(dnew < tolerance*tolerance*d0)
      break;
    niter++;

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    // Q = A*D
    Mult(tmp_Q, A, tmp_D, iter);

    double aden=Dot(tmp_D, tmp_Q, iter);
    
    double a=dnew/aden;
    // X = a*D+X
    ScMult_Add(X, a, tmp_D, X, iter);
    // R = -a*Q+R
    ScMult_Add(tmp_R, -a, tmp_Q, tmp_R, iter);

    // Simple Preconditioning...
    DivDiagonal(tmp_Q, tmp_R, A, iter);

    // Calculate coefficient bk and direction vectors p and pp
    double dold=dnew;
    dnew=Dot(tmp_Q, tmp_R, iter);

    double b=dnew/dold;
    // D = b*D+Q
    ScMult_Add(tmp_D, b, tmp_D, tmp_Q, iter);

    //double err=sqrt(dnew/d0);
  }
  if(niter >= toomany)
    return -1;
  else
    return niter;
}

void SimpleCFD::applyViscosity(const ProcessorGroup*,
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

      Vector dx(patch->dCell());
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
      Vector inv_dx2(inv_dx*inv_dx);

      constNCVariable<int> bctype;
      // This should probably be aroundNodes?
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundCells, 1);

      SFCXVariable<double> xvel;
      new_dw->getModifiable(xvel, lb_->xvelocity, matl, patch);

      SFCYVariable<double> yvel;
      new_dw->getModifiable(yvel, lb_->yvelocity, matl, patch);

      SFCZVariable<double> zvel;
      new_dw->getModifiable(zvel, lb_->zvelocity, matl, patch);
      
      NCVariable<Stencil7> A;
      new_dw->allocateTemporary(A,  patch);
      NCVariable<double> rhs;
      new_dw->allocateTemporary(rhs, patch);
      NCVariable<double> tmp1;
      new_dw->allocateTemporary(tmp1, patch);
      NCVariable<double> tmp2;
      new_dw->allocateTemporary(tmp2, patch);
      NCVariable<double> tmp3;
      new_dw->allocateTemporary(tmp3, patch);

      int niter;
      IntVector l, h, FN, FS, FW, FE;
      IntVector h2=patch->getNodeHighIndex();
      // Viscosity
      l=patch->getSFCXLowIndex();
      h=patch->getSFCXHighIndex();
      Condition<double>* xvel_bc = bcs.getCondition<double>("xvelocity", Patch::XFaceBased);
      Condition<double>* xvel_bc_yface = bcs.getCondition<double>("xvelocity", Patch::YFaceBased);
      Condition<double>* xvel_bc_zface = bcs.getCondition<double>("xvelocity", Patch::ZFaceBased);
      Condition<double>* xvel_bc_cc = bcs.getCondition<double>("xvelocity", Patch::CellBased);

      FN=IntVector(0,1,0);
      FS=IntVector(0,0,0);
      FW=IntVector(-1,0,0);
      FE=IntVector(0,0,0);
      for(CellIterator iter(patch->getSFCXIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	applybc(idx, l, h, h2, xvel, delt, inv_dx2, viscosity_,
		bctype, xvel_bc, xvel_bc_cc, xvel_bc_yface, xvel_bc_zface,
		FN, FS, FW, FE,	A, rhs);
      }
      niter=cg(patch->getSFCXIterator(), A, xvel, rhs, tmp1, tmp2, tmp3,
	       viscosity_tolerance_);
      cerr << "X viscosity solved in " << niter << " iterations\n";

      l=patch->getSFCYLowIndex();
      h=patch->getSFCYHighIndex();
      Condition<double>* yvel_bc = bcs.getCondition<double>("yvelocity", Patch::YFaceBased);
      Condition<double>* yvel_bc_xface = bcs.getCondition<double>("yvelocity", Patch::XFaceBased);
      Condition<double>* yvel_bc_zface = bcs.getCondition<double>("yvelocity", Patch::ZFaceBased);
      Condition<double>* yvel_bc_cc = bcs.getCondition<double>("yvelocity", Patch::CellBased);

      FN=IntVector(0,0,0);
      FS=IntVector(0,-1,0);
      FW=IntVector(0,0,0);
      FE=IntVector(1,0,0);
      A.initialize(Stencil7());
      for(CellIterator iter(patch->getSFCYIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	applybc(idx, l, h, h2, yvel, delt, inv_dx2, viscosity_,
		bctype, yvel_bc, yvel_bc_xface, yvel_bc_cc, yvel_bc_zface,
		FN, FS, FW, FE,	A, rhs);
      }
      niter=cg(patch->getSFCYIterator(), A, yvel, rhs, tmp1, tmp2, tmp3,
	       viscosity_tolerance_);
      cerr << "Y viscosity solved in " << niter << " iterations\n";

      l=patch->getSFCZLowIndex();
      h=patch->getSFCZHighIndex();
      Condition<double>* zvel_bc = bcs.getCondition<double>("zvelocity", Patch::ZFaceBased);
      Condition<double>* zvel_bc_xface = bcs.getCondition<double>("zvelocity", Patch::XFaceBased);
      Condition<double>* zvel_bc_yface = bcs.getCondition<double>("zvelocity", Patch::YFaceBased);
      Condition<double>* zvel_bc_cc = bcs.getCondition<double>("zvelocity", Patch::CellBased);

      FN=IntVector(0,1,0);
      FS=IntVector(0,0,0);
      FW=IntVector(0,0,0);
      FE=IntVector(1,0,0);
      for(CellIterator iter(patch->getSFCZIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	applybc(idx, l, h, h2, zvel, delt, inv_dx2, viscosity_,
		bctype, zvel_bc, zvel_bc_xface, zvel_bc_yface, zvel_bc_cc,
		FN, FS, FW, FE,	A, rhs);
      }
      niter=cg(patch->getSFCZIterator(), A, zvel, rhs, tmp1, tmp2, tmp3,
	       viscosity_tolerance_);
      cerr << "Z viscosity solved in " << niter << " iterations\n";
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

      constNCVariable<int> bctype;
      // This should probably be aroundNodes?
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundCells, 1);

      SFCXVariable<double> xvel;
      new_dw->getModifiable(xvel, lb_->xvelocity, matl, patch);

      SFCYVariable<double> yvel;
      new_dw->getModifiable(yvel, lb_->yvelocity, matl, patch);

      SFCZVariable<double> zvel;
      new_dw->getModifiable(zvel, lb_->zvelocity, matl, patch);
      
      CCVariable<Stencil7> A;
      new_dw->allocateTemporary(A,  patch);
      CCVariable<double> rhs;
      new_dw->allocateTemporary(rhs, patch);
      CCVariable<double> sol;
      new_dw->allocateTemporary(sol, patch);
      CCVariable<double> tmp1;
      new_dw->allocateTemporary(tmp1, patch);
      CCVariable<double> tmp2;
      new_dw->allocateTemporary(tmp2, patch);
      CCVariable<double> tmp3;
      new_dw->allocateTemporary(tmp3, patch);

      // Velocity correction...
      IntVector l(patch->getCellLowIndex());
      IntVector h(patch->getCellHighIndex());
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	A[idx].p = 4;
	A[idx].w = -1;
	A[idx].e = -1;
	A[idx].s = -1;
	A[idx].n = -1;
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
	rhs[idx] = -(gx+gy);
      }
      ASSERT(inside(pin_, l, h));
      rhs[pin_] = 0;
      if(inside(pin_+IntVector(1,0,0), l, h))
	A[pin_+IntVector(1,0,0)].w=0;
      if(inside(pin_+IntVector(-1,0,0), l, h))
	A[pin_+IntVector(-1,0,0)].e=0;
      if(inside(pin_+IntVector(0,1,0), l, h))
      A[pin_+IntVector(0,1,0)].s=0;
      if(inside(pin_+IntVector(0,-1,0), l, h))
	A[pin_+IntVector(0,-1,0)].n=0;
      A[pin_].p=1;
      A[pin_].w=0;
      A[pin_].e=0;
      A[pin_].s=0;
      A[pin_].n=0;
      A[pin_].w=0;
      A[pin_].s=0;
      sol.initialize(0);
      int niter=cg(patch->getCellIterator(), A, sol, rhs, tmp1, tmp2, tmp3, 
		   correction_tolerance_);
      cerr << "Correction solved in " << niter << " iterations\n";
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	if(idx.x()>l.x()){
	  double gx = sol[idx]-sol[idx+IntVector(-1,0,0)];
	  xvel[idx] -= gx;
	}
	if(idx.y()>l.y()){
	  double gy = sol[idx]-sol[idx+IntVector(0,-1,0)];
	  yvel[idx] -= gy;
	}
	if(idx.z()>l.z()){
	  double gz = sol[idx]-sol[idx+IntVector(0,0,-1)];
	  zvel[idx] -= gz;
	}
      }
      print(xvel, yvel, zvel, "Corrected velocity");
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
		  Ghost::AroundCells, 1);

      constSFCYVariable<double> yvel;
      new_dw->get(yvel, lb_->yvelocity, matl, patch,
		  Ghost::AroundCells, 1);

      constSFCZVariable<double> zvel;
      new_dw->get(zvel, lb_->zvelocity, matl, patch,
		  Ghost::AroundCells, 1);

      constCCVariable<double> den_old;
      old_dw->get(den_old, lb_->density, matl, patch,
		  Ghost::AroundCells, 1);
      CCVariable<double> den;
      new_dw->allocateAndPut(den, lb_->density, matl, patch);

      advect(den, den_old, patch->getCellIterator(), patch, delT,
	     Vector(0.5, 0.5, 0.5), xvel, yvel, zvel, bctype,
	     bcs.getCondition<double>("density", Patch::CellBased));
    }
  }      
}

void SimpleCFD::diffuseScalars(const ProcessorGroup*,
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

      Vector dx(patch->dCell());
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
      Vector inv_dx2(inv_dx*inv_dx);

      constNCVariable<int> bctype;
      // This should probably be aroundNodes?
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundCells, 1);

      CCVariable<double> den;
      new_dw->getModifiable(den, lb_->density, matl, patch);

      CCVariable<Stencil7> A;
      new_dw->allocateTemporary(A,  patch);
      CCVariable<double> rhs;
      new_dw->allocateTemporary(rhs, patch);
      CCVariable<double> tmp1;
      new_dw->allocateTemporary(tmp1, patch);
      CCVariable<double> tmp2;
      new_dw->allocateTemporary(tmp2, patch);
      CCVariable<double> tmp3;
      new_dw->allocateTemporary(tmp3, patch);

      // Diffusion
      IntVector l=patch->getCellLowIndex();
      IntVector h=patch->getCellHighIndex();
      IntVector h2=patch->getNodeHighIndex();
      Condition<double>* den_bc = bcs.getCondition<double>("density", Patch::CellBased);
      Condition<double>* xflux_bc = bcs.getCondition<double>("density", Patch::XFaceBased);
      Condition<double>* yflux_bc = bcs.getCondition<double>("density", Patch::YFaceBased);
      Condition<double>* zflux_bc = bcs.getCondition<double>("density", Patch::ZFaceBased);
      IntVector FN(0,1,0);
      IntVector FS(0,0,0);
      IntVector FW(0,0,0);
      IntVector FE(1,0,0);
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
	IntVector idx(*iter);
	applybc(idx, l, h, h2, den, delt, inv_dx2, density_diffusion_,
		bctype, den_bc, xflux_bc, yflux_bc, zflux_bc, FN, FS, FW, FE,
		A, rhs);
      }
      int niter = cg(patch->getCellIterator(), A, den, rhs, tmp1, tmp2, tmp3,
		     density_diffusion_tolerance_);
      cerr << "Diffusion solved in " << niter << " iterations\n";
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
      old_dw->get(bctype, lb_->bctype, matl, patch, Ghost::AroundCells, 1);

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
      print(den, "dissipated density");
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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {
  static MPI_Datatype makeMPI_Stencil7()
  {
    ASSERTEQ(sizeof(Stencil7), sizeof(double)*7);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 7, 7, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(Stencil7*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "Stencil7", true, 
				  &makeMPI_Stencil7);
    }
    return td;
  }
  
}

namespace SCIRun {

void swapbytes( Stencil7& a) {
  SWAP_8(a.p);
  SWAP_8(a.e);
  SWAP_8(a.w);
  SWAP_8(a.n);
  SWAP_8(a.s);
  SWAP_8(a.t);
  SWAP_8(a.b);
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1209
#endif


} // namespace SCIRun
