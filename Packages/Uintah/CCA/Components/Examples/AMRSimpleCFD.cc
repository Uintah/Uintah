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
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("SimpleCFD_doing", false);

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
//______________________________________________________________________
//
void AMRSimpleCFD::problemSetup(const ProblemSpecP& params, GridP& grid,
				SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup  \t\t\t AMRSimpleCFD" << '\n';
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
//______________________________________________________________________
//
void AMRSimpleCFD::scheduleInitialize(const LevelP& level,
				      SchedulerP& sched)
{
  cout_doing << "AMRSimpleCFD::scheduleInitialize on level " << level->getIndex() << '\n';
  SimpleCFD::scheduleInitialize(level, sched);
  if(keep_pressure){
    Task* task = scinew Task("initialize",
			     this, &AMRSimpleCFD::initialize);
    task->computes(lb_->pressure2);
    sched->addTask(task, level->eachPatch(), sharedState_->allMaterials());
  }
}
//______________________________________________________________________
//
void AMRSimpleCFD::initialize(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
{
  cout_doing << "Doing initialize  \t\t\t\t AMRSimpleCFD" << '\n';

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
//______________________________________________________________________
//
void AMRSimpleCFD::scheduleRefineInterface(const LevelP& /*fineLevel*/,
					   SchedulerP& /*sched*/,
					   int /*step*/, int /*nsteps*/)
{
  cout_doing << "AMRSimpleCFD::scheduleRefineInterface not implemented.\n";
}
//______________________________________________________________________
//
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                // This turns off SGI compiler warning.
template<class ArrayType, class constArrayType>
void refineFaces(const Patch* patch, 
                 const Level* level,
		 const Level* coarseLevel, 
                 const IntVector& dir,
		 Patch::FaceType lowFace, 
                 Patch::FaceType highFace,
		 ArrayType& xvel, 
                 const VarLabel* label,
		 double subCycleProgress_var, 
                 int matl, 
                 DataWarehouse* coarse_old_dw,
		 DataWarehouse* coarse_new_dw, 
                 Patch::VariableBasis basis)
{
  //  cout << "RANDY: AMRSimpleCFD::refineFaces() BGN" << endl;
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    if(patch->getBCType(face) != Patch::Coarse)
      continue;

    {
     //__________________________________
     //  determine low and high cell iter limits
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
      
      if(face != Patch::xminus && 
         face != Patch::xplus && patch->getBCType(Patch::xminus) == Patch::None){
	l+=IntVector(1,0,0);
      }
      if(face != Patch::xminus && 
         face != Patch::xplus && patch->getBCType(Patch::xplus) == Patch::None){
	h-=IntVector(1,0,0);
      }
      if(face != Patch::yminus && 
         face != Patch::yplus && patch->getBCType(Patch::yminus) == Patch::None){
	l+=IntVector(0,1,0);
      }
      if(face != Patch::yminus && 
         face != Patch::yplus && patch->getBCType(Patch::yplus) == Patch::None){
	h-=IntVector(0,1,0);
      }
      if(face != Patch::zminus && 
         face != Patch::zplus && patch->getBCType(Patch::zminus) == Patch::None){
	l+=IntVector(0,0,1);
      }
      if(face != Patch::zminus && 
         face != Patch::zplus && patch->getBCType(Patch::zplus) == Patch::None){
	h-=IntVector(0,0,1);
      }
      
      //__________________________________
      //  determine low and high coarse level
      //  iteration limits.
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
      default:
	break;
      }  // switch face
      
      
      l = Max(l, xvel.getLowIndex());
      h = Min(h, xvel.getHighIndex());
      //__________________________________
      //   subCycleProgress_var  = 0
      if(subCycleProgress_var < 1.e-10){
	constArrayType xvel0;
	coarse_old_dw->getRegion(xvel0, label, matl, coarseLevel,
				 coarseLow, coarseHigh);
                             
	for(CellIterator iter(l,h); !iter.done(); iter++){
	  IntVector idx = *iter;
         //_________________
         //  deterimine the interpolation weights
	  Vector w;
	  IntVector cidx = level->interpolateToCoarser(idx, dir, w);
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
	  default:
	    break;
	  }
         //_________________
         //  interpolation, using the coarse old_DW
	  double x0 = xvel0[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
	    + xvel0[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
	    + xvel0[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
	    + xvel0[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
	    + xvel0[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
	    + xvel0[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
	    + xvel0[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
	    + xvel0[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
	  xvel[idx] = x0;
	}  // cell iterator
      } else if(subCycleProgress_var > 1-1.e-10){        /// subCycleProgress_var near 1.0
	constArrayType xvel1;
	coarse_new_dw->getRegion(xvel1, label, matl, coarseLevel,
				 coarseLow, coarseHigh);
                             
	for(CellIterator iter(l,h); !iter.done(); iter++){
	  IntVector idx = *iter;
         //_________________
         //  deterimine the interpolation weights
	  Vector w;      
	  IntVector cidx = level->interpolateToCoarser(idx, dir, w);
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
	  default:
	    break;
	  }
         //_________________
         //  interpolation using the coarse_new_dw data
	  double x1 = xvel1[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
	    + xvel1[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
	    + xvel1[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
	    + xvel1[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
	    + xvel1[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
	    + xvel1[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
	  xvel[idx] = x1;
	}  // cell iterator
      } else {                    // subCycleProgress_var neither 0 or 1 
	constArrayType xvel0;
       constArrayType xvel1;
	coarse_old_dw->getRegion(xvel0, label, matl, coarseLevel,
				 coarseLow, coarseHigh);
	coarse_new_dw->getRegion(xvel1, label, matl, coarseLevel,
				 coarseLow, coarseHigh);

	for(CellIterator iter(l,h); !iter.done(); iter++){
	  IntVector idx = *iter;
         //_________________
         //  deterimine the interpolation weights
	  Vector w;
	  IntVector cidx = level->interpolateToCoarser(idx, dir, w);
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
	  default:
	    break;
	  }
         //_________________
         //  interpolation from both coarse new and old dw
         // coarse_old_dw data
	  double x0 = xvel0[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())	    
           + xvel0[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
	    + xvel0[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
	    + xvel0[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
	    + xvel0[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
	    + xvel0[cidx+IntVector(1,0,1)]*   w.x( )*(1-w.y())*   w.z()
	    + xvel0[cidx+IntVector(0,1,1)]*(1-w.x())*    w.y()*   w.z()
	    + xvel0[cidx+IntVector(1,1,1)]*   w.x() *    w.y()*   w.z();
          
          // coarse_new_dw data 
	  double x1 = xvel1[cidx+IntVector(0,0,0)]*(1-w.x())*(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(1,0,0)]*   w.x() *(1-w.y())*(1-w.z())
	    + xvel1[cidx+IntVector(0,1,0)]*(1-w.x())*   w.y() *(1-w.z())
	    + xvel1[cidx+IntVector(1,1,0)]*   w.x() *   w.y() *(1-w.z())
	    + xvel1[cidx+IntVector(0,0,1)]*(1-w.x())*(1-w.y())*   w.z()
	    + xvel1[cidx+IntVector(1,0,1)]*   w.x() *(1-w.y())*   w.z()
	    + xvel1[cidx+IntVector(0,1,1)]*(1-w.x())*   w.y() *   w.z()
	    + xvel1[cidx+IntVector(1,1,1)]*   w.x() *   w.y() *   w.z();
           
         // Interpolate temporally  
	  double x = (1-subCycleProgress_var)*x0 + subCycleProgress_var*x1;
	  xvel[idx] = x;
	}
      }
    }
  }
  //  cout << "RANDY: AMRSimpleCFD::refineFaces() END" << endl;
}
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif
//______________________________________________________________________
//
void AMRSimpleCFD::addRefineDependencies(Task* task, 
                                         const VarLabel* var,
					      int step, 
                                         int nsteps)
{
  cout_doing << "Doing addRefineDependencies  \t\t\t AMRSimpleCFD" 
             << " step " << step << " nsteps " << nsteps <<'\n';
  ASSERTRANGE(step, 0, nsteps+1);

  Ghost::GhostType gc = Ghost::AroundCells;
  /*
  TypeDescription::Type type = var->typeDescription()->getType();
  if (type == TypeDescription::SFCXVariable)
    gc = Ghost::AroundFacesX;
  else if (type == TypeDescription::SFCYVariable)
    gc = Ghost::AroundFacesY;
  else if (type == TypeDescription::SFCZVariable)
    gc = Ghost::AroundFacesZ;
  */
  if(step != nsteps)
    task->requires(Task::CoarseOldDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain, gc, 1);
  if(step != 0)
    task->requires(Task::CoarseNewDW, var,
		   0, Task::CoarseLevel, 0, Task::NormalDomain, gc, 1);
}
//______________________________________________________________________
//  CCVariable<double> version
void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				    CCVariable<double>& val,
				    DataWarehouse* new_dw,
				    const VarLabel* label,
				    int matl,
				    double subCycleProgress_var)
{
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(CC) BGN" << endl;
  cout_doing << "Doing refineBoundaries <double>  \t\t AMRSimpleCFD" << '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<CCVariable<double>, constCCVariable<double> >
    (patch, level, coarseLevel, IntVector(0,0,0), Patch::invalidFace,
     Patch::invalidFace, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::CellBased);
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(CC) END" << endl;
}
//______________________________________________________________________
//  SFCXVariable version
void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				        SFCXVariable<double>& val,
				        DataWarehouse* new_dw,
				        const VarLabel* label,
				        int matl,
				        double subCycleProgress_var)
{
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(SFCX) BGN" << endl;
  cout_doing << "Doing refineBoundaries <SFCXVariable> \t\t AMRSimpleCFD" << '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<SFCXVariable<double>, constSFCXVariable<double> >
    (patch, level, coarseLevel, IntVector(1,0,0), Patch::xminus,
     Patch::xplus, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::XFaceBased);
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(SFCX) END" << endl;
}
//______________________________________________________________________
//  SFCYVariable version
void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				        SFCYVariable<double>& val,
				        DataWarehouse* new_dw,
				        const VarLabel* label,
				        int matl,
				        double subCycleProgress_var)
{
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(SFCY) BGN" << endl;
  cout_doing << "Doing refineBoundaries <SFCYVariable> \t\t AMRSimpleCFD" << '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<SFCYVariable<double>, constSFCYVariable<double> >
    (patch, level, coarseLevel, IntVector(0,1,0), Patch::yminus,
     Patch::yplus, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::YFaceBased);
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(SFCY) END" << endl;
}
//______________________________________________________________________
//  SFCZVariable version
void AMRSimpleCFD::refineBoundaries(const Patch* patch,
				        SFCZVariable<double>& val,
				        DataWarehouse* new_dw,
				        const VarLabel* label,
				        int matl,
				        double subCycleProgress_var)
{
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(SFCZ) BGN" << endl;
  cout_doing << "Doing refineBoundaries <SFCZVariable> \t\t AMRSimpleCFD" << '\n';
  DataWarehouse* coarse_old_dw = new_dw->getOtherDataWarehouse(Task::CoarseOldDW);
  DataWarehouse* coarse_new_dw = new_dw->getOtherDataWarehouse(Task::CoarseNewDW);
  
  const Level* level = patch->getLevel();
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  
  refineFaces<SFCZVariable<double>, constSFCZVariable<double> >
    (patch, level, coarseLevel, IntVector(0,0,1), Patch::zminus,
     Patch::zplus, val, label, subCycleProgress_var, matl,
     coarse_old_dw, coarse_new_dw, Patch::ZFaceBased);
  //  cout << "RANDY: AMRSimpleCFD::refineBoundaries(SFCZ) END" << endl;
}
//______________________________________________________________________
//
void AMRSimpleCFD::scheduleCoarsen(const LevelP& coarseLevel,
				   SchedulerP& sched)
{
  Ghost::GhostType  gn = Ghost::None; 
  cout_doing << "AMRSimpleCFD::scheduleCoarsen on level " << coarseLevel->getIndex() << '\n';
  Task* task = scinew Task("coarsen",
			   this, &AMRSimpleCFD::coarsen);
  task->requires(Task::NewDW, lb_->density,
		 0, Task::FineLevel, 0, Task::NormalDomain, gn, 0);

  if(do_thermal){
    task->requires(Task::NewDW, lb_->temperature,
		   0, Task::FineLevel, 0, Task::NormalDomain,gn, 0);
    task->modifies(lb_->temperature);
  }

  // give these ghost cells to refine the faces
  task->requires(Task::NewDW, lb_->xvelocity,
		 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::AroundFacesX, 10);

  task->requires(Task::NewDW, lb_->yvelocity,
		 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::AroundFacesY, 10);

  task->requires(Task::NewDW, lb_->zvelocity,
		 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::AroundFacesZ, 10);

  task->modifies(lb_->density);
  task->modifies(lb_->xvelocity);
  task->modifies(lb_->yvelocity);
  task->modifies(lb_->zvelocity);  
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());

  
  //__________________________________
  // Re-solve/apply the pressure, using the pressure2 variable
  // you need to clean up the pressure field after you coarsen so that
  // you get a divergence free velocity field
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver)
    throw InternalError("SimpleCFD needs a solver component to work", __FILE__, __LINE__);
  schedulePressureSolve(coarseLevel, sched, solver, lb_->pressure2,
			lb_->pressure2_matrix, lb_->pressure2_rhs, false);
  
  releasePort("solver");
}
//______________________________________________________________________
//
void AMRSimpleCFD::scheduleRefine(const PatchSet* patches,
				   SchedulerP& sched)
{
  const PatchSet ps; // Have to declare this dummy var (which is not used) to get
                     // around a compiler bug with the '<<' operator 2 lines below.

  Ghost::GhostType  gn = Ghost::None; 
  cout_doing << "AMRSimpleCFD::scheduleRefine on patches " << *patches << '\n';
  Task* task = scinew Task("refine",
			   this, &AMRSimpleCFD::refine);
  task->requires(Task::NewDW, lb_->density,
		 0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);

  if(do_thermal){
    task->requires(Task::NewDW, lb_->temperature,
		   0, Task::CoarseLevel, 0, Task::NormalDomain,gn, 0);
    //task->computes(lb_->temperature);
  }

  task->requires(Task::NewDW, lb_->xvelocity,
		 0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);

  task->requires(Task::NewDW, lb_->yvelocity,
		 0, Task::CoarseLevel, 0, Task::NormalDomain, gn, 0);

  task->requires(Task::NewDW, lb_->zvelocity,
		 0, Task::CoarseLevel, 0, Task::NormalDomain,gn, 0);

  //task->requires(Task::NewDW, lb_->pressure2,
  //		 0, Task::CoarseLevel, 0, Task::NormalDomain,gn, 0);

  //task->computes(lb_->density);
  //task->computes(lb_->xvelocity);
  //task->computes(lb_->yvelocity);
  //task->computes(lb_->zvelocity);
  //  task->computes(lb_->pressure2);

  sched->addTask(task, patches, sharedState_->allMaterials());
  /*
  //__________________________________
  // Re-solve/apply the pressure, using the pressure2 variable
  // you need to clean up the pressure field after you coarsen so that
  // you get a divergence free velocity field
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver)
    throw InternalError("SimpleCFD needs a solver component to work");
  schedulePressureSolve(fineLevel, sched, solver, lb_->pressure2,
			lb_->pressure2_matrix, lb_->pressure2_rhs);
  releasePort("solver");
  */
}
//______________________________________________________________________
//  This is only used to flag indicating which cells need to be refined.
//  Nothing is done with this.
void AMRSimpleCFD::scheduleErrorEstimate(const LevelP& coarseLevel,
					 SchedulerP& sched)
{
  cout_doing << "AMRSimpleCFD::scheduleErrorEstimate on level " << coarseLevel->getIndex() << '\n';
  // Estimate error - this should probably be in it's own schedule,
  // and the simulation controller should not schedule it every time step
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task* task = scinew Task("errorEstimate", this, &AMRSimpleCFD::errorEstimate, false);
  task->requires(Task::NewDW, lb_->density,     gac, 1);
  task->requires(Task::NewDW, lb_->temperature, gac, 1);
  task->requires(Task::NewDW, lb_->pressure,    gac, 1);
  task->requires(Task::NewDW, lb_->density,     gac, 1);
  task->requires(Task::NewDW, lb_->ccvelocity,  gac, 1);
  
  task->modifies(sharedState_->get_refineFlag_label(), sharedState_->refineFlagMaterials());
  task->modifies(sharedState_->get_refinePatchFlag_label(), sharedState_->refineFlagMaterials());
  task->computes(lb_->density_gradient_mag);
  task->computes(lb_->temperature_gradient_mag);
  task->computes(lb_->pressure_gradient_mag);
  task->computes(lb_->ccvorticitymag);
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//  This is only used to flag indicating which cells need to be refined.
//  Nothing is done with this.
void AMRSimpleCFD::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                                SchedulerP& sched)
{
  cout_doing << "AMRSimpleCFD::scheduleInitialErrorEstimate on level " << coarseLevel->getIndex() << '\n';
  // Estimate error - this should probably be in it's own schedule,
  // and the simulation controller should not schedule it every time step
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task* task = scinew Task("errorEstimate", this, &AMRSimpleCFD::errorEstimate, true);
  task->requires(Task::NewDW, lb_->temperature, gac, 1);
  task->requires(Task::NewDW, lb_->pressure,    gac, 1);
  task->requires(Task::NewDW, lb_->density,     gac, 1);
  task->requires(Task::NewDW, lb_->ccvelocity,  gac, 1);
  
  task->modifies(sharedState_->get_refineFlag_label(), sharedState_->refineFlagMaterials());
  task->modifies(sharedState_->get_refinePatchFlag_label(), sharedState_->refineFlagMaterials());
  task->computes(lb_->temperature_gradient_mag);
  task->computes(lb_->pressure_gradient_mag);
  task->computes(lb_->ccvorticitymag);
  sched->addTask(task, coarseLevel->eachPatch(), sharedState_->allMaterials());
}
//______________________________________________________________________
//
void AMRSimpleCFD::coarsen(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw)
{
  cout_doing << "Doing coarsen \t\t\t AMRSimpleCFD" << '\n';
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  IntVector rr(fineLevel->getRefinementRatio());
  double ratio = 1./(rr.x()*rr.y()*rr.z());
  
  for(int p=0;p<patches->size();p++){  
    const Patch* coarsePatch = patches->get(p);
    cout_doing << "\t\t on patch " << coarsePatch->getID();
    // Find the overlapping regions...
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      //__________________________________
      //   D E N S I T Y
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
	  double rho_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
         
	  for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
	      !inside.done(); inside++){
	    rho_tmp+=fine_den[fineStart+*inside];
	  }
	  density[*iter]=rho_tmp*ratio;
	}
      }  // fine patch loop
      //print(density, "coarsened density");
      //__________________________________
      //      T E M P E R A T U R E
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
	    double temp_tmp=0;
	    IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	    
           for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
		!inside.done(); inside++){
	      temp_tmp+=fine_temp[fineStart+*inside];
	    }
	    temp[*iter]=temp_tmp*ratio;
	  }
	}
	//print(temp, "coarsened temperature");
      }  // do thermal
      
      //__________________________________
      //      X V E L
      SFCXVariable<double> xvel;
      new_dw->getModifiable(xvel, lb_->xvelocity, matl, coarsePatch);
      
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
	
       SFCXVariable<double> fine_xvel;
	new_dw->getCopy(fine_xvel, lb_->xvelocity, matl, finePatch,
			Ghost::AroundFacesX, 1);
                     
// 	refineFaces<SFCXVariable<double>, constSFCXVariable<double> >
// 	  (finePatch, fineLevel, coarseLevel, IntVector(1,0,0),
// 	   Patch::xminus, Patch::xplus, fine_xvel,
// 	   lb_->xvelocity, 1.0, matl, old_dw, new_dw, Patch::XFaceBased);
          
	IntVector fl(finePatch->getSFCXLowIndex());
	IntVector fh(finePatch->getSFCXHighIndex());
	IntVector l(fineLevel->mapCellToCoarser(fl));
	IntVector h(fineLevel->mapCellToCoarser(fh+IntVector(rr.x()-1, 0, 0)));
	l = Max(l, coarsePatch->getSFCXLowIndex());
	h = Min(h, coarsePatch->getSFCXHighIndex());
       
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  double xvel_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(-1,0,0), IntVector(0,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    xvel_tmp+=fine_xvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(1,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    xvel_tmp+=fine_xvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(1,0,0), IntVector(2,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    xvel_tmp +=fine_xvel[fineStart+*inside]*0.5;
	  }
	  xvel[*iter]=xvel_tmp*ratio;
	}
      }
      
      //print(xvel, "coarsened xvel");
      //__________________________________
      //     Y V E L
      SFCYVariable<double> yvel;
      new_dw->getModifiable(yvel, lb_->yvelocity, matl, coarsePatch);
      
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
       
	SFCYVariable<double> fine_yvel;
	new_dw->getCopy(fine_yvel, lb_->yvelocity, matl, finePatch,
			Ghost::AroundFacesY, 1);
                     
// 	refineFaces<SFCYVariable<double>, constSFCYVariable<double> >
// 	  (finePatch, fineLevel, coarseLevel, IntVector(0,1,0),
// 	   Patch::yminus, Patch::yplus, fine_yvel,
// 	   lb_->yvelocity, 1.0, matl, old_dw, new_dw, Patch::YFaceBased);
          
	IntVector fl(finePatch->getSFCYLowIndex());
	IntVector fh(finePatch->getSFCYHighIndex());
	IntVector l(fineLevel->mapCellToCoarser(fl));
	IntVector h(fineLevel->mapCellToCoarser(fh+IntVector(0, rr.y()-1, 0)));
	l = Max(l, coarsePatch->getSFCYLowIndex());
	h = Min(h, coarsePatch->getSFCYHighIndex());
       
	for(CellIterator iter(l, h); !iter.done(); iter++){
	  double yvel_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(0,-1,0), IntVector(rr.x(),0,rr.z()));
	      !inside.done(); inside++){
	    yvel_tmp+=fine_yvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(rr.x(),1,rr.z()));
	      !inside.done(); inside++){
	    yvel_tmp+=fine_yvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(0,1,0), IntVector(rr.x(),2,rr.z()));
	      !inside.done(); inside++){
	    yvel_tmp+=fine_yvel[fineStart+*inside]*0.5;
	  }
	  yvel[*iter]=yvel_tmp*ratio;
	}
      }
      //print(yvel, "coarsened yvel");
      //__________________________________
      //      Z   V E L
      SFCZVariable<double> zvel;
      new_dw->getModifiable(zvel, lb_->zvelocity, matl, coarsePatch);
      for(int i=0;i<finePatches.size();i++){
	const Patch* finePatch = finePatches[i];
       
	SFCZVariable<double> fine_zvel;
	new_dw->getCopy(fine_zvel, lb_->zvelocity, matl, finePatch,
			Ghost::AroundFacesZ, 1);
	
//        refineFaces<SFCZVariable<double>, constSFCZVariable<double> >
// 	  (finePatch, fineLevel, coarseLevel, IntVector(0,0,1),
// 	   Patch::zminus, Patch::zplus, fine_zvel,
// 	   lb_->zvelocity, 1.0, matl, old_dw, new_dw, Patch::ZFaceBased);
	
       IntVector fl(finePatch->getSFCZLowIndex());
	IntVector fh(finePatch->getSFCZHighIndex());
	IntVector l(fineLevel->mapCellToCoarser(fl));
	IntVector h(fineLevel->mapCellToCoarser(fh+IntVector(0, 0, rr.z()-1)));
	l = Max(l, coarsePatch->getSFCZLowIndex());
	h = Min(h, coarsePatch->getSFCZHighIndex());
	
       for(CellIterator iter(l, h); !iter.done(); iter++){
	  double zvel_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(0,0,-1), IntVector(rr.x(),rr.y(),0));
	      !inside.done(); inside++){
	    zvel_tmp+=fine_zvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(rr.x(),rr.y(),1));
	      !inside.done(); inside++){
	    zvel_tmp+=fine_zvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(0,0,1), IntVector(rr.x(),rr.y(),2));
	      !inside.done(); inside++){
	    zvel_tmp+=fine_zvel[fineStart+*inside]*0.5;
	  }
	  zvel[*iter]=zvel_tmp*ratio;
	}
      }
      //print(zvel, "coarsened zvel");      
    }
  }  // course patch loop 
}
//______________________________________________________________________
//
void AMRSimpleCFD::refine ( const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw )
{
  cout << "RANDY: AMRSimpleCFD::refine() BGN" << endl;
//    new_dw->transferFrom(old_dw, lb_->bctype, patches, matls);

  cout_doing << "Doing refine \t\t\t AMRSimpleCFD" << '\n';
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  IntVector rr(coarseLevel->getRefinementRatio());
  //double ratio = 1./(rr.x()*rr.y()*rr.z());

  //  IntVector rr(fineLevel->getRefinementRatio());
  //  double ratio = 1./(rr.x()*rr.y()*rr.z());

  for (int p = 0; p < patches->size(); p++) {  
    const Patch* finePatch = patches->get(p);
    cout_doing << "\t\t on patch " << finePatch->getID();
    cout << "RANDY: Patch = " << finePatch->getID() << endl;

    // Find the overlapping regions...
    Level::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      //__________________________________
      //   B C T Y P E
      if (old_dw->exists(lb_->bctype, m, finePatch)) {
	new_dw->transferFrom(old_dw, lb_->bctype, patches, matls);
      } else {
	NCVariable<int> bctype;
	new_dw->allocateAndPut(bctype, lb_->bctype, matl, finePatch);
	bcs.set(bctype, finePatch);
      }

      //__________________________________
      //   P R E S S U R E
      if(keep_pressure){
	if (old_dw->exists(lb_->pressure, m, finePatch)) {
	  new_dw->transferFrom(old_dw, lb_->pressure, patches, matls);
	} else {
	  CCVariable<double> pressure;
	  new_dw->allocateAndPut(pressure, lb_->pressure, matl, finePatch);

 /*
	  for ( int i = 0; i < coarsePatches.size(); i++ ) {
	    const Patch* coarsePatch = coarsePatches[i];
	    constCCVariable<double> coarse_pres;
	    new_dw->get(coarse_pres, lb_->pressure, matl, coarsePatch, Ghost::None, 0);
                  
	    IntVector cl(coarsePatch->getCellLowIndex());
	    IntVector ch(coarsePatch->getCellHighIndex());
	    IntVector l(coarseLevel->mapCellToFiner(cl));
	    IntVector h(coarseLevel->mapCellToFiner(ch));
	    l = Max(l, finePatch->getCellLowIndex());
	    h = Min(h, finePatch->getCellHighIndex());
       
	    for(CellIterator iter(l, h); !iter.done(); iter++){
	      double tmp=0;
	      IntVector coarseStart(fineLevel->mapCellToCoarser(*iter));
         
	      for(CellIterator inside(IntVector(0,0,0), coarseLevel->getRefinementRatio());
		  !inside.done(); inside++){
		tmp+=coarse_pres[coarseStart+*inside];
	      }
	      pressure[*iter]=tmp*ratio;
	    }
	  } // for(int i=0;i<coarsePatches.size();i++) {
*/

	} // if (old_dw->exists(lb_->pressure, m, finePatch)) {

      } // if(keep_pressure) {

      //__________________________________
      //   D E N S I T Y
      if (old_dw->exists(lb_->density, m, finePatch)) {
	new_dw->transferFrom(old_dw, lb_->density, patches, matls);
      } else {
	CCVariable<double> density;
	new_dw->allocateAndPut(density, lb_->density, matl, finePatch);
      }

      /*
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
	  double rho_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
         
	  for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
	      !inside.done(); inside++){
	    rho_tmp+=fine_den[fineStart+*inside];
	  }
	  density[*iter]=rho_tmp*ratio;
	}
      }  // fine patch loop
      //print(density, "coarsened density");
      */
      //__________________________________
      //      T E M P E R A T U R E
      if(do_thermal){
	if (old_dw->exists(lb_->temperature, m, finePatch)) {
	  new_dw->transferFrom(old_dw, lb_->temperature, patches, matls);
	} else {
	  CCVariable<double> temp;
	  new_dw->allocateAndPut(temp, lb_->temperature, matl, finePatch);
	}
	/*
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
	    double temp_tmp=0;
	    IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	    
           for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
		!inside.done(); inside++){
	      temp_tmp+=fine_temp[fineStart+*inside];
	    }
	    temp[*iter]=temp_tmp*ratio;
	  }
	}
	//print(temp, "coarsened temperature");
       */
      }  // do thermal
      
      //__________________________________
      //      X V E L
      if (old_dw->exists(lb_->xvelocity, m, finePatch)) {
	new_dw->transferFrom(old_dw, lb_->xvelocity, patches, matls);
      } else {
	SFCXVariable<double> xvel;
	new_dw->allocateAndPut(xvel, lb_->xvelocity, matl, finePatch);
      }

      /*
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
	  double xvel_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(-1,0,0), IntVector(0,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    xvel_tmp+=fine_xvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(1,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    xvel_tmp+=fine_xvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(1,0,0), IntVector(2,rr.y(),rr.z()));
	      !inside.done(); inside++){
	    xvel_tmp +=fine_xvel[fineStart+*inside]*0.5;
	  }
	  xvel[*iter]=xvel_tmp*ratio;
	}
      }
      */      
      //print(xvel, "coarsened xvel");
      //__________________________________
      //     Y V E L
      if (old_dw->exists(lb_->yvelocity, m, finePatch)) {
	new_dw->transferFrom(old_dw, lb_->yvelocity, patches, matls);
      } else {
	SFCYVariable<double> yvel;
	new_dw->allocateAndPut(yvel, lb_->yvelocity, matl, finePatch);
      }
      /*
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
	  double yvel_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(0,-1,0), IntVector(rr.x(),0,rr.z()));
	      !inside.done(); inside++){
	    yvel_tmp+=fine_yvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(rr.x(),1,rr.z()));
	      !inside.done(); inside++){
	    yvel_tmp+=fine_yvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(0,1,0), IntVector(rr.x(),2,rr.z()));
	      !inside.done(); inside++){
	    yvel_tmp+=fine_yvel[fineStart+*inside]*0.5;
	  }
	  yvel[*iter]=yvel_tmp*ratio;
	}
      }
      */
      //print(yvel, "coarsened yvel");
      //__________________________________
      //      Z   V E L
      if (old_dw->exists(lb_->zvelocity, m, finePatch)) {
	new_dw->transferFrom(old_dw, lb_->zvelocity, patches, matls);
      } else {
	SFCZVariable<double> zvel;
	new_dw->allocateAndPut(zvel, lb_->zvelocity, matl, finePatch);
      }
      /*
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
	  double zvel_tmp=0;
	  IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
	  for(CellIterator inside(IntVector(0,0,-1), IntVector(rr.x(),rr.y(),0));
	      !inside.done(); inside++){
	    zvel_tmp+=fine_zvel[fineStart+*inside]*0.5;
	  }
	  for(CellIterator inside(IntVector(0,0,0), IntVector(rr.x(),rr.y(),1));
	      !inside.done(); inside++){
	    zvel_tmp+=fine_zvel[fineStart+*inside];
	  }
	  for(CellIterator inside(IntVector(0,0,1), IntVector(rr.x(),rr.y(),2));
	      !inside.done(); inside++){
	    zvel_tmp+=fine_zvel[fineStart+*inside]*0.5;
	  }
	  zvel[*iter]=zvel_tmp*ratio;
	}
      }
      //print(zvel, "coarsened zvel");      
      */
    }
  }  // course patch loop 

  cout << "RANDY: AMRSimpleCFD::refine() END" << endl;
}
//______________________________________________________________________
//  A diagnosic flag is generated in this task.
void AMRSimpleCFD::errorEstimate(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse*,
				 DataWarehouse* new_dw,
                                 bool initial)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    CCVariable<int> refineFlag;
    PerPatch<PatchFlagP> refinePatchFlag;
    
    new_dw->getModifiable(refineFlag, sharedState_->get_refineFlag_label(),
                          0, patch);
    new_dw->get(refinePatchFlag, sharedState_->get_refinePatchFlag_label(),
                0, patch);

    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
    
    cout_doing << "Doing errorEstimate on patch "<< patch->getID()<<" \t\t\t AMRSimpleCFD" << '\n';
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      Vector dx(patch->dCell());
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());

      IntVector l(patch->getCellLowIndex());
      IntVector h(patch->getCellHighIndex());

      // don't do density on initialization estimation, we won't know enough
      // info to calculate density!
      if (!initial) {
        //__________________________________
        //     D E N S I T Y   G R A D
        constCCVariable<double> density;
        CCVariable<double> density_gradient_mag;
        new_dw->get(density, lb_->density, matl, patch, Ghost::AroundCells, 1);
        
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
          if(density_gradient_mag[idx] > err_density_grad) {
            refineFlag[idx]=true;
            refinePatch->set();
          }
          density_gradient_mag[idx] *= err_density_grad_inv;
        }
      }
      //__________________________________
      //   T E M P E R A T U R E   G R A D
      if(err_temperature_grad > 0){
	CCVariable<double> temperature_gradient_mag;
       constCCVariable<double> temperature;
	new_dw->allocateAndPut(temperature_gradient_mag, lb_->temperature_gradient_mag,
			       matl, patch);
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
	  if(temperature_gradient_mag[idx] > err_temperature_grad) {
	    refineFlag[idx]=true;
            refinePatch->set();
          }
	  temperature_gradient_mag[idx] *= inv_err_temperature_grad;
	}
      }
      //__________________________________
      //      P R E S S U R E   G R A D
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
	  if(pressure_gradient_mag[idx] > err_pressure_grad) {
	    refineFlag[idx]=true;
            refinePatch->set();
          }
	  pressure_gradient_mag[idx] *= inv_err_pressure_grad;
	}
      }
      //__________________________________
      //     V O R T I C I T Y   M A G .
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
	  if(ccvorticitymag[idx] > err_vorticity_mag) {
	    refineFlag[idx]=true;
            refinePatch->set();
          }
	  ccvorticitymag[idx] *= inv_err_vorticity_mag;
	}
      }
    }
  }
}
