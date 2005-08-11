#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h> 
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/InternalError.h>

using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_doing("ICE_DOING_COUT", false);
/*______________________________________________________________________
 Function~  ICE::scheduleCoarsenPressure--
 Purpose:  After the implicit pressure solve is performed on all levels 
 you need to project/coarsen the fine level solution onto the coarser level
 _____________________________________________________________________*/
void ICE::scheduleCoarsenPressure(SchedulerP& sched, 
                                  const LevelP& coarseLevel,
                                  const MaterialSubset* press_matl)
{
  if (d_doAMR &&  d_impICE){                                                                            
    cout_doing << "ICE::scheduleCoarsenPressure\t\t\t\tL-" 
               << coarseLevel->getIndex() << endl;

    Task* t = scinew Task("ICE::coarsenPressure",
                    this, &ICE::coarsenPressure);

    Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
    Ghost::GhostType  gn = Ghost::None;

    t->requires(Task::NewDW, lb->press_CCLabel,
                0, Task::FineLevel,  press_matl,oims, gn, 0);
                
    t->modifies(lb->press_CCLabel, d_press_matl, oims);        

    sched->addTask(t, coarseLevel->eachPatch(), d_sharedState->allICEMaterials());
  }
}

/* _____________________________________________________________________
 Function~  ICE::CoarsenPressure
 _____________________________________________________________________  */
void ICE::coarsenPressure(const ProcessorGroup*,
                          const PatchSubset* coarsePatches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse* new_dw)
{ 
  const Level* coarseLevel = getLevel(coarsePatches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  for(int p=0;p<coarsePatches->size();p++){
    const Patch* coarsePatch = coarsePatches->get(p);
    
    cout_doing << "Doing CoarsenPressure on patch "
               << coarsePatch->getID() << "\t ICE \tL-" <<coarseLevel->getIndex()<< endl;
    CCVariable<double> press_CC, notUsed;                  
    new_dw->getModifiable(press_CC, lb->press_CCLabel,  0,    coarsePatch);
   
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    Vector dx_c = coarseLevel->dCell();
    Vector dx_f = fineLevel->dCell();
    double coarseCellVol = dx_c.x()*dx_c.y()*dx_c.z();
    double fineCellVol   = dx_f.x()*dx_f.y()*dx_f.z();

    for(int i=0;i<finePatches.size();i++){
      const Patch* finePatch = finePatches[i];

      IntVector fl(finePatch->getInteriorCellLowIndex());
      IntVector fh(finePatch->getInteriorCellHighIndex());
      IntVector cl(fineLevel->mapCellToCoarser(fl));
      IntVector ch(fineLevel->mapCellToCoarser(fh));

      cl = Max(cl, coarsePatch->getCellLowIndex());
      ch = Min(ch, coarsePatch->getCellHighIndex());

      // get the region of the fine patch that overlaps the coarse patch
      // we might not have the entire patch in this proc's DW
      fl = coarseLevel->mapCellToFiner(cl);
      fh = coarseLevel->mapCellToFiner(ch);
      if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
        continue;
      }

      constCCVariable<double> fine_press_CC;
      new_dw->getRegion(fine_press_CC,  lb->press_CCLabel, 0, fineLevel, fl, fh);

      //cout << " fineToCoarseOperator: finePatch "<< fl << " " << fh 
      //         << " coarsePatch "<< cl << " " << ch << endl;

      IntVector refinementRatio = fineLevel->getRefinementRatio();

      // iterate over coarse level cells
      for(CellIterator iter(cl, ch); !iter.done(); iter++){
        IntVector c = *iter;
        double press_CC_tmp(0.0);
        IntVector fineStart = coarseLevel->mapCellToFiner(c);

        // for each coarse level cell iterate over the fine level cells   
        for(CellIterator inside(IntVector(0,0,0),refinementRatio );
                                            !inside.done(); inside++){
          IntVector fc = fineStart + *inside;

          press_CC_tmp += fine_press_CC[fc] * fineCellVol;
        }
        press_CC[c] =press_CC_tmp / coarseCellVol;
      }
    }
  }
}
