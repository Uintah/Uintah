/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- DORadiationModel.cc --------------------------------------------------
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadPetscSolver.h>
#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Thread/Time.h>
#include <cmath>
#include <sci_defs/hypre_defs.h>


#ifdef HAVE_HYPRE
#include <CCA/Components/Arches/Radiation/RadHypreSolver.h>
#endif

using namespace std;
using namespace Uintah;

#include <CCA/Components/Arches/Radiation/fortran/rordr_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radcoef_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radwsgg_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radcal_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomflux_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomvolq_fort.h>
static DebugStream dbg("ARCHES_RADIATION",false);
//****************************************************************************
// Default constructor for DORadiationModel
//****************************************************************************
DORadiationModel::DORadiationModel(const ArchesLabel* label,
                                   const MPMArchesLabel* MAlab,
                                   BoundaryCondition* bndry_cond,
                                   const ProcessorGroup* myworld):
                                   d_lab(label),
                                   d_MAlab(MAlab), 
                                   d_boundaryCondition(bndry_cond),
                                   d_myworld(myworld), lprobone(false), lprobtwo(false), lprobthree(false)
{

  d_linearSolver = 0;
  d_radCalcFreq = 0; 
  _props_calculator = 0;
  d_perproc_patches = 0;
  _using_props_calculator = false;
  d_use_abskp = false;
}

//****************************************************************************
// Destructor
//****************************************************************************
DORadiationModel::~DORadiationModel()
{
  delete d_linearSolver;
  delete _props_calculator;
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches; 
}

//****************************************************************************
// Problem Setup for DORadiationModel
//**************************************************************************** 

void 
DORadiationModel::problemSetup( ProblemSpecP& params, bool stand_alone_src )

{

  //stand_alone_src indicates if the DORadiation model is invoked from the 
  //EnthalpySolver (should be false)  or the new SourceTerm (should be true) 

  if ( !stand_alone_src ) { 
    params->getWithDefault("radiationCalcFreq",         d_radCalcFreq, 3);
    params->getWithDefault("radCalcForAllRKSteps",      d_radRKsteps,  false);
    params->getWithDefault("radCalcForAllImplicitSteps",d_radImpsteps, false);
  }
  
  string prop_model;
  ProblemSpecP db = params->findBlock("DORadiationModel");
  //__________________________________
  // property calculator
  if ( db->findBlock("property_calculator") ){ 

    std::string calculator_type; 
    db->findBlock("property_calculator")->getAttribute("type",calculator_type);

    if ( calculator_type == "constant" ){ 
      _props_calculator = scinew ConstantProperties(); 
    } else if ( calculator_type == "burns_christon" ){ 
      _props_calculator = scinew BurnsChriston(); 
    } else { 
      throw InvalidValue("Error: Property calculator not recognized.",__FILE__, __LINE__); 
    } 

    ProblemSpecP db_pc = db->findBlock("property_calculator"); 
    _using_props_calculator = _props_calculator->problemSetup( db_pc );

    if ( !_using_props_calculator ) {
      throw InvalidValue("Error: Property calculator specified in input file but I was unable to setup your calculator!",__FILE__, __LINE__); 
    } 
  } 


  //__________________________________
  // article absorption coefficient 
  PropertyModelFactory& prop_factory = PropertyModelFactory::self();
  d_use_abskp = prop_factory.find_property_model( "abskp");
  
  if(d_use_abskp){
    PropertyModelBase& abskpModel = prop_factory.retrieve_property_model( "abskp");
    d_abskpLabel = abskpModel.getPropLabel(); 
  }
  
 
  if (db) {
    db->getWithDefault("ordinates",d_sn,2);
    db->require("opl",d_opl);
    db->getWithDefault("property_model",prop_model,"radcoef");
  }
  else {
    d_sn=6;
    d_opl=0.18;
  }


  if (prop_model == "radcoef"){ 
    lradcal     = false;
    lwsgg       = false;
    d_lambda      = 1;
    lplanckmean = false;
    lpatchmean  = false;
  }

  if (prop_model == "patchmean"){ 
    cout << "WARNING! Serial and parallel results may deviate for this model" << endl;
    lradcal     = true;
    lwsgg       = false;
    d_lambda      = 6;
    lplanckmean = false;
    lpatchmean  = true;
  }

  if (prop_model == "wsggm"){ 
    throw InternalError("WSGG radiation model does not run in parallel and has been disabled", __FILE__, __LINE__);
    lradcal       = false;
    lwsgg         = true;
    d_lambda        = 4;
    lplanckmean   = false;
    lpatchmean    = false;
  }

  fraction.resize(1,100);
  fraction.initialize(0.0);

  fractiontwo.resize(1,100);
  fractiontwo.initialize(0.0);

  computeOrdinatesOPL();


    
  string linear_sol;
  db->findBlock("LinearSolver")->getAttribute("type",linear_sol);

  if (linear_sol == "petsc"){ 
    d_linearSolver = scinew RadPetscSolver(d_myworld);
#ifdef HAVE_HYPRE
  }else if (linear_sol == "hypre"){ 
    d_linearSolver = scinew RadHypreSolver(d_myworld);
#endif
  }
  
  d_linearSolver->problemSetup(db);
  
  ffield = -1;

  d_do_const_wall_T = false; 
  if ( db->findBlock("const_wall_temperature" )) { 
    d_do_const_wall_T = true; 
    d_wall_temperature = 293.0; 
  }
  db->getWithDefault("wall_abskg", d_wall_abskg, 1.0); 
  db->getWithDefault("intrusion_abskg", d_intrusion_abskg, 1.0); 

}
//______________________________________________________________________
//
void
DORadiationModel::computeOrdinatesOPL() {

  /*
  //  if(lprobone==true){
    d_opl = 1.0*d_opl; 
    //  }
  if(lprobtwo==true){
    d_opl = 1.76;
  }
  */
  
  d_totalOrds = d_sn*(d_sn+2);
// d_totalOrds = 8*d_sn*d_sn;

  omu.resize( 1,d_totalOrds + 1);
  oeta.resize(1,d_totalOrds + 1);
  oxi.resize( 1,d_totalOrds + 1);
  wt.resize(  1,d_totalOrds + 1);
  //  ord.resize(1,(d_sn/2) + 1);
   //   ord.resize(1,3);

   omu.initialize(0.0);
   oeta.initialize(0.0);
   oxi.initialize(0.0);
   wt.initialize(0.0);
   //   ord.initialize(0.0);

   fort_rordr(d_sn, oxi, omu, oeta, wt);
   //           fort_rordrss(d_sn, oxi, omu, oeta, wt);
   //           fort_rordrtn(d_sn, ord, oxi, omu, oeta, wt);
}

//______________________________________________________________________
//  This is the main task called by Enthalpysolver
void
DORadiationModel::sched_computeSource( const LevelP& level, 
                                       SchedulerP& sched, 
                                       const MaterialSet* matls,
                                       const TimeIntegratorLabel* timelabels, 
                                       const bool isFirstIntegrationStep )
{

  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
    
  LoadBalancer* lb  = sched->getLoadBalancer();
  d_perproc_patches = lb->getPerProcessorPatchSet(level);
  d_perproc_patches->addReference();
  
  string taskname =  "DORadiation::sched_computeSource";      
                       
  Task* tsk = scinew Task(taskname, this,
                        &DORadiationModel::computeSource,
                        timelabels, isFirstIntegrationStep);
  
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
  
  Task::WhichDW which_dw = Task::NewDW;
  if (isFirstIntegrationStep) {
    which_dw = Task::OldDW;
  }
  
  tsk->requires(which_dw,      d_lab->d_tempINLabel,   gac, 1);
  tsk->requires(which_dw,      d_lab->d_cpINLabel,     gn,  0);
  tsk->requires(which_dw,      d_lab->d_co2INLabel,    gn,  0);
  tsk->requires(which_dw,      d_lab->d_h2oINLabel,    gn,  0);
  tsk->requires(which_dw,      d_lab->d_sootFVINLabel, gn,  0);
  
  tsk->requires(Task::NewDW,   d_lab->d_cellInfoLabel, gn);
  tsk->requires(Task::NewDW,   d_lab->d_cellTypeLabel, gac, 1);

  if (isFirstIntegrationStep ) {

    tsk->requires(Task::OldDW, d_lab->d_radiationSRCINLabel,    gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationFluxEINLabel,  gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationFluxWINLabel,  gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationFluxNINLabel,  gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationFluxSINLabel,  gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationFluxTINLabel,  gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationFluxBINLabel,  gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_radiationVolqINLabel,   gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_abskgINLabel,           gn, 0);

    tsk->computes(d_lab->d_abskgINLabel);
    tsk->computes(d_lab->d_radiationSRCINLabel);
    tsk->computes(d_lab->d_radiationFluxEINLabel);
    tsk->computes(d_lab->d_radiationFluxWINLabel);
    tsk->computes(d_lab->d_radiationFluxNINLabel);
    tsk->computes(d_lab->d_radiationFluxSINLabel);
    tsk->computes(d_lab->d_radiationFluxTINLabel);
    tsk->computes(d_lab->d_radiationFluxBINLabel);
    tsk->modifies(d_lab->d_radiationVolqINLabel);
  } else {
    tsk->modifies(d_lab->d_abskgINLabel);
    tsk->modifies(d_lab->d_radiationSRCINLabel);
    tsk->modifies(d_lab->d_radiationFluxEINLabel);
    tsk->modifies(d_lab->d_radiationFluxWINLabel);
    tsk->modifies(d_lab->d_radiationFluxNINLabel);
    tsk->modifies(d_lab->d_radiationFluxSINLabel);
    tsk->modifies(d_lab->d_radiationFluxTINLabel);
    tsk->modifies(d_lab->d_radiationFluxBINLabel);
    tsk->modifies(d_lab->d_radiationVolqINLabel);
  }
  
  //__________________________________
  if(d_use_abskp){
    tsk->requires(Task::OldDW, d_abskpLabel,   gn, 0);
  }
  
  cout << " energy exchange " << d_boundaryCondition->getIfCalcEnergyExchange() << endl;
  if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {
    tsk->requires(Task::NewDW, d_MAlab->integTemp_CCLabel, gn, 0);
  }
  
  sched->addTask(tsk, d_perproc_patches , matls);
}

//______________________________________________________________________
//
void
DORadiationModel::computeSource( const ProcessorGroup* pc, 
                                 const PatchSubset* patches, 
                                 const MaterialSubset* matls, 
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw, 
                                 const TimeIntegratorLabel* timelabels,
                                 bool isFirstIntegrationStep )
{


  int radCounter = d_lab->d_sharedState->getCurrentTopLevelTimeStep();

  int archIndex = 0; // only one arches material
  int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
  double new_total_src = 0.0;
    
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"DORadiation::computeSource");
    

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
     DataWarehouse* which_dw = new_dw;
    if (isFirstIntegrationStep){
      which_dw = old_dw;
    }

    d_linearSolver->matrixCreate( d_perproc_patches, patches);

    ArchesVariables      radVars;
    ArchesConstVariables constRadVars;
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;
    
    // all integrator steps:
    which_dw->get(constRadVars.co2,     d_lab->d_co2INLabel,    indx, patch, gn,  0);
    which_dw->get(constRadVars.h2o,     d_lab->d_h2oINLabel,    indx, patch, gn,  0);
    which_dw->get(constRadVars.sootFV,  d_lab->d_sootFVINLabel, indx, patch, gn,  0);
    new_dw->get(  constRadVars.cellType,d_lab->d_cellTypeLabel, indx, patch, gac, 1);
    
    radVars.ESRCG.allocate(patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL),
                           patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL));

    //__________________________________
    //  abskp
    radVars.ABSKP.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    if(d_use_abskp){
      old_dw->copyOut(radVars.ABSKP, d_abskpLabel,indx, patch, gn, 0);
    } else {
      radVars.ABSKP.initialize(0.0);
    }
    
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      old_dw->getCopy(radVars.temperature, d_lab->d_tempINLabel, indx, patch, gac, 1);
    }
    else {
      new_dw->getCopy(radVars.temperature, d_lab->d_tempINLabel, indx, patch, gac, 1);
    }

    if ( isFirstIntegrationStep ){
      new_dw->allocateAndPut(radVars.qfluxe, d_lab->d_radiationFluxEINLabel, indx, patch);
      old_dw->copyOut(       radVars.qfluxe, d_lab->d_radiationFluxEINLabel, indx, patch, gn, 0);

      new_dw->allocateAndPut(radVars.qfluxw, d_lab->d_radiationFluxWINLabel, indx, patch);
      old_dw->copyOut(       radVars.qfluxw, d_lab->d_radiationFluxWINLabel, indx, patch, gn, 0);

      new_dw->allocateAndPut(radVars.qfluxn, d_lab->d_radiationFluxNINLabel, indx, patch);
      old_dw->copyOut(       radVars.qfluxn, d_lab->d_radiationFluxNINLabel, indx, patch, gn, 0);

      new_dw->allocateAndPut(radVars.qfluxs, d_lab->d_radiationFluxSINLabel, indx, patch);
      old_dw->copyOut(       radVars.qfluxs, d_lab->d_radiationFluxSINLabel, indx, patch, gn, 0);

      new_dw->allocateAndPut(radVars.qfluxt, d_lab->d_radiationFluxTINLabel, indx, patch);
      old_dw->copyOut(       radVars.qfluxt, d_lab->d_radiationFluxTINLabel, indx, patch, gn, 0);

      new_dw->allocateAndPut(radVars.qfluxb, d_lab->d_radiationFluxBINLabel, indx, patch);
      old_dw->copyOut(       radVars.qfluxb, d_lab->d_radiationFluxBINLabel, indx, patch, gn, 0);
      
      new_dw->allocateAndPut(radVars.src,    d_lab->d_radiationSRCINLabel,   indx, patch);
      old_dw->copyOut(       radVars.src,    d_lab->d_radiationSRCINLabel,   indx, patch, gn, 0);

      new_dw->allocateAndPut(radVars.ABSKG,  d_lab->d_abskgINLabel,          indx, patch);
      old_dw->copyOut(       radVars.ABSKG,  d_lab->d_abskgINLabel,          indx, patch, gn, 0);
      
      new_dw->getModifiable( radVars.volq,   d_lab->d_radiationVolqINLabel,  indx, patch);

    } else {         // after first step
      new_dw->getModifiable(radVars.qfluxe, d_lab->d_radiationFluxEINLabel, indx, patch);
      new_dw->getModifiable(radVars.qfluxw, d_lab->d_radiationFluxWINLabel, indx, patch);
      new_dw->getModifiable(radVars.qfluxn, d_lab->d_radiationFluxNINLabel, indx, patch);
      new_dw->getModifiable(radVars.qfluxs, d_lab->d_radiationFluxSINLabel, indx, patch);
      new_dw->getModifiable(radVars.qfluxt, d_lab->d_radiationFluxTINLabel, indx, patch);
      new_dw->getModifiable(radVars.qfluxb, d_lab->d_radiationFluxBINLabel, indx, patch);
      new_dw->getModifiable(radVars.volq,   d_lab->d_radiationVolqINLabel,  indx, patch);
      new_dw->getModifiable(radVars.src,    d_lab->d_radiationSRCINLabel,   indx, patch);
      new_dw->getModifiable(radVars.ABSKG,  d_lab->d_abskgINLabel,          indx, patch);
    }

    //__________________________________
    //Radiation calculation

    bool first_step = isFirstIntegrationStep;

    if ( radCounter%d_radCalcFreq == 0){
      if (  (first_step  && !timelabels->recursion )
           ||(!first_step && d_radRKsteps)
           ||(timelabels->recursion && d_radImpsteps) ) {
              
        radVars.src.initialize(0.0);
        radVars.qfluxe.initialize(0.0);
        radVars.qfluxw.initialize(0.0);
        radVars.qfluxn.initialize(0.0);
        radVars.qfluxs.initialize(0.0);
        radVars.qfluxt.initialize(0.0);
        radVars.qfluxb.initialize(0.0);
        radVars.ABSKG.initialize(0.0);
        radVars.ESRCG.initialize(0.0);

        computeRadiationProps(pc, patch, cellinfo, &radVars, &constRadVars);

        // apply boundary conditons
        boundarycondition(    pc, patch, cellinfo, &radVars, &constRadVars);


        if (d_MAlab && d_boundaryCondition->getIfCalcEnergyExchange()) {
          bool d_energyEx = true;
          constCCVariable<double> solidTemp;
          new_dw->get(solidTemp, d_MAlab->integTemp_CCLabel, indx, patch, gn, 0);

          d_boundaryCondition->mmWallTemperatureBC(patch, constRadVars.cellType,
                                                   solidTemp, radVars.temperature,
                                                   d_energyEx);
        }

        int wall = d_boundaryCondition->wallCellType();
        intensitysolve(pc, patch, cellinfo, &radVars, &constRadVars, wall );
      }
    }
  }
}


//****************************************************************************
//  Actually compute the properties here
//****************************************************************************

void 
DORadiationModel::computeRadiationProps(const ProcessorGroup*,
                                        const Patch* patch,
                                        CellInformation* cellinfo, 
                                        ArchesVariables* vars,
                                        ArchesConstVariables* constvars)

{
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  IntVector domLo = patch->getExtraCellLowIndex();
  IntVector domHi = patch->getExtraCellHighIndex();

  CCVariable<double> shgamma;
  vars->shgamma.allocate(domLo,domHi);
  vars->shgamma.initialize(0.0);

  fort_radcoef(idxLo, idxHi, vars->temperature,                                                 
               constvars->co2, constvars->h2o, constvars->cellType, ffield,
               d_opl, constvars->sootFV, vars->ABSKP, vars->ABSKG, vars->ESRCG, vars->shgamma,
               cellinfo->xx, cellinfo->yy, cellinfo->zz, fraction, fractiontwo,
               lprobone, lprobtwo, lprobthree, d_lambda, lradcal);

  if (_using_props_calculator){
    _props_calculator->computeProps( patch, vars->ABSKG );                                      
  }
}

//***************************************************************************
// Sets the radiation boundary conditions for the D.O method
//***************************************************************************
void 
DORadiationModel::boundarycondition(const ProcessorGroup*,
                                    const Patch* patch,
                                    CellInformation* cellinfo,
                                    ArchesVariables* vars,
                                    ArchesConstVariables* constvars)
{
           
  //__________________________________
  // loop over computational domain faces
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    
    Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
    
    if ( d_do_const_wall_T ) {   
      for (CellIterator iter =  patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
        IntVector c = *iter;
        if (constvars->cellType[c] != ffield ){
          vars->temperature[c] = d_wall_temperature; 
          vars->ABSKG[c]       = d_wall_abskg;
        }
      }
    } else { 
      for (CellIterator iter =  patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
        IntVector c = *iter;
        if (constvars->cellType[c] != ffield ){
          vars->ABSKG[c]       = d_wall_abskg;
        }
      }
    }
  }
}
//***************************************************************************
// Solves for intensity in the D.O method
//***************************************************************************
void 
DORadiationModel::intensitysolve(const ProcessorGroup* pg,
                                 const Patch* patch,
                                 CellInformation* cellinfo,
                                 ArchesVariables* vars,
                                 ArchesConstVariables* constvars, 
                                 int wall_type )
{
  double solve_start = Time::currentSeconds();
  rgamma.resize(1,29);    
  sd15.resize(1,481);     
  sd.resize(1,2257);      
  sd7.resize(1,49);       
  sd3.resize(1,97);       

  rgamma.initialize(0.0); 
  sd15.initialize(0.0);   
  sd.initialize(0.0);     
  sd7.initialize(0.0);    
  sd3.initialize(0.0);    

  if (d_lambda > 1) {
    fort_radarray(rgamma, sd15, sd, sd7, sd3);
  }

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  IntVector domLo = patch->getExtraCellLowIndex();
  IntVector domHi = patch->getExtraCellHighIndex();

  CCVariable<double> su;
  CCVariable<double> ae;
  CCVariable<double> aw;
  CCVariable<double> an;
  CCVariable<double> as;
  CCVariable<double> at;
  CCVariable<double> ab;
  CCVariable<double> ap;
  //CCVariable<double> volq;
  CCVariable<double> cenint;
  
  vars->cenint.allocate(domLo,domHi);

  su.allocate(domLo,domHi);
  ae.allocate(domLo,domHi);
  aw.allocate(domLo,domHi);
  an.allocate(domLo,domHi);
  as.allocate(domLo,domHi);
  at.allocate(domLo,domHi);
  ab.allocate(domLo,domHi);
  ap.allocate(domLo,domHi);
  //volq.allocate(domLo,domHi);
  
  srcbm.resize(domLo.x(),domHi.x());
  srcbm.initialize(0.0);
  srcpone.resize(domLo.x(),domHi.x());
  srcpone.initialize(0.0);
  qfluxbbm.resize(domLo.x(),domHi.x());
  qfluxbbm.initialize(0.0);

  vars->volq.initialize(0.0);
  vars->cenint.initialize(0.0);
  vars->src.initialize(0.0);

  //__________________________________
  //begin discrete ordinates

  for (int bands =1; bands <=d_lambda; bands++){

    vars->volq.initialize(0.0);

    if(lwsgg == true){    
      fort_radwsgg(idxLo, idxHi, vars->ABSKG, vars->ESRCG, vars->shgamma,
                   bands, constvars->cellType, ffield, 
                   constvars->co2, constvars->h2o, constvars->sootFV, 
                   vars->temperature, d_lambda, fraction, fractiontwo);
    }

    if(lradcal==true){    
      fort_radcal(idxLo, idxHi, vars->ABSKG, vars->ESRCG, vars->shgamma,
                  cellinfo->xx, cellinfo->yy, cellinfo->zz, bands, 
                  constvars->cellType, ffield, 
                  constvars->co2, constvars->h2o, constvars->sootFV, 
                  vars->temperature, lprobone, lprobtwo, lplanckmean, lpatchmean, d_lambda, fraction, rgamma, 
                  sd15, sd, sd7, sd3, d_opl);
    }

    for (int direcn = 1; direcn <=d_totalOrds; direcn++){
      vars->cenint.initialize(0.0);
      su.initialize(0.0);
      aw.initialize(0.0);
      as.initialize(0.0);
      ab.initialize(0.0);
      ap.initialize(0.0);
      ae.initialize(0.0);
      an.initialize(0.0);
      at.initialize(0.0);
      bool plusX, plusY, plusZ;
      
      fort_rdomsolve(idxLo, idxHi, constvars->cellType, ffield, 
                     cellinfo->sew, cellinfo->sns, cellinfo->stb, 
                     vars->ESRCG, direcn, oxi, omu,oeta, wt, 
                     vars->temperature, vars->ABSKG,
                     su, aw, as, ab, ap, ae, an, at,
                     plusX, plusY, plusZ, fraction, bands, d_intrusion_abskg);

      //      double timeSetMat = Time::currentSeconds();
      d_linearSolver->setMatrix(pg ,patch, vars, plusX, plusY, plusZ, 
                                su, ab, as, aw, ap, ae, an, at);
                                
      //      timeRadMatrix += Time::currentSeconds() - timeSetMat;
      bool converged =  d_linearSolver->radLinearSolve();
      
      if (converged) {
        d_linearSolver->copyRadSoln(patch, vars);
      }else {
        throw InternalError("Radiation solver not converged", __FILE__, __LINE__);
      }
      
      d_linearSolver->destroyMatrix();

      fort_rdomvolq(idxLo, idxHi, direcn, wt, vars->cenint, vars->volq);
      
      fort_rdomflux(idxLo, idxHi, direcn, oxi, omu, oeta, wt, vars->cenint,
                    plusX, plusY, plusZ, 
                    vars->qfluxe, vars->qfluxw,
                    vars->qfluxn, vars->qfluxs,
                    vars->qfluxt, vars->qfluxb);
    }  // ordinate loop

    fort_rdomsrc(idxLo, idxHi, vars->ABSKG, vars->ESRCG,vars->volq, vars->src);
  }  // bands loop

  if(d_myworld->myrank() == 0) {
    cerr << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";
  }
}
DORadiationModel::PropertyCalculatorBase::PropertyCalculatorBase(){};
DORadiationModel::PropertyCalculatorBase::~PropertyCalculatorBase(){};
DORadiationModel::ConstantProperties::ConstantProperties(){};
DORadiationModel::ConstantProperties::~ConstantProperties(){};
DORadiationModel::BurnsChriston::BurnsChriston(){};
DORadiationModel::BurnsChriston::~BurnsChriston(){};















