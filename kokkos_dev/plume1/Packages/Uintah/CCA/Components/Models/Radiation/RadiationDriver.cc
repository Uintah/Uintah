//----- RadiationDriver.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_DORadiationModel.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationDriver.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_RadiationModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Ports/ModelMaker.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("MODELS_DOING_COUT", false);

RadiationDriver::RadiationDriver(const ProcessorGroup* myworld,
                                 ProblemSpecP& params):
  ModelInterface (myworld), params(params), d_myworld(myworld)
{
  d_perproc_patches = 0;
  d_DORadiation = 0;
  d_radCounter = -1; //to decide how often radiation calc is done
  d_radCalcFreq = 0; 
  d_matl_set = 0;
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();

  d_cellInfoLabel = VarLabel::create("radCellInformation",
                                     PerPatch<Models_CellInformationP>::getTypeDescription());

  // cellType_CC is the variable to determine the location of
  // boundaries, whether they are of the immersed solid or those
  // of open or wall boundaries. This variable is necessary because
  // radiation is a surface-dependent process.

  cellType_CCLabel = VarLabel::create("cellType_CC", td_CCdouble);

  // shgamma is simply 1/(3*abskg). It is calculated and stored 
  // this way because it is more efficient to do so than just
  // calculating it on the fly each time.  This is done because
  // gradients of shgamma are needed in the P1 calculation.

  shgamma_CCLabel = VarLabel::create("shgamma", td_CCdouble);

  // abskg is the absorption coefficient, units 1/m.  This is
  // the cardinal property of absorbing and emitting (participating)
  // media. esrcg is the blackbody emissivity per unit steradian,
  // abskg*Temp**4/pi. cenint is the temporary storage for the 
  // intensity. The intensities for individual solid angles are not
  // stored because of storage reasons ... in DO, we have n(n+2)
  // directions, where n is the order of the approximation.
  // So, for n = 2, we are talking about 8 directions, and for
  // n = 4, we are talking about 24 directions.  So there is a lot
  // of storage involved
  abskg_CCLabel  = VarLabel::create("abskg", td_CCdouble);
  esrcg_CCLabel  = VarLabel::create("esrcg", td_CCdouble);
  cenint_CCLabel = VarLabel::create("cenint",td_CCdouble);

  // qfluxE, qfluxW, qfluxN, qfluxS, qfluxT, and qfluxB are the
  // heat fluxes in the east, west, north, south, top, and bottom
  // directions
  qfluxE_CCLabel = VarLabel::create("qfluxE", td_CCdouble);
  qfluxW_CCLabel = VarLabel::create("qfluxW", td_CCdouble);
  qfluxN_CCLabel = VarLabel::create("qfluxN", td_CCdouble);
  qfluxS_CCLabel = VarLabel::create("qfluxS", td_CCdouble);
  qfluxT_CCLabel = VarLabel::create("qfluxT", td_CCdouble);
  qfluxB_CCLabel = VarLabel::create("qfluxB", td_CCdouble);

  // CO2 and H2O are table values, and are used as constants.
  co2_CCLabel = VarLabel::create("CO2", td_CCdouble);
  h2o_CCLabel = VarLabel::create("H2O", td_CCdouble);

  // radCO2 and radH2O are local variables, which are set here.
  radCO2_CCLabel = VarLabel::create("radCO2", td_CCdouble);
  radH2O_CCLabel = VarLabel::create("radH2O", td_CCdouble);

  // scalar-f comes from table (an ICE value, though)
  mixfrac_CCLabel = VarLabel::create("scalar-f", td_CCdouble);
  mixfracCopy_CCLabel = VarLabel::create("mixfrac", td_CCdouble);

  // density comes from the table
  density_CCLabel = VarLabel::create("density", td_CCdouble);

  // rho_CC is the density from ICE
  iceDensity_CCLabel = VarLabel::create("rho_CC", td_CCdouble);

  // Temp is the temperature from the table; right now, since fire in ICE does
  // not give correct values (algorithm needs reworking), I am
  // using table temperatures. Later, when adiabatic fire works
  // properly, we can start using ICE temperatures for radiation
  // calculations
  temp_CCLabel = VarLabel::create("Temp", td_CCdouble);

  // temp_CC is the temperature from ICE
  iceTemp_CCLabel = VarLabel::create("temp_CC", td_CCdouble);

  // TempCopy is the temperature that is used in all actual
  // calculations in these radiative heat transfer calculations.
  // It is simply a copy of either the table temperature or the
  // ICE temperature.  This variable is needed because the radiation
  // calculations modify the temperature at the boundary (see
  // comments for boundaryCondition), and we do not want to 
  // change the boundary conditions for the real temperature.
  tempCopy_CCLabel = VarLabel::create("TempCopy", td_CCdouble);

  // sootVF is the variable that will eventually come from the
  // table. This will then be copied to sootVFCopy and used.
  // Right now, the table does not calculate sootVF.  The empirical
  // soot model that is used here can be put into the table,
  // but I have not done that yet; it should be done eventually
  // because that abstraction will allow the table to calculate
  // the temperature in whatever way it thinks necessary.
  // sootVFCopy is the copy of the soot volume fraction that is
  // actually used in all the radiation calculations.
  sootVF_CCLabel = VarLabel::create("sootVF", td_CCdouble);
  sootVFCopy_CCLabel = VarLabel::create("sootVFCopy", td_CCdouble);

  // radiationSrc is the source term that is used in the energy
  // equation. This variable and the fluxes are the outputs from
  // the radiation calculation to the energy transport.
  radiationSrc_CCLabel = VarLabel::create("radiationSrc", td_CCdouble);
  scalar_CCLabel = VarLabel::create("scalar-f", td_CCdouble);
}

//****************************************************************************
// Destructor
//****************************************************************************
RadiationDriver::~RadiationDriver()
{
  delete d_DORadiation;
  if(d_perproc_patches && d_perproc_patches->removeReference()){
    delete d_perproc_patches;
  }
  
  VarLabel::destroy(d_cellInfoLabel);

  VarLabel::destroy(cellType_CCLabel);  

  VarLabel::destroy(shgamma_CCLabel);  
  VarLabel::destroy(abskg_CCLabel);  
  VarLabel::destroy(esrcg_CCLabel);  
  VarLabel::destroy(cenint_CCLabel);  

  VarLabel::destroy(qfluxE_CCLabel);  
  VarLabel::destroy(qfluxW_CCLabel);  
  VarLabel::destroy(qfluxN_CCLabel);  
  VarLabel::destroy(qfluxS_CCLabel);  
  VarLabel::destroy(qfluxT_CCLabel);  
  VarLabel::destroy(qfluxB_CCLabel);

  VarLabel::destroy(co2_CCLabel);
  VarLabel::destroy(h2o_CCLabel);
  VarLabel::destroy(radCO2_CCLabel);
  VarLabel::destroy(radH2O_CCLabel);
  VarLabel::destroy(mixfrac_CCLabel);
  VarLabel::destroy(mixfracCopy_CCLabel);
  VarLabel::destroy(density_CCLabel);
  VarLabel::destroy(iceDensity_CCLabel);
  VarLabel::destroy(temp_CCLabel);
  VarLabel::destroy(iceTemp_CCLabel);
  VarLabel::destroy(tempCopy_CCLabel);
  VarLabel::destroy(sootVF_CCLabel);
  VarLabel::destroy(sootVFCopy_CCLabel);

  VarLabel::destroy(radiationSrc_CCLabel);
  VarLabel::destroy(scalar_CCLabel);
  
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void 
RadiationDriver::problemSetup(GridP& grid,
                              SimulationStateP& sharedState,
                              ModelSetup* setup)
{
  d_sharedState = sharedState;
  d_matl = d_sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = new MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();


  ProblemSpecP db = params->findBlock("RadiationModel");

  // how often radiation calculations are performed (not yet
  // operational ... July 25, 2005)
  db->getWithDefault("radiationCalcFreq",d_radCalcFreq,5);

  // logical to decide which temperature field to use; table
  // temperature or ICE temperature
  db->getWithDefault("useIceTemp",d_useIceTemp,false);

  // logical to decide whether table values should be used
  // for co2, h2o, sootVF, and temperature (but for temperature,
  // there is the additional d_useIceTemp variable) above.
  db->getWithDefault("useTableValues",d_useTableValues,true);

  db->getWithDefault("computeCO2_H2O_from_f",d_computeCO2_H2O_from_f,false);

  if (!d_useTableValues) {
    d_useIceTemp = true;
  }
  d_DORadiation = scinew Models_DORadiationModel(d_myworld);
  d_DORadiation->problemSetup(db);
}

//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void RadiationDriver::scheduleInitialize(SchedulerP& sched,
                                         const LevelP& level,
                                         const ModelInfo*)
                                         
{
  cout_doing << " Radiation:scheduleInitialize\t\t\tL-" 
             << level->getIndex() << endl;
  Task* t = scinew Task("RadiationDriver::initialize", this, 
                        &RadiationDriver::initialize);

  const PatchSet* patches= level->eachPatch();

  t->computes(qfluxE_CCLabel);
  t->computes(qfluxW_CCLabel);
  t->computes(qfluxN_CCLabel);
  t->computes(qfluxS_CCLabel);
  t->computes(qfluxT_CCLabel);
  t->computes(qfluxB_CCLabel);
  t->computes(radiationSrc_CCLabel);

  t->computes(radCO2_CCLabel);
  t->computes(radH2O_CCLabel);
  t->computes(sootVFCopy_CCLabel);
  t->computes(mixfracCopy_CCLabel);
  t->computes(abskg_CCLabel);
  t->computes(esrcg_CCLabel);
  t->computes(shgamma_CCLabel);
  t->computes(tempCopy_CCLabel);

  t->computes(cellType_CCLabel);
  
  if(d_computeCO2_H2O_from_f){
    t->computes(co2_CCLabel);
    t->computes(h2o_CCLabel);
  }
  
  sched->addTask(t, patches, d_matl_set);
}

//****************************************************************************
// Actually initialize variables at first time step
//****************************************************************************
// This function initializes the fluxes and the source terms to zero.
// But the more significant function it performs is to calculate
// the cellTypes that are needed.  The first cut for this is to
// do this for the pure fire case, in which there are no embedded
// solids. In this case, the only boundaries that need to be set
// are the open, inlet, and wall boundaries, all of which are
// at the domain boundaries.

void
RadiationDriver::initialize(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tRadiation" << endl;
  for (int p=0; p<patches->size();p++){
    const Patch* patch = patches->get(p);

    RadiationVariables vars;
    int indx = d_matl->getDWIndex();

    new_dw->allocateAndPut(vars.cellType, cellType_CCLabel, indx, patch);

    new_dw->allocateAndPut(vars.qfluxe, qfluxE_CCLabel, indx, patch);
    new_dw->allocateAndPut(vars.qfluxw, qfluxW_CCLabel, indx, patch);
    new_dw->allocateAndPut(vars.qfluxn, qfluxN_CCLabel, indx, patch);
    new_dw->allocateAndPut(vars.qfluxs, qfluxS_CCLabel, indx, patch);
    new_dw->allocateAndPut(vars.qfluxt, qfluxT_CCLabel, indx, patch);
    new_dw->allocateAndPut(vars.qfluxb, qfluxB_CCLabel, indx, patch);
    new_dw->allocateAndPut(vars.src,    radiationSrc_CCLabel, indx, patch);

    new_dw->allocateAndPut(vars.co2,    radCO2_CCLabel,        indx, patch);
    new_dw->allocateAndPut(vars.h2o,    radH2O_CCLabel,        indx, patch);
    new_dw->allocateAndPut(vars.sootVF, sootVFCopy_CCLabel,    indx, patch);
    new_dw->allocateAndPut(vars.mixfrac,mixfracCopy_CCLabel,   indx, patch);
    new_dw->allocateAndPut(vars.ABSKG,  abskg_CCLabel,         indx, patch);
    new_dw->allocateAndPut(vars.ESRCG,  esrcg_CCLabel,         indx, patch);
    new_dw->allocateAndPut(vars.shgamma,shgamma_CCLabel,       indx, patch);
    new_dw->allocateAndPut(vars.temperature, tempCopy_CCLabel, indx, patch);
    

    vars.qfluxe.initialize(0.0);
    vars.qfluxw.initialize(0.0);
    vars.qfluxn.initialize(0.0);
    vars.qfluxs.initialize(0.0);
    vars.qfluxt.initialize(0.0);
    vars.qfluxb.initialize(0.0);
    vars.src.initialize(0.0);
    vars.co2.initialize(0.03);
    vars.h2o.initialize(0.0);
    vars.sootVF.initialize(0.0);
    vars.mixfrac.initialize(0.0);
    vars.ABSKG.initialize(0.0);
    vars.ESRCG.initialize(0.0);
    vars.shgamma.initialize(0.0);
    vars.temperature.initialize(298.0);
    
    //__________________________________
    //  If we're computing CO2 & H2O concentrations
    //  from scalar-f
    if(d_computeCO2_H2O_from_f){
      CCVariable<double> H2O_concentration, CO2_concentration;
      new_dw->allocateAndPut(H2O_concentration, h2o_CCLabel,indx, patch); 
      new_dw->allocateAndPut(CO2_concentration, co2_CCLabel,indx, patch);

      constCCVariable<double> scalar_f;
      new_dw->get(scalar_f, scalar_CCLabel,indx,patch,Ghost::None,0);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        H2O_concentration[c] = 0.5 * scalar_f[c];    // CHANGE THIS EQUATION
        CO2_concentration[c] = 0.5 * scalar_f[c];
      }
    }

    //__________________________________
    //  specify cell types on the boundary
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    // set cellType in domain to be ffield, and at boundaries to be 10
    // We can generalize this with volume fraction information to account
    // for intrusions

    int ffield = -1;
    vars.cellType.initialize(ffield);

    if (xminus) {
      int colX = idxLo.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector xminusCell(colX-1, colY, colZ);
          vars.cellType[xminusCell] = 10;
        }
      }
    }
    
    if (xplus) {
      int colX = idxHi.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector xplusCell(colX+1, colY, colZ);
          vars.cellType[xplusCell] = 10;
        }
      }
    }
    
    if (yminus) {
      int colY = idxLo.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector yminusCell(colX, colY-1, colZ);
          vars.cellType[yminusCell] = 10;
        }
      }
    }

    if (yplus) {
      int colY = idxHi.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector yplusCell(colX, colY+1, colZ);
          vars.cellType[yplusCell] = 10;
        }
      }
    }

    if (zminus) {
      int colZ = idxLo.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector zminusCell(colX, colY, colZ-1);
          vars.cellType[zminusCell] = 10;
        }
      }
    }
      
    if (zplus) {
      int colZ = idxHi.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector zplusCell(colX, colY, colZ+1);
          vars.cellType[zplusCell] = 10;
        }
      }
    }
    // Here you would add the section that would look like:
    // for CellIterator inside domain, if volume fraction
    // of solid (which you would need to get in fluid-solid
    // problems, from MPM) is greater than 0.5, set cellType
    // to be wall (something other than -1 and 10). For this
    // case, you would get the temperature of this cell from
    // MPM. The computation of radiative emission from these
    // cells is already taken care of by the code in the
    // radiation modules. The resulting fluxes that are 
    // transmitted to the solid boundary are also obtained
    // in the same stairstepped fashion.
  }
  //end of patches loop
}

//****************************************************************************
// dummy
//****************************************************************************
void RadiationDriver::scheduleComputeStableTimestep(SchedulerP&,
                                                    const LevelP&,
                                                    const ModelInfo*)
{
  // not applicable
}

//****************************************************************************
// schedule the computation of radiation fluxes and the source term
//****************************************************************************
void
RadiationDriver::scheduleComputeModelSources(SchedulerP& sched,
                                             const LevelP& level,
                                             const ModelInfo* mi)
{
  cout_doing << "RADIATION::scheduleComputeModelSources\t\t\tL-" 
             << level->getIndex() << endl;
  const PatchSet* patches = level->eachPatch();

  string taskname = "RadiationDriver::buildLinearMatrix";

  Task* t = scinew Task("RadiationDriver::buildLinearMatrix", this, 
                        &RadiationDriver::buildLinearMatrix);

  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->createPerProcessorPatchSet(level);
  d_perproc_patches->addReference();

  sched->addTask(t, d_perproc_patches, d_matl_set);
  
  scheduleComputeCO2_H2O(   level,sched, patches, d_matl_set);     
  scheduleCopyValues(       level,sched, patches, d_matl_set);
  scheduleComputeProps(     level,sched, patches, d_matl_set);    
  scheduleBoundaryCondition(level,sched, patches, d_matl_set);    
  scheduleIntensitySolve(   level,sched, patches, d_matl_set, mi);
}

//****************************************************************************
// Initialize linear solver matrix for petsc/hypre
//****************************************************************************
void
RadiationDriver::buildLinearMatrix(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();
  cout_doing << "Doing buildLinearMatrix on level "<<L_indx
               << "\t\t\t Radiation" << endl;
  d_DORadiation->d_linearSolver->matrixCreate(d_perproc_patches, patches);
}
/*______________________________________________________________________
 Task~  scheduleComputeCO2_H2O--
 Purpose: Backout the CO2 and H2O concentrations from the mixture
          fraction.  The mixture fraction (scalar-f) is computed in the
          passiveScalar model
 _____________________________________________________________________  */
void
RadiationDriver::scheduleComputeCO2_H2O(const LevelP& level,
                                    SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  if(d_computeCO2_H2O_from_f){
    cout_doing << "RADIATION::scheduleComputeCO2_H2O \t\tL-" 
               << level->getIndex() << endl;
    Task* t = scinew Task("RadiationDriver::computeCO2_H2O",
                    this, &RadiationDriver::computeCO2_H2O);
    Ghost::GhostType  gn = Ghost::None;
    t->requires(Task::OldDW, scalar_CCLabel, gn, 0);
                    
    t->computes(co2_CCLabel);
    t->computes(h2o_CCLabel);  
    sched->addTask(t, patches, matls);               
  }
}
//______________________________________________________________________
void 
RadiationDriver::computeCO2_H2O(const ProcessorGroup*, 
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing computeCO2_H2O on patch "<<patch->getID()
               << "\t\t\t\t Radiation" << endl;
    int indx = d_matl->getDWIndex();
    
    CCVariable<double> H2O_concentration, CO2_concentration;
    new_dw->allocateAndPut(H2O_concentration, h2o_CCLabel,indx, patch); 
    new_dw->allocateAndPut(CO2_concentration, co2_CCLabel,indx, patch);
    
    constCCVariable<double> scalar_f;
    old_dw->get(scalar_f, scalar_CCLabel,indx,patch,Ghost::None,0);
        
    for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
      IntVector c = *iter;
      H2O_concentration[c] = 0.5 * scalar_f[c];    // CHANGE THIS EQUATION
      CO2_concentration[c] = 0.5 * scalar_f[c];
    }
  }
}
//****************************************************************************
// schedule copy of values from previous time step; if the radiation is to
// be updated using the radcounter, then we perform radiation calculations,
// else we just use the values from previous time step
//****************************************************************************
void
RadiationDriver::scheduleCopyValues(const LevelP& level,
                                    SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  cout_doing << "RADIATION::scheduleCopyValues \t\tL-" 
             << level->getIndex() << endl;
  Task* t = scinew Task("RadiationDriver::copyValues",
                      this, &RadiationDriver::copyValues);

  Ghost::GhostType  gn = Ghost::None;
  t->requires(Task::OldDW, cellType_CCLabel,    gn, 0);  
  t->requires(Task::OldDW, qfluxE_CCLabel,      gn, 0);
  t->requires(Task::OldDW, qfluxW_CCLabel,      gn, 0);
  t->requires(Task::OldDW, qfluxN_CCLabel,      gn, 0);
  t->requires(Task::OldDW, qfluxS_CCLabel,      gn, 0);
  t->requires(Task::OldDW, qfluxT_CCLabel,      gn, 0);
  t->requires(Task::OldDW, qfluxB_CCLabel,      gn, 0);
  t->requires(Task::OldDW, radiationSrc_CCLabel, gn, 0);

  t->requires(Task::OldDW, radCO2_CCLabel,      gn, 0);
  t->requires(Task::OldDW, radH2O_CCLabel,      gn, 0);  
  t->requires(Task::OldDW, sootVFCopy_CCLabel,  gn, 0);
  t->requires(Task::OldDW, mixfracCopy_CCLabel, gn, 0);
  t->requires(Task::OldDW, abskg_CCLabel,       gn, 0);
  t->requires(Task::OldDW, esrcg_CCLabel,       gn, 0);
  t->requires(Task::OldDW, shgamma_CCLabel,     gn, 0);
  t->requires(Task::OldDW, tempCopy_CCLabel,    gn, 0);

  t->computes(cellType_CCLabel);
  t->computes(qfluxE_CCLabel);
  t->computes(qfluxW_CCLabel);
  t->computes(qfluxN_CCLabel);
  t->computes(qfluxS_CCLabel);
  t->computes(qfluxT_CCLabel);
  t->computes(qfluxB_CCLabel);
  t->computes(radiationSrc_CCLabel);

  t->computes(radCO2_CCLabel);
  t->computes(radH2O_CCLabel);
  t->computes(sootVFCopy_CCLabel);
  t->computes(mixfracCopy_CCLabel);
  t->computes(abskg_CCLabel);
  t->computes(esrcg_CCLabel);
  t->computes(shgamma_CCLabel);
  t->computes(tempCopy_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual copy of old values of fluxes and source term
//****************************************************************************

void
RadiationDriver::copyValues(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    cout_doing << "Doing copyValues on patch "<<patch->getID()
               << "\t\t\t\t Radiation" << endl;
    int indx = d_matl->getDWIndex();

    constCCVariable<int> oldPcell;
    constCCVariable<double> oldFluxE;
    constCCVariable<double> oldFluxW;
    constCCVariable<double> oldFluxN;
    constCCVariable<double> oldFluxS;
    constCCVariable<double> oldFluxT;
    constCCVariable<double> oldFluxB;
    constCCVariable<double> oldRadiationSrc;

    constCCVariable<double> oldRadCO2;
    constCCVariable<double> oldRadH2O;
    constCCVariable<double> oldSootVF;
    constCCVariable<double> oldMixfrac;
    constCCVariable<double> oldAbskg;
    constCCVariable<double> oldEsrcg;
    constCCVariable<double> oldShgamma;
    constCCVariable<double> oldTempCopy;
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(oldPcell, cellType_CCLabel, indx, patch,gn, 0);
    old_dw->get(oldFluxE, qfluxE_CCLabel,   indx, patch,gn, 0);
    old_dw->get(oldFluxW, qfluxW_CCLabel,   indx, patch,gn, 0);
    old_dw->get(oldFluxN, qfluxN_CCLabel,   indx, patch,gn, 0);
    old_dw->get(oldFluxS, qfluxS_CCLabel,   indx, patch,gn, 0);
    old_dw->get(oldFluxT, qfluxT_CCLabel,   indx, patch,gn, 0);
    old_dw->get(oldFluxB, qfluxB_CCLabel,   indx, patch,gn, 0);
    old_dw->get(oldRadiationSrc, radiationSrc_CCLabel, indx, patch,gn, 0);

    old_dw->get(oldRadCO2,  radCO2_CCLabel,     indx, patch,gn, 0);
    old_dw->get(oldRadH2O,  radH2O_CCLabel,     indx, patch,gn, 0);
    old_dw->get(oldSootVF,  sootVFCopy_CCLabel, indx, patch,gn, 0);
    old_dw->get(oldMixfrac, mixfracCopy_CCLabel,indx, patch,gn, 0);
    old_dw->get(oldAbskg,   abskg_CCLabel,      indx, patch,gn, 0);
    old_dw->get(oldEsrcg,   esrcg_CCLabel,      indx, patch,gn, 0);
    old_dw->get(oldShgamma, shgamma_CCLabel,    indx, patch,gn, 0);
    old_dw->get(oldTempCopy,tempCopy_CCLabel,   indx, patch,gn, 0);

    CCVariable<int> pcell;
    CCVariable<double> fluxE;
    CCVariable<double> fluxW;
    CCVariable<double> fluxN;
    CCVariable<double> fluxS;
    CCVariable<double> fluxT;
    CCVariable<double> fluxB;
    CCVariable<double> radiationSrc;

    CCVariable<double> radCO2;
    CCVariable<double> radH2O;
    CCVariable<double> sootVF;
    CCVariable<double> mixfrac;
    CCVariable<double> abskg;
    CCVariable<double> esrcg;
    CCVariable<double> shgamma;
    CCVariable<double> tempCopy;
    
    new_dw->allocateAndPut(pcell, cellType_CCLabel, indx, patch);    
    new_dw->allocateAndPut(fluxE, qfluxE_CCLabel,   indx, patch);   
    new_dw->allocateAndPut(fluxW, qfluxW_CCLabel,   indx, patch);   
    new_dw->allocateAndPut(fluxN, qfluxN_CCLabel,   indx, patch);   
    new_dw->allocateAndPut(fluxS, qfluxS_CCLabel,   indx, patch);   
    new_dw->allocateAndPut(fluxT, qfluxT_CCLabel,   indx, patch);   
    new_dw->allocateAndPut(fluxB, qfluxB_CCLabel,   indx, patch);   
    new_dw->allocateAndPut(radiationSrc, radiationSrc_CCLabel, indx, patch);    

    new_dw->allocateAndPut(radCO2,  radCO2_CCLabel,      indx, patch); 
    new_dw->allocateAndPut(radH2O,  radH2O_CCLabel,      indx, patch); 
    new_dw->allocateAndPut(sootVF,  sootVFCopy_CCLabel,  indx, patch); 
    new_dw->allocateAndPut(mixfrac, mixfracCopy_CCLabel, indx, patch); 
    new_dw->allocateAndPut(abskg,   abskg_CCLabel,       indx, patch); 
    new_dw->allocateAndPut(esrcg,   esrcg_CCLabel,       indx, patch); 
    new_dw->allocateAndPut(shgamma, shgamma_CCLabel,     indx, patch); 
    new_dw->allocateAndPut(tempCopy,tempCopy_CCLabel,    indx, patch); 

    pcell.copyData(oldPcell);
    fluxE.copyData(oldFluxE);
    fluxW.copyData(oldFluxW);
    fluxN.copyData(oldFluxN);
    fluxS.copyData(oldFluxS);
    fluxT.copyData(oldFluxT);
    fluxB.copyData(oldFluxB);
    radiationSrc.copyData(oldRadiationSrc);

    radCO2.copyData(oldRadCO2);
    radH2O.copyData(oldRadH2O);
    sootVF.copyData(oldSootVF);
    mixfrac.copyData(oldMixfrac);
    abskg.copyData(oldAbskg);
    esrcg.copyData(oldEsrcg);
    shgamma.copyData(oldShgamma);
    tempCopy.copyData(oldTempCopy);
    
    
    
  }
#if 0   // furture changes
  new_dw->transferFrom(old_dw,radCO2_CCLabel,     patches, matls);
  new_dw->transferFrom(old_dw,radH2O_CCLabel,     patches, matls);
  new_dw->transferFrom(old_dw,sootVFCopy_CCLabel, patches, matls);
  new_dw->transferFrom(old_dw,mixfracCopy_CCLabel,patches, matls);
  new_dw->transferFrom(old_dw,abskg_CCLabel,      patches, matls);
  new_dw->transferFrom(old_dw,esrcg_CCLabel,      patches, matls);
  new_dw->transferFrom(old_dw,shgamma_CCLabel,    patches, matls);
  new_dw->transferFrom(old_dw,tempCopy_CCLabel,   patches, matls);
#endif
  // end patches loop
}

//****************************************************************************
// schedule computation of radiative properties
//****************************************************************************
void
RadiationDriver::scheduleComputeProps(const LevelP& level,
                                       SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{

  cout_doing << "RADIATION::scheduleComputeProps \t\t\tL-" 
             << level->getIndex() << endl;
             
  Task* t=scinew Task("RadiationDriver::computeProps",
                this, &RadiationDriver::computeProps);

  Ghost::GhostType  gn = Ghost::None;
  
  if (d_useTableValues) {
    t->requires(Task::NewDW, co2_CCLabel, gn, 0);
    t->requires(Task::NewDW, h2o_CCLabel, gn, 0);
  }
  t->modifies(radCO2_CCLabel);
  t->modifies(radH2O_CCLabel);

  if (d_useTableValues){ 
    if (d_useIceTemp){
      t->requires(Task::OldDW, iceTemp_CCLabel, gn, 0);
    }else{
      t->requires(Task::NewDW, temp_CCLabel, gn, 0);
    }
  }else{
    t->requires(Task::OldDW, iceTemp_CCLabel, gn, 0);
  }
  t->modifies(tempCopy_CCLabel);

  // Below is for later, when we get sootVF from reaction table.  But
  // currently we compute sootVF from mixture fraction and temperature inside
  // the properties function
  //  t->requires(Task::NewDW, sootVF_CCLabel, gn, 0);

  t->modifies(sootVFCopy_CCLabel);

  if (d_useTableValues){
    t->requires(Task::OldDW, mixfrac_CCLabel, gn, 0);
  }
  t->modifies(mixfracCopy_CCLabel);

  if (d_useTableValues){
    t->requires(Task::NewDW, density_CCLabel, gn, 0);
  }else{
    t->requires(Task::NewDW, iceDensity_CCLabel, gn, 0);
  }
  
  t->modifies(abskg_CCLabel);
  t->modifies(esrcg_CCLabel);
  t->modifies(shgamma_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual compute of properties
//****************************************************************************

// This function computes soot volume fraction, absorption coefficient,
// blackbody emissive intensity (and shgamma (1/3k) for spherical
// harmonics), using co2, h2o, mixture fraction, and temperature 
// values.  The soot
// values are calculated using the Sarofim model at present;
// this model calculates soot as a function of stoichiometry
// (mixture fraction), density, and temperature; this is then fed
// to the property model that calculates absorption coefficient
// (abskg), blackbody emissive intensity (esrcg), shgamma 
// (= 1/(3*abskg)) as a function of co2, h2o, soot volume fraction,
// and temperature.  Modification to properties are also made for
// the test problems. 
// There is an option to get temperature values from the table, which is
// how you would do reacting flow cases.  But if this option is
// not used (i.e., if d_useIceTemp = true), the temperature that is
// used comes from the ICE calculations.  In an ideal world, we
// would always use the ICE temperatures, but since those temperatures
// are way off for the fire at the time of writing (July 25,2005),
// we use the table temperatures for fire.
// Another logical used here is d_useTableValues, which tells
// you whether you want to use the table values at all.
// If this option is false, we HAVE to use the ICE values for
// temperature, AND hardwire co2 and h2o concentrations.
// You can have d_useTableValues = true but d_useIceTemp = true,
// which means that all values other than temperature (i.e.,
// co2, h2o, and soot volume fraction) are calculated from
// the table.  This is the eventual option for doing fire in
// ICE.
// If you are performing a nongray calculation, then the only
// thing this function does is to calculate the soot volume 
// fraction used in radiation calculations. (Well, the test
// problems are also affected.)

void
RadiationDriver::computeProps(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    cout_doing << "Doing computeProps on patch "<<patch->getID()
               << "\t\t\t\t Radiation" << endl;
    int indx = d_matl->getDWIndex();
    
    Ghost::GhostType  gn = Ghost::None;
        
    RadiationVariables radVars;
    RadiationConstVariables constRadVars;
    
    PerPatch<Models_CellInformationP> cellInfoP;
    if (new_dw->exists(d_cellInfoLabel, indx, patch)) 
      new_dw->get(cellInfoP, d_cellInfoLabel, indx, patch);
    else {
      cellInfoP.setData(scinew Models_CellInformation(patch));
      new_dw->put(cellInfoP, d_cellInfoLabel, indx, patch);
    }
    Models_CellInformation* cellinfo = cellInfoP.get().get_rep();

    if (d_useTableValues) {
      new_dw->get(constRadVars.co2, co2_CCLabel, indx, patch,gn, 0);
      new_dw->get(constRadVars.h2o, h2o_CCLabel, indx, patch,gn, 0);
    }
    new_dw->getModifiable(radVars.co2, radCO2_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.h2o, radH2O_CCLabel, indx, patch);
    if (d_useTableValues) {
      radVars.co2.copyData(constRadVars.co2);
      radVars.h2o.copyData(constRadVars.h2o);
    }

    if (d_useTableValues){ 
      if (d_useIceTemp){ 
        new_dw->get(constRadVars.temperature, temp_CCLabel,    indx, patch,gn, 0);
      }else{
        old_dw->get(constRadVars.temperature, iceTemp_CCLabel, indx, patch,gn, 0);
      }
    }else{
      old_dw->get(constRadVars.temperature, iceTemp_CCLabel, indx, patch,gn, 0);
    }
    new_dw->getModifiable(radVars.temperature, tempCopy_CCLabel, indx, patch);
    
    radVars.temperature.copyData(constRadVars.temperature);
    
    // We will use constRadVars.sootVF when we get it from the table.
    // For now, calculate sootVF in radcoef.F
    // As long as the test routines are embedded in the properties and
    // boundary conditions, we will need radVars.sootVF, because those
    // test routines modify the soot properties (bad design).
    // So until that is fixed, the idea is to get constRadVars.sootVF
    // and copy it to radVars.sootVF, and use radVars.sootVF in our
    // calculations.  Instead, we now compute sootVF from mixture fraction
    // and temperature inside the properties function
    // 
    //      new_dw->get(constRadVars.sootVF, sootVF_CCLabel, indx, patch,
    //            gn, 0);
    new_dw->getModifiable(radVars.sootVF, sootVFCopy_CCLabel, 
                          indx, patch);

    new_dw->getModifiable(radVars.mixfrac, mixfracCopy_CCLabel, indx, patch);
    if (d_useTableValues) {
      old_dw->get(constRadVars.mixfrac, mixfrac_CCLabel, indx, patch,gn, 0);
      radVars.mixfrac.copyData(constRadVars.mixfrac);
    }
      
    if (d_useTableValues){
      new_dw->get(constRadVars.density, density_CCLabel,   indx, patch,gn, 0);
    }else{
      new_dw->get(constRadVars.density, iceDensity_CCLabel, indx, patch,gn, 0);
    }
    new_dw->getModifiable(radVars.ABSKG,   abskg_CCLabel,   indx, patch);
    new_dw->getModifiable(radVars.ESRCG,   esrcg_CCLabel,   indx, patch);
    new_dw->getModifiable(radVars.shgamma, shgamma_CCLabel, indx, patch);

    d_radCounter = d_sharedState->getCurrentTopLevelTimeStep();
    if (d_radCounter%d_radCalcFreq == 0) {
      d_DORadiation->computeRadiationProps(pc, patch, cellinfo,
                                           &radVars, &constRadVars);
    }
  }
}

//****************************************************************************
// schedule computation of radiative boundary condition
//****************************************************************************
void 
RadiationDriver::scheduleBoundaryCondition(const LevelP& level,
                                           SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  cout_doing << "RADIATION::scheduleBoundaryCondition\t\t\tL-" 
             << level->getIndex() << endl;
  Task* t=scinew Task("RadiationDriver::boundaryCondition",
                this, &RadiationDriver::boundaryCondition);

  t->requires(Task::NewDW, cellType_CCLabel, Ghost::AroundCells, 1);
  t->modifies(tempCopy_CCLabel);
  t->modifies(abskg_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual boundary condition for radiation
//****************************************************************************
// This function sets boundary conditions for the radiative heat transfer.
// The boundary conditions are for temperature and abskg.
// The temperature boundary conditions for the open domain fire
// are 293 K at all boundaries, regardless of the type of 
// simulation.  This is because there is ambiguity in the 
// specification of the temperature at the boundary in open
// systems.  What temperature does the fire see?  The use of
// a Neumann bc at the edges is not accurate, because the fire
// sees distances up to infinity in open domains.  We have seen
// (in Arches) that setting Neumann boundary conditions at the
// edges of the fire leads to instability and an unbounded
// reduction in the temperature. So we do the sanest thing:
// we set the temperature to be what the fire will see at 
// infinity.
// The other boundary condition is that for abskg. Here, it should
// be noted that abskg at the boundary really is storage for
// the emissivity at the boundaries. Absorption coefficient is
// a property of the participating medium (co2, h2o, soot),
// whereas emissivity is a property of a surface. The emissivity
// of an imaginary boundary plane cannot be specified in an 
// open domain, much as the temperature cannot be set at that
// plane in any exact fashion. So we set the emissivity to be
// one; this means that the boundary is a perfect emitter and 
// absorber.

void
RadiationDriver::boundaryCondition(const ProcessorGroup* pc,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* ,
                                   DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    cout_doing << "Doing boundaryCondition on patch "<<patch->getID()
               << "\t\t\t Radiation" << endl;
    
    int indx = d_matl->getDWIndex();
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    RadiationVariables radVars;
    RadiationConstVariables constRadVars;
    d_radCounter = d_sharedState->getCurrentTopLevelTimeStep();

    new_dw->get(constRadVars.cellType,         cellType_CCLabel,indx, patch,gac, 1);
    new_dw->getModifiable(radVars.temperature, tempCopy_CCLabel,indx, patch);
    new_dw->getModifiable(radVars.ABSKG,        abskg_CCLabel,  indx, patch);

    if (d_radCounter%d_radCalcFreq == 0) {
      d_DORadiation->boundaryCondition(pc, patch, &radVars, &constRadVars);
    }
  }
}

//****************************************************************************
// schedule solve for radiative intensities, radiative source, and heat fluxes
//****************************************************************************
void
RadiationDriver::scheduleIntensitySolve(const LevelP& level,
                                        SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls,
                                        const ModelInfo* mi)
{
  cout_doing << "RADIATION::scheduleIntensitySolve\t\t\tL-" 
             << level->getIndex() << endl;
  Task* t=scinew Task("RadiationDriver::intensitySolve",
                this, &RadiationDriver::intensitySolve, mi);
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  t->requires(Task::NewDW, radCO2_CCLabel,    gac, 1);
  t->requires(Task::NewDW, radH2O_CCLabel,    gac, 1);
  t->requires(Task::NewDW, sootVFCopy_CCLabel,gac, 1);
  t->requires(Task::NewDW, cellType_CCLabel,  gac, 1);
  t->requires(Task::NewDW, tempCopy_CCLabel,  gac, 1);
  t->requires(Task::NewDW, shgamma_CCLabel,   gac, 1);
  t->requires(Task::NewDW, abskg_CCLabel,     gac, 1);
  t->requires(Task::NewDW, esrcg_CCLabel,     gac, 1);

  t->modifies(qfluxE_CCLabel);
  t->modifies(qfluxW_CCLabel);
  t->modifies(qfluxN_CCLabel);
  t->modifies(qfluxS_CCLabel);
  t->modifies(qfluxT_CCLabel);
  t->modifies(qfluxB_CCLabel);
  t->modifies(radiationSrc_CCLabel);

  t->modifies(mi->energy_source_CCLabel);

  sched->addTask(t, patches, matls);
}

//****************************************************************************
// Actual solve for radiative intensities, radiative source and heat fluxes
//****************************************************************************

void
RadiationDriver::intensitySolve(const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* ,
                                DataWarehouse* new_dw,
                                const ModelInfo* mi)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing intensitySolve on patch "<<patch->getID()
               << "\t\t\t\t Radiation" << endl;
    int indx = d_matl->getDWIndex();
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    RadiationVariables radVars;
    RadiationConstVariables constRadVars;
    CCVariable<double> energySource;

    IntVector domLo = patch->getCellLowIndex();
    IntVector domHi = patch->getCellHighIndex();
    CCVariable<double> zeroSource;
    zeroSource.allocate(domLo,domHi);
    zeroSource.initialize(0.0);

    PerPatch<Models_CellInformationP> cellInfoP;
    if (new_dw->exists(d_cellInfoLabel, indx, patch)) 
      new_dw->get(cellInfoP, d_cellInfoLabel, indx, patch);
    else {
      cellInfoP.setData(scinew Models_CellInformation(patch));
      new_dw->put(cellInfoP, d_cellInfoLabel, indx, patch);
    }
    Models_CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constRadVars.co2,        radCO2_CCLabel,    indx, patch,gac,1);
    new_dw->get(constRadVars.h2o,        radH2O_CCLabel,    indx, patch,gac,1);
    new_dw->get(constRadVars.sootVF,     sootVFCopy_CCLabel,indx, patch,gac,1);
    new_dw->get(constRadVars.temperature,tempCopy_CCLabel,  indx, patch,gac,1);
    new_dw->get(constRadVars.cellType,   cellType_CCLabel,  indx, patch,gac,1);

    new_dw->getCopy(radVars.ABSKG,  abskg_CCLabel,   indx, patch, gac,1);
    new_dw->getCopy(radVars.shgamma, shgamma_CCLabel,indx, patch, gac,1);
    new_dw->getCopy(radVars.ESRCG,   esrcg_CCLabel,  indx, patch, gac,1);

    new_dw->getModifiable(radVars.qfluxe, qfluxE_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.qfluxw, qfluxW_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.qfluxn, qfluxN_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.qfluxs, qfluxS_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.qfluxt, qfluxT_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.qfluxb, qfluxB_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.src, radiationSrc_CCLabel, indx, patch);

    new_dw->getModifiable(energySource, mi->energy_source_CCLabel, indx, patch);

    d_radCounter = d_sharedState->getCurrentTopLevelTimeStep();
    if (d_radCounter%d_radCalcFreq == 0) {

      radVars.qfluxe.initialize(0.0);
      radVars.qfluxw.initialize(0.0);
      radVars.qfluxn.initialize(0.0);
      radVars.qfluxs.initialize(0.0);
      radVars.qfluxt.initialize(0.0);
      radVars.qfluxb.initialize(0.0);
      radVars.src.initialize(0.0);

      d_DORadiation->intensitysolve(pc, patch, cellinfo, &radVars, &constRadVars);
    }
    
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          double vol=cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
          energySource[currCell] += vol*radVars.src[currCell];
        }
      }
    }
  }
}
//______________________________________________________________________
void RadiationDriver::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                              const LevelP&,
                                                              const MaterialSet*)
{
  // not applicable  
}
void RadiationDriver::computeSpecificHeat(CCVariable<double>&,
                                          const Patch*,
                                          DataWarehouse*,
                                          const int)
{
  // not applicable
}
//______________________________________________________________________
//
void RadiationDriver::scheduleErrorEstimate(const LevelP&,
                                            SchedulerP&)
{
  // This may not be ever implemented, since it is known that the radiation
  // calculations work fine with a coarse mesh.
}
//__________________________________
void RadiationDriver::scheduleTestConservation(SchedulerP&,
                                               const PatchSet*,
                                               const ModelInfo*)
{
  // do nothing; there is a conservation test in Models_DORadiationModel;
  // perhaps at a later time, I will move it here.
}
