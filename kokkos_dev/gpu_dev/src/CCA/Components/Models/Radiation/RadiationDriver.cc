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


#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/Models/Radiation/Models_CellInformationP.h>
#include <CCA/Components/Models/Radiation/Models_DORadiationModel.h>
#include <CCA/Components/Models/Radiation/RadiationDriver.h>
#include <CCA/Components/Models/Radiation/Models_RadiationModel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/ModelMaker.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
//______________________________________________________________________
/* To Do:
    
*/
//______________________________________________________________________


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
  d_sigma = 5.670e-8;  //Boltzmann constant   W/(M^2 K^4)
  
  Ilb  = scinew ICELabel();
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();

  // This is PerPatch structure inherited from Arches.
  // It is used to store all sorts of cell dimentions (e.g. dx, dy, dz plus
  // everything else for nonuniform staggered grid).
  // Currently, and overkill to use, but removing it could be a pain.
  d_cellInfoLabel = VarLabel::create("radCellInformation",
                                     PerPatch<Models_CellInformationP>::getTypeDescription());

  // cellType_CC is the variable to determine the location of
  // boundaries, whether they are of the immersed solid or those
  // of open or wall boundaries. This variable is necessary because
  // radiation is a surface-dependent process.

  cellType_CCLabel = VarLabel::create("cellType_CC", CCVariable<int>::getTypeDescription());

  // shgamma is simply 1/(3*abskg). It is calculated and stored 
  // this way because it is more efficient to do so than just
  // calculating it on the fly each time.  This is done because
  // gradients of shgamma are needed in the P1 calculation.

  shgamma_CCLabel  = VarLabel::create("shgamma", td_CCdouble);

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
  
  // diagnostic variable
  solidEmissionLabel = VarLabel::create("solidEmission", td_CCdouble);
  
  // defines what cells are on the solid surface
  isGasSolidInterfaceLabel = VarLabel::create("isGasSolidInterface", td_CCdouble);
  
  // defines what cells have an absorbing solid
  insideSolidLabel = VarLabel::create("insideSolid", td_CCdouble);
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
  VarLabel::destroy(solidEmissionLabel);
  VarLabel::destroy(isGasSolidInterfaceLabel);
  VarLabel::destroy(insideSolidLabel);
  
  delete Ilb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void 
RadiationDriver::problemSetup(GridP& grid,
                              SimulationStateP& sharedState,
                              ModelSetup* setup)
{
  cout_doing << " RadiationDriver::problemSetup "<< endl;
  
  d_sharedState = sharedState;
  ProblemSpecP db = params->findBlock("RadiationModel");
  d_matl_G= d_sharedState->parseAndLookupMaterial(db, "radiatingGas");

  string absSolid("");
  ProblemSpecP abs_ps = db->get("absorbingSolid",absSolid);

  if (abs_ps){
    d_hasAbsorbingSolid = true;
  }else{
    d_hasAbsorbingSolid = false;
  }
  
  
  
  //__________________________________
  // absorbing solid
  //  db->getWithDefault("absorbingSolid",d_hasAbsorbingSolid, false);
  if(d_hasAbsorbingSolid){
    if (d_sharedState->getNumMPMMatls() == 0){
      ostringstream warn;
      warn<<"ERROR\n Radiation: If you have an absorbing solid you must"
        " have at least 1 mpm material and use -mpmice/-rmpmice\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }    
    d_matl_S = d_sharedState->parseAndLookupMaterial(db, "absorbingSolid");
    
    //__________________________________
    //  Read in the geometry objects of the absorbing solid 
    for (ProblemSpecP geom_obj_ps = db->findBlock("geom_object");
        geom_obj_ps != 0;
        geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

      vector<GeometryPieceP> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPieceP mainpiece;
      if(pieces.size() == 0){
        throw ProblemSetupException("\n ERROR: RADIATION MODEL: No piece specified in geom_object", __FILE__, __LINE__);
      } else if(pieces.size() > 1){
        mainpiece = scinew UnionGeometryPiece(pieces);
      } else {
        mainpiece = pieces[0];
      }
      d_geom_pieces.push_back(mainpiece);
    }

    if(d_geom_pieces.size() == 0) {
      throw ProblemSetupException("\n ERROR: RADIATION MODEL: geometry objects specified for the absorbing solid",
                                __FILE__, __LINE__);
    }
  }
 
  // how often radiation calculations are performed 
  db->getWithDefault("calcFreq",     d_radCalcFreq,     999999);  
  db->getWithDefault("calcInterval", d_radCalc_interval,999999);
  d_radCalc_nextTime = d_radCalc_interval;
  if(d_radCalcFreq == 999999 && d_radCalc_interval == 999999){
    ostringstream warn;
    warn<<"ERROR\n Radiation: If you must specify either <radiationCalcFreq> or <radiationCalc_interval>\n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }   



  // use ICE or Table density and temperature
  db->getWithDefault("table_or_ice_temp_density",d_table_or_ice_temp_density,"ice"); 
  if(d_table_or_ice_temp_density != "ice" && d_table_or_ice_temp_density != "table"){
    ostringstream warn;
    warn<<"ERROR\n Radiation: If you must specify either ice/table in <table_or_ice_temp_density>\n";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  } 

  // logical to decide whether table values should be used
  // for co2, h2o, sootVF
  // If set to false, CO2 AND h2O would be computed from scalar F
  db->getWithDefault("useTableValues",d_useTableValues,true);


  d_DORadiation = scinew Models_DORadiationModel(d_myworld);
  d_DORadiation->problemSetup(db);
}

//______________________________________________________________________
//
void RadiationDriver::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","Radiation");

  ProblemSpecP rad_ps = model_ps->appendChild("RadiationModel");
 
  rad_ps->appendElement("radiatingGas",d_matl_G->getName());
  if (d_hasAbsorbingSolid) {
    rad_ps->appendElement("absorbingSolid",d_matl_S->getName());
  }
  rad_ps->appendElement("calcFreq",d_radCalcFreq);
  rad_ps->appendElement("calcInterval",d_radCalc_interval);
  rad_ps->appendElement("table_or_ice_temp_density",
                        d_table_or_ice_temp_density);
  rad_ps->appendElement("useTableValues",d_useTableValues);

  ProblemSpecP geom_ps = rad_ps->appendChild("geom_object");
  for (vector<GeometryPieceP>::iterator it = d_geom_pieces.begin();
       it != d_geom_pieces.end(); it++) {
    (*it)->outputProblemSpec(geom_ps);
  }

  d_DORadiation->outputProblemSpec(rad_ps);

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
    
  if (!d_useTableValues){
    t->computes(co2_CCLabel);
    t->computes(h2o_CCLabel);
  }
  if(d_hasAbsorbingSolid){
    t->computes(insideSolidLabel);
  }
  
 
  int m = d_matl_G->getDWIndex();
  MaterialSet* matl_set = scinew MaterialSet();
  matl_set->add(m);
  matl_set->addReference();
  
  sched->addTask(t, patches, matl_set);
  
  if(matl_set && matl_set->removeReference()) {
    delete matl_set;
  }
}

/*______________________________________________________________________
 Function~  RadiationDriver::initialize--
 This function initializes the fluxes and the source terms to zero.
 _____________________________________________________________________*/
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
    int indx = d_matl_G->getDWIndex();

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
    vars.co2.initialize(0.0);
    vars.h2o.initialize(0.0);
    vars.sootVF.initialize(0.0);
    vars.mixfrac.initialize(0.0);
    vars.ABSKG.initialize(0.0);
    vars.ESRCG.initialize(0.0);
    vars.shgamma.initialize(0.0);
    vars.temperature.initialize(298.0);
    //__________________________________
    // If there's an absorbing solid 
    // set where the inside of the solid is using
    // the geometry pieces
    if(d_hasAbsorbingSolid){
    
      int indx_S = d_matl_S->getDWIndex();
      CCVariable<double> insideSolid;
      new_dw->allocateAndPut(insideSolid, insideSolidLabel,indx_S, patch);
      insideSolid.initialize(0.0);
      
      for(vector<GeometryPieceP>::iterator iter = d_geom_pieces.begin();
                                    iter != d_geom_pieces.end(); iter++){
        GeometryPieceP piece = *iter;
        
        for(CellIterator iter = patch->getExtraCellIterator();!iter.done(); iter++){
          IntVector c = *iter;
          Point p = patch->cellPosition(c);            
          if(piece->inside(p)) {
            insideSolid[c] = 1.0;
          }
        } // Over cells
      } // geometry pieces
    }    
    
    
    //__________________________________
    //  If we're computing CO2 & H2O concentrations
    //  from scalar-f initialize them
    if(!d_useTableValues){
      CCVariable<double> H2O_concentration, CO2_concentration;
      new_dw->allocateAndPut(H2O_concentration, h2o_CCLabel,indx, patch); 
      new_dw->allocateAndPut(CO2_concentration, co2_CCLabel,indx, patch);

      constCCVariable<double> scalar_f;
      new_dw->get(scalar_f, scalar_CCLabel,indx,patch,Ghost::None,0);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        H2O_concentration[c] = 0.085 * scalar_f[c];
        CO2_concentration[c] = 0.18 * scalar_f[c];
      }
    }
  }
}

//****************************************************************************
// schedule the computation of radiation fluxes and the source term
//****************************************************************************
void
RadiationDriver::scheduleComputeModelSources(SchedulerP& sched,
                                             const LevelP& level,
                                             const ModelInfo* mi)
{
  cout_doing << "RADIATION::scheduleComputeModelSources\t\t\t\tL-" 
             << level->getIndex() << endl;
  const PatchSet* patches = level->eachPatch();

  string taskname = "RadiationDriver::buildLinearMatrix";

  Task* t = scinew Task("RadiationDriver::buildLinearMatrix", this, 
                        &RadiationDriver::buildLinearMatrix);

  // d_perproc_patches is a set of all patches on all processors for a given
  // problem. It is used by Petsc to figure out the local-to-global mapping it
  // needs. This annoyance (l2g mapping) is not required by Hypre.
  if(d_perproc_patches && d_perproc_patches->removeReference())
    delete d_perproc_patches;
  LoadBalancer* lb = sched->getLoadBalancer();
  d_perproc_patches = lb->getPerProcessorPatchSet(level);
  d_perproc_patches->addReference();

  // compute the material set and subset
  int m  = d_matl_G->getDWIndex();
  MaterialSet* matl_set_G  = scinew MaterialSet();
  MaterialSet* matl_set_GS = scinew MaterialSet();
  const MaterialSubset* mss_G = d_matl_G->thisMaterial();
  const MaterialSubset* mss_S = NULL; 
  
  vector<int> g;
  g.push_back(m);
  if(d_hasAbsorbingSolid){
    g.push_back(d_matl_S->getDWIndex());
    mss_S = d_matl_S->thisMaterial();
  }
  matl_set_G->add(m);
  matl_set_GS->addAll(g);
  
  matl_set_G->addReference();
  matl_set_GS->addReference();
  
  //__________________________________
  // shedule the tasts
  sched->addTask(t, d_perproc_patches, matl_set_G);
  
  scheduleSet_cellType(     level,sched, patches, mss_G,
                                                  mss_S,
                                                  matl_set_GS);
                                                  
  scheduleComputeCO2_H2O(   level,sched, patches, matl_set_G);     
  scheduleCopyValues(       level,sched, patches, mss_G, 
                                                  mss_S,
                                                  matl_set_GS);
                                                  
  scheduleComputeProps(     level,sched, patches, mss_G,
                                                  mss_S,
                                                  matl_set_GS);    
  scheduleBoundaryCondition(level,sched, patches, matl_set_G);    
  scheduleIntensitySolve(   level,sched, patches, mss_G,
                                                  mss_S,  
                                                  matl_set_GS, mi);
  
  if(matl_set_G && matl_set_G->removeReference()) {
    delete matl_set_G;
  }
  if(matl_set_GS && matl_set_GS->removeReference()) {
    delete matl_set_GS;
  }
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
  cout_doing << "Doing buildLinearMatrix on level "<<level->getIndex()
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
  if(!d_useTableValues){
    cout_doing << "RADIATION::scheduleComputeCO2_H2O \t\t\t\tL-" 
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
    int indx = d_matl_G->getDWIndex();
    
    CCVariable<double> H2O_concentration, CO2_concentration;
    new_dw->allocateAndPut(H2O_concentration, h2o_CCLabel,indx, patch); 
    new_dw->allocateAndPut(CO2_concentration, co2_CCLabel,indx, patch);
    
    constCCVariable<double> scalar_f;
    old_dw->get(scalar_f, scalar_CCLabel,indx,patch,Ghost::None,0);
        
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();  iter++){
      IntVector c = *iter;
      H2O_concentration[c] = 0.085 * scalar_f[c];   // Hardwired for JP8
      CO2_concentration[c] = 0.18 * scalar_f[c];
    }
  }
}

/*______________________________________________________________________
 Task~  scheduleSet_cellType--
 Purpose:  set the cell type through out the domain  
 _____________________________________________________________________  */
void
RadiationDriver::scheduleSet_cellType(const LevelP&         level,
                                      SchedulerP&           sched,
                                      const PatchSet*       patches,
                                      const MaterialSubset* mss_G,
                                      const MaterialSubset* mss_S,
                                      const MaterialSet*    matls)
{
  // TODO:  We don't need to do this every timestep
  cout_doing << "RADIATION::scheduleSet_cellType \t\t\t\tL-" 
             << level->getIndex() << endl;

  Task* t = scinew Task("RadiationDriver::set_cellType",
                        this, &RadiationDriver::set_cellType);

  Ghost::GhostType  gn = Ghost::None;

  if(d_hasAbsorbingSolid){
    t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, mss_S, gn,0);
    t->requires(Task::OldDW, insideSolidLabel,      mss_S, gn,0);
    t->computes(isGasSolidInterfaceLabel, mss_S);       
  }

  t->requires(Task::OldDW, abskg_CCLabel,  mss_G, gn,0);

  t->computes(cellType_CCLabel, mss_G);
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
void 
RadiationDriver::set_cellType(const ProcessorGroup*, 
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{ 
  const Level* level = getLevel(patches);
  int levelIndex = level->getIndex();
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing set_cellType on patch "<<patch->getID()
               << "\t\t\t\t Radiation L-" << levelIndex<< endl;
               
    //__________________________________
    // Is it time to radiate?
    // You need to check this in the first task
    // and only on the first patch of level 0
    d_radCounter = d_sharedState->getCurrentTopLevelTimeStep();
    
    d_doRadCalc = false;
    if (d_radCalcFreq > 0){
      if (d_radCounter%d_radCalcFreq == 0 ) {
        d_doRadCalc = true;
      }
    }
    double time= d_dataArchiver->getCurrentTime() + 1E-100;

    if (time >= d_radCalc_nextTime) {
      d_doRadCalc  = true;
      d_radCalc_nextTime = d_radCalc_interval 
                        * ceil(time/d_radCalc_interval + 1E-100); 
    }
               
    int indx_G = d_matl_G->getDWIndex();
    //__________________________________
    //default value
    RadiationVariables vars;
    Ghost::GhostType gn = Ghost::None;
    
    int flowField = -1;
    new_dw->allocateAndPut(vars.cellType,cellType_CCLabel,indx_G,patch);
    vars.cellType.initialize(flowField);

    //__________________________________
    //if there's an absorbing solid
    // 1) set cellType = 1
    // 2) find the gas-solid interface
    if(d_hasAbsorbingSolid){
      
      constCCVariable<double> vol_frac_solid, vol_frac_gas, insideSolid;
      CCVariable<double> isGasSolidInterface;
      int indx_S = d_matl_S->getDWIndex();
      
      new_dw->get(vol_frac_solid,Ilb->vol_frac_CCLabel,indx_S,patch,gn,0);
      new_dw->get(vol_frac_gas,  Ilb->vol_frac_CCLabel,indx_G,patch,gn,0);
      old_dw->get(insideSolid,   insideSolidLabel,     indx_S,patch,gn,0);
      new_dw->allocateAndPut(isGasSolidInterface,isGasSolidInterfaceLabel,indx_S,patch);
      isGasSolidInterface.initialize(0.0);
      
      IntVector X(1,0,0);
      IntVector Y(0,1,0);   // cell offsets
      IntVector Z(0,0,1);
      
      for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter; 
        if(insideSolid[c] == 1){
          vars.cellType[c] = 1;

          if( insideSolid[c-X] == 0 ||  insideSolid[c+X] == 0){
            isGasSolidInterface[c] = 1;        // x+, x-
          }
          if( insideSolid[c-Y] == 0 ||  insideSolid[c+Y] == 0){
            isGasSolidInterface[c] = 1;        // y+, y-
          }
          if( insideSolid[c-Z] == 0 ||  insideSolid[c+Z] == 0){
            isGasSolidInterface[c] = 1;        // z+, z-
          }   
        }        
      }
    }     
    //__________________________________
    //  set boundary conditions
    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for (iter  = bf.begin(); iter != bf.end(); ++iter){
      Patch::FaceType face = *iter;
      for(CellIterator itr = patch->getFaceIterator(face, Patch::ExtraPlusEdgeCells); 
          !itr.done();  itr++){
        IntVector c = *itr;
        vars.cellType[c] = 10;
      } 
    }  // domain faces
  }  // patches
}

/*______________________________________________________________________
 Task~  scheduleCopyValues--
 Purpose:  move the data from the old_dw to the new_dw.  
 We need to move the data forward whenever the radiation calc isn't performed.
 _____________________________________________________________________  */
void
RadiationDriver::scheduleCopyValues(const LevelP& level,
                                    SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSubset* mss_G,         
                                    const MaterialSubset* mss_S,         
                                    const MaterialSet*    matls_set_GS)  
{
  cout_doing << "RADIATION::scheduleCopyValues \t\t\t\t\tL-" 
             << level->getIndex() << endl;
  Task* t = scinew Task("RadiationDriver::copyValues",
                        this, &RadiationDriver::copyValues);

  Ghost::GhostType  gn = Ghost::None;
  t->requires(Task::OldDW, qfluxE_CCLabel,      mss_G, gn, 0);
  t->requires(Task::OldDW, qfluxW_CCLabel,      mss_G, gn, 0);
  t->requires(Task::OldDW, qfluxN_CCLabel,      mss_G, gn, 0);
  t->requires(Task::OldDW, qfluxS_CCLabel,      mss_G, gn, 0);
  t->requires(Task::OldDW, qfluxT_CCLabel,      mss_G, gn, 0);
  t->requires(Task::OldDW, qfluxB_CCLabel,      mss_G, gn, 0);
  t->requires(Task::OldDW, radiationSrc_CCLabel,mss_G, gn, 0);

  t->requires(Task::OldDW, radCO2_CCLabel,      mss_G, gn, 0);
  t->requires(Task::OldDW, radH2O_CCLabel,      mss_G, gn, 0);  
  t->requires(Task::OldDW, sootVFCopy_CCLabel,  mss_G, gn, 0);
  t->requires(Task::OldDW, mixfracCopy_CCLabel, mss_G, gn, 0);
  t->requires(Task::OldDW, abskg_CCLabel,       mss_G, gn, 0);
  t->requires(Task::OldDW, esrcg_CCLabel,       mss_G, gn, 0);
  t->requires(Task::OldDW, shgamma_CCLabel,     mss_G, gn, 0);
  t->requires(Task::OldDW, tempCopy_CCLabel,    mss_G, gn, 0);

  
  t->computes(qfluxE_CCLabel,       mss_G);
  t->computes(qfluxW_CCLabel,       mss_G);
  t->computes(qfluxN_CCLabel,       mss_G);
  t->computes(qfluxS_CCLabel,       mss_G);
  t->computes(qfluxT_CCLabel,       mss_G);
  t->computes(qfluxB_CCLabel,       mss_G);
  t->computes(radiationSrc_CCLabel, mss_G);
  t->computes(radCO2_CCLabel,       mss_G);
  t->computes(radH2O_CCLabel,       mss_G);
  t->computes(sootVFCopy_CCLabel,   mss_G);
  t->computes(mixfracCopy_CCLabel,  mss_G);
  t->computes(abskg_CCLabel,        mss_G);
  t->computes(esrcg_CCLabel,        mss_G);
  t->computes(shgamma_CCLabel,      mss_G);
  t->computes(tempCopy_CCLabel,     mss_G);
  
  if(d_hasAbsorbingSolid){
    t->requires(Task::OldDW,insideSolidLabel, mss_S, gn, 0);
    t->computes(insideSolidLabel, mss_S);
  }

  sched->addTask(t, patches, matls_set_GS);
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
  const MaterialSubset* mss_G = d_matl_G->thisMaterial();
  new_dw->transferFrom(old_dw, qfluxE_CCLabel,   patches, mss_G);   
  new_dw->transferFrom(old_dw, qfluxW_CCLabel,   patches, mss_G);   
  new_dw->transferFrom(old_dw, qfluxN_CCLabel,   patches, mss_G);   
  new_dw->transferFrom(old_dw, qfluxS_CCLabel,   patches, mss_G);   
  new_dw->transferFrom(old_dw, qfluxT_CCLabel,   patches, mss_G);   
  new_dw->transferFrom(old_dw, qfluxB_CCLabel,   patches, mss_G);   
  new_dw->transferFrom(old_dw, radiationSrc_CCLabel, patches, mss_G);

  new_dw->transferFrom(old_dw,radCO2_CCLabel,     patches, mss_G);
  new_dw->transferFrom(old_dw,radH2O_CCLabel,     patches, mss_G);
  new_dw->transferFrom(old_dw,sootVFCopy_CCLabel, patches, mss_G);
  new_dw->transferFrom(old_dw,mixfracCopy_CCLabel,patches, mss_G);
  new_dw->transferFrom(old_dw,abskg_CCLabel,      patches, mss_G);
  new_dw->transferFrom(old_dw,esrcg_CCLabel,      patches, mss_G);
  new_dw->transferFrom(old_dw,shgamma_CCLabel,    patches, mss_G);
  new_dw->transferFrom(old_dw,tempCopy_CCLabel,   patches, mss_G);
  if(d_hasAbsorbingSolid){
    const MaterialSubset* mss_S = d_matl_S->thisMaterial();
    new_dw->transferFrom(old_dw,insideSolidLabel, patches, mss_S);
  }
}

//****************************************************************************
// schedule computation of radiative properties
//****************************************************************************
void
RadiationDriver::scheduleComputeProps(const LevelP&        level,
                                      SchedulerP&           sched,
                                      const PatchSet*       patches,
                                      const MaterialSubset* mss_G,
                                      const MaterialSubset* mss_S,
                                      const MaterialSet*    matls_set_GS)
{
  cout_doing << "RADIATION::scheduleComputeProps \t\t\t\tL-" 
             << level->getIndex() << endl;
             
  Task* t=scinew Task("RadiationDriver::computeProps",
                this, &RadiationDriver::computeProps);
                
  t->requires( Task::OldDW, Ilb->delTLabel);
  
  Ghost::GhostType  gn = Ghost::None;

  if(d_table_or_ice_temp_density == "ice"){
   t->requires(Task::OldDW, iceTemp_CCLabel,    mss_G,gn, 0);
   t->requires(Task::NewDW, iceDensity_CCLabel, mss_G,gn, 0);
  }else{
   t->requires(Task::NewDW, temp_CCLabel,       mss_G,gn, 0);
   t->requires(Task::NewDW, density_CCLabel,    mss_G,gn, 0);
  }

  if (d_useTableValues){
    t->requires(Task::NewDW, co2_CCLabel,    mss_G, gn, 0);
    t->requires(Task::NewDW, h2o_CCLabel,    mss_G, gn, 0);
    t->requires(Task::OldDW, mixfrac_CCLabel,mss_G, gn, 0);
  }
  
  // Below is for later, when we get sootVF from reaction table.  But
  // currently we compute sootVF from mixture fraction and temperature inside
  // the properties function
  //  t->requires(Task::NewDW, sootVF_CCLabel,  mss_G, gn, 0);
  
  t->modifies(radCO2_CCLabel,     mss_G);
  t->modifies(radH2O_CCLabel,     mss_G);  
  t->modifies(tempCopy_CCLabel,   mss_G);
  t->modifies(sootVFCopy_CCLabel, mss_G);
  t->modifies(mixfracCopy_CCLabel,mss_G);
  t->requires(Task::NewDW, cellType_CCLabel, mss_G, gn, 0);

  if(d_hasAbsorbingSolid){
    t->requires(Task::NewDW,Ilb->vol_frac_CCLabel, mss_S, gn,0);              
  } 
  
  t->modifies(abskg_CCLabel,   mss_G);
  t->modifies(esrcg_CCLabel,   mss_G);
  t->modifies(shgamma_CCLabel, mss_G);

  t->computes(d_cellInfoLabel);

  sched->addTask(t, patches, matls_set_GS);
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
    int indx = d_matl_G->getDWIndex();
    
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
//----

    if(d_table_or_ice_temp_density == "ice"){
      new_dw->get(constRadVars.density,     iceDensity_CCLabel, indx, patch,gn,0);
      old_dw->get(constRadVars.temperature, iceTemp_CCLabel,    indx, patch,gn,0);
    }else{
      new_dw->get(constRadVars.density,     density_CCLabel,    indx, patch,gn,0);
      new_dw->get(constRadVars.temperature, temp_CCLabel,       indx, patch,gn,0);
    }   
    
    new_dw->getModifiable(radVars.co2,        radCO2_CCLabel,     indx, patch);
    new_dw->getModifiable(radVars.h2o,        radH2O_CCLabel,     indx, patch); 
    new_dw->getModifiable(radVars.temperature,tempCopy_CCLabel,   indx, patch);
    new_dw->getModifiable(radVars.sootVF,     sootVFCopy_CCLabel, indx, patch);
    new_dw->getModifiable(radVars.mixfrac,    mixfracCopy_CCLabel,indx, patch);
    new_dw->getModifiable(radVars.ABSKG,      abskg_CCLabel,      indx, patch);
    new_dw->getModifiable(radVars.ESRCG,      esrcg_CCLabel,      indx, patch);
    new_dw->getModifiable(radVars.shgamma,    shgamma_CCLabel,    indx, patch);
    new_dw->get(constRadVars.cellType,        cellType_CCLabel,   indx, patch,gn,0);
    
    radVars.temperature.copyData(constRadVars.temperature);
    
    if (d_useTableValues) {
      new_dw->get(constRadVars.co2,     co2_CCLabel,    indx, patch,gn, 0);
      new_dw->get(constRadVars.h2o,     h2o_CCLabel,    indx, patch,gn, 0);
      old_dw->get(constRadVars.mixfrac, mixfrac_CCLabel,indx, patch,gn, 0);
      radVars.co2.copyData(constRadVars.co2);
      radVars.h2o.copyData(constRadVars.h2o);
      radVars.mixfrac.copyData(constRadVars.mixfrac);
    }
    
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


    if (d_doRadCalc) {
      d_DORadiation->computeRadiationProps(pc, patch, cellinfo,
                                           &radVars, &constRadVars);
      //__________________________________
      //if there's an absorbing solid
      // set abskg = 1 in those cells
      // This is actually useless for radcoef property model 
      // since abskg is hardcoded to be 1 if not flow field in m_radcoef.F
      // Still, may be needed for other things
      if(d_hasAbsorbingSolid){
        constCCVariable<double> vol_frac_solid;
        int indx_S = d_matl_S->getDWIndex();
        new_dw->get(vol_frac_solid,Ilb->vol_frac_CCLabel,indx_S,patch,gn,0);

        for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter; 
          if(vol_frac_solid[c] > 0.5){
            radVars.ABSKG[c] = 1.0;
          }
        }
      }  // has absorbingSolid                                         
    }  // radiation timestep
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
  cout_doing << "RADIATION::scheduleBoundaryCondition\t\t\t\tL-" 
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
    
    int indx = d_matl_G->getDWIndex();
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    RadiationVariables radVars;
    RadiationConstVariables constRadVars;

    new_dw->get(constRadVars.cellType,         cellType_CCLabel,indx, patch,gac, 1);
    new_dw->getModifiable(radVars.temperature, tempCopy_CCLabel,indx, patch);
    new_dw->getModifiable(radVars.ABSKG,       abskg_CCLabel,   indx, patch);

    if (d_doRadCalc) {
      d_DORadiation->boundaryCondition(pc, patch, &radVars, &constRadVars);
      
#if 0     
      //__________________________________
      //  set boundary conditions  -- DEBUGGING ONLY  
      vector<Patch::FaceType>::const_iterator iter;

      for (iter  = patch->getBoundaryFaces()->begin(); 
           iter != patch->getBoundaryFaces()->end(); ++iter){
        Patch::FaceType face = *iter;
        for(CellIterator itr = patch->getFaceCellIterator(face, "plusEdgeCells"); 
            !itr.done();  itr++){
          IntVector c = *itr;
          radVars.temperature[c] = 293;
          radVars.ABSKG[c] = 0.223561452340239;
        } 
      }  // domain faces
#endif
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
                                        const MaterialSubset* mss_G,
                                        const MaterialSubset* mss_S,
                                        const MaterialSet*    matls_set_GS,
                                        const ModelInfo* mi)
{
  cout_doing << "RADIATION::scheduleIntensitySolve\t\t\t\tL-" 
             << level->getIndex() << endl;
  Task* t=scinew Task("RadiationDriver::intensitySolve",
                      this, &RadiationDriver::intensitySolve, mi);
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  
  t->requires(Task::OldDW, mi->delT_Label,level.get_rep()); 
  t->requires(Task::NewDW, d_cellInfoLabel,gn);
  // why are we using gac 1????
  
  t->requires(Task::NewDW, radCO2_CCLabel,       mss_G, gac, 1);
  t->requires(Task::NewDW, radH2O_CCLabel,       mss_G, gac, 1);
  t->requires(Task::NewDW, sootVFCopy_CCLabel,   mss_G, gac, 1);
  t->requires(Task::NewDW, cellType_CCLabel,     mss_G, gac, 1);
  t->requires(Task::NewDW, tempCopy_CCLabel,     mss_G, gac, 1);
  t->requires(Task::NewDW, shgamma_CCLabel,      mss_G, gac, 1);
  t->requires(Task::NewDW, abskg_CCLabel,        mss_G, gac, 1);
  t->requires(Task::NewDW, esrcg_CCLabel,        mss_G, gac, 1);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,mss_G, gn, 0);
  
  t->modifies(qfluxE_CCLabel,       mss_G);
  t->modifies(qfluxW_CCLabel,       mss_G);
  t->modifies(qfluxN_CCLabel,       mss_G);
  t->modifies(qfluxS_CCLabel,       mss_G);
  t->modifies(qfluxT_CCLabel,       mss_G);
  t->modifies(qfluxB_CCLabel,       mss_G);
  t->modifies(radiationSrc_CCLabel, mss_G);
  t->modifies(mi->modelEng_srcLabel,mss_G);
  
  if(d_hasAbsorbingSolid){
    t->requires(Task::NewDW, insideSolidLabel,    mss_S, gn, 0);
    t->requires(Task::NewDW, Ilb->temp_CCLabel,   mss_S, gn, 0);

    t->modifies(mi->modelEng_srcLabel, mss_S);
    t->computes(solidEmissionLabel,    mss_S);
  }

  sched->addTask(t, patches, matls_set_GS);
}

//****************************************************************************
// Actual solve for radiative intensities, radiative source and heat fluxes
//****************************************************************************
void
RadiationDriver::intensitySolve(const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing intensitySolve on patch "<<patch->getID()
               << "\t\t\t\t Radiation" << endl;
    int indx_G = d_matl_G->getDWIndex();
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    
    RadiationVariables radVars;
    RadiationConstVariables constRadVars;
    CCVariable<double> energySrc_Gas;
    constCCVariable<double> vol_frac_gas;

    PerPatch<Models_CellInformationP> cellInfoP;
    if (new_dw->exists(d_cellInfoLabel, indx_G, patch)) 
      new_dw->get(cellInfoP, d_cellInfoLabel, indx_G, patch);
    else {
      cellInfoP.setData(scinew Models_CellInformation(patch));
      new_dw->put(cellInfoP, d_cellInfoLabel, indx_G, patch);
    }
    Models_CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constRadVars.co2,        radCO2_CCLabel,        indx_G, patch,gac,1);
    new_dw->get(constRadVars.h2o,        radH2O_CCLabel,        indx_G, patch,gac,1);
    new_dw->get(constRadVars.sootVF,     sootVFCopy_CCLabel,    indx_G, patch,gac,1);
    new_dw->get(constRadVars.temperature,tempCopy_CCLabel,      indx_G, patch,gac,1);
    new_dw->get(constRadVars.cellType,   cellType_CCLabel,      indx_G, patch,gac,1);
    new_dw->get(vol_frac_gas,            Ilb->vol_frac_CCLabel, indx_G, patch,gn,0);

    new_dw->getCopy(radVars.ABSKG,   abskg_CCLabel,  indx_G, patch, gac,1);
    new_dw->getCopy(radVars.shgamma, shgamma_CCLabel,indx_G, patch, gac,1);
    new_dw->getCopy(radVars.ESRCG,   esrcg_CCLabel,  indx_G, patch, gac,1);

    new_dw->getModifiable(radVars.qfluxe, qfluxE_CCLabel, indx_G, patch);
    new_dw->getModifiable(radVars.qfluxw, qfluxW_CCLabel, indx_G, patch);
    new_dw->getModifiable(radVars.qfluxn, qfluxN_CCLabel, indx_G, patch);
    new_dw->getModifiable(radVars.qfluxs, qfluxS_CCLabel, indx_G, patch);
    new_dw->getModifiable(radVars.qfluxt, qfluxT_CCLabel, indx_G, patch);
    new_dw->getModifiable(radVars.qfluxb, qfluxB_CCLabel, indx_G, patch);
    new_dw->getModifiable(radVars.src, radiationSrc_CCLabel, indx_G, patch);

    new_dw->getModifiable(energySrc_Gas, mi->modelEng_srcLabel, indx_G, patch);

    if (d_doRadCalc) {
      radVars.qfluxe.initialize(0.0);
      radVars.qfluxw.initialize(0.0);
      radVars.qfluxn.initialize(0.0);
      radVars.qfluxs.initialize(0.0);
      radVars.qfluxt.initialize(0.0);
      radVars.qfluxb.initialize(0.0);
      radVars.src.initialize(0.0);

      d_DORadiation->intensitysolve(pc, patch, cellinfo, &radVars, &constRadVars);
    }
    
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    
    //__________________________________
    //  Gas Phase
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      energySrc_Gas[c] += delT * cell_vol * vol_frac_gas[c] * radVars.src[c];
    }
    //__________________________________
    // solid Phase   
    // This assumes that the solids absorptivity = 1.0
    // TODO:  generate a composite temperature field that is used in the radiation calc.
    // 
    if(d_hasAbsorbingSolid){
      int indx_S = d_matl_S->getDWIndex();
      constCCVariable<double> vol_frac_solid;
      CCVariable<double> energySrc_solid, emit_solid;

      new_dw->get(vol_frac_solid, Ilb->vol_frac_CCLabel,    indx_S, patch,gn,0);
      new_dw->getModifiable(energySrc_solid, mi->modelEng_srcLabel, indx_S, patch);
      new_dw->allocateAndPut(emit_solid,     solidEmissionLabel,    indx_S, patch);
       
      emit_solid.initialize(0.0); 
       
      for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
    //  energySrc_solid[c] += delT * cell_vol * vol_frac_solid[c] * radVars.src[c];
        energySrc_solid[c] += 100 * vol_frac_solid[c];    //  DEBUGGING
      }

      solidEmission(energySrc_solid, vol_frac_gas, emit_solid, delT, patch,new_dw); 
      
    }  // absorbing solid 
  }  // patches loop
}
//______________________________________________________________________                      
void RadiationDriver::solidEmission(CCVariable<double>& energySrc_solid,
                                    constCCVariable<double>& vol_frac_gas,
                                    CCVariable<double>& emit_solid,
                                    const double delT,
                                    const Patch* patch,                        
                                    DataWarehouse* new_dw)                          
{   
  cout_doing << "Doing solidEmission on patch "<<patch->getID()
             << "\t\t\t\t Radiation" << endl;

  Ghost::GhostType  gn  = Ghost::None; 
  int indx_S = d_matl_S->getDWIndex();  
  constCCVariable<double> Temp_solid;
  constCCVariable<double> insideSolid;

  new_dw->get(Temp_solid,  Ilb->temp_CCLabel,indx_S, patch,gn,0);         
  new_dw->get(insideSolid, insideSolidLabel, indx_S, patch,gn,0);         

  Vector dx = patch->dCell();
  double areaYZ = dx.y() * dx.z();
  double areaXZ = dx.x() * dx.z();
  double areaXY = dx.x() * dx.y();

  IntVector X(1,0,0);
  IntVector Y(0,1,0);
  IntVector Z(0,0,1);

  // Stair step approarch to finding the emitting surface area
  // Look at the surrounding cells, if they are not inside the absorbing solid
  // then that cell face *may* be at a gas solid interface. To be on the surface 
  // the volume fraction of the gas in the adjacent cell must be > small number.
  for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
    IntVector c = *iter;
    if(insideSolid[c] == 1.0){

      double cellFaceAreas = 0;
      double min_vol_frac = 1e-3;  
      if( insideSolid[c-X] == 0 && vol_frac_gas[c-X] > min_vol_frac){
        cellFaceAreas += areaYZ;        // x+
      }
      if( insideSolid[c+X] == 0 && vol_frac_gas[c+X] > min_vol_frac){
        cellFaceAreas += areaYZ;        // x-
      }  
      if( insideSolid[c-Y] == 0 && vol_frac_gas[c-Y] > min_vol_frac){
        cellFaceAreas += areaXZ;        // Y+
      }
      if( insideSolid[c+Y] == 0 && vol_frac_gas[c+Y] > min_vol_frac){
        cellFaceAreas += areaXZ;        // Y-
      }
      if( insideSolid[c-Z] == 0 && vol_frac_gas[c-Z] > min_vol_frac){
        cellFaceAreas += areaXY;        // Z+
      }
      if( insideSolid[c+Z] == 0 && vol_frac_gas[c+Z] > min_vol_frac){
        cellFaceAreas += areaXY;        // Z-
      }
      emit_solid[c] = 100;    // DEBUGGING
        
      // This assumes that the solid surface temperature equals the the cell centered temperature  
      // emit_solid[c] =  cellFaceAreas * d_sigma
      //               * Temp_solid[c] * Temp_solid[c] * Temp_solid[c] * Temp_solid[c];
                    
      energySrc_solid[c] -= emit_solid[c];
                    
    }  
  }
}

//______________________________________________________________________
// not applicable
void RadiationDriver::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                              const LevelP&,
                                                              const MaterialSet*)
{    
}
void RadiationDriver::computeSpecificHeat(CCVariable<double>&,
                                          const Patch*,
                                          DataWarehouse*,
                                          const int)
{
}
void RadiationDriver::scheduleErrorEstimate(const LevelP&,
                                            SchedulerP&)
{
}
void RadiationDriver::scheduleTestConservation(SchedulerP&,
                                               const PatchSet*,
                                               const ModelInfo*)
{
}
void RadiationDriver::scheduleComputeStableTimestep(SchedulerP&,
                                                    const LevelP&,
                                                    const ModelInfo*)
{
}
