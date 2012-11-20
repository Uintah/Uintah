/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// MPMArches.cc

#include <cstring>
#include <fstream>
#include <CCA/Components/MPMArches/MPMArches.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/Point.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/EnthalpySolver.h>
#include <CCA/Components/Arches/NonlinearSolver.h>
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/LinearInterpolator.h>
#include <CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/MPMArches/CutCellInfo.h>
#include <CCA/Components/MPMArches/CutCellInfoP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <CCA/Ports/SolverInterface.h>

using namespace Uintah;
using namespace std;

#include <CCA/Components/MPMArches/fortran/collect_drag_cc_fort.h>
#include <CCA/Components/MPMArches/fortran/collect_scalar_fctocc_fort.h>
#include <CCA/Components/MPMArches/fortran/energy_exchange_term_fort.h>
#include <CCA/Components/MPMArches/fortran/interp_centertoface_fort.h>
#include <CCA/Components/MPMArches/fortran/momentum_exchange_term_continuous_cc_fort.h>
#include <CCA/Components/MPMArches/fortran/pressure_force_fort.h>
#include <CCA/Components/MPMArches/fortran/read_complex_geometry_fort.h>
#include <CCA/Components/MPMArches/fortran/read_complex_geometry_walls_fort.h>

// ****************************************************************************
// Actual constructor for MPMArches
// ****************************************************************************

#undef RIGID_MPM
// #define RIGID_MPM

  MPMArches::MPMArches(const ProcessorGroup* myworld)
: UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  d_MAlb = scinew MPMArchesLabel();
#ifdef RIGID_MPM
  d_mpm      = scinew RigidMPM(myworld);
#else
  d_mpm      = scinew SerialMPM(myworld);
#endif
  d_arches      = scinew Arches(myworld);
  d_SMALL_NUM = 1.e-100;
  nofTimeSteps = 0;
  d_doingRestart = false; 
}

// ****************************************************************************
// Destructor
// ****************************************************************************

MPMArches::~MPMArches()
{
  delete d_MAlb;
  delete d_mpm;
  delete d_arches;
  
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      delete *iter;
    }
  }
}

// ****************************************************************************
// problem set up
// ****************************************************************************

void MPMArches::problemSetup(const ProblemSpecP& prob_spec, 
    const ProblemSpecP& materials_ps,
    GridP& grid, SimulationStateP& sharedState)
{
  d_sharedState = sharedState;

  ProblemSpecP restart_mat_ps = 0;
  if (materials_ps){
    restart_mat_ps = materials_ps;
  }
  else{
    restart_mat_ps = prob_spec;
  }


  ProblemSpecP db = prob_spec->findBlock("Multimaterial");

  if( db.get_rep() == NULL ) { // Make sure the Multimaterial block exists...
    printf("\n");
    printf("ERROR: It appears that the <Multimaterial> tag is missing from the problem specification (.ups file)...\n");
    printf("\n");
  }
  d_mpm->setWithARCHES();
  d_arches->setWithMPMARCHES();

  db->require("heatExchange", d_calcEnergyExchange);
  db->require("fluidThermalConductivity", d_tcond);
  db->require("turbulentPrandtNo",prturb);
  db->require("fluidHeatCapacity",cpfluid);
  db->require("IfCutCell",d_useCutCell);
  db->getWithDefault("StationarySolid",d_stationarySolid,true);
  db->require("inviscid",d_inviscid);
  db->require("restart",d_restart);
  db->getWithDefault("fixedCellType",fixCellType,true);
  db->getWithDefault("fixedTemp",d_fixTemp,false);
  db->getWithDefault("TestCutCells",d_ifTestingCutCells,true);
  db->getWithDefault("stairstep",d_stairstep,true);

  fixCellType = d_stationarySolid;

  bool dontCalcVel = d_stationarySolid || d_useCutCell;
  // cut cells do not currently support moving geometries
  calcVel = !dontCalcVel;

  Output* dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!dataArchiver){
    throw InternalError("MPMARCHES:couldn't get output port", __FILE__, __LINE__);
  }
  d_arches->attachPort("output",dataArchiver);
  d_mpm->attachPort("output",dataArchiver);

  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
  if(!sched){
    throw InternalError("MPMARCHES:couldn't get scheduler port", __FILE__, __LINE__);
  }

  d_mpm->attachPort("scheduler",sched);

  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver){
    throw InternalError("ARCHES needs a solver component to work", __FILE__, __LINE__);
  }
  d_arches->attachPort("solver", solver);

  d_mpm->setMPMLabel(Mlb);
  d_mpm->problemSetup(prob_spec, materials_ps,grid, d_sharedState);
  // set multimaterial label in Arches to access interface variables
  d_arches->setMPMArchesLabel(d_MAlb);
  d_Alab = d_arches->getArchesLabel();
  d_arches->problemSetup(prob_spec, materials_ps,grid, d_sharedState);

  d_arches->getBoundaryCondition()->setIfCalcEnergyExchange(d_calcEnergyExchange);
  d_arches->getBoundaryCondition()->setIfFixTemp(d_fixTemp);
  d_arches->getBoundaryCondition()->setCutCells(d_useCutCell);

  if (d_arches->checkSolveEnthalpy()) {
    d_radiation = d_arches->getNonlinearSolver()->getEnthalpySolver()->checkRadiation();
    if (d_radiation) 
      d_DORad = d_arches->getNonlinearSolver()->getEnthalpySolver()->checkDORadiation();
    else
      d_DORad = false;
  }
  else {
    d_DORad = false;
    d_radiation = false;
  }
  
  //__________________________________
  //  create analysis modules
  // call problemSetup  
  d_analysisModules = AnalysisModuleFactory::create(prob_spec, sharedState, dataArchiver);
  
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->problemSetup(prob_spec, grid, sharedState);
    }
  }

  // make an allowance for an enthalpy variable with a different name: 
  d_enthalpy_name = "enthalpySP"; 
  if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ClassicTable") ){ 
    if ( db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ClassicTable")->findBlock("enthalpy_label")){ 
      db->getRootNode()->
        findBlock("CFD")->
        findBlock("ARCHES")->
        findBlock("Properties")->
        findBlock("ClassicTable")->getWithDefault( "enthalpy_label", d_enthalpy_name, string("enthalpySP") );
    }
  } 
  
  d_enthalpy_label = VarLabel::find( d_enthalpy_name ); 

}

void MPMArches::outputProblemSpec(ProblemSpecP& root_ps)
{
  d_mpm->outputProblemSpec(root_ps);
}

void
MPMArches::restartInitialize()
{
  d_doingRestart = true;
  d_arches->restartInitialize();
}


// ****************************************************************************
// Schedule initialization
// ****************************************************************************

void MPMArches::scheduleInitialize(const LevelP& level,
    SchedulerP& sched)
{
  d_mpm->scheduleInitialize(      level, sched);

#ifdef ExactMPMArchesInitialize
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* arches_matls = d_sharedState->allArchesMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();

  scheduleInterpolateParticlesToGrid(sched, patches, mpm_matls);
  scheduleInterpolateNCToCC(sched, patches, mpm_matls);
  scheduleComputeVoidFracMPM(sched, patches, arches_matls, mpm_matls, all_matls);
#endif

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* arches_matls = d_sharedState->allArchesMaterials();
  scheduleInitializeKStability(sched, patches, arches_matls);
  if (d_useCutCell) 
    scheduleInitializeCutCells(sched, patches, arches_matls);

  d_arches->scheduleInitialize(      level, sched);

  //  cerr << "Doing Initialization \t\t\t MPMArches" <<endl;
  //  cerr << "--------------------------------\n"<<endl; 
  
  // dataAnalysis 
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleInitialize( sched, level);
    }
  }
  
}

//______________________________________________________________________
//

void MPMArches::scheduleInitializeKStability(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls)
{
  const MaterialSubset* mpm_matls = d_sharedState->allMPMMaterials()->getUnion();
  // set initial values for Stability factors due to drag
  Task* t = scinew Task("MPMArches::initializeKStability",
      this, &MPMArches::initializeKStability);
  t->computes(d_MAlb->KStabilityULabel);
  t->computes(d_MAlb->KStabilityVLabel);
  t->computes(d_MAlb->KStabilityWLabel);
  t->computes(d_MAlb->KStabilityHLabel);

  t->computes(d_MAlb->void_frac_CCLabel);
  t->computes(d_MAlb->void_frac_MPM_CCLabel);
  t->computes(d_MAlb->solid_fraction_CCLabel);

  t->computes(d_Alab->d_mmcellTypeLabel);
  t->computes(d_MAlb->mmCellType_MPMLabel);
  t->computes(d_MAlb->solid_fraction_CCLabel,mpm_matls);

  sched->addTask(t, patches, arches_matls);
}

//______________________________________________________________________
//

void MPMArches::initializeKStability(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* arches_matls,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)

{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlindex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    StaticArray<CCVariable<double> > solid_fraction_cc(numMPMMatls);

    for (int m = 0; m < numMPMMatls; m++) {

      Material* matl = d_sharedState->getMPMMaterial( m );
      int dwindex = matl->getDWIndex();
      new_dw->allocateAndPut(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
          dwindex, patch);
      solid_fraction_cc[m].initialize(1.0);

    }

    CCVariable<double> epsg;
    new_dw->allocateAndPut(epsg, d_MAlb->void_frac_CCLabel,
        matlindex, patch);
    epsg.initialize(1.0);
    CCVariable<double> epsgMPM;
    new_dw->allocateAndPut(epsgMPM, d_MAlb->void_frac_MPM_CCLabel,
        matlindex, patch);
    epsgMPM.initialize(1.0);

    CCVariable<int> mmcelltype;
    new_dw->allocateAndPut(mmcelltype, d_Alab->d_mmcellTypeLabel,
        matlindex, patch);
    mmcelltype.initialize(1);

    CCVariable<int> mmcelltypeMPM;
    new_dw->allocateAndPut(mmcelltypeMPM, d_MAlb->mmCellType_MPMLabel,
        matlindex, patch);
    mmcelltypeMPM.initialize(1);

    CCVariable<double> KStabilityU;
    CCVariable<double> KStabilityV;
    CCVariable<double> KStabilityW;
    CCVariable<double> KStabilityH;
    new_dw->allocateAndPut(KStabilityU, d_MAlb->KStabilityULabel,
        matlindex, patch); 
    KStabilityU.initialize(0.);
    new_dw->allocateAndPut(KStabilityV, d_MAlb->KStabilityULabel,
        matlindex, patch); 
    KStabilityV.initialize(0.);
    new_dw->allocateAndPut(KStabilityW, d_MAlb->KStabilityULabel,
        matlindex, patch); 
    KStabilityW.initialize(0.);
    new_dw->allocateAndPut(KStabilityH, d_MAlb->KStabilityHLabel,
        matlindex, patch); 
    KStabilityH.initialize(0.);
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleInitializeCutCells(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls)
{
  Task* t = scinew Task("MPMArches::initializeCutCells",
      this, &MPMArches::initializeCutCells);

  t->computes(d_MAlb->cutCellLabel);
  t->computes(d_MAlb->d_normal1Label);
  t->computes(d_MAlb->d_normal2Label);
  t->computes(d_MAlb->d_normal3Label);
  t->computes(d_MAlb->d_normalLabel);
  t->computes(d_MAlb->d_centroid1Label);
  t->computes(d_MAlb->d_centroid2Label);
  t->computes(d_MAlb->d_centroid3Label);
  t->computes(d_MAlb->d_totAreaLabel);

  t->computes(d_MAlb->d_pGasAreaFracXPLabel);
  t->computes(d_MAlb->d_pGasAreaFracXELabel);
  t->computes(d_MAlb->d_pGasAreaFracYPLabel);
  t->computes(d_MAlb->d_pGasAreaFracYNLabel);
  t->computes(d_MAlb->d_pGasAreaFracZPLabel);
  t->computes(d_MAlb->d_pGasAreaFracZTLabel);

  /*
     Above stuff should be removed once
     we get d_cutcell to work ...
     normal1 to pGasAreaFracZT
     */

  t->computes(d_MAlb->d_nextCutCellILabel);
  t->computes(d_MAlb->d_nextCutCellJLabel);
  t->computes(d_MAlb->d_nextCutCellKLabel);

  t->computes(d_MAlb->d_nextWallILabel);
  t->computes(d_MAlb->d_nextWallJLabel);
  t->computes(d_MAlb->d_nextWallKLabel);

  t->computes(d_MAlb->void_frac_CutCell_CCLabel);
  t->modifies(d_MAlb->void_frac_CCLabel);
  t->computes(d_MAlb->mmCellType_CutCellLabel);

  sched->addTask(t, patches, arches_matls);
}

//______________________________________________________________________
//

void MPMArches::initializeCutCells(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* arches_matls,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)

{

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlindex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<cutcell> d_CCell;
    new_dw->allocateAndPut(d_CCell, d_MAlb->cutCellLabel, 
        matlindex, patch);

    PerPatch<CutCellInfoP> cutCellInfoP;
    if (new_dw->exists(d_MAlb->d_cutCellInfoLabel, matlindex, patch)) 
      new_dw->get(cutCellInfoP, d_MAlb->d_cutCellInfoLabel, matlindex, patch);
    else {
      cutCellInfoP.setData(scinew CutCellInfo());
      new_dw->put(cutCellInfoP, d_MAlb->d_cutCellInfoLabel, matlindex, patch);
    }
    CutCellInfo* ccinfo = cutCellInfoP.get().get_rep();

    // Initialize cutcell struct

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      for (int kk = CENX; kk <= AREAB; kk++) {
        d_CCell[c].d_cutcell[kk] = 0.0;
      }
    }

    ifstream file("patch_table.dat");
    if (!file) {
      cerr << "Failed to open patch_table\n" << endl;
      exit(EXIT_FAILURE);
    }

    int numPatches = patches->size();

    // Read patch table to determine patch index
    // and total numbers of p, u, v, and w cells

    int istart;
    int jstart;
    int kstart;
    int iend;
    int jend;
    int kend;

    IntVector dimLo = patch->getFortranCellLowIndex();
    IntVector dimHi = patch->getFortranCellHighIndex();

    string s1,s2;

    int ierr = 0;

    file.seekg(0);
    for (int ii=0; ii< numPatches; ii++) {

      while (ierr == 0) {

        file >> ccinfo->patchindex;
        file >> istart >> jstart >> kstart;
        file >> iend >> jend >> kend;
        file >> ccinfo->tot_cutp >> ccinfo->tot_cutu >> ccinfo->tot_cutv >> ccinfo->tot_cutw >> ccinfo->tot_walls;
        /*
           cout << "tot_cutp = " << ccinfo->tot_cutp << endl;
           cout << "tot_cutu = " << ccinfo->tot_cutu << endl;
           cout << "tot_cutv = " << ccinfo->tot_cutv << endl;
           cout << "tot_cutw = " << ccinfo->tot_cutw << endl;
           cout << "tot_walls = " << ccinfo->tot_walls << endl;
           */
        IntVector start(istart,jstart,kstart);
        IntVector end(iend,jend,kend);
        if ((dimLo.x() >= istart) && (dimHi.x() <= iend) &&
            (dimLo.y() >= jstart) && (dimHi.y() <= jend) &&
            (dimLo.z() >= kstart) && (dimHi.z() <= kend)) {
          ierr = 1;
          // we have found the right patch
        }
      }
    }

    /*
       stuff below should be removed if we wish to 
       use the d_cutcell struct
       */

    CCVariable<int> mmCellTypeCutCell;
    new_dw->allocateAndPut(mmCellTypeCutCell, d_MAlb->mmCellType_CutCellLabel,
        matlindex, patch);
    mmCellTypeCutCell.initialize(1);

    CCVariable<double> epsg;
    new_dw->getModifiable(epsg, d_MAlb->void_frac_CCLabel,
        matlindex, patch); 

    CCVariable<double> epsg_cut;
    new_dw->allocateAndPut(epsg_cut, d_MAlb->void_frac_CutCell_CCLabel,
        matlindex, patch); 
    epsg_cut.copyData(epsg);

    CCVariable<double> nbar1;
    new_dw->allocateAndPut(nbar1, d_MAlb->d_normal1Label,
        matlindex, patch); 
    nbar1.initialize(0.0);

    if (patch->containsCell(IntVector(6,53,80))) {
      cerr << "[6,53,80] x normal" << nbar1[IntVector(6,53,80)] << endl;
    }

    CCVariable<double> nbar2;
    new_dw->allocateAndPut(nbar2, d_MAlb->d_normal2Label,
        matlindex, patch); 
    nbar2.initialize(0.0);

    CCVariable<double> nbar3;
    new_dw->allocateAndPut(nbar3, d_MAlb->d_normal3Label,
        matlindex, patch); 
    nbar3.initialize(0.0);

    CCVariable<Vector> normal;
    new_dw->allocateAndPut(normal, d_MAlb->d_normalLabel,
        matlindex, patch); 
    normal.initialize(Vector(0.0,0.0,0.0));

    CCVariable<double> cbar1;
    new_dw->allocateAndPut(cbar1, d_MAlb->d_centroid1Label,
        matlindex, patch); 
    cbar1.initialize(0.0);

    CCVariable<double> cbar2;
    new_dw->allocateAndPut(cbar2, d_MAlb->d_centroid2Label,
        matlindex, patch); 
    cbar2.initialize(0.0);

    CCVariable<double> cbar3;
    new_dw->allocateAndPut(cbar3, d_MAlb->d_centroid3Label,
        matlindex, patch);  
    cbar3.initialize(0.0);

    CCVariable<double> totArea;
    new_dw->allocateAndPut(totArea, d_MAlb->d_totAreaLabel,
        matlindex, patch);
    totArea.initialize(0.0);

    CCVariable<double> gaf_x;
    new_dw->allocateAndPut(gaf_x, d_MAlb->d_pGasAreaFracXPLabel,
        matlindex, patch);
    gaf_x.initialize(1.0);

    CCVariable<double> gaf_xe;
    new_dw->allocateAndPut(gaf_xe, d_MAlb->d_pGasAreaFracXELabel,
        matlindex, patch);
    gaf_xe.initialize(1.0);

    CCVariable<double> gaf_y;
    new_dw->allocateAndPut(gaf_y, d_MAlb->d_pGasAreaFracYPLabel,
        matlindex, patch);
    gaf_y.initialize(1.0);

    CCVariable<double> gaf_yn;
    new_dw->allocateAndPut(gaf_yn, d_MAlb->d_pGasAreaFracYNLabel,
        matlindex, patch);
    gaf_yn.initialize(1.0);

    CCVariable<double> gaf_z;
    new_dw->allocateAndPut(gaf_z, d_MAlb->d_pGasAreaFracZPLabel,
        matlindex, patch);
    gaf_z.initialize(1.0);

    CCVariable<double> gaf_zt;
    new_dw->allocateAndPut(gaf_zt, d_MAlb->d_pGasAreaFracZTLabel,
        matlindex, patch);
    gaf_zt.initialize(1.0);

    /*
       Above stuff should be removed if we use d_cutcell instead;
       for now we will use CCVariables for each of these attributes,
       viz. centroids, normals, area fractions, total area

*/

    CCVariable<int> inext;
    new_dw->allocateAndPut(inext, d_MAlb->d_nextCutCellILabel,
        matlindex, patch);
    inext.initialize(100000);

    CCVariable<int> jnext;
    new_dw->allocateAndPut(jnext, d_MAlb->d_nextCutCellJLabel,
        matlindex, patch);
    jnext.initialize(100000);

    CCVariable<int> knext;
    new_dw->allocateAndPut(knext, d_MAlb->d_nextCutCellKLabel,
        matlindex, patch);
    knext.initialize(100000);

    CCVariable<int> inextWall;
    new_dw->allocateAndPut(inextWall, d_MAlb->d_nextWallILabel,
        matlindex, patch);
    inextWall.initialize(100000);

    CCVariable<int> jnextWall;
    new_dw->allocateAndPut(jnextWall, d_MAlb->d_nextWallJLabel,
        matlindex, patch);
    jnextWall.initialize(100000);

    CCVariable<int> knextWall;
    new_dw->allocateAndPut(knextWall, d_MAlb->d_nextWallKLabel,
        matlindex, patch);
    knextWall.initialize(100000);

    if (patch->containsCell(IntVector(6,53,80))) {
      cerr << "[6,53,80] x normal" << nbar1[IntVector(6,53,80)] << endl;
    }

    if (ccinfo->tot_cutp != 0) {

      fort_read_complex_geometry(ccinfo->iccst, ccinfo->jccst, ccinfo->kccst,
          inext,
          jnext,
          knext,
          epsg,
          totArea,
          nbar1,
          nbar2,
          nbar3,
          cbar1,
          cbar2,
          cbar3,
          gaf_x,
          gaf_xe,
          gaf_y,
          gaf_yn,
          gaf_z,
          gaf_zt,
          ccinfo->patchindex,
          ccinfo->tot_cutp);

      //	replace with stuff below when we use d_cutcell

      /*
         fort_read_complex_geometry(ccinfo->iccst, ccinfo->jccst, ccinfo->kccst,
         inext,
         jnext,
         knext,
         epsg,
         d_CCell[dimLo].d_cutcell[TOTAREA],
         d_CCell[dimLo].d_cutcell[NORMX],
         d_CCell[dimLo].d_cutcell[NORMY],
         d_CCell[dimLo].d_cutcell[NORMZ],
         d_CCell[dimLo].d_cutcell[CENX],
         d_CCell[dimLo].d_cutcell[CENY],
         d_CCell[dimLo].d_cutcell[CENZ],
         d_CCell[dimLo].d_cutcell[AREAW],
         d_CCell[dimLo].d_cutcell[AREAE],
         d_CCell[dimLo].d_cutcell[AREAS],
         d_CCell[dimLo].d_cutcell[AREAN],
         d_CCell[dimLo].d_cutcell[AREAB],
         d_CCell[dimLo].d_cutcell[AREAT],
         ccinfo->patchindex,
         ccinfo->tot_cutp);

*/

      if (patch->containsCell(IntVector(6,53,80))) {
        cerr << "[6,53,80] x normal" << nbar1[IntVector(6,53,80)] << endl;
      }

      if (ccinfo->tot_walls != 0) {

        fort_read_complex_geometry_walls(ccinfo->iccst, ccinfo->jccst, ccinfo->kccst,
            inextWall,
            jnextWall,
            knextWall,
            epsg,
            totArea,
            nbar1,
            nbar2,
            nbar3,
            cbar1,
            cbar2,
            cbar3,
            gaf_x,
            gaf_xe,
            gaf_y,
            gaf_yn,
            gaf_z,
            gaf_zt,
            ccinfo->patchindex,
            ccinfo->tot_walls);
      }

      // Apply Boundary Conditions to Cut Cell Data

      IntVector idxLo = patch->getFortranCellLowIndex();
      IntVector idxHi = patch->getFortranCellHighIndex();

      bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
      bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
      bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
      bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

      if (xminus) {
        int colX = idxLo.x();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xminusCell(colX-1, colY, colZ);
            nbar1[xminusCell] = nbar1[currCell];
            nbar2[xminusCell] = nbar2[currCell];
            nbar3[xminusCell] = nbar3[currCell];
            epsg[xminusCell] = epsg[currCell];

          }
        }
      }

      if (xplus) {
        int colX = idxHi.x();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector xplusCell(colX+1, colY, colZ);
            nbar1[xplusCell] = nbar1[currCell];
            nbar2[xplusCell] = nbar2[currCell];
            nbar3[xplusCell] = nbar3[currCell];
            epsg[xplusCell] = epsg[currCell];
          }
        }
      }

      if (yminus) {
        int colY = idxLo.y();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector yminusCell(colX, colY-1, colZ);
            nbar1[yminusCell] = nbar1[currCell];
            nbar2[yminusCell] = nbar2[currCell];
            nbar3[yminusCell] = nbar3[currCell];
            epsg[yminusCell] = epsg[currCell];

          }
        }
      }

      if (yplus) {
        int colY = idxHi.y();
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector yplusCell(colX, colY+1, colZ);
            nbar1[yplusCell] = nbar1[currCell];
            nbar2[yplusCell] = nbar2[currCell];
            nbar3[yplusCell] = nbar3[currCell];
            epsg[yplusCell] = epsg[currCell];

          }
        }
      }

      if (zminus) {
        int colZ = idxLo.z();
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector zminusCell(colX, colY, colZ-1);
            nbar1[zminusCell] = nbar1[currCell];
            nbar2[zminusCell] = nbar2[currCell];
            nbar3[zminusCell] = nbar3[currCell];
            epsg[zminusCell] = epsg[currCell];

          }
        }
      }

      if (zplus) {
        int colZ = idxHi.z();
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            IntVector zplusCell(colX, colY, colZ+1);
            nbar1[zplusCell] = nbar1[currCell];
            nbar2[zplusCell] = nbar2[currCell];
            nbar3[zplusCell] = nbar3[currCell];
            epsg[zplusCell] = epsg[currCell];

          }
        }
      }

      epsg_cut.copyData(epsg);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        normal[c] = Vector(nbar1[c],nbar2[c],nbar3[c]);
      }
    }
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */

  Task* t = scinew Task("MPMArches::interpolateParticlesToGrid",
      this,&MPMArches::interpolateParticlesToGrid);
  int numGhostCells = 1;

  t->requires(Task::NewDW, Mlb->pXLabel,             
      Ghost::AroundNodes, numGhostCells);

#ifdef debugExactInitializeMPMArches
#else
  t->requires(Task::NewDW, Mlb->pMassLabel,          
      Ghost::AroundNodes, numGhostCells);
  t->requires(Task::NewDW, Mlb->pVolumeLabel,        
      Ghost::AroundNodes, numGhostCells);
  if (!d_stationarySolid) 
    t->requires(Task::NewDW, Mlb->pVelocityLabel,      
        Ghost::AroundNodes, numGhostCells);
  t->requires(Task::NewDW, Mlb->pTemperatureLabel,   
      Ghost::AroundNodes, numGhostCells);

  t->computes(Mlb->gMassLabel);
  t->computes(Mlb->gMassLabel, d_sharedState->getAllInOneMatl(),
      Task::OutOfDomain);
  t->computes(Mlb->gTemperatureLabel, d_sharedState->getAllInOneMatl(),
      Task::OutOfDomain);
  t->computes(Mlb->gVolumeLabel);
  t->computes(Mlb->gVelocityLabel);
  t->computes(Mlb->gTemperatureLabel);
  t->computes(Mlb->TotalMassLabel);
#endif

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//

void MPMArches::interpolateParticlesToGrid(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* ,
    DataWarehouse* ,
    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    LinearInterpolator* interpolator = scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());


    int numMatls = d_sharedState->getNumMPMMatls();
    int numGhostCells = 1;

#ifdef debugExactInitializeMPMArches
#else
    NCVariable<double> gmassglobal;
    NCVariable<double> gtempglobal;
    new_dw->allocateAndPut(gmassglobal,Mlb->gMassLabel,
        d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gtempglobal,Mlb->gTemperatureLabel,
        d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_mpm->d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);
#endif

    for(int m = 0; m < numMatls; m++){

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();

      constParticleVariable<Point>  px;
#ifdef debugExactInitializeMPMArches
#else
      constParticleVariable<double> pmass;
      constParticleVariable<double> pvolume;
      constParticleVariable<double> pTemperature;
      constParticleVariable<Vector> pvelocity;
#endif

      ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch,
          Ghost::AroundNodes, 
          numGhostCells, 
          Mlb->pXLabel);

      new_dw->get(px,             Mlb->pXLabel,             pset);

#ifdef debugExactInitializeMPMArches
#else
      new_dw->get(pmass,          Mlb->pMassLabel,          pset);
      new_dw->get(pvolume,        Mlb->pVolumeLabel,        pset);
      if (!d_stationarySolid)
        new_dw->get(pvelocity,      Mlb->pVelocityLabel,      pset);
      new_dw->get(pTemperature,   Mlb->pTemperatureLabel,   pset);

      // Create arrays for the grid data

      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<double> gTemperature;

      new_dw->allocateAndPut(gmass,            Mlb->gMassLabel,        
          matlindex, patch);
      new_dw->allocateAndPut(gvolume,          Mlb->gVolumeLabel,      
          matlindex, patch);
      new_dw->allocateAndPut(gvelocity,        Mlb->gVelocityLabel,    
          matlindex, patch);
      new_dw->allocateAndPut(gTemperature,     Mlb->gTemperatureLabel, 
          matlindex, patch);

      gmass.initialize(d_mpm->d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gTemperature.initialize(0);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity

      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);

      // Here we do interpolation only without fracture.  I've
      // taken the section from Jim's SerialMPM that is in the else
      // part of the if (d_fracture) loop.

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell

        interpolator->findCellAndWeights(px[idx], ni, S);

        if (!d_stationarySolid)
          total_mom += pvelocity[idx]*pmass[idx];

        // Add each particles contribution to the local mass & velocity 
        // Must use the node indices

        for(int k = 0; k < 8; k++) {

          if(patch->containsNode(ni[k])) {
            gmass[ni[k]]          += pmass[idx]                     * S[k];
            gmassglobal[ni[k]]    += pmass[idx]                     * S[k];
            gvolume[ni[k]]        += pvolume[idx]                   * S[k];
            if (d_stationarySolid)
              gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
            gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
            gtempglobal[ni[k]]    += pTemperature[idx] * pmass[idx] * S[k];
            totalmass             += pmass[idx]                     * S[k];

          }
        }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        gvelocity[*iter] /= gmass[*iter];
        gTemperature[*iter] /= gmass[*iter];
      }

      // Apply grid boundary conditions to the velocity before storing the data
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,matlindex,"Velocity",gvelocity,"linear");
      bc.setBoundaryCondition(patch,matlindex,"Symmetric",gvelocity,"linear");
      bc.setBoundaryCondition(patch,matlindex,"Temperature",gTemperature,"linear");

    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      gtempglobal[*iter] /= gmassglobal[*iter];
    }

#endif



    // End loop over patches
    delete interpolator;
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleComputeStableTimestep(const LevelP& level,
    SchedulerP& sched)
{
  // Schedule computing the Arches stable timestep
  d_arches->scheduleComputeStableTimestep(level, sched);
  // MPM stable timestep is a by-product of the CM
}

//______________________________________________________________________
//

  void
MPMArches::scheduleTimeAdvance( const LevelP & level,
    SchedulerP   & sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* arches_matls = d_sharedState->allArchesMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();

  //double time = d_Alab->d_sharedState->getElapsedTime();

  nofTimeSteps++ ;
  // note: this counter will only get incremented each
  // time the taskgraph is recompiled

//  //  if (nofTimeSteps < 2 && !d_restart) {
//  if (time < 1.0E-10) {
//    d_recompile = true;
//  }
//  else
//    d_recompile = false;

  d_mpm->scheduleApplyExternalLoads(sched, patches, mpm_matls);
  d_mpm->scheduleInterpolateParticlesToGrid(sched, patches, mpm_matls);
  d_mpm->scheduleComputeHeatExchange(       sched, patches, mpm_matls);

  // interpolate mpm properties from node center to cell center
  // and subsequently to face center
  // these computed variables are used by void fraction and mom exchange

  scheduleInterpolateNCToCC(sched, patches, mpm_matls);
  scheduleInterpolateCCToFC(sched, patches, mpm_matls);

  scheduleComputeVoidFracMPM(sched, patches, arches_matls, mpm_matls, all_matls);

  if (d_useCutCell)
    scheduleCopyCutCells(sched,patches,arches_matls);

  // Using computed MPM void fractions and Cut Cell void fractions (if applicable),
  // choose actual void fraction for case

  scheduleComputeVoidFrac(sched, patches, arches_matls, mpm_matls, all_matls);

  // compute celltypeinit for both MPM and cutcells (if applicable).  We do this
  // even though we have already chosen one of these two, because we want to see
  // the difference in the celltypes that we get from the two methods.

  d_arches->getBoundaryCondition()->sched_mmWallCellTypeInit(sched, patches, arches_matls, fixCellType);

  // for explicit calculation, exchange will be at the beginning

  scheduleMomExchange(sched, patches, arches_matls, mpm_matls, all_matls);

  if (d_calcEnergyExchange) 
    scheduleEnergyExchange(sched, patches, arches_matls, mpm_matls, all_matls);

  schedulePutAllForcesOnCC(sched, patches, mpm_matls);
  schedulePutAllForcesOnNC(sched, patches, mpm_matls);

  if (d_calcEnergyExchange) {
    scheduleComputeIntegratedSolidProps(sched, patches, arches_matls, mpm_matls, all_matls);
    scheduleComputeTotalHT(sched, patches, arches_matls);
  }

  // we may also need mass exchange here later.  This is 
  // not implemented yet.  This will be of the form
  // scheduleMassExchange(level, sched, old_dw, new_dw)

  // both Arches and MPM now have all the information they need
  // to proceed independently with their solution
  // Arches steps are identical with those in single-material code
  // once exchange terms are determined

  if ( d_doingRestart ) { 
    d_arches->MPMArchesIntrusionSetupForResart( level, sched, d_recompile, d_doingRestart ); 
    d_doingRestart = false; 
  }

  d_arches->scheduleTimeAdvance( level, sched );

  // remaining MPM steps are explicitly shown here.
  d_mpm->scheduleExMomInterpolated(sched, patches, mpm_matls);

  d_mpm->scheduleComputeInternalForce(sched, patches, mpm_matls);
  d_mpm->scheduleComputeInternalHeatRate(sched, patches, mpm_matls);
  d_mpm->scheduleComputeNodalHeatFlux(sched, patches, mpm_matls);
  scheduleComputeAndIntegrateAcceleration(sched, patches, mpm_matls);
  scheduleSolveHeatEquations(sched, patches, mpm_matls);
  d_mpm->scheduleIntegrateTemperatureRate(sched, patches, mpm_matls);
  d_mpm->scheduleExMomIntegrated(sched, patches, mpm_matls);
  d_mpm->scheduleComputeStressTensor(sched, patches, mpm_matls);
  d_mpm->scheduleInterpolateToParticlesAndUpdate(sched, patches, mpm_matls);

  sched->scheduleParticleRelocation(level, 
                                    Mlb->pXLabel_preReloc,
                                    d_sharedState->d_particleState_preReloc,
                                    Mlb->pXLabel, 
                                    d_sharedState->d_particleState,
                                    Mlb->pParticleIDLabel,
                                    mpm_matls);
      
  //__________________________________
  //  on the fly analysis
  if(d_analysisModules.size() != 0){
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++){
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis( sched, level);
    }
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleInterpolateNCToCC(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)
{
  /* interpolate variables from node centers to cell center */

  // primitive variable initialization
  Task* t=scinew Task("MPMArches::interpolateNCToCC",
      this, &MPMArches::interpolateNCToCC);
  int numGhostCells = 1;

  t->requires(Task::NewDW, Mlb->gMassLabel, 
      Ghost::AroundNodes, numGhostCells);
  t->computes(d_MAlb->cMassLabel);

  t->requires(Task::NewDW, Mlb->gVolumeLabel,
      Ghost::AroundNodes, numGhostCells);
  t->computes(d_MAlb->cVolumeLabel);

  t->requires(Task::NewDW, Mlb->gVelocityLabel,
      Ghost::AroundNodes, numGhostCells);
  t->computes(d_MAlb->vel_CCLabel);

  if (d_calcEnergyExchange) {
    t->requires(Task::NewDW, Mlb->gTemperatureLabel,
        Ghost::AroundNodes, numGhostCells);
    t->computes(d_MAlb->tempSolid_CCLabel);
  }

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//

void MPMArches::interpolateNCToCC(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* matls,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){

      Vector zero(0.0,0.0,0.);
      int numGhostCells = 1;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();

      // Create arrays for the grid data

      constNCVariable<double > gmass, gvolume, gTemperature;
      constNCVariable<Vector > gvelocity;
      CCVariable<double > cmass, cvolume, tempSolid_CC;
      CCVariable<Vector > vel_CC;

      new_dw->get(gmass,     Mlb->gMassLabel,        matlindex, 
          patch, Ghost::AroundNodes, numGhostCells);
      new_dw->allocateAndPut(cmass, d_MAlb->cMassLabel,         
          matlindex, patch);
      cmass.initialize(0.);

      new_dw->get(gvolume,   Mlb->gVolumeLabel,      matlindex, 
          patch, Ghost::AroundNodes, numGhostCells);
      new_dw->allocateAndPut(cvolume, d_MAlb->cVolumeLabel,       
          matlindex, patch);
      cvolume.initialize(0.);

      new_dw->get(gvelocity, Mlb->gVelocityLabel,    matlindex, 
          patch, Ghost::AroundNodes, numGhostCells);
      new_dw->allocateAndPut(vel_CC, d_MAlb->vel_CCLabel,        
          matlindex, patch);
      vel_CC.initialize(zero); 

      if (d_calcEnergyExchange) {
        new_dw->get(gTemperature, Mlb->gTemperatureLabel, matlindex, 
            patch, Ghost::AroundNodes, numGhostCells);
        new_dw->allocateAndPut(tempSolid_CC, d_MAlb->tempSolid_CCLabel, 
            matlindex, patch);
        tempSolid_CC.initialize(0.);

      }

      IntVector nodeIdx[8];

      for (CellIterator iter =patch->getExtraCellIterator();
          !iter.done();iter++){

        patch->findNodesFromCell(*iter,nodeIdx);
        for (int in=0;in<8;in++){

          cmass[*iter]    += .125*gmass[nodeIdx[in]];
          cvolume[*iter]  += .125*gvolume[nodeIdx[in]];

          if (calcVel) 
            vel_CC[*iter]   += gvelocity[nodeIdx[in]]*.125*
              gmass[nodeIdx[in]];

          if (d_calcEnergyExchange) {
            tempSolid_CC[*iter] += gTemperature[nodeIdx[in]]*.125*
              gmass[nodeIdx[in]];
          }
        }

        if (calcVel){
          vel_CC[*iter]      /= (cmass[*iter]     + d_SMALL_NUM);
        }

        if (d_calcEnergyExchange){
          tempSolid_CC[*iter]   /= (cmass[*iter]     + d_SMALL_NUM);
        }

      }
    }
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleInterpolateCCToFC(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)
{
  Task* t=scinew Task("MPMArches::interpolateCCToFC",
      this, &MPMArches::interpolateCCToFC);
  int numGhostCells = 1;

  t->requires(Task::NewDW, d_MAlb->cMassLabel,
      Ghost::AroundCells, numGhostCells);
  if (calcVel)
    t->requires(Task::NewDW, d_MAlb->vel_CCLabel,
        Ghost::AroundCells, numGhostCells);
  if (d_calcEnergyExchange) 
    t->requires(Task::NewDW, d_MAlb->tempSolid_CCLabel,
        Ghost::AroundCells, numGhostCells);

  t->computes(d_MAlb->xvel_CCLabel);
  t->computes(d_MAlb->yvel_CCLabel);
  t->computes(d_MAlb->zvel_CCLabel);

  t->computes(d_MAlb->xvel_FCXLabel);
  t->computes(d_MAlb->xvel_FCYLabel);
  t->computes(d_MAlb->xvel_FCZLabel);

  t->computes(d_MAlb->yvel_FCXLabel);
  t->computes(d_MAlb->yvel_FCYLabel);
  t->computes(d_MAlb->yvel_FCZLabel);

  t->computes(d_MAlb->zvel_FCXLabel);
  t->computes(d_MAlb->zvel_FCYLabel);
  t->computes(d_MAlb->zvel_FCZLabel);

  if (d_calcEnergyExchange) {

    t->computes(d_MAlb->tempSolid_FCXLabel);
    t->computes(d_MAlb->tempSolid_FCYLabel);
    t->computes(d_MAlb->tempSolid_FCZLabel);

  }

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//

void MPMArches::interpolateCCToFC(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* matls,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){

      int numGhostCells = 1;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();

      constCCVariable<double > cmass;
      constCCVariable<Vector > vel_CC;
      constCCVariable<double > tempSolid_CC;

      CCVariable<double> xvel_CC;
      CCVariable<double> yvel_CC;
      CCVariable<double> zvel_CC;

      SFCXVariable<double> xvelFCX;
      SFCYVariable<double> xvelFCY;
      SFCZVariable<double> xvelFCZ;

      SFCXVariable<double> yvelFCX;
      SFCYVariable<double> yvelFCY;
      SFCZVariable<double> yvelFCZ;

      SFCXVariable<double> zvelFCX;
      SFCYVariable<double> zvelFCY;
      SFCZVariable<double> zvelFCZ;

      SFCXVariable<double> tempSolid_FCX;
      SFCYVariable<double> tempSolid_FCY;
      SFCZVariable<double> tempSolid_FCZ;

      new_dw->get(cmass,    d_MAlb->cMassLabel,         matlindex, 
          patch, Ghost::AroundCells, numGhostCells);
      if (calcVel)
        new_dw->get(vel_CC,   d_MAlb->vel_CCLabel,        matlindex, 
            patch, Ghost::AroundCells, numGhostCells);
      if (d_calcEnergyExchange) 
        new_dw->get(tempSolid_CC, d_MAlb->tempSolid_CCLabel, matlindex, 
            patch, Ghost::AroundCells, numGhostCells);

      new_dw->allocateAndPut(xvel_CC, d_MAlb->xvel_CCLabel,
          matlindex, patch);
      new_dw->allocateAndPut(yvel_CC, d_MAlb->yvel_CCLabel,
          matlindex, patch);
      new_dw->allocateAndPut(zvel_CC, d_MAlb->zvel_CCLabel,
          matlindex, patch);

      new_dw->allocateAndPut(xvelFCX, d_MAlb->xvel_FCXLabel, 
          matlindex, patch);
      new_dw->allocateAndPut(xvelFCY, d_MAlb->xvel_FCYLabel, 
          matlindex, patch);
      new_dw->allocateAndPut(xvelFCZ, d_MAlb->xvel_FCZLabel, 
          matlindex, patch);

      new_dw->allocateAndPut(yvelFCX, d_MAlb->yvel_FCXLabel, 
          matlindex, patch);
      new_dw->allocateAndPut(yvelFCY, d_MAlb->yvel_FCYLabel, 
          matlindex, patch);
      new_dw->allocateAndPut(yvelFCZ, d_MAlb->yvel_FCZLabel, 
          matlindex, patch);

      new_dw->allocateAndPut(zvelFCX, d_MAlb->zvel_FCXLabel, 
          matlindex, patch);
      new_dw->allocateAndPut(zvelFCY, d_MAlb->zvel_FCYLabel, 
          matlindex, patch);
      new_dw->allocateAndPut(zvelFCZ, d_MAlb->zvel_FCZLabel, 
          matlindex, patch);

      if (d_calcEnergyExchange) {

        new_dw->allocateAndPut(tempSolid_FCX, d_MAlb->tempSolid_FCXLabel,
            matlindex, patch);
        new_dw->allocateAndPut(tempSolid_FCY, d_MAlb->tempSolid_FCYLabel,
            matlindex, patch);
        new_dw->allocateAndPut(tempSolid_FCZ, d_MAlb->tempSolid_FCZLabel,
            matlindex, patch);

      }

      xvel_CC.initialize(0.);
      yvel_CC.initialize(0.);
      zvel_CC.initialize(0.);

      xvelFCX.initialize(0.);
      xvelFCY.initialize(0.);
      xvelFCZ.initialize(0.);

      yvelFCX.initialize(0.);
      yvelFCY.initialize(0.);
      yvelFCZ.initialize(0.);

      zvelFCX.initialize(0.);
      zvelFCY.initialize(0.);
      zvelFCZ.initialize(0.);

      if (d_calcEnergyExchange) {
        tempSolid_FCX.initialize(0.);
        tempSolid_FCY.initialize(0.);
        tempSolid_FCZ.initialize(0.);
      }

      double mass;

      if (d_calcEnergyExchange) {

        for(CellIterator iter = patch->getExtraCellIterator();
            !iter.done(); iter++){

          IntVector curcell = *iter;
          if (curcell.x() >= (patch->getCellLowIndex()).x()) {

            IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());
            mass = cmass[curcell] + cmass[adjcell];
            tempSolid_FCX[curcell] = (tempSolid_CC[curcell] * cmass[curcell] +
                tempSolid_CC[adjcell] * cmass[adjcell])/mass;
            
          }
          //_____________________________________
          //   S O U T H   F A C E S (FCY Values)

          if (curcell.y() >= (patch->getCellLowIndex()).y()) {

            IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
            mass = cmass[curcell] + cmass[adjcell];
            tempSolid_FCY[curcell] = (tempSolid_CC[curcell] * cmass[curcell] +
                tempSolid_CC[adjcell] * cmass[adjcell])/mass;
          }
          //_______________________________________
          //   B O T T O M   F A C E S (FCZ Values)

          if (curcell.z() >= (patch->getCellLowIndex()).z()) {

            IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
            mass = cmass[curcell] + cmass[adjcell];
            tempSolid_FCZ[curcell] = (tempSolid_CC[curcell] * cmass[curcell] +
                tempSolid_CC[adjcell] * cmass[adjcell])/mass;
          }
        }
      }

      if (calcVel) {

        for(CellIterator iter = patch->getExtraCellIterator();
            !iter.done(); iter++){

          IntVector curcell = *iter;
          xvel_CC[curcell] = vel_CC[curcell].x();
          yvel_CC[curcell] = vel_CC[curcell].y();
          zvel_CC[curcell] = vel_CC[curcell].z();

          //___________________________________
          //   L E F T   F A C E S (FCX Values)

          if (curcell.x() >= (patch->getCellLowIndex()).x()) {

            IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());
            mass = cmass[curcell] + cmass[adjcell];

            xvelFCX[curcell] = (vel_CC[curcell].x() * cmass[curcell] +
                vel_CC[adjcell].x() * cmass[adjcell])/mass;
            yvelFCX[curcell] = (vel_CC[curcell].y() * cmass[curcell] +
                vel_CC[adjcell].y() * cmass[adjcell])/mass;
            zvelFCX[curcell] = (vel_CC[curcell].z() * cmass[curcell] +
                vel_CC[adjcell].z() * cmass[adjcell])/mass;

          }
          //_____________________________________
          //   S O U T H   F A C E S (FCY Values)

          if (curcell.y() >= (patch->getCellLowIndex()).y()) {

            IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
            mass = cmass[curcell] + cmass[adjcell];

            xvelFCY[curcell] = (vel_CC[curcell].x() * cmass[curcell] +
                vel_CC[adjcell].x() * cmass[adjcell])/mass;
            yvelFCY[curcell] = (vel_CC[curcell].y() * cmass[curcell] +
                vel_CC[adjcell].y() * cmass[adjcell])/mass;
            zvelFCY[curcell] = (vel_CC[curcell].z() * cmass[curcell] +
                vel_CC[adjcell].z() * cmass[adjcell])/mass;

          }
          //_______________________________________
          //   B O T T O M   F A C E S (FCZ Values)

          if (curcell.z() >= (patch->getCellLowIndex()).z()) {

            IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
            mass = cmass[curcell] + cmass[adjcell];

            xvelFCZ[curcell] = (vel_CC[curcell].x() * cmass[curcell] +
                vel_CC[adjcell].x() * cmass[adjcell])/mass;
            yvelFCZ[curcell] = (vel_CC[curcell].y() * cmass[curcell] +
                vel_CC[adjcell].y() * cmass[adjcell])/mass;
            zvelFCZ[curcell] = (vel_CC[curcell].z() * cmass[curcell] +
                vel_CC[adjcell].z() * cmass[adjcell])/mass;

          }
        }
      }
    }
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleComputeVoidFracMPM(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls,
    const MaterialSet* mpm_matls,
    const MaterialSet* all_matls)
{
  // primitive variable initialization

  Task* t=scinew Task("MPMArches::computeVoidFracMPM",
      this, &MPMArches::computeVoidFracMPM);

  int zeroGhostCells = 0;

//  double time = d_sharedState->getElapsedTime();
//  bool recalculateVoidFrac = false;
//  if (time < 1.0e-10 || !d_stationarySolid) recalculateVoidFrac = true;

  t->requires(Task::NewDW, d_MAlb->cVolumeLabel,   
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, d_MAlb->void_frac_MPM_CCLabel,   
      arches_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->requires(Task::OldDW, d_MAlb->solid_fraction_CCLabel,   
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);

//  if (time < 1.0E-10)
//    d_recompile = true;

  t->computes(d_MAlb->solid_fraction_CCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->void_frac_MPM_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->solid_frac_sum_CCLabel, arches_matls->getUnion());

  sched->addTask(t, patches, all_matls);
}

//______________________________________________________________________
//

void MPMArches::computeVoidFracMPM(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw) 

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlindex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int zeroGhostCells = 0;

    StaticArray<constCCVariable<double> > mat_vol(numMPMMatls);
    StaticArray<constCCVariable<double> > oldSolidFrac(numMPMMatls);
    constCCVariable<double> oldVoidFrac;

    StaticArray<CCVariable<double> > solid_fraction_cc(numMPMMatls);    
    CCVariable<double> void_frac;
    CCVariable<double> solid_sum;

    double time = d_sharedState->getElapsedTime();
    bool recalculateVoidFrac = false;
    if (time < 1.0e-10 || !d_stationarySolid) recalculateVoidFrac = true;
    bool recalculateSolidFrac = recalculateVoidFrac;

    // get and allocate

    for (int m = 0; m < numMPMMatls; m++) {	
      Material* matl = d_sharedState->getMPMMaterial( m );
      int dwindex = matl->getDWIndex();
      new_dw->get(mat_vol[m], d_MAlb->cVolumeLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
    }
    old_dw->get(oldVoidFrac, d_MAlb->void_frac_MPM_CCLabel,
        matlindex, patch, Ghost::None, zeroGhostCells);
    for (int m = 0; m < numMPMMatls; m++) {
      Material* matl = d_sharedState->getMPMMaterial( m );
      int dwindex = matl->getDWIndex();
      old_dw->get(oldSolidFrac[m], d_MAlb->solid_fraction_CCLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
    }

    for (int m = 0; m < numMPMMatls; m++) {	
      Material* matl = d_sharedState->getMPMMaterial( m );
      int dwindex = matl->getDWIndex();
      new_dw->allocateAndPut(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
          dwindex, patch);
      solid_fraction_cc[m].initialize(1.0);
    }
    new_dw->allocateAndPut(void_frac, d_MAlb->void_frac_MPM_CCLabel, 
        matlindex, patch); 
    void_frac.initialize(1.0);
    new_dw->allocateAndPut(solid_sum, d_MAlb->solid_frac_sum_CCLabel, 
        matlindex, patch); 
    solid_sum.initialize(0.0);

    // actual computation

    // First calculate solid fractions and solid fraction sum
    // alone.  We will use this later (below) to get the void fraction.

    if (recalculateSolidFrac) {
      for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
        double total_vol = patch->dCell().x()*patch->dCell().y()*patch->dCell().z();
        double solid_frac_sum = 0.0;
        solid_sum[*iter] = 0.0;
        for (int m = 0; m < numMPMMatls; m++) {
          solid_fraction_cc[m][*iter] = mat_vol[m][*iter]/total_vol;
          solid_frac_sum += solid_fraction_cc[m][*iter];
          solid_sum[*iter] += solid_fraction_cc[m][*iter];
        }
        if (solid_sum[*iter] > 1.0)
          solid_sum[*iter] = 1.0;	
        if (solid_frac_sum > 0.0) {
          for (int m = 0; m < numMPMMatls; m++) {
            solid_fraction_cc[m][*iter] = solid_fraction_cc[m][*iter]/solid_frac_sum;
          }
          if (numMPMMatls < 1)
            solid_fraction_cc[numMPMMatls][*iter] = 1.0;
        }
      }
    }
    else {
      for (int m = 0; m < numMPMMatls; m++) {
        solid_fraction_cc[m].copyData(oldSolidFrac[m]);
      }
    }

    if (recalculateVoidFrac) {
      for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
        double solid_frac_sum = solid_sum[*iter];
        void_frac[*iter] = 1.0 - solid_frac_sum;
        if (void_frac[*iter] < 0.0)
          void_frac[*iter] = 0.0;
      }
    }
    else
      void_frac.copyData(oldVoidFrac);
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleCopyCutCells(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls)

{ 
  // primitive variable initialization
  Task* t=scinew Task("MPMArches::copyCutCells",
      this, &MPMArches::copyCutCells);
  int numGhostCells = 0;

  t->requires(Task::OldDW, d_MAlb->d_normal1Label,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_normal2Label,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_normal3Label,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_normalLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);

  t->requires(Task::OldDW, d_MAlb->d_centroid1Label,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_centroid2Label,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_centroid3Label,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_totAreaLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);

  t->requires(Task::OldDW, d_MAlb->d_pGasAreaFracXPLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_pGasAreaFracXELabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_pGasAreaFracYPLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_pGasAreaFracYNLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_pGasAreaFracZPLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);
  t->requires(Task::OldDW, d_MAlb->d_pGasAreaFracZTLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);

  t->requires(Task::OldDW, d_MAlb->void_frac_CutCell_CCLabel,
      arches_matls->getUnion(),
      Ghost::None, numGhostCells);

  t->computes(d_MAlb->d_normal1Label, arches_matls->getUnion());
  t->computes(d_MAlb->d_normal2Label, arches_matls->getUnion());
  t->computes(d_MAlb->d_normal3Label, arches_matls->getUnion());
  t->computes(d_MAlb->d_normalLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_centroid1Label, arches_matls->getUnion());
  t->computes(d_MAlb->d_centroid2Label, arches_matls->getUnion());
  t->computes(d_MAlb->d_centroid3Label, arches_matls->getUnion());
  t->computes(d_MAlb->d_totAreaLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_pGasAreaFracXPLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_pGasAreaFracXELabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_pGasAreaFracYPLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_pGasAreaFracYNLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_pGasAreaFracZPLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_pGasAreaFracZTLabel, arches_matls->getUnion());

  t->computes(d_MAlb->void_frac_CutCell_CCLabel, arches_matls->getUnion());

  sched->addTask(t, patches, arches_matls);
}

//______________________________________________________________________
//

void MPMArches::copyCutCells(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> oldNormal1;
    CCVariable<double> normal1;
    constCCVariable<double> oldNormal2;
    CCVariable<double> normal2;
    constCCVariable<double> oldNormal3;
    CCVariable<double> normal3;
    constCCVariable<Vector> oldNormal;
    CCVariable<Vector> normal;

    constCCVariable<double> oldCentroid1;
    CCVariable<double> centroid1;
    constCCVariable<double> oldCentroid2;
    CCVariable<double> centroid2;
    constCCVariable<double> oldCentroid3;
    CCVariable<double> centroid3;
    constCCVariable<double> oldTotArea;
    CCVariable<double> totArea;

    constCCVariable<double> oldPGasAreaFracXP;
    CCVariable<double> pGasAreaFracXP;
    constCCVariable<double> oldPGasAreaFracXE;
    CCVariable<double> pGasAreaFracXE;
    constCCVariable<double> oldPGasAreaFracYP;
    CCVariable<double> pGasAreaFracYP;
    constCCVariable<double> oldPGasAreaFracYN;
    CCVariable<double> pGasAreaFracYN;
    constCCVariable<double> oldPGasAreaFracZP;
    CCVariable<double> pGasAreaFracZP;
    constCCVariable<double> oldPGasAreaFracZT;
    CCVariable<double> pGasAreaFracZT;

    constCCVariable<double> oldepsg;
    CCVariable<double> epsg;

    //    
    old_dw->get(oldNormal1, d_MAlb->d_normal1Label, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(normal1, d_MAlb->d_normal1Label,
        matlIndex, patch);
    normal1.initialize(0.0);
    normal1.copyData(oldNormal1);

    if (patch->containsCell(IntVector(6,53,80))) {
      cerr << "[6,53,80] x normal" << normal1[IntVector(6,53,80)] << endl;
    }

    old_dw->get(oldNormal2, d_MAlb->d_normal2Label, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(normal2, d_MAlb->d_normal2Label,
        matlIndex, patch);
    normal2.copyData(oldNormal2);

    old_dw->get(oldNormal3, d_MAlb->d_normal3Label, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(normal3, d_MAlb->d_normal3Label,
        matlIndex, patch);
    normal3.copyData(oldNormal3);

    old_dw->get(oldNormal, d_MAlb->d_normalLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(normal, d_MAlb->d_normalLabel,
        matlIndex, patch);
    normal.copyData(oldNormal);
    //

    //
    old_dw->get(oldCentroid1, d_MAlb->d_centroid1Label, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(centroid1, d_MAlb->d_centroid1Label,
        matlIndex, patch);
    centroid1.copyData(oldCentroid1);

    old_dw->get(oldCentroid2, d_MAlb->d_centroid2Label, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(centroid2, d_MAlb->d_centroid2Label,
        matlIndex, patch);
    centroid2.copyData(oldCentroid2);

    old_dw->get(oldCentroid3, d_MAlb->d_centroid3Label, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(centroid3, d_MAlb->d_centroid3Label,
        matlIndex, patch);
    centroid3.copyData(oldCentroid3);

    old_dw->get(oldTotArea, d_MAlb->d_totAreaLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(totArea, d_MAlb->d_totAreaLabel,
        matlIndex, patch);
    totArea.copyData(oldTotArea);
    //

    //
    old_dw->get(oldPGasAreaFracXP, d_MAlb->d_pGasAreaFracXPLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(pGasAreaFracXP, d_MAlb->d_pGasAreaFracXPLabel,
        matlIndex, patch);
    pGasAreaFracXP.copyData(oldPGasAreaFracXP);

    old_dw->get(oldPGasAreaFracXE, d_MAlb->d_pGasAreaFracXELabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(pGasAreaFracXE, d_MAlb->d_pGasAreaFracXELabel,
        matlIndex, patch);
    pGasAreaFracXE.copyData(oldPGasAreaFracXE);

    old_dw->get(oldPGasAreaFracYP, d_MAlb->d_pGasAreaFracYPLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(pGasAreaFracYP, d_MAlb->d_pGasAreaFracYPLabel,
        matlIndex, patch);
    pGasAreaFracYP.copyData(oldPGasAreaFracYP);

    old_dw->get(oldPGasAreaFracYN, d_MAlb->d_pGasAreaFracYNLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(pGasAreaFracYN, d_MAlb->d_pGasAreaFracYNLabel,
        matlIndex, patch);
    pGasAreaFracYN.copyData(oldPGasAreaFracYN);

    old_dw->get(oldPGasAreaFracZP, d_MAlb->d_pGasAreaFracZPLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(pGasAreaFracZP, d_MAlb->d_pGasAreaFracZPLabel,
        matlIndex, patch);
    pGasAreaFracZP.copyData(oldPGasAreaFracZP);

    old_dw->get(oldPGasAreaFracZT, d_MAlb->d_pGasAreaFracZTLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(pGasAreaFracZT, d_MAlb->d_pGasAreaFracZTLabel,
        matlIndex, patch);
    pGasAreaFracZT.copyData(oldPGasAreaFracZT);
    //

    old_dw->get(oldepsg, d_MAlb->void_frac_CutCell_CCLabel, matlIndex, 
        patch, Ghost::None, 0);    
    new_dw->allocateAndPut(epsg, d_MAlb->void_frac_CutCell_CCLabel,
        matlIndex, patch);
    epsg.copyData(oldepsg);

  }
}

//______________________________________________________________________
//

void MPMArches::scheduleComputeVoidFrac(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls,
    const MaterialSet* mpm_matls,
    const MaterialSet* all_matls)
{
  // primitive variable initialization

  Task* t=scinew Task("MPMArches::computeVoidFrac",
      this, &MPMArches::computeVoidFrac);

  int zeroGhostCells = 0;

//  double time = d_sharedState->getElapsedTime();
//  if (time < 1.0E-10)
//    d_recompile = true;

  if (d_useCutCell)
    t->requires(Task::NewDW, d_MAlb->void_frac_CutCell_CCLabel, 
        arches_matls->getUnion(), Ghost::None, zeroGhostCells);
  else
    t->requires(Task::NewDW, d_MAlb->void_frac_MPM_CCLabel, 
        arches_matls->getUnion(), Ghost::None, zeroGhostCells);

  t->computes(d_MAlb->void_frac_CCLabel, arches_matls->getUnion());

  sched->addTask(t, patches, all_matls);
}

//______________________________________________________________________
//

void MPMArches::computeVoidFrac(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw) 

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    StaticArray<constCCVariable<double> > solid_fraction_cc_old(numMPMMatls);
    StaticArray<CCVariable<double> > solid_fraction_cc(numMPMMatls);

    // int zeroGhostCells = 0;

    // get and allocate

    CCVariable<double> void_frac;
    new_dw->allocateAndPut(void_frac, d_MAlb->void_frac_CCLabel, 
        matlIndex, patch); 

    constCCVariable<double> void_frac_CutCell;
    constCCVariable<double> void_frac_MPM;
    if (d_useCutCell)
      new_dw->get(void_frac_CutCell, d_MAlb->void_frac_CutCell_CCLabel, 
          matlIndex, patch, Ghost::None, 0);    
    else
      new_dw->get(void_frac_MPM, d_MAlb->void_frac_MPM_CCLabel, 
          matlIndex, patch, Ghost::None, 0);    

    // actual computation

    if (d_useCutCell)
      void_frac.copyData(void_frac_CutCell);
    else
      void_frac.copyData(void_frac_MPM);

  }
}

//______________________________________________________________________
//

void MPMArches::scheduleComputeIntegratedSolidProps(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls,
    const MaterialSet* mpm_matls,
    const MaterialSet* all_matls)
{
  // primitive variable initialization

  Task* t=scinew Task("MPMArches::getIntegratedProps",
      this, &MPMArches::computeIntegratedSolidProps);

  int zeroGhostCells = 0;

  t->requires(Task::NewDW, d_MAlb->solid_fraction_CCLabel,   
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->tempSolid_CCLabel,   
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->heaTranSolid_CCLabel,
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->heaTranSolid_FCXLabel,
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->heaTranSolid_FCYLabel,
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->heaTranSolid_FCZLabel,
      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);

  t->computes(d_MAlb->integTemp_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->integHTS_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->integHTS_FCXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->integHTS_FCYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->integHTS_FCZLabel, arches_matls->getUnion());

  sched->addTask(t, patches, all_matls);
}

//______________________________________________________________________
//

void MPMArches::computeIntegratedSolidProps(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw) 

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlindex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    StaticArray<constCCVariable<double> > solid_fraction_cc(numMPMMatls);
    StaticArray<constCCVariable<double> > tempSolid_CC(numMPMMatls);
    StaticArray<constCCVariable<double> > hTSolid_CC(numMPMMatls);
    StaticArray<constSFCXVariable<double> > hTSolid_FCX(numMPMMatls);
    StaticArray<constSFCYVariable<double> > hTSolid_FCY(numMPMMatls);
    StaticArray<constSFCZVariable<double> > hTSolid_FCZ(numMPMMatls);

    int zeroGhostCells = 0;

    // get and allocate

    for (int m = 0; m < numMPMMatls; m++) {

      Material* matl = d_sharedState->getMPMMaterial( m );
      int dwindex = matl->getDWIndex();

      new_dw->get(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
      new_dw->get(tempSolid_CC[m], d_MAlb->tempSolid_CCLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
      new_dw->get(hTSolid_CC[m], d_MAlb->heaTranSolid_CCLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
      new_dw->get(hTSolid_FCX[m], d_MAlb->heaTranSolid_FCXLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
      new_dw->get(hTSolid_FCY[m], d_MAlb->heaTranSolid_FCYLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
      new_dw->get(hTSolid_FCZ[m], d_MAlb->heaTranSolid_FCZLabel,
          dwindex, patch, Ghost::None, zeroGhostCells);
    }

    CCVariable<double> integTemp;
    new_dw->allocateAndPut(integTemp, d_MAlb->integTemp_CCLabel,
        matlindex, patch); 
    CCVariable<double> integHTS_CC;
    new_dw->allocateAndPut(integHTS_CC, d_MAlb->integHTS_CCLabel,
        matlindex, patch); 
    SFCXVariable<double> integHTS_FCX;
    new_dw->allocateAndPut(integHTS_FCX, d_MAlb->integHTS_FCXLabel,
        matlindex, patch); 
    SFCYVariable<double> integHTS_FCY;
    new_dw->allocateAndPut(integHTS_FCY, d_MAlb->integHTS_FCYLabel,
        matlindex, patch); 
    SFCZVariable<double> integHTS_FCZ;
    new_dw->allocateAndPut(integHTS_FCZ, d_MAlb->integHTS_FCZLabel,
        matlindex, patch); 


    // actual computation

    integTemp.initialize(0.0);
    integHTS_CC.initialize(0.0);
    integHTS_FCX.initialize(0.0);
    integHTS_FCY.initialize(0.0);
    integHTS_FCZ.initialize(0.0);

    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {

      for (int m = 0; m < numMPMMatls; m++) {

        integTemp[*iter] += solid_fraction_cc[m][*iter]*tempSolid_CC[m][*iter];
        integHTS_CC[*iter] += solid_fraction_cc[m][*iter]*hTSolid_CC[m][*iter];
        integHTS_FCX[*iter] += solid_fraction_cc[m][*iter]*hTSolid_FCX[m][*iter];
        integHTS_FCY[*iter] += solid_fraction_cc[m][*iter]*hTSolid_FCY[m][*iter];
        integHTS_FCZ[*iter] += solid_fraction_cc[m][*iter]*hTSolid_FCZ[m][*iter];

      }
    }
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleComputeTotalHT(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls)
{
  // primitive variable initialization

  Task* t=scinew Task("MPMArches::getTotalHeatFlux",
      this, &MPMArches::computeTotalHT);

  // Purposes: 1. To calculate the TOTAL heat transfer to 
  // a cell, adding face energy transfers from all directions 
  // and from all adjacent partial cells
  // 
  // 2. To calculate the directional heat rates incident on 
  // any faces, essentially SFCXhtrate(i,j,k) = SFCXhtrate(i,j,k)
  // + partialcellhtrate(i+1,j,k) + partialcellhtrate(i-1,j,k)
  // and similarly for the other two directions
  //
  // 3. To calculate the directional heat fluxes incident on 
  // any faces, essentially SFCXflux(i,j,k) = SFCXflux(i,j,k)
  // + partialcellflux(i+1,j,k) + partialcellflux(i-1,j,k)
  // and similarly for the other two directions

  int numGhostCells = 1;

  t->requires(Task::NewDW, d_MAlb->integHTS_CCLabel,
      arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->integHTS_FCXLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->integHTS_FCYLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->integHTS_FCZLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->htfluxXLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->htfluxYLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->htfluxZLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->htfluxConvCCLabel,
      arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);

  t->computes(d_MAlb->totHT_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->totHT_FCXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->totHT_FCYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->totHT_FCZLabel, arches_matls->getUnion());

  t->computes(d_MAlb->totHtFluxXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->totHtFluxYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->totHtFluxZLabel, arches_matls->getUnion());

  sched->addTask(t, patches, arches_matls);
}

//______________________________________________________________________
//

void MPMArches::computeTotalHT(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw) 

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlindex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> integHTS_CC;
    constSFCXVariable<double> integHTS_FCX;
    constSFCYVariable<double> integHTS_FCY;
    constSFCZVariable<double> integHTS_FCZ;

    constSFCXVariable<double> htfluxX;
    constSFCYVariable<double> htfluxY;
    constSFCZVariable<double> htfluxZ;
    constCCVariable<double> htfluxCC;

    int numGhostCells = 1;

    // get and allocate

    new_dw->get(integHTS_CC, d_MAlb->integHTS_CCLabel,
        matlindex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(integHTS_FCX, d_MAlb->integHTS_FCXLabel,
        matlindex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(integHTS_FCY, d_MAlb->integHTS_FCYLabel,
        matlindex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(integHTS_FCZ, d_MAlb->integHTS_FCZLabel,
        matlindex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->get(htfluxX, d_MAlb->htfluxXLabel,
        matlindex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(htfluxY, d_MAlb->htfluxYLabel,
        matlindex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(htfluxZ, d_MAlb->htfluxZLabel,
        matlindex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(htfluxCC, d_MAlb->htfluxConvCCLabel,
        matlindex, patch, Ghost::AroundCells, numGhostCells);

    CCVariable<double> totalHT;
    new_dw->allocateAndPut(totalHT, d_MAlb->totHT_CCLabel,
        matlindex, patch); 
    SFCXVariable<double> totalHT_FCX;
    new_dw->allocateAndPut(totalHT_FCX, d_MAlb->totHT_FCXLabel,
        matlindex, patch); 
    SFCYVariable<double> totalHT_FCY;
    new_dw->allocateAndPut(totalHT_FCY, d_MAlb->totHT_FCYLabel,
        matlindex, patch); 
    SFCZVariable<double> totalHT_FCZ;
    new_dw->allocateAndPut(totalHT_FCZ, d_MAlb->totHT_FCZLabel,
        matlindex, patch); 

    SFCXVariable<double> totHtFluxX;
    new_dw->allocateAndPut(totHtFluxX, d_MAlb->totHtFluxXLabel,
        matlindex, patch); 
    SFCYVariable<double> totHtFluxY;
    new_dw->allocateAndPut(totHtFluxY, d_MAlb->totHtFluxYLabel,
        matlindex, patch); 
    SFCZVariable<double> totHtFluxZ;
    new_dw->allocateAndPut(totHtFluxZ, d_MAlb->totHtFluxZLabel,
        matlindex, patch); 

    // actual computation

    totalHT.initialize(0.0);
    totalHT_FCX.initialize(0.0);
    totalHT_FCY.initialize(0.0);
    totalHT_FCZ.initialize(0.0);
    totHtFluxX.initialize(0.0);
    totHtFluxY.initialize(0.0);
    totHtFluxZ.initialize(0.0);

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++) {

      IntVector curcell = *iter;
      IntVector eastcell(curcell.x()+1,curcell.y(),curcell.z());
      IntVector westcell(curcell.x()-1,curcell.y(),curcell.z());
      IntVector northcell(curcell.x(),curcell.y()+1,curcell.z());
      IntVector southcell(curcell.x(),curcell.y()-1,curcell.z());
      IntVector topcell(curcell.x(),curcell.y(),curcell.z()+1);
      IntVector botcell(curcell.x(),curcell.y(),curcell.z()-1);

      totalHT[curcell] = totalHT[curcell] + integHTS_FCX[curcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_FCX[eastcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_FCY[curcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_FCY[northcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_FCZ[curcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_FCZ[topcell];

      totalHT[curcell] = totalHT[curcell] + integHTS_CC[westcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_CC[eastcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_CC[southcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_CC[northcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_CC[botcell];
      totalHT[curcell] = totalHT[curcell] + integHTS_CC[topcell];

      totalHT_FCX[curcell] = totalHT_FCX[curcell] + integHTS_FCX[curcell];
      totalHT_FCX[curcell] = totalHT_FCX[curcell] + integHTS_CC[westcell];
      totalHT_FCX[curcell] = totalHT_FCX[curcell] + integHTS_CC[curcell];

      totalHT_FCY[curcell] = totalHT_FCY[curcell] + integHTS_FCY[curcell];
      totalHT_FCY[curcell] = totalHT_FCY[curcell] + integHTS_CC[southcell];
      totalHT_FCY[curcell] = totalHT_FCY[curcell] + integHTS_CC[curcell];

      totalHT_FCZ[curcell] = totalHT_FCZ[curcell] + integHTS_FCZ[curcell];
      totalHT_FCZ[curcell] = totalHT_FCZ[curcell] + integHTS_CC[botcell];
      totalHT_FCZ[curcell] = totalHT_FCZ[curcell] + integHTS_CC[curcell];

      if (htfluxX[curcell] >= 1.0e-12) {
        totHtFluxX[curcell] = totHtFluxX[curcell] + htfluxX[curcell] + htfluxCC[curcell] + htfluxCC[westcell];
      }
      if (htfluxY[curcell] >= 1.0e-12) {
        totHtFluxY[curcell] = totHtFluxY[curcell] + htfluxY[curcell] + htfluxCC[curcell] + htfluxCC[southcell];
      }
      if (htfluxZ[curcell] >= 1.0e-12) {
        totHtFluxZ[curcell] = totHtFluxZ[curcell] + htfluxZ[curcell] + htfluxCC[curcell] + htfluxCC[botcell];
      }
    }
  }
}

//______________________________________________________________________
//

void MPMArches::scheduleMomExchange(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls,
    const MaterialSet* mpm_matls,
    const MaterialSet* all_matls)

// first step: su_drag and sp_drag for arches are calculated
// at face centers and cell centers, using face-centered
// solid velocities and cell-centered solid velocities, along
// with cell-centered gas velocities.  In this step, 
// pressure forces at face centers are also calculated.

{ 
  // primitive variable initialization
  Task* t=scinew Task("MPMArches::doMomExchange",
      this, &MPMArches::doMomExchange);
  int numGhostCells = 1;

  // requires from Arches: celltype, pressure, velocity at cc.
  // also, from mpmarches, void fraction
  // use old_dw since using at the beginning of the time advance loop

  t->computes(d_Alab->d_cellInfoLabel, arches_matls->getUnion());

  // use modified celltype

  t->requires(Task::NewDW, d_Alab->d_mmcellTypeLabel, 
      arches_matls->getUnion(), 
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::OldDW, d_Alab->d_pressPlusHydroLabel,  
      arches_matls->getUnion(), 
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::OldDW, d_Alab->d_CCVelocityLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::OldDW, d_Alab->d_densityCPLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::OldDW, d_Alab->d_densityMicroLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW,  d_Alab->d_mmgasVolFracLabel,   
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  // computes, for arches, su_drag[x,y,z], sp_drag[x,y,z] at the
  // face centers and cell centers
  // Also computes stability factors for u,v, and w equations
  // due to fluid drag

  t->computes(d_MAlb->d_uVel_mmLinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmLinSrc_FCYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmLinSrc_FCZLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_uVel_mmNonlinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_vVel_mmLinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmLinSrc_FCZLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmLinSrc_FCXLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_vVel_mmNonlinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_wVel_mmLinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmLinSrc_FCXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmLinSrc_FCYLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_wVel_mmNonlinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion());

  t->computes(d_MAlb->KStabilityULabel, arches_matls->getUnion());
  t->computes(d_MAlb->KStabilityVLabel, arches_matls->getUnion());
  t->computes(d_MAlb->KStabilityWLabel, arches_matls->getUnion());

  // requires, from mpm, solid velocities at cc, fcx, fcy, and fcz

  t->requires(Task::NewDW, d_MAlb->solid_fraction_CCLabel, 
      mpm_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->xvel_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->xvel_FCXLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_FCYLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_FCZLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->xvel_FCYLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->xvel_FCZLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->yvel_FCZLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_FCXLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->zvel_FCXLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_FCYLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  // computes, for mpm, pressure forces and drag forces
  // at all face centers

  t->computes(d_MAlb->DragForceX_CCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceY_CCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceZ_CCLabel, mpm_matls->getUnion());

  t->computes(d_MAlb->DragForceX_FCYLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceX_FCZLabel, mpm_matls->getUnion());

  t->computes(d_MAlb->DragForceY_FCZLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceY_FCXLabel, mpm_matls->getUnion());

  t->computes(d_MAlb->DragForceZ_FCXLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceZ_FCYLabel, mpm_matls->getUnion());

  t->computes(d_MAlb->PressureForce_FCXLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->PressureForce_FCYLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->PressureForce_FCZLabel, mpm_matls->getUnion());


  sched->addTask(t, patches, all_matls);


  // second step: interpolate/collect sources from previous step to 
  // cell-centered sources

  // primitive variable initialization
  t=scinew Task("MPMArches::collectToCCGasMomExchSrcs",
      this, &MPMArches::collectToCCGasMomExchSrcs);

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_CCLabel, arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_CCLabel, arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_CCLabel, arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_CCLabel, arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_CCLabel, arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_CCLabel, arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_FCYLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_FCZLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_FCZLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_FCXLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_FCXLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_FCYLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion(),
      Ghost::AroundFaces, numGhostCells);

  // computes 

  t->computes(d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel, arches_matls->getUnion());

  sched->addTask(t, patches, arches_matls);


  // third step: interpolates sources from previous step to 
  // SFC(X,Y,Z) source arrays that arches actually uses in the 
  // momentum equations

  // primitive variable initialization

  t=scinew Task("MPMArches::interpolateCCToFCGasMomExchSrcs",
      this, &MPMArches::interpolateCCToFCGasMomExchSrcs);
  // requires

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel,
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  // computes 

  t->computes(d_MAlb->d_uVel_mmLinSrcLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrcLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_vVel_mmLinSrcLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrcLabel, arches_matls->getUnion());

  t->computes(d_MAlb->d_wVel_mmLinSrcLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrcLabel, arches_matls->getUnion());

  sched->addTask(t, patches, arches_matls);

#if 0

  // fourth step: redistributes the drag force calculated at the
  // cell-center (due to partially filled cells) to face centers
  // to supply to mpm

  // primitive variable initialization
  t=scinew Task("MPMArches::redistributeDragForceFromCCtoFC",
      this, &MPMArches::redistributeDragForceFromCCtoFC);
  numGhostCells = 1;

  // redistributes the drag forces calculated at cell center to 
  // staggered face centers in the direction of flow
  t->requires(Task::NewDW, d_MAlb->DragForceX_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->DragForceY_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->DragForceZ_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  //computes 

  t->computes(d_MAlb->DragForceX_FCXLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceY_FCYLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceZ_FCZLabel, mpm_matls->getUnion());
  sched->addTask(t, patches, mpm_matls);

#endif

}

//______________________________________________________________________
//

void MPMArches::doMomExchange(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numMPMMatls  = d_sharedState->getNumMPMMatls();

    // MPM stuff

    StaticArray<constCCVariable<double> > solid_fraction_cc(numMPMMatls);

    StaticArray<constCCVariable<double> > xvelCC_solid(numMPMMatls);
    StaticArray<constCCVariable<double> > yvelCC_solid(numMPMMatls);
    StaticArray<constCCVariable<double> > zvelCC_solid(numMPMMatls);

    StaticArray<constSFCXVariable<double> > xvelFCX_solid(numMPMMatls);
    StaticArray<constSFCXVariable<double> > yvelFCX_solid(numMPMMatls);
    StaticArray<constSFCXVariable<double> > zvelFCX_solid(numMPMMatls);

    StaticArray<constSFCYVariable<double> > xvelFCY_solid(numMPMMatls);
    StaticArray<constSFCYVariable<double> > yvelFCY_solid(numMPMMatls);
    StaticArray<constSFCYVariable<double> > zvelFCY_solid(numMPMMatls);

    StaticArray<constSFCZVariable<double> > xvelFCZ_solid(numMPMMatls);
    StaticArray<constSFCZVariable<double> > yvelFCZ_solid(numMPMMatls);
    StaticArray<constSFCZVariable<double> > zvelFCZ_solid(numMPMMatls);

    StaticArray<CCVariable<double> >   dragForceX_cc(numMPMMatls);
    StaticArray<CCVariable<double> >   dragForceY_cc(numMPMMatls);
    StaticArray<CCVariable<double> >   dragForceZ_cc(numMPMMatls);

    StaticArray<SFCYVariable<double> > dragForceX_fcy(numMPMMatls);
    StaticArray<SFCZVariable<double> > dragForceX_fcz(numMPMMatls);

    StaticArray<SFCZVariable<double> > dragForceY_fcz(numMPMMatls);
    StaticArray<SFCXVariable<double> > dragForceY_fcx(numMPMMatls);

    StaticArray<SFCXVariable<double> > dragForceZ_fcx(numMPMMatls);
    StaticArray<SFCYVariable<double> > dragForceZ_fcy(numMPMMatls);

    StaticArray<SFCXVariable<double> > pressForceX(numMPMMatls);
    StaticArray<SFCYVariable<double> > pressForceY(numMPMMatls);
    StaticArray<SFCZVariable<double> > pressForceZ(numMPMMatls);

    // Arches stuff

    constCCVariable<Vector> oldNormal;
    CCVariable<Vector> normal;

    constCCVariable<int> cellType;
    constCCVariable<double> pressure;

    constCCVariable<double> gas_fraction_cc;

    constCCVariable<double> density;
    constCCVariable<double> denMicro;

    // multimaterial contribution to SP and SU terms 
    // in Arches momentum eqns currently at cc and fcs.  
    // Later we will interpolate to where Arches wants 
    // them.  Also, stability factors for u, v, and w
    // due to fluid drag.

    CCVariable<double> uVelLinearSrc_cc; 
    SFCYVariable<double> uVelLinearSrc_fcy; 
    SFCZVariable<double> uVelLinearSrc_fcz; 

    CCVariable<double> uVelNonlinearSrc_cc;
    SFCYVariable<double> uVelNonlinearSrc_fcy;
    SFCZVariable<double> uVelNonlinearSrc_fcz;

    CCVariable<double> vVelLinearSrc_cc; 
    SFCZVariable<double> vVelLinearSrc_fcz; 
    SFCXVariable<double> vVelLinearSrc_fcx; 

    CCVariable<double> vVelNonlinearSrc_cc;
    SFCZVariable<double> vVelNonlinearSrc_fcz;
    SFCXVariable<double> vVelNonlinearSrc_fcx;

    CCVariable<double> wVelLinearSrc_cc; 
    SFCXVariable<double> wVelLinearSrc_fcx; 
    SFCYVariable<double> wVelLinearSrc_fcy; 

    CCVariable<double> wVelNonlinearSrc_cc; 
    SFCXVariable<double> wVelNonlinearSrc_fcx; 
    SFCYVariable<double> wVelNonlinearSrc_fcy; 

    CCVariable<double> KStabilityU; 
    CCVariable<double> KStabilityV; 
    CCVariable<double> KStabilityW; 

    constCCVariable<Vector> CCVelocity; 
    CCVariable<double> xvelCC_gas; 
    CCVariable<double> yvelCC_gas; 
    CCVariable<double> zvelCC_gas; 

    int numGhostCells = 1;
    int numGhostCellsG = 1;

    new_dw->get(cellType, d_Alab->d_mmcellTypeLabel,          matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);
    //    old_dw->get(pressure, d_Alab->d_pressureSPBCLabel,        matlIndex, 
    //		patch, Ghost::AroundCells, numGhostCellsG);
    old_dw->get(pressure, d_Alab->d_pressPlusHydroLabel,        matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);

    old_dw->get(CCVelocity, d_Alab->d_CCVelocityLabel,   matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);

    new_dw->allocateTemporary( xvelCC_gas, patch, Ghost::AroundCells, 1 ); 
    new_dw->allocateTemporary( yvelCC_gas, patch, Ghost::AroundCells, 1 ); 
    new_dw->allocateTemporary( zvelCC_gas, patch, Ghost::AroundCells, 1 ); 

    //copy over components of velocity.  Not very efficient but would 
    //require a rewrite of the momexch from fortran to c++
    for ( CellIterator iter = patch->getCellIterator( numGhostCellsG ); !iter.done(); ++iter ){ 

      IntVector c = *iter; 

      xvelCC_gas[c] = CCVelocity[c].x(); 
      yvelCC_gas[c] = CCVelocity[c].y(); 
      zvelCC_gas[c] = CCVelocity[c].z(); 

    } 

    new_dw->get(gas_fraction_cc, d_Alab->d_mmgasVolFracLabel, matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);

    old_dw->get(density, d_Alab->d_densityCPLabel, matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);
    old_dw->get(denMicro, d_Alab->d_densityMicroLabel, matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);    

    // Get the PerPatch CellInformation data from oldDW, initialize it if it is
    // not there
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_Alab->d_cellInfoLabel, matlIndex, patch)) 
      throw InvalidValue("cellInformation should not be initialized yet",
          __FILE__, __LINE__);
    if (old_dw->exists(d_Alab->d_cellInfoLabel, matlIndex, patch)) {
      old_dw->get(cellInfoP, d_Alab->d_cellInfoLabel, matlIndex, patch); }
    else {
      cellInfoP.setData(scinew CellInformation(patch));
    }
    new_dw->put(cellInfoP, d_Alab->d_cellInfoLabel, matlIndex, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // computes su_drag[x,y,z], sp_drag[x,y,z] for arches at cell centers
    // and face centers; also, stability factors due to drag
    // for u-, v-, and w-momentum equations

    new_dw->allocateAndPut(uVelLinearSrc_cc, d_MAlb->d_uVel_mmLinSrc_CCLabel, 
        matlIndex, patch);
    new_dw->allocateAndPut(uVelLinearSrc_fcy, d_MAlb->d_uVel_mmLinSrc_FCYLabel, 
        matlIndex, patch);
    new_dw->allocateAndPut(uVelLinearSrc_fcz, d_MAlb->d_uVel_mmLinSrc_FCZLabel, 
        matlIndex, patch);

    new_dw->allocateAndPut(uVelNonlinearSrc_cc, d_MAlb->d_uVel_mmNonlinSrc_CCLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(uVelNonlinearSrc_fcy, d_MAlb->d_uVel_mmNonlinSrc_FCYLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(uVelNonlinearSrc_fcz, d_MAlb->d_uVel_mmNonlinSrc_FCZLabel,
        matlIndex, patch);

    new_dw->allocateAndPut(vVelLinearSrc_cc, d_MAlb->d_vVel_mmLinSrc_CCLabel, 
        matlIndex, patch);
    new_dw->allocateAndPut(vVelLinearSrc_fcz, d_MAlb->d_vVel_mmLinSrc_FCZLabel, 
        matlIndex, patch);
    new_dw->allocateAndPut(vVelLinearSrc_fcx, d_MAlb->d_vVel_mmLinSrc_FCXLabel, 
        matlIndex, patch);

    new_dw->allocateAndPut(vVelNonlinearSrc_cc, d_MAlb->d_vVel_mmNonlinSrc_CCLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(vVelNonlinearSrc_fcz, d_MAlb->d_vVel_mmNonlinSrc_FCZLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(vVelNonlinearSrc_fcx, d_MAlb->d_vVel_mmNonlinSrc_FCXLabel,
        matlIndex, patch);

    new_dw->allocateAndPut(wVelLinearSrc_cc, d_MAlb->d_wVel_mmLinSrc_CCLabel, 
        matlIndex, patch);
    new_dw->allocateAndPut(wVelLinearSrc_fcx, d_MAlb->d_wVel_mmLinSrc_FCXLabel, 
        matlIndex, patch);
    new_dw->allocateAndPut(wVelLinearSrc_fcy, d_MAlb->d_wVel_mmLinSrc_FCYLabel, 
        matlIndex, patch);

    new_dw->allocateAndPut(wVelNonlinearSrc_cc, d_MAlb->d_wVel_mmNonlinSrc_CCLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(wVelNonlinearSrc_fcx, d_MAlb->d_wVel_mmNonlinSrc_FCXLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(wVelNonlinearSrc_fcy, d_MAlb->d_wVel_mmNonlinSrc_FCYLabel,
        matlIndex, patch);

    new_dw->allocateAndPut(KStabilityU, d_MAlb->KStabilityULabel,
        matlIndex, patch);
    KStabilityU.initialize(0.);
    new_dw->allocateAndPut(KStabilityV, d_MAlb->KStabilityVLabel,
        matlIndex, patch);
    KStabilityV.initialize(0.);
    new_dw->allocateAndPut(KStabilityW, d_MAlb->KStabilityWLabel,
        matlIndex, patch);
    KStabilityW.initialize(0.);

    uVelLinearSrc_cc.initialize(0.);
    uVelLinearSrc_fcy.initialize(0.);
    uVelLinearSrc_fcz.initialize(0.);

    uVelNonlinearSrc_cc.initialize(0.);
    uVelNonlinearSrc_fcy.initialize(0.);
    uVelNonlinearSrc_fcz.initialize(0.);

    vVelLinearSrc_cc.initialize(0.);
    vVelLinearSrc_fcz.initialize(0.);
    vVelLinearSrc_fcx.initialize(0.);

    vVelNonlinearSrc_cc.initialize(0.);
    vVelNonlinearSrc_fcz.initialize(0.);
    vVelNonlinearSrc_fcx.initialize(0.);

    wVelLinearSrc_cc.initialize(0.);
    wVelLinearSrc_fcx.initialize(0.);
    wVelLinearSrc_fcy.initialize(0.);

    wVelNonlinearSrc_cc.initialize(0.);
    wVelNonlinearSrc_fcx.initialize(0.);
    wVelNonlinearSrc_fcy.initialize(0.);

    for (int m = 0; m < numMPMMatls; m++) {

      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();

      new_dw->get(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
          idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(xvelCC_solid[m], d_MAlb->xvel_CCLabel, 
          idx, patch, Ghost::AroundCells, numGhostCells);
      new_dw->get(yvelCC_solid[m], d_MAlb->yvel_CCLabel, 
          idx, patch, Ghost::AroundCells, numGhostCells);
      new_dw->get(zvelCC_solid[m], d_MAlb->zvel_CCLabel, 
          idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(xvelFCX_solid[m], d_MAlb->xvel_FCXLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->get(yvelFCY_solid[m], d_MAlb->yvel_FCYLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->get(zvelFCZ_solid[m], d_MAlb->zvel_FCZLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);      

      new_dw->get(xvelFCY_solid[m], d_MAlb->xvel_FCYLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->get(xvelFCZ_solid[m], d_MAlb->xvel_FCZLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);

      new_dw->get(yvelFCZ_solid[m], d_MAlb->yvel_FCZLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->get(yvelFCX_solid[m], d_MAlb->yvel_FCXLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);

      new_dw->get(zvelFCX_solid[m], d_MAlb->zvel_FCXLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);
      new_dw->get(zvelFCY_solid[m], d_MAlb->zvel_FCYLabel, 
          idx, patch, Ghost::AroundFaces, numGhostCells);

      new_dw->allocateAndPut(dragForceX_cc[m], d_MAlb->DragForceX_CCLabel,
          idx, patch);
      dragForceX_cc[m].initialize(0.);

      new_dw->allocateAndPut(dragForceY_cc[m], d_MAlb->DragForceY_CCLabel,
          idx, patch);
      dragForceY_cc[m].initialize(0.);

      new_dw->allocateAndPut(dragForceZ_cc[m], d_MAlb->DragForceZ_CCLabel,
          idx, patch);
      dragForceZ_cc[m].initialize(0.);

      new_dw->allocateAndPut(dragForceX_fcy[m], d_MAlb->DragForceX_FCYLabel, 
          idx, patch);
      dragForceX_fcy[m].initialize(0.);

      new_dw->allocateAndPut(dragForceX_fcz[m], d_MAlb->DragForceX_FCZLabel, 
          idx, patch);
      dragForceX_fcz[m].initialize(0.);

      new_dw->allocateAndPut(dragForceY_fcz[m], d_MAlb->DragForceY_FCZLabel, 
          idx, patch);
      dragForceY_fcz[m].initialize(0.);

      new_dw->allocateAndPut(dragForceY_fcx[m], d_MAlb->DragForceY_FCXLabel, 
          idx, patch);
      dragForceY_fcx[m].initialize(0.);

      new_dw->allocateAndPut(dragForceZ_fcx[m], d_MAlb->DragForceZ_FCXLabel, 
          idx, patch);
      dragForceZ_fcx[m].initialize(0.);

      new_dw->allocateAndPut(dragForceZ_fcy[m], d_MAlb->DragForceZ_FCYLabel, 
          idx, patch);
      dragForceZ_fcy[m].initialize(0.);

      new_dw->allocateAndPut(pressForceX[m], d_MAlb->PressureForce_FCXLabel,
          idx, patch);
      pressForceX[m].initialize(0.);

      new_dw->allocateAndPut(pressForceY[m], d_MAlb->PressureForce_FCYLabel,
          idx, patch);
      pressForceY[m].initialize(0.);

      new_dw->allocateAndPut(pressForceZ[m], d_MAlb->PressureForce_FCZLabel,
          idx, patch);
      pressForceZ[m].initialize(0.);

    }

    // Begin loop to calculate gas-solid exchange terms for each
    // solid material with the gas phase

    int ffieldid = d_arches->getBoundaryCondition()->flowCellType();
    int mmwallid = d_arches->getBoundaryCondition()->getMMWallId();

    double viscos = d_arches->getTurbulenceModel()->getMolecularViscosity();
    //    double csmag = d_arches->getTurbulenceModel()->getSmagorinskyConst();
    double csmag = 0.17;

    IntVector valid_lo;
    IntVector valid_hi;

    for (int m = 0; m < numMPMMatls; m++) {

      valid_lo = patch->getFortranCellLowIndex();
      valid_hi = patch->getFortranCellHighIndex();

      // code for x-direction momentum exchange

      int ioff = 1;
      int joff = 0;
      int koff = 0;

      int indexflo = 1;
      int indext1 =  2;
      int indext2 =  3;

      fort_momentum_exchange_cont_cc(uVelNonlinearSrc_fcy,
          uVelLinearSrc_fcy,
          uVelNonlinearSrc_fcz,
          uVelLinearSrc_fcz,
          uVelNonlinearSrc_cc,
          uVelLinearSrc_cc,
          KStabilityU,
          dragForceX_fcy[m],
          dragForceX_fcz[m],
          dragForceX_cc[m],
          xvelCC_gas,
          xvelCC_solid[m],
          xvelFCY_solid[m],
          xvelFCZ_solid[m],
          gas_fraction_cc,
          density,
          denMicro,
          solid_fraction_cc[m],
          viscos, csmag,
          cellinfo->sew, cellinfo->sns, cellinfo->stb, 
          cellinfo->yy, cellinfo->zz, 
          cellinfo->yv, cellinfo->zw,
          valid_lo, valid_hi,
          ioff, joff, koff,
          indexflo, indext1, indext2,
          cellType, mmwallid, ffieldid);

      // code for y-direction momentum exchange

      ioff = 0;
      joff = 1;
      koff = 0;

      indexflo = 2;
      indext1 =  3;
      indext2 =  1;

      fort_momentum_exchange_cont_cc(vVelNonlinearSrc_fcz,
          vVelLinearSrc_fcz,
          vVelNonlinearSrc_fcx,
          vVelLinearSrc_fcx,
          vVelNonlinearSrc_cc,
          vVelLinearSrc_cc,
          KStabilityV,
          dragForceY_fcz[m],
          dragForceY_fcx[m],
          dragForceY_cc[m],
          yvelCC_gas,
          yvelCC_solid[m],
          yvelFCZ_solid[m],
          yvelFCX_solid[m],
          gas_fraction_cc,
          density,
          denMicro,
          solid_fraction_cc[m],
          viscos, csmag,
          cellinfo->sns, cellinfo->stb, cellinfo->sew, 
          cellinfo->zz, cellinfo->xx, 
          cellinfo->zw, cellinfo->xu,
          valid_lo, valid_hi,
          ioff, joff, koff,
          indexflo, indext1, indext2,
          cellType, mmwallid, ffieldid);

      // code for z-direction momentum exchange

      ioff = 0;
      joff = 0;
      koff = 1;

      indexflo = 3;
      indext1 =  1;
      indext2 =  2;

      fort_momentum_exchange_cont_cc(wVelNonlinearSrc_fcx,
          wVelLinearSrc_fcx,
          wVelNonlinearSrc_fcy,
          wVelLinearSrc_fcy,
          wVelNonlinearSrc_cc,
          wVelLinearSrc_cc,
          KStabilityW,
          dragForceZ_fcx[m],
          dragForceZ_fcy[m],
          dragForceZ_cc[m],
          zvelCC_gas,
          zvelCC_solid[m],
          zvelFCX_solid[m],
          zvelFCY_solid[m],
          gas_fraction_cc,
          density,
          denMicro,
          solid_fraction_cc[m],
          viscos, csmag,
          cellinfo->stb, cellinfo->sew, cellinfo->sns, 
          cellinfo->xx, cellinfo->yy, 
          cellinfo->xu, cellinfo->yv,
          valid_lo, valid_hi,
          ioff, joff, koff,
          indexflo, indext1, indext2,
          cellType, mmwallid, ffieldid);

      // code for pressure forces (direction-independent)

      valid_lo = patch->getFortranCellLowIndex();
      valid_hi = patch->getFortranCellHighIndex();

      fort_pressure_force(pressForceX[m], pressForceY[m], pressForceZ[m],
          gas_fraction_cc, solid_fraction_cc[m],
          pressure, cellinfo->sew, cellinfo->sns,
          cellinfo->stb, valid_lo, valid_hi, cellType,
          mmwallid, ffieldid);

      // debug for testing inviscid option
      // September 18, 2003, SK

      if (d_inviscid) {

        uVelLinearSrc_cc.initialize(0.);
        uVelLinearSrc_fcy.initialize(0.);
        uVelLinearSrc_fcz.initialize(0.);
        uVelNonlinearSrc_cc.initialize(0.);
        uVelNonlinearSrc_fcy.initialize(0.);
        uVelNonlinearSrc_fcz.initialize(0.);
        vVelLinearSrc_cc.initialize(0.);
        vVelLinearSrc_fcz.initialize(0.);
        vVelLinearSrc_fcx.initialize(0.);
        vVelNonlinearSrc_cc.initialize(0.);
        vVelNonlinearSrc_fcz.initialize(0.);
        vVelNonlinearSrc_fcx.initialize(0.);
        wVelLinearSrc_cc.initialize(0.);
        wVelLinearSrc_fcx.initialize(0.);
        wVelLinearSrc_fcy.initialize(0.);
        wVelNonlinearSrc_cc.initialize(0.);
        wVelNonlinearSrc_fcx.initialize(0.);
        wVelNonlinearSrc_fcy.initialize(0.);

      }
    }
  }
}

//______________________________________________________________________
//

void MPMArches::collectToCCGasMomExchSrcs(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)

{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<double> su_dragx_cc;
    constSFCYVariable<double> su_dragx_fcy;
    constSFCZVariable<double> su_dragx_fcz;

    CCVariable<double> sp_dragx_cc;
    constSFCYVariable<double> sp_dragx_fcy;
    constSFCZVariable<double> sp_dragx_fcz;

    CCVariable<double> su_dragy_cc;
    constSFCZVariable<double> su_dragy_fcz;
    constSFCXVariable<double> su_dragy_fcx;

    CCVariable<double> sp_dragy_cc;
    constSFCZVariable<double> sp_dragy_fcz;
    constSFCXVariable<double> sp_dragy_fcx;

    CCVariable<double> su_dragz_cc;
    constSFCXVariable<double> su_dragz_fcx;
    constSFCYVariable<double> su_dragz_fcy;

    CCVariable<double> sp_dragz_cc;
    constSFCXVariable<double> sp_dragz_fcx;
    constSFCYVariable<double> sp_dragz_fcy;

    int numGhostCells = 1;

    new_dw->allocateAndPut(sp_dragx_cc, d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(sp_dragx_cc, d_MAlb->d_uVel_mmLinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->allocateAndPut(sp_dragy_cc, d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(sp_dragy_cc, d_MAlb->d_vVel_mmLinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->allocateAndPut(sp_dragz_cc, d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(sp_dragz_cc, d_MAlb->d_wVel_mmLinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);

    new_dw->allocateAndPut(su_dragx_cc, d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(su_dragx_cc, d_MAlb->d_uVel_mmNonlinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->allocateAndPut(su_dragy_cc, d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(su_dragy_cc, d_MAlb->d_vVel_mmNonlinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->allocateAndPut(su_dragz_cc, d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(su_dragz_cc, d_MAlb->d_wVel_mmNonlinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);

    new_dw->get(su_dragx_fcy, d_MAlb->d_uVel_mmNonlinSrc_FCYLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(su_dragx_fcz, d_MAlb->d_uVel_mmNonlinSrc_FCZLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->get(sp_dragx_fcy, d_MAlb->d_uVel_mmLinSrc_FCYLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(sp_dragx_fcz, d_MAlb->d_uVel_mmLinSrc_FCZLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->get(su_dragy_fcz, d_MAlb->d_vVel_mmNonlinSrc_FCZLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(su_dragy_fcx, d_MAlb->d_vVel_mmNonlinSrc_FCXLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->get(sp_dragy_fcz, d_MAlb->d_vVel_mmLinSrc_FCZLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(sp_dragy_fcx, d_MAlb->d_vVel_mmLinSrc_FCXLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->get(su_dragz_fcx, d_MAlb->d_wVel_mmNonlinSrc_FCXLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(su_dragz_fcy, d_MAlb->d_wVel_mmNonlinSrc_FCYLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->get(sp_dragz_fcx, d_MAlb->d_wVel_mmLinSrc_FCXLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(sp_dragz_fcy, d_MAlb->d_wVel_mmLinSrc_FCYLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    IntVector valid_lo;
    IntVector valid_hi;

    int ioff;
    int joff;
    int koff;

    // collect x-direction sources from face centers to cell center

    valid_lo = patch->getSFCXFORTLowIndex__Old();
    valid_hi = patch->getSFCXFORTHighIndex__Old();

    ioff = 1;
    joff = 0;
    koff = 0;

    // for first transverse direction, i.e., y

    fort_collect_drag_cc(su_dragx_cc, sp_dragx_cc,
        su_dragx_fcy, sp_dragx_fcy,
        koff, ioff, joff,
        valid_lo, valid_hi);

    // for second transverse direction, i.e., z

    fort_collect_drag_cc(su_dragx_cc, sp_dragx_cc,
        su_dragx_fcz, sp_dragx_fcz,
        joff, joff, ioff,
        valid_lo, valid_hi);

    // collect y-direction sources from face centers to cell center

    valid_lo = patch->getSFCYFORTLowIndex__Old();
    valid_hi = patch->getSFCYFORTHighIndex__Old();

    ioff = 0;
    joff = 1;
    koff = 0;

    // for first transverse direction, i.e., z

    fort_collect_drag_cc(su_dragy_cc, sp_dragy_cc,
        su_dragy_fcz, sp_dragy_fcz,
        koff, ioff, joff,
        valid_lo, valid_hi);


    // for second transverse direction, i.e., x

    fort_collect_drag_cc(su_dragy_cc, sp_dragy_cc,
        su_dragy_fcx, sp_dragy_fcx,
        joff, koff, ioff,
        valid_lo, valid_hi);

    // collect z-direction sources from face centers to cell center

    valid_lo = patch->getSFCZFORTLowIndex__Old();
    valid_hi = patch->getSFCZFORTHighIndex__Old();

    ioff = 0;
    joff = 0;
    koff = 1;

    // for first transverse direction, i.e., x

    fort_collect_drag_cc(su_dragz_cc, sp_dragz_cc,
        su_dragz_fcx, sp_dragz_fcx,
        koff, ioff, joff,
        valid_lo, valid_hi);


    // for second transverse direction, i.e., y

    fort_collect_drag_cc(su_dragz_cc, sp_dragz_cc,
        su_dragz_fcy, sp_dragz_fcy,
        joff, koff, ioff,
        valid_lo, valid_hi);
  }
}

//______________________________________________________________________
//

void MPMArches::interpolateCCToFCGasMomExchSrcs(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* ,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)

// This function interpolates the source terms that are calculated 
// and collected at the cell center to the staggered face centers
// for each momentum equation of the gas phase.  At the end of this
// function execution, the gas phase has all the momentum exchange
// source terms it needs for its calculations.

{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> su_dragx_cc;
    constCCVariable<double> sp_dragx_cc;
    constCCVariable<double> su_dragy_cc;
    constCCVariable<double> sp_dragy_cc;
    constCCVariable<double> su_dragz_cc;
    constCCVariable<double> sp_dragz_cc;

    SFCXVariable<double> su_dragx_fcx;
    SFCXVariable<double> sp_dragx_fcx;
    SFCYVariable<double> su_dragy_fcy;
    SFCYVariable<double> sp_dragy_fcy;
    SFCZVariable<double> su_dragz_fcz;
    SFCZVariable<double> sp_dragz_fcz;

    int numGhostCells = 1;

    // gets CC variables

    new_dw->get(sp_dragx_cc, d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragy_cc, d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragz_cc, d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);

    new_dw->get(su_dragx_cc, d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);    
    new_dw->get(su_dragy_cc, d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);        
    new_dw->get(su_dragz_cc, d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);    

    // computes FC interpolants

    new_dw->allocateAndPut(su_dragx_fcx, d_MAlb->d_uVel_mmNonlinSrcLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(sp_dragx_fcx, d_MAlb->d_uVel_mmLinSrcLabel,
        matlIndex, patch);

    new_dw->allocateAndPut(su_dragy_fcy, d_MAlb->d_vVel_mmNonlinSrcLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(sp_dragy_fcy, d_MAlb->d_vVel_mmLinSrcLabel,
        matlIndex, patch);

    new_dw->allocateAndPut(su_dragz_fcz, d_MAlb->d_wVel_mmNonlinSrcLabel,
        matlIndex, patch);
    new_dw->allocateAndPut(sp_dragz_fcz, d_MAlb->d_wVel_mmLinSrcLabel,
        matlIndex, patch);

    // initialize fc interpolants so that values in non-mpm areas
    // are zero

    su_dragx_fcx.initialize(0);
    sp_dragx_fcx.initialize(0);
    su_dragy_fcy.initialize(0);
    sp_dragy_fcy.initialize(0);
    su_dragz_fcz.initialize(0);
    sp_dragz_fcz.initialize(0);

    IntVector valid_lo;
    IntVector valid_hi;

    int ioff;
    int joff;
    int koff;

    // Interpolate x-momentum source terms

    ioff = 1;
    joff = 0;
    koff = 0;

    valid_lo = patch->getSFCXFORTLowIndex__Old();
    valid_hi = patch->getSFCXFORTHighIndex__Old();

    // nonlinear source

    fort_interp_centertoface(su_dragx_fcx,
        su_dragx_cc, 
        ioff, joff, koff,
        valid_lo, valid_hi);

    // linear source

    fort_interp_centertoface(sp_dragx_fcx,
        sp_dragx_cc, 
        ioff, joff, koff,
        valid_lo, valid_hi);

    // Interpolate y-momentum source terms

    ioff = 0;
    joff = 1;
    koff = 0;

    valid_lo = patch->getSFCYFORTLowIndex__Old();
    valid_hi = patch->getSFCYFORTHighIndex__Old();

    // nonlinear source

    fort_interp_centertoface(su_dragy_fcy,
        su_dragy_cc, 
        ioff, joff, koff,
        valid_lo, valid_hi);

    // linear source

    fort_interp_centertoface(sp_dragy_fcy,
        sp_dragy_cc, 
        ioff, joff, koff,
        valid_lo, valid_hi);

    // Interpolate z-momentum source terms

    ioff = 0;
    joff = 0;
    koff = 1;

    valid_lo = patch->getSFCZFORTLowIndex__Old();
    valid_hi = patch->getSFCZFORTHighIndex__Old();

    // nonlinear source

    fort_interp_centertoface(su_dragz_fcz,
        su_dragz_cc, 
        ioff, joff, koff,
        valid_lo, valid_hi);

    // linear source

    fort_interp_centertoface(sp_dragz_fcz,
        sp_dragz_cc, 
        ioff, joff, koff,
        valid_lo, valid_hi);
  }
}

//______________________________________________________________________
//

#if 0
void MPMArches::redistributeDragForceFromCCtoFC(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* ,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)

//
// redistributes the drag forces experienced by the solid materials,
// which are calculated at cell centers for partially filled 
// cells in the previous step, to face centers
//

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int numMPMMatls  = d_sharedState->getNumMPMMatls();
    // MPM stuff

    CCVariable<double> dragForceX_cc;
    CCVariable<double> dragForceY_cc;
    CCVariable<double> dragForceZ_cc;

    SFCXVariable<double> dragForceX_fcx;
    SFCYVariable<double> dragForceY_fcy;
    SFCZVariable<double> dragForceZ_fcz;

    int numGhostCells = 1;

    for (int m = 0; m < numMPMMatls; m++) {

      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();

      new_dw->get(dragForceX_cc, d_MAlb->DragForceX_CCLabel,
          idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(dragForceY_cc, d_MAlb->DragForceY_CCLabel,
          idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(dragForceZ_cc, d_MAlb->DragForceZ_CCLabel,
          idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->allocate(dragForceX_fcx, d_MAlb->DragForceX_FCXLabel,
          idx, patch);

      new_dw->allocate(dragForceY_fcy, d_MAlb->DragForceY_FCYLabel,
          idx, patch);

      new_dw->allocate(dragForceZ_fcz, d_MAlb->DragForceZ_FCZLabel,
          idx, patch);

      IntVector valid_lo;
      IntVector valid_hi;

      // redistribute x-direction drag forces

      valid_lo = patch->getSFCXFORTLowIndex__Old();
      valid_hi = patch->getSFCXFORTHighIndex__Old();

      int ioff = 1;
      int joff = 0;
      int koff = 0;

      fort_mm_redistribute_drag(dragForceX_fcx, 
          dragForceX_cc,
          ioff, joff, koff, 
          valid_lo, valid_hi);

      // redistribute y-direction drag forces

      valid_lo = patch->getSFCYFORTLowIndex__Old();
      valid_hi = patch->getSFCYFORTHighIndex__Old();

      ioff = 0;
      joff = 1;
      koff = 0;

      fort_mm_redistribute_drag(dragForceY_fcx, 
          dragForceY_cc,
          ioff, joff, koff, 
          valid_lo, valid_hi);

      // redistribute z-direction drag forces

      valid_lo = patch->getSFCZFORTLowIndex__Old();
      valid_hi = patch->getSFCZFORTHighIndex__Old();

      ioff = 0;
      joff = 0;
      koff = 1;

      fort_mm_redistribute_drag(dragForceZ_fcx, 
          dragForceZ_cc,
          iof,f joff, koff, 
          valid_lo, valid_hi);

      // Calculation done; now put things back in DW

      new_dw->put(dragForceX_fcx, d_MAlb->DragForceX_FCXLabel, 
          idx, patch);
      new_dw->put(dragForceY_fcy, d_MAlb->DragForceY_FCYLabel, 
          idx, patch);
      new_dw->put(dragForceZ_fcz, d_MAlb->DragForceZ_FCZLabel, 
          idx, patch);

    }  
  }

}
#endif

//______________________________________________________________________
//

void MPMArches::scheduleEnergyExchange(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* arches_matls,
    const MaterialSet* mpm_matls,
    const MaterialSet* all_matls)

{ 

  // first step: calculate heat fluxes at cell centers and faces
  // and store them where they are calculated.  MPM is fine with
  // this; Arches requires a further processing step to put sources
  // at cell centers.

  Task* t=scinew Task("MPMArches::doEnergyExchange",
      this, &MPMArches::doEnergyExchange);

  int numGhostCells = 1;
  int zeroGhostCells = 0;

  // requires, from mpmarches, solid temperatures at cc, fcx, fcy, and 
  // fcz, solid fraction

  t->requires(Task::NewDW, d_MAlb->solid_fraction_CCLabel, 
      mpm_matls->getUnion(), Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->tempSolid_CCLabel, 
      mpm_matls->getUnion(), Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->tempSolid_FCXLabel, 
      mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->tempSolid_FCYLabel, 
      mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->tempSolid_FCZLabel, 
      mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->xvel_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_CCLabel, mpm_matls->getUnion(),
      Ghost::AroundCells, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->xvel_FCYLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->xvel_FCZLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->yvel_FCZLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_FCXLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->zvel_FCXLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_FCYLabel, mpm_matls->getUnion(),
      Ghost::AroundFaces, zeroGhostCells);

  // computes, for mpm, heat transferred to solid at cell centers 
  // and at all face centers

  t->computes(d_MAlb->heaTranSolid_tmp_CCLabel,  mpm_matls->getUnion());
  t->computes(d_MAlb->heaTranSolid_FCXLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->heaTranSolid_FCYLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->heaTranSolid_FCZLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->heaTranSolid_FCX_RadLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->heaTranSolid_FCY_RadLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->heaTranSolid_FCZ_RadLabel, mpm_matls->getUnion());

  // requires from Arches: celltype, gas temperature
  // also, from mpmarches, void fraction
  // use old_dw since using at the beginning of the time advance loop

  // use modified celltype

  t->requires(Task::NewDW, d_Alab->d_mmcellTypeLabel,      
      arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
  t->requires(Task::OldDW, d_Alab->d_tempINLabel,      
      arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW,  d_Alab->d_mmgasVolFracLabel,   
      arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);

  t->requires(Task::OldDW, d_Alab->d_CCVelocityLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::OldDW, d_Alab->d_densityMicroLabel, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  t->requires(Task::OldDW, d_enthalpy_label, 
      arches_matls->getUnion(),
      Ghost::AroundCells, numGhostCells);

  if (d_DORad) {
    // stuff for radiative heat flux to intrusions
    t->requires(Task::OldDW, d_Alab->d_radiationFluxEINLabel, 
        arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
    t->requires(Task::OldDW, d_Alab->d_radiationFluxWINLabel, 
        arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
    t->requires(Task::OldDW, d_Alab->d_radiationFluxNINLabel, 
        arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
    t->requires(Task::OldDW, d_Alab->d_radiationFluxSINLabel, 
        arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
    t->requires(Task::OldDW, d_Alab->d_radiationFluxTINLabel, 
        arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
    t->requires(Task::OldDW, d_Alab->d_radiationFluxBINLabel, 
        arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);  
  }

  // computes, for arches, su_enth and sp_enth at the
  // face centers and cell centers

  t->computes(d_MAlb->d_enth_mmLinSrc_tmp_CCLabel,  arches_matls->getUnion());
  t->computes(d_MAlb->d_enth_mmLinSrc_FCXLabel,     arches_matls->getUnion());
  t->computes(d_MAlb->d_enth_mmLinSrc_FCYLabel,     arches_matls->getUnion());
  t->computes(d_MAlb->d_enth_mmLinSrc_FCZLabel,     arches_matls->getUnion());

  t->computes(d_MAlb->d_enth_mmNonLinSrc_tmp_CCLabel,  arches_matls->getUnion());
  t->computes(d_MAlb->d_enth_mmNonLinSrc_FCXLabel,     arches_matls->getUnion());
  t->computes(d_MAlb->d_enth_mmNonLinSrc_FCYLabel,     arches_matls->getUnion());
  t->computes(d_MAlb->d_enth_mmNonLinSrc_FCZLabel,     arches_matls->getUnion());

  // computes Stability Factor for multimaterial heat exchange

  t->computes(d_MAlb->KStabilityHLabel, arches_matls->getUnion());

  // computes heat fluxes at face centers for all three 
  // directions and convective heat flux at cell centers

  t->computes(d_MAlb->htfluxConvXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->htfluxRadXLabel,  arches_matls->getUnion());
  t->computes(d_MAlb->htfluxXLabel,     arches_matls->getUnion());
  t->computes(d_MAlb->htfluxConvYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->htfluxRadYLabel,  arches_matls->getUnion());
  t->computes(d_MAlb->htfluxYLabel,     arches_matls->getUnion());
  t->computes(d_MAlb->htfluxConvZLabel, arches_matls->getUnion());
  t->computes(d_MAlb->htfluxRadZLabel,  arches_matls->getUnion());
  t->computes(d_MAlb->htfluxZLabel,     arches_matls->getUnion());
  t->computes(d_MAlb->htfluxConvCCLabel,arches_matls->getUnion());

  sched->addTask(t, patches, all_matls);

  // second step: interpolate/collect sources from face centers and
  // add them to cell-centered source calculated in the last step.
  // This is the source that the gas-phase equations use

  // primitive variable initialization

  t=scinew Task("MPMArches::collectToCCGasEnergyExchSrcs",
      this, &MPMArches::collectToCCGasEnergyExchSrcs);

  numGhostCells = 1;

  t->requires(Task::NewDW, d_MAlb->d_enth_mmLinSrc_tmp_CCLabel,
      arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_enth_mmLinSrc_FCXLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_enth_mmLinSrc_FCYLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_enth_mmLinSrc_FCZLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_enth_mmNonLinSrc_tmp_CCLabel,
      arches_matls->getUnion(), Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_enth_mmNonLinSrc_FCXLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_enth_mmNonLinSrc_FCYLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_enth_mmNonLinSrc_FCZLabel,
      arches_matls->getUnion(), Ghost::AroundFaces, numGhostCells);

  // computes 

  t->computes(d_MAlb->d_enth_mmLinSrc_CCLabel,    arches_matls->getUnion());
  t->computes(d_MAlb->d_enth_mmNonLinSrc_CCLabel, arches_matls->getUnion());

  sched->addTask(t, patches, arches_matls);

}

//______________________________________________________________________
//

void MPMArches::doEnergyExchange(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)

{

  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numMPMMatls  = d_sharedState->getNumMPMMatls();
    Vector Dx = patch->dCell(); 

    // MPM stuff

    StaticArray<constCCVariable<double> > solid_fraction_cc(numMPMMatls);
    StaticArray<constCCVariable<double> > tempSolid_cc(numMPMMatls);
    StaticArray<constSFCXVariable<double> > tempSolid_fcx(numMPMMatls);
    StaticArray<constSFCYVariable<double> > tempSolid_fcy(numMPMMatls);
    StaticArray<constSFCZVariable<double> > tempSolid_fcz(numMPMMatls);

    StaticArray<constCCVariable<double> > upCC(numMPMMatls);
    StaticArray<constCCVariable<double> > vpCC(numMPMMatls);
    StaticArray<constCCVariable<double> > wpCC(numMPMMatls);
    StaticArray<constSFCXVariable<double> > vpFCX(numMPMMatls);
    StaticArray<constSFCXVariable<double> > wpFCX(numMPMMatls);
    StaticArray<constSFCYVariable<double> > upFCY(numMPMMatls);
    StaticArray<constSFCYVariable<double> > wpFCY(numMPMMatls);
    StaticArray<constSFCZVariable<double> > upFCZ(numMPMMatls);
    StaticArray<constSFCZVariable<double> > vpFCZ(numMPMMatls);

    StaticArray<CCVariable<double> >   heaTranSolid_cc(numMPMMatls);
    StaticArray<SFCXVariable<double> > heaTranSolid_fcx(numMPMMatls);
    StaticArray<SFCYVariable<double> > heaTranSolid_fcy(numMPMMatls);
    StaticArray<SFCZVariable<double> > heaTranSolid_fcz(numMPMMatls);
    StaticArray<SFCXVariable<double> > heaTranSolid_fcx_rad(numMPMMatls);
    StaticArray<SFCYVariable<double> > heaTranSolid_fcy_rad(numMPMMatls);
    StaticArray<SFCZVariable<double> > heaTranSolid_fcz_rad(numMPMMatls);

    // Arches stuff

    constCCVariable<int> cellType;
    constCCVariable<double> tempGas;
    constCCVariable<double> gas_fraction_cc;
    constCCVariable<Vector> CCVelocity; 
    constCCVariable<double> denMicro;

    // stuff for radiative heat flux to intrusions

    CCVariable<double> radfluxE;
    CCVariable<double> radfluxW;
    CCVariable<double> radfluxN;
    CCVariable<double> radfluxS;
    CCVariable<double> radfluxT;
    CCVariable<double> radfluxB;

    // multimaterial contribution to SP and SU terms in Arches 
    // enthalpy eqn., stored where calculated

    CCVariable<double> su_enth_cc;
    SFCXVariable<double> su_enth_fcx;
    SFCYVariable<double> su_enth_fcy;
    SFCZVariable<double> su_enth_fcz;

    CCVariable<double> sp_enth_cc;
    SFCXVariable<double> sp_enth_fcx;
    SFCYVariable<double> sp_enth_fcy;
    SFCZVariable<double> sp_enth_fcz;

    // heat fluxes

    SFCXVariable<double> htfluxConvX;
    SFCXVariable<double> htfluxRadX;
    SFCXVariable<double> htfluxX;
    SFCYVariable<double> htfluxConvY;
    SFCYVariable<double> htfluxRadY;
    SFCYVariable<double> htfluxY;
    SFCZVariable<double> htfluxConvZ;
    SFCZVariable<double> htfluxRadZ;
    SFCZVariable<double> htfluxZ;
    CCVariable<double> htfluxConvCC;

    // Stability factor

    CCVariable<double> KStabilityH;
    constCCVariable<double> enthalpy;

    int numGhostCells = 1;

    int numGhostCellsG = 1;

    // patch geometry information

    // memory for MPM

    for (int m = 0; m < numMPMMatls; m++) {

      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();

      int zeroGhostCells = 0;

      // gets

      new_dw->get(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
          idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(tempSolid_cc[m], d_MAlb->tempSolid_CCLabel,
          idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(tempSolid_fcx[m], d_MAlb->tempSolid_FCXLabel,
          idx, patch, Ghost::AroundFaces, numGhostCells);

      new_dw->get(tempSolid_fcy[m], d_MAlb->tempSolid_FCYLabel,
          idx, patch, Ghost::AroundFaces, numGhostCells);

      new_dw->get(tempSolid_fcz[m], d_MAlb->tempSolid_FCZLabel,
          idx, patch, Ghost::AroundFaces, numGhostCells);

      new_dw->get(upCC[m], d_MAlb->xvel_CCLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(vpCC[m], d_MAlb->yvel_CCLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(wpCC[m], d_MAlb->zvel_CCLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(upFCY[m], d_MAlb->xvel_FCYLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(upFCZ[m], d_MAlb->xvel_FCZLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(vpFCX[m], d_MAlb->yvel_FCXLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(vpFCZ[m], d_MAlb->yvel_FCZLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(wpFCX[m], d_MAlb->zvel_FCXLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      new_dw->get(wpFCY[m], d_MAlb->zvel_FCYLabel,
          idx, patch, Ghost::None, zeroGhostCells);

      // allocates

      new_dw->allocateAndPut(heaTranSolid_cc[m], d_MAlb->heaTranSolid_tmp_CCLabel,
          idx, patch);
      heaTranSolid_cc[m].initialize(0.);

      new_dw->allocateAndPut(heaTranSolid_fcx[m], d_MAlb->heaTranSolid_FCXLabel,
          idx, patch);
      heaTranSolid_fcx[m].initialize(0.);

      new_dw->allocateAndPut(heaTranSolid_fcy[m], d_MAlb->heaTranSolid_FCYLabel,
          idx, patch);
      heaTranSolid_fcy[m].initialize(0.);

      new_dw->allocateAndPut(heaTranSolid_fcz[m], d_MAlb->heaTranSolid_FCZLabel,
          idx, patch);
      heaTranSolid_fcz[m].initialize(0.);

      new_dw->allocateAndPut(heaTranSolid_fcx_rad[m], d_MAlb->heaTranSolid_FCX_RadLabel,
          idx, patch);
      heaTranSolid_fcx_rad[m].initialize(0.);

      new_dw->allocateAndPut(heaTranSolid_fcy_rad[m], d_MAlb->heaTranSolid_FCY_RadLabel,
          idx, patch);
      heaTranSolid_fcy_rad[m].initialize(0.);

      new_dw->allocateAndPut(heaTranSolid_fcz_rad[m], d_MAlb->heaTranSolid_FCZ_RadLabel,
          idx, patch);
      heaTranSolid_fcz_rad[m].initialize(0.);

    }

    // memory for Arches

    // gets

    new_dw->get(cellType, d_Alab->d_mmcellTypeLabel, 
        matlIndex, patch, Ghost::AroundCells, numGhostCellsG);

    old_dw->get(tempGas, d_Alab->d_tempINLabel,   
        matlIndex, patch, Ghost::AroundCells, numGhostCellsG);

    new_dw->get(gas_fraction_cc, d_Alab->d_mmgasVolFracLabel, matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);

    old_dw->get(CCVelocity, d_Alab->d_CCVelocityLabel, matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);

    CCVariable<double> ugCC; 
    CCVariable<double> vgCC; 
    CCVariable<double> wgCC; 

    new_dw->allocateTemporary( ugCC, patch, Ghost::AroundCells, numGhostCellsG ); 
    new_dw->allocateTemporary( vgCC, patch, Ghost::AroundCells, numGhostCellsG ); 
    new_dw->allocateTemporary( wgCC, patch, Ghost::AroundCells, numGhostCellsG ); 

    //copy over components of velocity.  Not very efficient but would 
    //require a rewrite of the momexch from fortran to c++
    for ( CellIterator iter = patch->getCellIterator( numGhostCellsG ); !iter.done(); ++iter ){ 

      IntVector c = *iter; 

      ugCC[c] = CCVelocity[c].x(); 
      vgCC[c] = CCVelocity[c].y(); 
      wgCC[c] = CCVelocity[c].z(); 

    } 

    old_dw->get(denMicro, d_Alab->d_densityMicroLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCellsG);

    if (d_DORad && d_calcEnergyExchange) {

      // for radiative heat transfer to intrusions

      old_dw->getCopy(radfluxE, d_Alab->d_radiationFluxEINLabel,
          matlIndex, patch, Ghost::AroundCells, numGhostCellsG);
      old_dw->getCopy(radfluxW, d_Alab->d_radiationFluxWINLabel,
          matlIndex, patch, Ghost::AroundCells, numGhostCellsG);
      old_dw->getCopy(radfluxN, d_Alab->d_radiationFluxNINLabel,
          matlIndex, patch, Ghost::AroundCells, numGhostCellsG);
      old_dw->getCopy(radfluxS, d_Alab->d_radiationFluxSINLabel,
          matlIndex, patch, Ghost::AroundCells, numGhostCellsG);
      old_dw->getCopy(radfluxT, d_Alab->d_radiationFluxTINLabel,
          matlIndex, patch, Ghost::AroundCells, numGhostCellsG);
      old_dw->getCopy(radfluxB, d_Alab->d_radiationFluxBINLabel,
          matlIndex, patch, Ghost::AroundCells, numGhostCellsG);
    }
    else {
      radfluxE.allocate(patch->getExtraCellLowIndex(numGhostCellsG), 
          patch->getExtraCellHighIndex(numGhostCellsG));
      radfluxE.initialize(0.);
      radfluxW.allocate(patch->getExtraCellLowIndex(numGhostCellsG), 
          patch->getExtraCellHighIndex(numGhostCellsG));
      radfluxW.initialize(0.);
      radfluxN.allocate(patch->getExtraCellLowIndex(numGhostCellsG), 
          patch->getExtraCellHighIndex(numGhostCellsG));
      radfluxN.initialize(0.);
      radfluxS.allocate(patch->getExtraCellLowIndex(numGhostCellsG), 
          patch->getExtraCellHighIndex(numGhostCellsG));
      radfluxS.initialize(0.);
      radfluxT.allocate(patch->getExtraCellLowIndex(numGhostCellsG), 
          patch->getExtraCellHighIndex(numGhostCellsG));
      radfluxT.initialize(0.);
      radfluxB.allocate(patch->getExtraCellLowIndex(numGhostCellsG), 
          patch->getExtraCellHighIndex(numGhostCellsG));
      radfluxB.initialize(0.);
    }

    old_dw->get(enthalpy, d_enthalpy_label, matlIndex, 
        patch, Ghost::AroundCells, numGhostCellsG);
    // allocates

    new_dw->allocateAndPut(sp_enth_cc, d_MAlb->d_enth_mmLinSrc_tmp_CCLabel,
        matlIndex, patch);
    sp_enth_cc.initialize(0.);

    new_dw->allocateAndPut(sp_enth_fcx, d_MAlb->d_enth_mmLinSrc_FCXLabel,
        matlIndex, patch);
    sp_enth_fcx.initialize(0.);

    new_dw->allocateAndPut(sp_enth_fcy, d_MAlb->d_enth_mmLinSrc_FCYLabel,
        matlIndex, patch);
    sp_enth_fcy.initialize(0.);

    new_dw->allocateAndPut(sp_enth_fcz, d_MAlb->d_enth_mmLinSrc_FCZLabel,
        matlIndex, patch);
    sp_enth_fcz.initialize(0.);

    new_dw->allocateAndPut(su_enth_cc, d_MAlb->d_enth_mmNonLinSrc_tmp_CCLabel,
        matlIndex, patch);
    su_enth_cc.initialize(0.);

    new_dw->allocateAndPut(su_enth_fcx, d_MAlb->d_enth_mmNonLinSrc_FCXLabel,
        matlIndex, patch);
    su_enth_fcx.initialize(0.);

    new_dw->allocateAndPut(su_enth_fcy, d_MAlb->d_enth_mmNonLinSrc_FCYLabel,
        matlIndex, patch);
    su_enth_fcy.initialize(0.);

    new_dw->allocateAndPut(su_enth_fcz, d_MAlb->d_enth_mmNonLinSrc_FCZLabel,
        matlIndex, patch);
    su_enth_fcz.initialize(0.);

    new_dw->allocateAndPut(htfluxConvX, d_MAlb->htfluxConvXLabel,
        matlIndex, patch);
    htfluxConvX.initialize(0.);
    new_dw->allocateAndPut(htfluxRadX, d_MAlb->htfluxRadXLabel,
        matlIndex, patch);
    htfluxRadX.initialize(0.);
    new_dw->allocateAndPut(htfluxX, d_MAlb->htfluxXLabel,
        matlIndex, patch);
    htfluxX.initialize(0.);

    new_dw->allocateAndPut(htfluxConvY, d_MAlb->htfluxConvYLabel,
        matlIndex, patch);
    htfluxConvY.initialize(0.);
    new_dw->allocateAndPut(htfluxRadY, d_MAlb->htfluxRadYLabel,
        matlIndex, patch);
    htfluxRadY.initialize(0.);
    new_dw->allocateAndPut(htfluxY, d_MAlb->htfluxYLabel,
        matlIndex, patch);
    htfluxY.initialize(0.);

    new_dw->allocateAndPut(htfluxConvZ, d_MAlb->htfluxConvZLabel,
        matlIndex, patch);
    htfluxConvZ.initialize(0.);

    new_dw->allocateAndPut(htfluxRadZ, d_MAlb->htfluxRadZLabel,
        matlIndex, patch);
    htfluxRadZ.initialize(0.);

    new_dw->allocateAndPut(htfluxZ, d_MAlb->htfluxZLabel,
        matlIndex, patch);
    htfluxZ.initialize(0.);

    new_dw->allocateAndPut(htfluxConvCC, d_MAlb->htfluxConvCCLabel,
        matlIndex, patch);
    htfluxConvCC.initialize(0.);

    new_dw->allocateAndPut(KStabilityH, d_MAlb->KStabilityHLabel,
        matlIndex, patch);
    KStabilityH.initialize(0.);

    // Begin loop to calculate gas-solid exchange terms for each
    // solid material with the gas phase

    int ffieldid = d_arches->getBoundaryCondition()->flowCellType();
    int mmwallid = d_arches->getBoundaryCondition()->getMMWallId();

    // csmag = d_arches->getTurbulenceModel()->getSmagorinskyConst();
    double csmag = 0.17;

    IntVector valid_lo = patch->getFortranCellLowIndex();
    IntVector valid_hi = patch->getFortranCellHighIndex();

    double dx = Dx.x();
    double dy = Dx.y(); 
    double dz = Dx.z(); 

    for (int m = 0; m < numMPMMatls; m++) {

      fort_energy_exchange_term(
          heaTranSolid_fcx[m],
          heaTranSolid_fcy[m],
          heaTranSolid_fcz[m],
          heaTranSolid_fcx_rad[m],
          heaTranSolid_fcy_rad[m],
          heaTranSolid_fcz_rad[m],
          heaTranSolid_cc[m],
          htfluxConvX,
          htfluxRadX,
          htfluxX,
          htfluxConvY,
          htfluxRadY,
          htfluxY,
          htfluxConvZ,
          htfluxRadZ,
          htfluxZ,
          htfluxConvCC,
          su_enth_cc, 
          sp_enth_cc,
          su_enth_fcx, 
          sp_enth_fcx,
          su_enth_fcy, 
          sp_enth_fcy,
          su_enth_fcz, 
          sp_enth_fcz,
          KStabilityH,
          tempGas, 
          tempSolid_cc[m],
          tempSolid_fcx[m],
          tempSolid_fcy[m],
          tempSolid_fcz[m],
          ugCC,
          vgCC,
          wgCC,
          upCC[m],
          vpCC[m],
          wpCC[m],
          vpFCX[m],
          wpFCX[m],
          upFCY[m],
          wpFCY[m],
          upFCZ[m],
          vpFCZ[m],
          denMicro,
          enthalpy,
          radfluxE,
          radfluxW,
          radfluxN,
          radfluxS,
          radfluxT,
          radfluxB,
          gas_fraction_cc, 
          solid_fraction_cc[m],
          dx, 
          dy, 
          dz, 
          d_tcond,
          csmag,
          prturb,
          cpfluid,
          valid_lo, valid_hi,
          cellType, mmwallid, ffieldid);

    }
  }
}

//______________________________________________________________________
//

void MPMArches::collectToCCGasEnergyExchSrcs(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse* ,
    DataWarehouse* new_dw)

{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constSFCXVariable<double> su_enth_fcx;
    constSFCYVariable<double> su_enth_fcy;
    constSFCZVariable<double> su_enth_fcz;

    constSFCXVariable<double> sp_enth_fcx;
    constSFCYVariable<double> sp_enth_fcy;
    constSFCZVariable<double> sp_enth_fcz;

    CCVariable<double> su_enth_cc;
    CCVariable<double> sp_enth_cc;

    int numGhostCells = 1;

    new_dw->get(su_enth_fcx, d_MAlb->d_enth_mmNonLinSrc_FCXLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(su_enth_fcy, d_MAlb->d_enth_mmNonLinSrc_FCYLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(su_enth_fcz, d_MAlb->d_enth_mmNonLinSrc_FCZLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->get(sp_enth_fcx, d_MAlb->d_enth_mmLinSrc_FCXLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(sp_enth_fcy, d_MAlb->d_enth_mmLinSrc_FCYLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);
    new_dw->get(sp_enth_fcz, d_MAlb->d_enth_mmLinSrc_FCZLabel,
        matlIndex, patch, Ghost::AroundFaces, numGhostCells);

    new_dw->allocateAndPut(su_enth_cc, d_MAlb->d_enth_mmNonLinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(su_enth_cc, d_MAlb->d_enth_mmNonLinSrc_tmp_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);

    new_dw->allocateAndPut(sp_enth_cc, d_MAlb->d_enth_mmLinSrc_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->copyOut(sp_enth_cc, d_MAlb->d_enth_mmLinSrc_tmp_CCLabel,
        matlIndex, patch, Ghost::AroundCells, numGhostCells);

    IntVector valid_lo = patch->getFortranCellLowIndex();
    IntVector valid_hi = patch->getFortranCellHighIndex();

    fort_collect_scalar_fctocc(
        su_enth_cc,  sp_enth_cc,
        su_enth_fcx, sp_enth_fcx,
        su_enth_fcy, sp_enth_fcy,
        su_enth_fcz, sp_enth_fcz,
        valid_lo, valid_hi);

  }
}

//______________________________________________________________________
//

void MPMArches::schedulePutAllForcesOnCC(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* mpm_matls)
{
  // Grab all of the forces and energy fluxes which Arches wants to 
  // give to MPM and accumulate them on the cell centers

  Task* t=scinew Task("MPMArches::putAllForcesOnCC",
      this, &MPMArches::putAllForcesOnCC);

  int zeroGhostCells = 0;
  int numGhostCells = 1;

  t->requires(Task::NewDW, d_MAlb->cMassLabel, Ghost::None, zeroGhostCells);

  if (!d_stationarySolid) {
    t->requires(Task::NewDW, d_MAlb->DragForceX_CCLabel, 
        mpm_matls->getUnion(), Ghost::None, zeroGhostCells);  
    t->requires(Task::NewDW, d_MAlb->DragForceY_CCLabel, 
        mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
    t->requires(Task::NewDW, d_MAlb->DragForceZ_CCLabel, 
        mpm_matls->getUnion(), Ghost::None, zeroGhostCells);

    t->requires(Task::NewDW, d_MAlb->PressureForce_FCXLabel,
        mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
    t->requires(Task::NewDW, d_MAlb->PressureForce_FCYLabel,
        mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
    t->requires(Task::NewDW, d_MAlb->PressureForce_FCZLabel,
        mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
  }

  if (d_calcEnergyExchange) {

    t->requires(Task::NewDW, d_MAlb->heaTranSolid_tmp_CCLabel,
        mpm_matls->getUnion(), Ghost::AroundCells, numGhostCells);

    t->requires(Task::NewDW, d_MAlb->heaTranSolid_FCXLabel,
        mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
    t->requires(Task::NewDW, d_MAlb->heaTranSolid_FCYLabel,
        mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);
    t->requires(Task::NewDW, d_MAlb->heaTranSolid_FCZLabel,
        mpm_matls->getUnion(), Ghost::AroundFaces, numGhostCells);

    t->computes(d_MAlb->heaTranSolid_CCLabel, mpm_matls->getUnion());

  }

  t->computes(d_MAlb->SumAllForcesCCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->AccArchesCCLabel,    mpm_matls->getUnion());

  sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//

void MPMArches::putAllForcesOnCC(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* matls,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)

{
  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      int numGhostCells = 1;
      int zeroGhostCells = 0;
      CCVariable<Vector> totalforce;
      CCVariable<Vector> acc_arches;
      constCCVariable<double> DFX_CC, DFY_CC, DFZ_CC, cmass;
      constSFCXVariable<double> PRX_FC;
      constSFCYVariable<double> PRY_FC;
      constSFCZVariable<double> PRZ_FC;

      CCVariable<double> htrate_cc;
      constSFCXVariable<double> htrate_fcx;
      constSFCYVariable<double> htrate_fcy;
      constSFCZVariable<double> htrate_fcz;

      new_dw->allocateAndPut(totalforce, d_MAlb->SumAllForcesCCLabel,
          matlindex, patch);
      new_dw->allocateAndPut(acc_arches, d_MAlb->AccArchesCCLabel,   
          matlindex, patch);

      if (!d_stationarySolid) {
        new_dw->get(cmass,  d_MAlb->cMassLabel,         matlindex, patch,
            Ghost::None, zeroGhostCells);

        new_dw->get(DFX_CC, d_MAlb->DragForceX_CCLabel, matlindex, patch,
            Ghost::None, zeroGhostCells);

        new_dw->get(DFY_CC, d_MAlb->DragForceY_CCLabel, matlindex, patch,
            Ghost::None, zeroGhostCells);

        new_dw->get(DFZ_CC, d_MAlb->DragForceZ_CCLabel, matlindex, patch,
            Ghost::None, zeroGhostCells);

        new_dw->get(PRX_FC, d_MAlb->PressureForce_FCXLabel, matlindex, 
            patch, Ghost::AroundFaces, numGhostCells);

        new_dw->get(PRY_FC, d_MAlb->PressureForce_FCYLabel, matlindex, 
            patch, Ghost::AroundFaces, numGhostCells);

        new_dw->get(PRZ_FC, d_MAlb->PressureForce_FCZLabel, matlindex, 
            patch, Ghost::AroundFaces, numGhostCells);
      }

      if (d_calcEnergyExchange) {

        new_dw->allocateAndPut(htrate_cc, d_MAlb->heaTranSolid_CCLabel, matlindex, 
            patch, Ghost::AroundCells, numGhostCells);

        new_dw->copyOut(htrate_cc, d_MAlb->heaTranSolid_tmp_CCLabel, matlindex, 
            patch, Ghost::AroundCells, numGhostCells);

        new_dw->get(htrate_fcx, d_MAlb->heaTranSolid_FCXLabel, matlindex, 
            patch, Ghost::AroundFaces, numGhostCells);

        new_dw->get(htrate_fcy, d_MAlb->heaTranSolid_FCYLabel, matlindex, 
            patch, Ghost::AroundFaces, numGhostCells);

        new_dw->get(htrate_fcz, d_MAlb->heaTranSolid_FCZLabel, matlindex, 
            patch, Ghost::AroundFaces, numGhostCells);

      }

      acc_arches.initialize(Vector(0.,0.,0.));

      if (!d_stationarySolid) {

        for(CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){

          totalforce[*iter] = Vector(DFX_CC[*iter], DFY_CC[*iter], DFZ_CC[*iter]);
          IntVector curcell = *iter;
          double XCPF = 0.0, YCPF = 0.0, ZCPF = 0.0;

          if (curcell.x() >= (patch->getCellLowIndex()).x()) {
            IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());
            XCPF = .5*(PRX_FC[curcell] + PRX_FC[adjcell]);
          }

          if (curcell.y() >= (patch->getCellLowIndex()).y()) {
            IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
            YCPF = .5*(PRY_FC[curcell] + PRY_FC[adjcell]);
          }

          if (curcell.z() >= (patch->getCellLowIndex()).z()) {
            IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
            ZCPF = .5*(PRZ_FC[curcell] + PRZ_FC[adjcell]);
          }

          totalforce[*iter] += Vector(XCPF, YCPF, ZCPF);
          if(cmass[*iter] > d_SMALL_NUM){
            acc_arches[*iter] = totalforce[*iter]/cmass[*iter];
          }
        }
      }
      else {
        acc_arches.initialize(Vector(0.,0.,0.));
        totalforce.initialize(Vector(0.,0.,0.));
      }

      if (d_calcEnergyExchange) {

        for(CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){

          IntVector curcell = *iter;

          if (curcell.x() >= (patch->getCellLowIndex()).x()) {

            IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());
            htrate_cc[curcell] = htrate_cc[curcell] + 
              .5*(htrate_fcx[curcell] + htrate_fcx[adjcell]);
          }

          if (curcell.y() >= (patch->getCellLowIndex()).y()) {

            IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
            htrate_cc[curcell] = htrate_cc[curcell] + 
              .5*(htrate_fcy[curcell] + htrate_fcy[adjcell]);
          }

          if (curcell.z() >= (patch->getCellLowIndex()).z()) {
            IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
            htrate_cc[curcell] = htrate_cc[curcell] + 
              .5*(htrate_fcz[curcell] + htrate_fcz[adjcell]);
          }
        }
      }
    }
  }
}

//______________________________________________________________________
//

void MPMArches::schedulePutAllForcesOnNC(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* mpm_matls)
{
  // Take the cell centered forces from Arches and put them on the
  // nodes where SerialMPM can grab and use them
  Task* t=scinew Task("MPMArches::putAllForcesOnNC",
      this, &MPMArches::putAllForcesOnNC);

  int numGhostCells = 1;

  if (!d_stationarySolid) {

    t->requires(Task::NewDW,d_MAlb->AccArchesCCLabel, mpm_matls->getUnion(),
        Ghost::AroundCells, numGhostCells);
  }
  t->computes(d_MAlb->AccArchesNCLabel,             mpm_matls->getUnion());


  if (d_calcEnergyExchange) {

    t->requires(Task::NewDW,d_MAlb->heaTranSolid_CCLabel, 
        mpm_matls->getUnion(), Ghost::AroundCells, numGhostCells);

  }
  t->computes(d_MAlb->heaTranSolid_NCLabel, mpm_matls->getUnion());
  sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//

void MPMArches::putAllForcesOnNC(const ProcessorGroup*,
    const PatchSubset* patches,
    const MaterialSubset* matls,
    DataWarehouse* /*old_dw*/,
    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector cIdx[8];

    for(int m=0;m<matls->size();m++){

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      constCCVariable<Vector> acc_archesCC;
      NCVariable<Vector> acc_archesNC;

      constCCVariable<double> htrate_cc;
      NCVariable<double> htrate_nc;

      int numGhostCells = 1;
      Vector zero(0.0,0.0,0.0);      

      if (!d_stationarySolid) {
        new_dw->get(acc_archesCC, d_MAlb->AccArchesCCLabel,   
            matlindex, patch, Ghost::AroundCells, numGhostCells);

      }

      new_dw->allocateAndPut(acc_archesNC, d_MAlb->AccArchesNCLabel, 
          matlindex, patch);
      acc_archesNC.initialize(zero);

      if (d_calcEnergyExchange) {

        new_dw->get(htrate_cc, d_MAlb->heaTranSolid_CCLabel,   
            matlindex, patch, Ghost::AroundCells, numGhostCells);
      }

      new_dw->allocateAndPut(htrate_nc, d_MAlb->heaTranSolid_NCLabel, 
          matlindex, patch);
      htrate_nc.initialize(0.0);

      if (d_stationarySolid)
        acc_archesNC.initialize(zero);
      else {

        for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){

          patch->findCellsFromNode(*iter,cIdx);
          for (int in=0;in<8;in++){

            acc_archesNC[*iter]  += acc_archesCC[cIdx[in]]*.125;

          }
        }
      }

      if (d_calcEnergyExchange) {

        for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){

          patch->findCellsFromNode(*iter,cIdx);
          for (int in=0;in<8;in++){

            htrate_nc[*iter] += htrate_cc[cIdx[in]]*.125;

          }
        }
      }
    }
  }
}

void MPMArches::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)
{
  Task* t = scinew Task("MPMArches::computeAndIntegrateAcceleration",
      this, &MPMArches::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, Mlb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, Mlb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, Mlb->gExternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, Mlb->gVelocityLabel,      Ghost::None);
  t->requires(Task::NewDW, d_MAlb->AccArchesNCLabel,Ghost::None);

  t->computes(Mlb->gVelocityStarLabel);
  t->computes(Mlb->gAccelerationLabel);
  sched->addTask(t,patches,matls);
}

void MPMArches::computeAndIntegrateAcceleration(const ProcessorGroup* pg,
    const PatchSubset* patches,
    const MaterialSubset* ms,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gnone = Ghost::None;
    //    Vector gravity = d_mpm->flags->d_gravity;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<Vector> internalforce, externalforce, velocity;
      constNCVariable<double> mass;

      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      new_dw->get(internalforce,Mlb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce,Mlb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(mass,         Mlb->gMassLabel,          dwi, patch, gnone, 0);
      new_dw->get(velocity,     Mlb->gVelocityLabel,      dwi, patch, gnone, 0);

      constNCVariable<Vector> AccArchesNC; 
      new_dw->get(AccArchesNC,d_MAlb->AccArchesNCLabel,dwi,patch,Ghost::None,0);

      // Create variables for the results
      NCVariable<Vector> velocity_star,acceleration;
      new_dw->allocateAndPut(velocity_star,Mlb->gVelocityStarLabel, dwi, patch);
      new_dw->allocateAndPut(acceleration, Mlb->gAccelerationLabel, dwi, patch);

      acceleration.initialize(Vector(0.,0.,0.));

      for(NodeIterator iter = patch->getExtraNodeIterator(); !iter.done();
          iter++){
        IntVector c = *iter;
        Vector acc(0.,0.,0.);
        acceleration[c] = (internalforce[c] + externalforce[c])/mass[c];
        //         acceleration[c] +=  gravity;
        acceleration[c] += AccArchesNC[c];
        velocity_star[c] = velocity[c] + acceleration[c] * delT;
      }
    }
  }
}

void MPMArches::scheduleSolveHeatEquations(SchedulerP& sched,
    const PatchSet* patches,
    const MaterialSet* matls)
{
  d_mpm->scheduleSolveHeatEquations(sched,patches,matls);

  Task* t = scinew Task("MPMArches::solveHeatEquations",
      this, &MPMArches::solveHeatEquations);

  t->requires(Task::NewDW, Mlb->heaTranSolid_NCLabel,  Ghost::None);
  t->requires(Task::NewDW, Mlb->gMassLabel,Ghost::None);
  t->modifies(Mlb->gTemperatureRateLabel);
  sched->addTask(t,patches,matls);
}

void MPMArches::solveHeatEquations(const ProcessorGroup* pg,
    const PatchSubset* patches,
    const MaterialSubset* ms,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double Cv = mpm_matl->getSpecificHeat();

      constNCVariable<double> htrate_gasNC,mass; 
      NCVariable<double> tempRate;

      new_dw->get(mass,    Mlb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->getModifiable(tempRate,Mlb->gTemperatureRateLabel,dwi, patch);
      new_dw->get(htrate_gasNC,Mlb->heaTranSolid_NCLabel,dwi,patch,
          Ghost::None, 0);

      for(NodeIterator iter = patch->getExtraNodeIterator();
          !iter.done();iter++)
        tempRate[*iter] += htrate_gasNC[*iter]/(mass[*iter]*Cv);
    }

  }

}

// ****************************************************************************
// Function to return boolean for recompiling taskgraph
// ****************************************************************************
bool MPMArches::needRecompile(double time, double dt, 
    const GridP& grid) {

  if ( d_recompile ) { 
    d_recompile = false; 
    return true; 
  } else { 
    return d_recompile;
  }
}
double MPMArches::recomputeTimestep(double current_dt) {
  return d_arches->recomputeTimestep(current_dt);
}

bool MPMArches::restartableTimesteps() {
  return d_arches->restartableTimesteps();
}

namespace Uintah {

  static MPI_Datatype makeMPI_cutcell()
  {
    ASSERTEQ(sizeof(cutcell), sizeof(double)*13);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 13, 13, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }

  const TypeDescription* fun_getTypeDescription(cutcell*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
          "cutcell", true, 
          &makeMPI_cutcell);
    }
    return td;
  }

} // namespace Uintah

//______________________________________________________________________
//  
namespace SCIRun {

  void swapbytes( Uintah::cutcell& c) {
    double *p = c.d_cutcell;
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p);
  }

} // namespace SCIRun
