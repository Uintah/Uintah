/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


//----- BoundaryCondition.cc ----------------------------------------------

#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/CellInformationP.h>

#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/ExtraScalarSolver.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>

#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
#include <sstream>


using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <CCA/Components/Arches/fortran/celltypeInit_fort.h>
#include <CCA/Components/Arches/fortran/areain_fort.h>
#include <CCA/Components/Arches/fortran/profscalar_fort.h>
#include <CCA/Components/Arches/fortran/inlbcs_fort.h>
#include <CCA/Components/Arches/fortran/inlpresbcinout_fort.h>
#include <CCA/Components/Arches/fortran/bcscalar_fort.h>
#include <CCA/Components/Arches/fortran/bcuvel_fort.h>
#include <CCA/Components/Arches/fortran/bcvvel_fort.h>
#include <CCA/Components/Arches/fortran/bcwvel_fort.h>
#include <CCA/Components/Arches/fortran/profv_fort.h>
#include <CCA/Components/Arches/fortran/intrusion_computevel_fort.h>
#include <CCA/Components/Arches/fortran/mmbcenthalpy_energyex_fort.h>
#include <CCA/Components/Arches/fortran/mmbcvelocity_momex_fort.h>
#include <CCA/Components/Arches/fortran/mmbcvelocity_fort.h>
#include <CCA/Components/Arches/fortran/mmcelltypeinit_fort.h>
#include <CCA/Components/Arches/fortran/mmwallbc_fort.h>
#include <CCA/Components/Arches/fortran/mmwallbc_trans_fort.h>
#include <CCA/Components/Arches/fortran/mm_computevel_fort.h>
#include <CCA/Components/Arches/fortran/mm_explicit_fort.h>
#include <CCA/Components/Arches/fortran/mm_explicit_oldvalue_fort.h>
#include <CCA/Components/Arches/fortran/mm_explicit_vel_fort.h>
#include <CCA/Components/Arches/fortran/get_ramping_factor_fort.h>

//****************************************************************************
// Constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition(const ArchesLabel* label,
                                     const MPMArchesLabel* MAlb,
                                     PhysicalConstants* phys_const,
                                     Properties* props,
                                     bool calcReactScalar,
                                     bool calcEnthalpy,
                                     bool calcVariance):
                                     d_lab(label), d_MAlab(MAlb),
                                     d_physicalConsts(phys_const), 
                                     d_props(props),
                                     d_reactingScalarSolve(calcReactScalar),
                                     d_enthalpySolve(calcEnthalpy),
                                     d_calcVariance(calcVariance)
{
  MM_CUTOFF_VOID_FRAC = 0.5;
  d_wallBdry = 0;
  d_pressureBC = 0;
  d_outletBC = 0;
  d_intrusionBC = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
BoundaryCondition::~BoundaryCondition()
{
  if(d_wallBdry){     
    delete d_wallBdry;   
  }
  if(d_pressureBC){   
    delete d_pressureBC;
  }
  if(d_outletBC){
    delete d_outletBC;
  }
  for (int ii = 0; ii < d_numInlets; ii++){
    delete d_flowInlets[ii];
  }
  if (d_calcExtraScalars){
    for (int i=0; i < static_cast<int>(d_extraScalarBCs.size()); i++){
      delete d_extraScalarBCs[i];
    }
  }
  if(d_intrusionBC){
    delete d_intrusionBC;
  }
  for ( EfficiencyMap::iterator iter = d_effVars.begin(); iter != d_effVars.end(); iter++){
    VarLabel::destroy(iter->second.label);
  }

  for ( SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){
    VarLabel::destroy(iter->second.flowRateLabel);
  }
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
BoundaryCondition::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params->findBlock("BoundaryConditions");
  d_flowfieldCellTypeVal = -1;
  d_numInlets = 0;
  d_numSourceBoundaries = 0;
  int total_cellTypes = 0;
 
  if(db.get_rep()==0)
  {
    proc0cout << "No Boundary Conditions Specified\n";
    d_inletBoundary = false;
    d_wallBoundary = false;
    d_pressureBoundary = false;
    d_outletBoundary = false;
    d_intrusionBoundary = false;
    d_carbon_balance=false;
    d_sulfur_balance=false;
    d_carbon_balance_es=false;
    d_sulfur_balance_es=false;
  }
  else
  {
    db->getWithDefault("carbon_balance", d_carbon_balance, false);
    db->getWithDefault("sulfur_balance", d_sulfur_balance, false);
    db->getWithDefault("carbon_balance_es", d_carbon_balance_es, false);
    db->getWithDefault("sulfur_balance_es", d_sulfur_balance_es, false);
    //--- instrusions with boundary sources -----
    if (ProblemSpecP intrusionbcs_db = db->findBlock("IntrusionWithBCSource")){
      for (ProblemSpecP intrusionbcs_db = db->findBlock("IntrusionWithBCSource");
          intrusionbcs_db != 0; intrusionbcs_db = intrusionbcs_db->findNextBlock("IntrusionWithBCSource")){
        d_sourceBoundaryInfo.push_back(scinew BCSourceInfo(d_calcVariance, d_reactingScalarSolve));

        d_sourceBoundaryInfo[d_numSourceBoundaries]->problemSetup(intrusionbcs_db);
        //compute the density and other properties for this inlet stream
        d_sourceBoundaryInfo[d_numSourceBoundaries]->streamMixturefraction.d_initEnthalpy = true;
        d_sourceBoundaryInfo[d_numSourceBoundaries]->streamMixturefraction.d_scalarDisp=0.0;
        d_sourceBoundaryInfo[d_numSourceBoundaries]->streamMixturefraction.d_mixVarVariance.push_back(0.0);
        d_props->computeInletProperties(d_sourceBoundaryInfo[d_numSourceBoundaries]->streamMixturefraction,
            d_sourceBoundaryInfo[d_numSourceBoundaries]->calcStream);
        ++d_numSourceBoundaries;

      }
    }

    // --- new efficiency calculator --- 
    ProblemSpecP eff_db = db->findBlock("ScalarEfficiency");
    if (eff_db) {
      for (ProblemSpecP scalareff_db = eff_db->findBlock("scalar"); scalareff_db != 0; 
            scalareff_db = scalareff_db->findNextBlock("scalar")) {

        std::string scalar_name;
        std::string fuel_ratio; 
        std::string air_ratio; 
        vector<std::string> species; 
        std::vector<string> which_inlets; 

        scalareff_db->getAttribute("label",scalar_name);
        scalareff_db->getAttribute("fuel_ratio",fuel_ratio);
        scalareff_db->getAttribute("air_ratio",  air_ratio); 

        double dfuel_ratio = atof(fuel_ratio.c_str());
        double dair_ratio  = atof(air_ratio.c_str());

        for (ProblemSpecP inlet_db = scalareff_db->findBlock("inlet"); inlet_db != 0;
            inlet_db = inlet_db->findNextBlock("inlet")){
          std::string inletName = inlet_db->getNodeValue();

          which_inlets.push_back(inletName);
        }

        // now get the species it needs to compute efficiency
        for (ProblemSpecP species_db = scalareff_db->findBlock("species"); species_db != 0;
            species_db = species_db->findNextBlock("species")){

          std::string species_name;
          std::string mol_ratio; 

          species_db->getAttribute("label",species_name);
          species_db->getAttribute("mol_ratio",mol_ratio);

          double dmol_ratio = atof(mol_ratio.c_str());

          species.push_back(species_name);
          this->insertIntoSpeciesMap( species_name, dmol_ratio ); 

        }

        this->insertIntoEffMap( scalar_name, dfuel_ratio, dair_ratio, species, which_inlets ); 
      }
    }

    if (ProblemSpecP inlet_db = db->findBlock("FlowInlet")) {
      d_inletBoundary = true;
      for (ProblemSpecP inlet_db = db->findBlock("FlowInlet");
          inlet_db != 0; inlet_db = inlet_db->findNextBlock("FlowInlet")) {
        d_flowInlets.push_back(scinew FlowInlet(total_cellTypes, d_calcVariance,
              d_reactingScalarSolve));
        d_flowInlets[d_numInlets]->problemSetup(inlet_db);
        // compute density and other dependent properties
        d_flowInlets[d_numInlets]->streamMixturefraction.d_initEnthalpy=true;
        d_flowInlets[d_numInlets]->streamMixturefraction.d_scalarDisp=0.0;
        d_props->computeInletProperties(
            d_flowInlets[d_numInlets]->streamMixturefraction,
            d_flowInlets[d_numInlets]->calcStream);
        double f = d_flowInlets[d_numInlets]->streamMixturefraction.d_mixVars[0];
        if (f > 0.0){
          d_flowInlets[d_numInlets]->fcr = d_props->getCarbonContent(f);
          
        }
        if (d_calcExtraScalars) {

          ProblemSpecP extra_scalar_db = inlet_db->findBlock("ExtraScalars");
          for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {

            string curr_name = d_extraScalars->at(i)->getScalarName();
  
            for (ProblemSpecP scalar_db = extra_scalar_db->findBlock("scalar"); 
                 scalar_db != 0; scalar_db = scalar_db->findNextBlock("scalar")){

              std::string scalar_name;
              scalar_db->getAttribute("label",scalar_name);

              if (scalar_name == curr_name){

                char* value;
                string svalue;  
                double dvalue = 0; 
                scalar_db->getAttribute("value", svalue);
                value = &svalue[0]; 
                dvalue = std::atof(value);                 

                d_extraScalarBC* bc = scinew d_extraScalarBC;
                bc->d_scalar_name = curr_name;
                bc->d_scalarBC_value = dvalue;
                bc->d_BC_ID = total_cellTypes;
                d_extraScalarBCs.push_back(bc);
              }
            }
          }
        }
        ++total_cellTypes;
        ++d_numInlets;
      }
    }
    else {
      proc0cout << "Flow inlet boundary not specified" << endl;
      d_inletBoundary = false;
    }

    if (ProblemSpecP wall_db = db->findBlock("WallBC")) {
      d_wallBoundary = true;
      d_wallBdry = scinew WallBdry(total_cellTypes);
      d_wallBdry->problemSetup(wall_db);
      ++total_cellTypes;
    }
    else {
      proc0cout << "Wall boundary not specified"<<endl;
      d_wallBoundary = false;
    }

    if (ProblemSpecP press_db = db->findBlock("PressureBC")) {
      d_pressureBoundary = true;
      d_pressureBC = scinew PressureInlet(total_cellTypes, d_calcVariance,
          d_reactingScalarSolve);
      d_pressureBC->problemSetup(press_db);
      // compute density and other dependent properties
      d_pressureBC->streamMixturefraction.d_initEnthalpy=true;
      d_pressureBC->streamMixturefraction.d_scalarDisp=0.0;
      d_props->computeInletProperties(d_pressureBC->streamMixturefraction, 
          d_pressureBC->calcStream);
      if (d_calcExtraScalars) {

        ProblemSpecP extra_scalar_db = press_db->findBlock("ExtraScalars");
        for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {

          string curr_name = d_extraScalars->at(i)->getScalarName();

          for (ProblemSpecP scalar_db = extra_scalar_db->findBlock("scalar"); 
               scalar_db != 0; scalar_db = scalar_db->findNextBlock("scalar")){
            std::string scalar_name;
            scalar_db->getAttribute("label",scalar_name);

            if (scalar_name == curr_name){
              char* value;
              string svalue;  
              double dvalue = 0; 
              scalar_db->getAttribute("value", svalue);
              value = &svalue[0]; 
              dvalue = std::atof(value);                 

              d_extraScalarBC* bc = scinew d_extraScalarBC;
              bc->d_scalar_name = curr_name;
              bc->d_scalarBC_value = dvalue;
              bc->d_BC_ID = total_cellTypes;
              d_extraScalarBCs.push_back(bc);

            }
          }
        }
      }
/*      if (d_calcExtraScalars) {
        ProblemSpecP extra_scalar_db = press_db->findBlock("ExtraScalars");
        for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
          double value;
          string name = d_extraScalars->at(i)->getScalarName();
          extra_scalar_db->require(name, value);
          d_extraScalarBC* bc = scinew d_extraScalarBC;
          bc->d_scalar_name = name;
          bc->d_scalarBC_value = value;
          bc->d_BC_ID = total_cellTypes;
          d_extraScalarBCs.push_back(bc);
        }
      }*/
      ++total_cellTypes;
    }
    else {
      proc0cout << "Pressure boundary not specified"<< endl;
      d_pressureBoundary = false;
    }

    if (ProblemSpecP outlet_db = db->findBlock("OutletBC")) {
      d_outletBoundary = true;
      d_outletBC = scinew FlowOutlet(total_cellTypes, d_calcVariance,
          d_reactingScalarSolve);
      d_outletBC->problemSetup(outlet_db);
      // compute density and other dependent properties
      d_outletBC->streamMixturefraction.d_initEnthalpy=true;
      d_outletBC->streamMixturefraction.d_scalarDisp=0.0;
      d_props->computeInletProperties(d_outletBC->streamMixturefraction, 
          d_outletBC->calcStream);
      if (d_calcExtraScalars) {

        ProblemSpecP extra_scalar_db = outlet_db->findBlock("ExtraScalars");
        for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {

          string curr_name = d_extraScalars->at(i)->getScalarName();

          for (ProblemSpecP scalar_db = extra_scalar_db->findBlock("scalar"); 
               scalar_db != 0; scalar_db = scalar_db->findNextBlock("scalar")){
            std::string scalar_name;
            scalar_db->getAttribute("label",scalar_name);

            if (scalar_name == curr_name){
              char* value;
              string svalue;  
              double dvalue = 0; 
              scalar_db->getAttribute("value", svalue);
              value = &svalue[0]; 
              dvalue = std::atof(value);                 

              d_extraScalarBC* bc = scinew d_extraScalarBC;
              bc->d_scalar_name = curr_name;
              bc->d_scalarBC_value = dvalue;
              bc->d_BC_ID = total_cellTypes;
              d_extraScalarBCs.push_back(bc);

            }
          }
        }
      }
/*      if (d_calcExtraScalars) {
        ProblemSpecP extra_scalar_db = outlet_db->findBlock("ExtraScalars");
        for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
          double value;
          string name = d_extraScalars->at(i)->getScalarName();
          extra_scalar_db->require(name, value);
          d_extraScalarBC* bc = scinew d_extraScalarBC;
          bc->d_scalar_name = name;
          bc->d_scalarBC_value = value;
          bc->d_BC_ID = total_cellTypes;
          d_extraScalarBCs.push_back(bc);
        }
      }*/
      ++total_cellTypes;
    }
    else {
      proc0cout << "Outlet boundary not specified"<<endl;
      d_outletBoundary = false;
    }

    if (ProblemSpecP intrusion_db = db->findBlock("intrusions")) {
      d_intrusionBoundary = true;
      d_intrusionBC = scinew IntrusionBdry(total_cellTypes);
      d_intrusionBC->problemSetup(intrusion_db);
      ++total_cellTypes;
    }
    else {
      proc0cout << "Intrusion boundary not specified"<<endl;
      d_intrusionBoundary = false;
    }
  }

  d_mmWallID = -10; // invalid cell type
  // if multimaterial then add an id for multimaterial wall
  if (d_MAlab){ 
    d_mmWallID = total_cellTypes;
  }
  if ((d_MAlab)&&(d_intrusionBoundary)){
    d_mmWallID = d_intrusionBC->d_cellTypeID;
  }
  //adding mms access
  if (d_doMMS) {

    ProblemSpecP params_non_constant = params;
    const ProblemSpecP params_root = params_non_constant->getRootNode();
    ProblemSpecP db_mmsblock=params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("MMS");
    
    if(!db_mmsblock->getAttribute("whichMMS",d_mms))
      d_mms="constantMMS";

    if (d_mms == "constantMMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("constantMMS");
      db_whichmms->getWithDefault("cu",cu,1.0);
      db_whichmms->getWithDefault("cv",cv,1.0);
      db_whichmms->getWithDefault("cw",cw,1.0);
      db_whichmms->getWithDefault("cp",cp,1.0);
      db_whichmms->getWithDefault("phi0",phi0,0.5);
    }
   else if (d_mms == "almgrenMMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("almgrenMMS");
      db_whichmms->getWithDefault("amplitude",amp,0.0);
      db_whichmms->require("viscosity",d_viscosity);
    }
    else
      throw InvalidValue("current MMS "
                         "not supported: " + d_mms, __FILE__, __LINE__);
  }
}

//****************************************************************************
// schedule the initialization of cell types
//****************************************************************************
void 
BoundaryCondition::sched_cellTypeInit(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::cellTypeInit",
                          this, &BoundaryCondition::cellTypeInit);
  tsk->computes(d_lab->d_cellTypeLabel);
  sched->addTask(tsk, patches, matls);

}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::cellTypeInit(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<int> cellType;
    new_dw->allocateAndPut(cellType, d_lab->d_cellTypeLabel, indx, patch);

    IntVector domLo = cellType.getFortLowIndex();
    IntVector domHi = cellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;
 
    // initialize CCVariable to -1 which corresponds to flowfield
    // fort_celltypeinit(idxLo, idxHi, cellType, d_flowfieldCellTypeVal);
    cellType.initialize(-1);
    
    // Find the geometry of the patch
    Box patchBox = patch->getExtraBox();

    int celltypeval;
    // initialization for pressure boundary
    if (d_pressureBoundary) {
      int nofGeomPieces = (int)d_pressureBC->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_pressureBC->d_geomPiece[ii];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchBox);
        // check for another geometry
        if (!(b.degenerate())) {
          CellIterator iter = patch->getCellCenterIterator(b);
          IntVector idxLo = iter.begin();
          IntVector idxHi = iter.end() - IntVector(1,1,1);
          celltypeval = d_pressureBC->d_cellTypeID;
          fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
        }
      }
    }
    // wall boundary type
    if (d_wallBoundary) {
      int nofGeomPieces = (int)d_wallBdry->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_wallBdry->d_geomPiece[ii];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchBox);
        // check for another geometry
        if (!(b.degenerate())) {
          /*CellIterator iter = patch->getCellCenterIterator(b);
          IntVector idxLo = iter.begin();
          IntVector idxHi = iter.end() - IntVector(1,1,1);
          celltypeval = d_wallBdry->d_cellTypeID;
          fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);*/
          for (CellIterator iter = patch->getCellCenterIterator(b);
               !iter.done(); iter++) {
            Point p = patch->cellPosition(*iter);
            if (piece->inside(p)) 
            cellType[*iter] = d_wallBdry->d_cellTypeID;
          }
        }
      }
    }
    // initialization for outlet boundary
    if (d_outletBoundary) {
      int nofGeomPieces = (int)d_outletBC->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_outletBC->d_geomPiece[ii];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchBox);
        // check for another geometry
        if (!(b.degenerate())) {
          CellIterator iter = patch->getCellCenterIterator(b);
          IntVector idxLo = iter.begin();
          IntVector idxHi = iter.end() - IntVector(1,1,1);
          celltypeval = d_outletBC->d_cellTypeID;
          fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);
        }
      }
    }
    // set boundary type for inlet flow field
    if (d_inletBoundary) {
      for (int ii = 0; ii < d_numInlets; ii++) {
        int nofGeomPieces = (int)d_flowInlets[ii]->d_geomPiece.size();
        for (int jj = 0; jj < nofGeomPieces; jj++) {
          GeometryPieceP  piece = d_flowInlets[ii]->d_geomPiece[jj];
          Box geomBox = piece->getBoundingBox();
          Box b = geomBox.intersect(patchBox);
          // check for another geometry
          if (b.degenerate())
            continue; // continue the loop for other inlets
            // iterates thru box b, converts from geometry space to index space
            // make sure this works
          /*CellIterator iter = patch->getCellCenterIterator(b);
          IntVector idxLo = iter.begin();
          IntVector idxHi = iter.end() - IntVector(1,1,1);
          celltypeval = d_flowInlets[ii].d_cellTypeID;
          fort_celltypeinit(idxLo, idxHi, cellType, celltypeval);*/
          for (CellIterator iter = patch->getCellCenterIterator(b);
               !iter.done(); iter++) {
            Point p = patch->cellPosition(*iter);
            if (piece->inside(p)) 
              cellType[*iter] = d_flowInlets[ii]->d_cellTypeID;
          }
        }
      }
    }
    if (d_intrusionBoundary) {
      Box patchInteriorBox = patch->getBox();
      int nofGeomPieces = (int)d_intrusionBC->d_geomPiece.size();
      for (int ii = 0; ii < nofGeomPieces; ii++) {
        GeometryPieceP  piece = d_intrusionBC->d_geomPiece[ii];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchInteriorBox);
        if (!(b.degenerate())) {
          for (CellIterator iter = patch->getCellCenterIterator(b);!iter.done(); iter++) {
            Point p = patch->cellPosition(*iter);
            if (piece->inside(p)) 
              cellType[*iter] = d_intrusionBC->d_cellTypeID;
          }
        }
      }
    }
  }
}  


//****************************************************************************
// copy_stencil7:
//   Copies data into and out of the stencil7 arrays so the fortran BC routines can 
//   handle it.
//****************************************************************************
template<class V, class T> void
BoundaryCondition::copy_stencil7(DataWarehouse* new_dw,
                                const Patch* patch,
                                const string& whichWay,
                                CellIterator iter,
                                V& A,
                                T& AP,
                                T& AE,
                                T& AW,
                                T& AN,
                                T& AS,
                                T& AT,
                                T& AB)
{
  if (whichWay == "copyInto"){
  
    new_dw->allocateTemporary(AP,patch);
    new_dw->allocateTemporary(AE,patch);
    new_dw->allocateTemporary(AW,patch);
    new_dw->allocateTemporary(AN,patch);
    new_dw->allocateTemporary(AS,patch);
    new_dw->allocateTemporary(AT,patch);
    new_dw->allocateTemporary(AB,patch);
    
    for(; !iter.done();iter++) {
      IntVector c = *iter;
      AP[c] = A[c].p;
      AE[c] = A[c].e;
      AW[c] = A[c].w;
      AN[c] = A[c].n;
      AS[c] = A[c].s;
      AT[c] = A[c].t;
      AB[c] = A[c].b;
    }
  }else{
    for(; !iter.done();iter++) { 
      IntVector c = *iter;
      A[c].p = AP[c];
      A[c].e = AE[c];
      A[c].w = AW[c];
      A[c].n = AN[c];
      A[c].s = AS[c];
      A[c].t = AT[c];
      A[c].b = AB[c];
    }
  }
}


// for multimaterial
//****************************************************************************
// schedule the initialization of mm wall cell types
//****************************************************************************
void 
BoundaryCondition::sched_mmWallCellTypeInit(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls,
                                            bool fixCellType)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::mmWallCellTypeInit",
                          this,
                          &BoundaryCondition::mmWallCellTypeInit,
                          fixCellType);
  
  

  // New DW warehouse variables to calculate cell types if we are
  // recalculating cell types and resetting void fractions

  //  double time = d_lab->d_sharedState->getElapsedTime();
  bool recalculateCellType = false;
  int dwnumber = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  
  if (dwnumber < 2 || !fixCellType){
    recalculateCellType = true;
  }

  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::OldDW, d_lab->d_mmgasVolFracLabel,   gn, 0);
  tsk->requires(Task::OldDW, d_lab->d_mmcellTypeLabel,     gn, 0);
  tsk->requires(Task::OldDW, d_MAlab->mmCellType_MPMLabel, gn, 0);
  if (d_cutCells)
    tsk->requires(Task::OldDW, d_MAlab->mmCellType_CutCellLabel, gn, 0);

  if (recalculateCellType) {

    tsk->requires(Task::NewDW, d_MAlab->void_frac_CCLabel, gn, 0);
    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel,     gn, 0);

    tsk->computes(d_lab->d_mmgasVolFracLabel);
    tsk->computes(d_lab->d_mmcellTypeLabel);
    tsk->computes(d_MAlab->mmCellType_MPMLabel);

    if (d_cutCells){
      tsk->computes(d_MAlab->mmCellType_CutCellLabel);
    }
    tsk->modifies(d_MAlab->void_frac_MPM_CCLabel);
    if (d_cutCells){
      tsk->modifies(d_MAlab->void_frac_CutCell_CCLabel);
    }
  }
  else {

    tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, gn, 0);
    tsk->computes(d_lab->d_mmgasVolFracLabel);
    tsk->computes(d_lab->d_mmcellTypeLabel);
    tsk->computes(d_MAlab->mmCellType_MPMLabel);
    if (d_cutCells){
      tsk->computes(d_MAlab->mmCellType_CutCellLabel);
    }
  }
  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::mmWallCellTypeInit(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset*,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      bool fixCellType)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    
    bool recalculateCellType = false;
    int dwnumber = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
    if (dwnumber < 2 || !fixCellType){
      recalculateCellType = true;
    }
    
    // New DW void fraction to decide cell types and reset void fractions

    constCCVariable<double> voidFrac;
    constCCVariable<int> cellType;

    CCVariable<double> mmGasVolFrac;
    CCVariable<int> mmCellType;
    CCVariable<int> mmCellTypeMPM;
    CCVariable<int> mmCellTypeCutCell;
    CCVariable<double> voidFracMPM;
    CCVariable<double> voidFracCutCell;

    constCCVariable<double> oldGasVolFrac;
    constCCVariable<int> mmCellTypeOld;
    constCCVariable<int> mmCellTypeMPMOld;
    constCCVariable<int> mmCellTypeCutCellOld;
    constCCVariable<double> voidFracMPMOld;
    constCCVariable<double> voidFracCutCellOld;
    
    
    Ghost::GhostType  gn = Ghost::None;
    old_dw->get(oldGasVolFrac,    d_lab->d_mmgasVolFracLabel,   indx, patch,gn, 0);
    old_dw->get(mmCellTypeOld,    d_lab->d_mmcellTypeLabel,     indx, patch,gn, 0);
    old_dw->get(mmCellTypeMPMOld, d_MAlab->mmCellType_MPMLabel, indx, patch,gn, 0);
    if (d_cutCells){
      old_dw->get(mmCellTypeCutCellOld, d_MAlab->mmCellType_CutCellLabel, indx, patch,gn, 0);
    }

    if (recalculateCellType) {
      new_dw->get(voidFrac, d_MAlab->void_frac_CCLabel, indx, patch,gn, 0);
      old_dw->get(cellType, d_lab->d_cellTypeLabel,     indx, patch,gn, 0);

      new_dw->allocateAndPut(mmGasVolFrac,  d_lab->d_mmgasVolFracLabel,   indx, patch);
      new_dw->allocateAndPut(mmCellType,    d_lab->d_mmcellTypeLabel,     indx, patch);
      new_dw->allocateAndPut(mmCellTypeMPM, d_MAlab->mmCellType_MPMLabel, indx, patch);
      if (d_cutCells){
        new_dw->allocateAndPut(mmCellTypeCutCell, d_MAlab->mmCellType_CutCellLabel, indx, patch);
      }

      new_dw->getModifiable(voidFracMPM, d_MAlab->void_frac_MPM_CCLabel, indx, patch);
      if (d_cutCells){
        new_dw->getModifiable(voidFracCutCell, d_MAlab->void_frac_CutCell_CCLabel, indx, patch);
      }
    }
    else {
      old_dw->get(cellType,                 d_lab->d_cellTypeLabel,       indx, patch,gn, 0);
      new_dw->allocateAndPut(mmGasVolFrac,  d_lab->d_mmgasVolFracLabel,   indx, patch);
      new_dw->allocateAndPut(mmCellType,    d_lab->d_mmcellTypeLabel,     indx, patch);
      new_dw->allocateAndPut(mmCellTypeMPM, d_MAlab->mmCellType_MPMLabel, indx, patch);
      if (d_cutCells){
        new_dw->allocateAndPut(mmCellTypeCutCell, d_MAlab->mmCellType_CutCellLabel, indx, patch);
      }
    }

    IntVector domLo = mmCellType.getFortLowIndex();
    IntVector domHi = mmCellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;

    if (recalculateCellType) {

      mmGasVolFrac.copyData(voidFrac);
      mmCellType.copyData(cellType);
      mmCellTypeMPM.copyData(cellType);
      if (d_cutCells){
        mmCellTypeCutCell.copyData(cellType);
      }
      // resets old mmwall type back to flow field and sets cells with void fraction
      // of less than .5 to mmWall

      if (d_intrusionBoundary) {
        for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);

              if (cellType[currCell] == d_mmWallID) {
                mmGasVolFrac[currCell] = 0.0;
                voidFracMPM[currCell] = 0.0;
              }
              else {
                mmGasVolFrac[currCell] = 1.0;
                voidFracMPM[currCell] = 1.0;
              }
            }
          }
        }
      }
      else {
        fort_mmcelltypeinit(idxLo, idxHi, mmGasVolFrac, mmCellType, d_mmWallID,
                                  d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  

        fort_mmcelltypeinit(idxLo, idxHi, voidFracMPM, mmCellTypeMPM, d_mmWallID,
                                  d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  
        if (d_cutCells)
          fort_mmcelltypeinit(idxLo, idxHi, voidFracCutCell, mmCellTypeCutCell, d_mmWallID,
                              d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);  
      }
    }
    else {
      mmGasVolFrac.copyData(oldGasVolFrac);
      mmCellType.copyData(mmCellTypeOld);
      mmCellTypeMPM.copyData(mmCellTypeMPMOld);
      if (d_cutCells){
        mmCellTypeCutCell.copyData(mmCellTypeCutCellOld);
      }
    }
  }  
}

// for multimaterial
//****************************************************************************
// schedule the initialization of mm wall cell types for the very first
// time step
//****************************************************************************
void 
BoundaryCondition::sched_mmWallCellTypeInit_first(SchedulerP& sched, 
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::mmWallCellTypeInit_first",
                          this,
                          &BoundaryCondition::mmWallCellTypeInit_first);
  
  Ghost::GhostType  gn = Ghost::None;
  tsk->requires(Task::NewDW, d_MAlab->void_frac_CCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,     gn, 0);
  tsk->computes(d_lab->d_mmcellTypeLabel);
  tsk->modifies(d_lab->d_mmgasVolFracLabel);
  
  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::mmWallCellTypeInit_first(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset*,
                                            DataWarehouse* ,
                                            DataWarehouse* new_dw)        
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    
    constCCVariable<int> cellType;
    constCCVariable<double> voidFrac;
    CCVariable<int> mmcellType;
    CCVariable<double> mmvoidFrac;
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(cellType,              d_lab->d_cellTypeLabel,      indx, patch, gn, 0);
    new_dw->get(voidFrac,              d_MAlab->void_frac_CCLabel,  indx, patch, gn, 0);
    new_dw->allocateAndPut(mmcellType, d_lab->d_mmcellTypeLabel,    indx, patch);
    new_dw->getModifiable(mmvoidFrac,  d_lab->d_mmgasVolFracLabel,  indx, patch);
    mmcellType.copyData(cellType);
    mmvoidFrac.copyData(voidFrac);
        
    IntVector domLo = mmcellType.getFortLowIndex();
    IntVector domHi = mmcellType.getFortHighIndex();
    IntVector idxLo = domLo;
    IntVector idxHi = domHi;

    // resets old mmwall type back to flow field and sets cells with void fraction
    // of less than .01 to mmWall

    fort_mmcelltypeinit(idxLo, idxHi, mmvoidFrac, mmcellType, d_mmWallID,
                        d_flowfieldCellTypeVal, MM_CUTOFF_VOID_FRAC);

    // allocateAndPut instead:
    /* new_dw->put(mmcellType, d_lab->d_mmcellTypeLabel, indx, patch); */;
    // save in arches label
    // allocateAndPut instead:
    /* new_dw->put(mmvoidFrac, d_lab->d_mmgasVolFracLabel, indx, patch); */;
  }  
}


    
//****************************************************************************
// Actual initialization of celltype
//****************************************************************************
void 
BoundaryCondition::computeInletFlowArea(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse*,
                                        DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Create the cellType variable
    constCCVariable<int> cellType;
    
    // Get the cell type data from the old_dw
    // **WARNING** numGhostcells, Ghost::None may change in the future
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::None, 0);
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, indx, patch)){ 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }else{ 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // Get the low and high index for the variable and the patch
    IntVector domLo = cellType.getFortLowIndex();
    IntVector domHi = cellType.getFortHighIndex();
    
    // Get the geometry of the patch
    Box patchBox = patch->getExtraBox();
    
    // Go thru the number of inlets
    for (int ii = 0; ii < d_numInlets; ii++) {
      
      // Loop thru the number of geometry pieces in each inlet
      int nofGeomPieces = (int)d_flowInlets[ii]->d_geomPiece.size();
      for (int jj = 0; jj < nofGeomPieces; jj++) {
        
        // Intersect the geometry piece with the patch box
        GeometryPieceP  piece = d_flowInlets[ii]->d_geomPiece[jj];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchBox);
        // check for another geometry
        if (b.degenerate()){
          new_dw->put(sum_vartype(0),d_flowInlets[ii]->d_area_label);
          continue; // continue the loop for other inlets
        }
        
        // iterates thru box b, converts from geometry space to index space
        // make sure this works
        CellIterator iter = patch->getCellCenterIterator(b);
        IntVector idxLo = iter.begin();
        IntVector idxHi = iter.end() - IntVector(1,1,1);
        
        // Calculate the inlet area
        double inlet_area;
        int cellid = d_flowInlets[ii]->d_cellTypeID;
        
        bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
        bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
        bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
        bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
        bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
        bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

        fort_areain(domLo, domHi, idxLo, idxHi, cellinfo->sew, cellinfo->sns,
                    cellinfo->stb, inlet_area, cellType, cellid,
                    d_flowfieldCellTypeVal,
                    xminus, xplus, yminus, yplus, zminus, zplus);
        
        // Write the inlet area to the old_dw
        new_dw->put(sum_vartype(inlet_area),d_flowInlets[ii]->d_area_label);
      }
    }
  }
}
    
//****************************************************************************
// Schedule computes inlet areas
// computes inlet area for inlet bc
//****************************************************************************
void 
BoundaryCondition::sched_calculateArea(SchedulerP& sched, 
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::calculateArea",this,
                           &BoundaryCondition::computeInletFlowArea);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None, 0);
  // ***warning checkpointing
  //      tsk->computes(old_dw, d_lab->d_cellInfoLabel, indx, patch);
  for (int ii = 0; ii < d_numInlets; ii++){ 
    tsk->computes(d_flowInlets[ii]->d_area_label);
  }

  sched->addTask(tsk, patches, matls);
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::calculateArea",
                           patch, old_dw, new_dw, this,
                           &BoundaryCondition::computeInletFlowArea);
      int indx = 0;
      tsk->requires(old_dw, d_lab->d_cellTypeLabel, indx, patch, Ghost::None,
                    Arches::ZEROGHOSTCELLS);
      // ***warning checkpointing
      //      tsk->computes(old_dw, d_lab->d_cellInfoLabel, indx, patch);
      for (int ii = 0; ii < d_numInlets; ii++) {
        // make it simple by adding matlindex for reduction vars
        tsk->computes(old_dw, d_flowInlets[ii].d_area_label);
      }
      sched->addTask(tsk);
    }
  }
#endif
}

//****************************************************************************
// Schedule set profile
// assigns flat velocity profiles for primary and secondary inlets
// Also sets flat profiles for density
//****************************************************************************
void 
BoundaryCondition::sched_setProfile(SchedulerP& sched, 
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::setProfile",
                          this,
                          &BoundaryCondition::setProfile);

  // This task requires cellTypeVariable and areaLabel for inlet boundary
  // Also densityIN, [u,v,w] velocityIN, scalarIN
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None, 0);
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::NewDW, d_flowInlets[ii]->d_area_label);
  }
  if (d_enthalpySolve) {
    tsk->modifies(d_lab->d_enthalpySPLabel);
  }
  if (d_reactingScalarSolve) {
    tsk->modifies(d_lab->d_reactscalarSPLabel);
  }
    
  // This task computes new density, uVelocity, vVelocity and wVelocity, scalars
  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);
  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  tsk->modifies(d_lab->d_scalarSPLabel);

  for (int ii = 0; ii < d_numInlets; ii++){ 
    tsk->computes(d_flowInlets[ii]->d_flowRate_label);
  }

  if (d_calcExtraScalars){
    for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++){
      tsk->modifies(d_extraScalars->at(i)->getScalarLabel());
    }
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually set flat profile at flow inlet boundary
//****************************************************************************
void 
BoundaryCondition::setProfile(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse*,
                              DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    CCVariable<double> density;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;
    CCVariable<double> scalar;
    CCVariable<double> reactscalar;
    CCVariable<double> enthalpy;

    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(uVelRhoHat, d_lab->d_uVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(vVelRhoHat, d_lab->d_vVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(wVelRhoHat, d_lab->d_wVelRhoHatLabel, indx, patch);
    
    // get cellType, density and velocity
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::None,
                Arches::ZEROGHOSTCELLS);
    new_dw->getModifiable(density, d_lab->d_densityCPLabel, indx, patch);
    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, indx, patch);
      // reactscalar will be zero at the boundaries, so no further calculation
      // is required.
    if (d_reactingScalarSolve){
      new_dw->getModifiable(reactscalar, d_lab->d_reactscalarSPLabel, indx, patch);
    }
    
    if (d_enthalpySolve){ 
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel, indx, patch);
    }

    IntVector idxLo = patch->getFortranCellLowIndex__New();
    IntVector idxHi = patch->getFortranCellHighIndex__New();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    // loop thru the flow inlets to set all the components of velocity and density
    if (d_inletBoundary) {
      double time = 0.0; 
      for (int indx = 0; indx < d_numInlets; indx++) {
        sum_vartype area_var;
        new_dw->get(area_var, d_flowInlets[indx]->d_area_label);
        double area = area_var;
        double actual_flow_rate;
      
        // Get a copy of the current flow inlet
        // check if given patch intersects with the inlet boundary of type index
        FlowInlet* fi = d_flowInlets[indx];
        //cerr << " inlet area" << area << " flowrate" << fi.flowRate << endl;
        //cerr << "density=" << fi.calcStream.d_density << endl;
        fort_profv(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
                   cellType, area, fi->d_cellTypeID, fi->flowRate, fi->inletVel,
                   fi->calcStream.d_density,
                   xminus, xplus, yminus, yplus, zminus, zplus, time,
                   fi->d_ramping_inlet_flowrate, actual_flow_rate);

        d_flowInlets[indx]->flowRate = actual_flow_rate;
        new_dw->put(delt_vartype(actual_flow_rate),
                    d_flowInlets[indx]->d_flowRate_label);

        fort_profscalar(idxLo, idxHi, density, cellType,
                        fi->calcStream.d_density, fi->d_cellTypeID,
                        xminus, xplus, yminus, yplus, zminus, zplus);
        if (d_enthalpySolve){
          fort_profscalar(idxLo, idxHi, enthalpy, cellType,
                          fi->calcStream.d_enthalpy, fi->d_cellTypeID,
                          xminus, xplus, yminus, yplus, zminus, zplus);
        }
      }
    }

    if (d_pressureBoundary) {
      // set density
      fort_profscalar(idxLo, idxHi, density, cellType,
                      d_pressureBC->calcStream.d_density,
                      d_pressureBC->d_cellTypeID,
                      xminus, xplus, yminus, yplus, zminus, zplus);
      if (d_enthalpySolve){
        fort_profscalar(idxLo, idxHi, enthalpy, cellType,
                        d_pressureBC->calcStream.d_enthalpy,
                        d_pressureBC->d_cellTypeID,
                        xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }

    if (d_outletBoundary) {
      // set density
      fort_profscalar(idxLo, idxHi, density, cellType,
                      d_outletBC->calcStream.d_density,
                      d_outletBC->d_cellTypeID,
                      xminus, xplus, yminus, yplus, zminus, zplus);
      if (d_enthalpySolve){
        fort_profscalar(idxLo, idxHi, enthalpy, cellType,
                        d_outletBC->calcStream.d_enthalpy,
                        d_outletBC->d_cellTypeID,
                        xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }

    if (d_inletBoundary) {
      for (int ii = 0; ii < d_numInlets; ii++) {
        double scalarValue = 
               d_flowInlets[ii]->streamMixturefraction.d_mixVars[0];
        fort_profscalar(idxLo, idxHi, scalar, cellType,
                        scalarValue, d_flowInlets[ii]->d_cellTypeID,
                        xminus, xplus, yminus, yplus, zminus, zplus);
        double reactScalarValue;
        if (d_reactingScalarSolve) {
          reactScalarValue = 
               d_flowInlets[ii]->streamMixturefraction.d_rxnVars[0];
          fort_profscalar(idxLo, idxHi, reactscalar, cellType,
                          reactScalarValue, d_flowInlets[ii]->d_cellTypeID,
                          xminus, xplus, yminus, yplus, zminus, zplus);
        }
      }
    }

    if (d_pressureBoundary) {
      double scalarValue = 
             d_pressureBC->streamMixturefraction.d_mixVars[0];
      fort_profscalar(idxLo, idxHi, scalar, cellType, scalarValue,
                      d_pressureBC->d_cellTypeID,
                      xminus, xplus, yminus, yplus, zminus, zplus);
      double reactScalarValue;
      if (d_reactingScalarSolve) {
        reactScalarValue = 
             d_pressureBC->streamMixturefraction.d_rxnVars[0];
        fort_profscalar(idxLo, idxHi, reactscalar, cellType,
                        reactScalarValue, d_pressureBC->d_cellTypeID,
                        xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }

    if (d_outletBoundary) {
      double scalarValue = 
             d_outletBC->streamMixturefraction.d_mixVars[0];
      fort_profscalar(idxLo, idxHi, scalar, cellType, scalarValue,
                      d_outletBC->d_cellTypeID,
                      xminus, xplus, yminus, yplus, zminus, zplus);
      double reactScalarValue;
      if (d_reactingScalarSolve) {
        reactScalarValue = 
             d_outletBC->streamMixturefraction.d_rxnVars[0];
        fort_profscalar(idxLo, idxHi, reactscalar, cellType,
                        reactScalarValue, d_outletBC->d_cellTypeID,
                        xminus, xplus, yminus, yplus, zminus, zplus);
      }
    }
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 

    if (d_calcExtraScalars) {
      for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
        string extra_scalar_name = d_extraScalars->at(i)->getScalarName();
        CCVariable<double> extra_scalar;
        new_dw->getModifiable(extra_scalar,
                              d_extraScalars->at(i)->getScalarLabel(),indx, patch);

        if (d_inletBoundary) {
          for (int ii = 0; ii < d_numInlets; ii++) {
            int BC_ID = d_flowInlets[ii]->d_cellTypeID;

            double extra_scalar_value;
            for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++)
              if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                  (d_extraScalarBCs[j]->d_BC_ID) == BC_ID)
                extra_scalar_value = d_extraScalarBCs[j]->d_scalarBC_value;

                fort_profscalar(idxLo, idxHi, extra_scalar, cellType,
                                extra_scalar_value, BC_ID,
                                xminus, xplus, yminus, yplus, zminus, zplus);
          }
        }

        if (d_pressureBoundary) {
          int BC_ID = d_pressureBC->d_cellTypeID;

          double extra_scalar_value;
          for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++){
            if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                (d_extraScalarBCs[j]->d_BC_ID) == BC_ID){
              extra_scalar_value = d_extraScalarBCs[j]->d_scalarBC_value;
            }
          }
          fort_profscalar(idxLo, idxHi, extra_scalar, cellType,
                          extra_scalar_value, BC_ID,
                          xminus, xplus, yminus, yplus, zminus, zplus);
        }

        if (d_outletBoundary) {
          int BC_ID = d_outletBC->d_cellTypeID;

          double extra_scalar_value;
          for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++){
            if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                (d_extraScalarBCs[j]->d_BC_ID) == BC_ID){
              extra_scalar_value = d_extraScalarBCs[j]->d_scalarBC_value;
            }
          }

          fort_profscalar(idxLo, idxHi, extra_scalar, cellType,
                          extra_scalar_value, BC_ID,
                                xminus, xplus, yminus, yplus, zminus, zplus);
        }
      }
    }
  }
}

//****************************************************************************
// Actually calculate the velocity BC
//****************************************************************************
void 
BoundaryCondition::velocityBC(const Patch* patch,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars) 
{
  int wall_celltypeval = wallCellType();
 
  // computes momentum source term due to wall
  // uses total viscosity for wall source, not just molecular viscosity
  //double molViscosity = d_physicalConsts->getMolecularViscosity();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
  
  //__________________________________
  //    X Dir
  IntVector idxLo = patch->getSFCXFORTLowIndex();
  IntVector idxHi = patch->getSFCXFORTHighIndex();
  fort_bcuvel(idxLo, idxHi,
              vars->uVelocityCoeff[Arches::AE],
              vars->uVelocityCoeff[Arches::AW], 
              vars->uVelocityCoeff[Arches::AN], 
              vars->uVelocityCoeff[Arches::AS], 
              vars->uVelocityCoeff[Arches::AT], 
              vars->uVelocityCoeff[Arches::AB], 
              vars->uVelNonlinearSrc, vars->uVelLinearSrc,
              vars->uVelocityConvectCoeff[Arches::AE],
              vars->uVelocityConvectCoeff[Arches::AW],
              vars->uVelocityConvectCoeff[Arches::AN],
              vars->uVelocityConvectCoeff[Arches::AS],
              vars->uVelocityConvectCoeff[Arches::AT],
              vars->uVelocityConvectCoeff[Arches::AB],
              constvars->cellType, wall_celltypeval, 
              cellinfo->sewu, cellinfo->sns, cellinfo->stb,
              constvars->viscosity,
              cellinfo->yy, cellinfo->yv, cellinfo->zz, cellinfo->zw,
              xminus, xplus, yminus, yplus, zminus, zplus);

  //__________________________________
  //    Y Dir
  idxLo = patch->getSFCYFORTLowIndex();
  idxHi = patch->getSFCYFORTHighIndex();

  fort_bcvvel(idxLo, idxHi,
              vars->vVelocityCoeff[Arches::AE],
              vars->vVelocityCoeff[Arches::AW], 
              vars->vVelocityCoeff[Arches::AN], 
              vars->vVelocityCoeff[Arches::AS], 
              vars->vVelocityCoeff[Arches::AT], 
              vars->vVelocityCoeff[Arches::AB], 
              vars->vVelNonlinearSrc, vars->vVelLinearSrc,
              vars->vVelocityConvectCoeff[Arches::AE],
              vars->vVelocityConvectCoeff[Arches::AW],
              vars->vVelocityConvectCoeff[Arches::AN],
              vars->vVelocityConvectCoeff[Arches::AS],
              vars->vVelocityConvectCoeff[Arches::AT],
              vars->vVelocityConvectCoeff[Arches::AB],
              constvars->cellType, wall_celltypeval,
              cellinfo->sew, cellinfo->snsv, cellinfo->stb,
              constvars->viscosity,
              cellinfo->xx, cellinfo->xu, cellinfo->zz, cellinfo->zw,
              xminus, xplus, yminus, yplus, zminus, zplus);

  //__________________________________
  //    Z Dir
  idxLo = patch->getSFCZFORTLowIndex();
  idxHi = patch->getSFCZFORTHighIndex();

  fort_bcwvel(idxLo, idxHi,
              vars->wVelocityCoeff[Arches::AE],
              vars->wVelocityCoeff[Arches::AW], 
              vars->wVelocityCoeff[Arches::AN], 
              vars->wVelocityCoeff[Arches::AS], 
              vars->wVelocityCoeff[Arches::AT], 
              vars->wVelocityCoeff[Arches::AB], 
              vars->wVelNonlinearSrc, vars->wVelLinearSrc,
              vars->wVelocityConvectCoeff[Arches::AE],
              vars->wVelocityConvectCoeff[Arches::AW],
              vars->wVelocityConvectCoeff[Arches::AN],
              vars->wVelocityConvectCoeff[Arches::AS],
              vars->wVelocityConvectCoeff[Arches::AT],
              vars->wVelocityConvectCoeff[Arches::AB],
              constvars->cellType, wall_celltypeval,
              cellinfo->sew, cellinfo->sns, cellinfo->stbw,
              constvars->viscosity,
              cellinfo->xx, cellinfo->xu, cellinfo->yy, cellinfo->yv,
              xminus, xplus, yminus, yplus, zminus, zplus);
}

//______________________________________________________________________
//  Set the boundary conditions on the pressure stencil.
// This will change when we move to the UCF based boundary conditions

void 
BoundaryCondition::pressureBC(const Patch* patch,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars)
{
  // shortcuts
  int wall_BC = wallCellType();
  int pressure_BC = pressureCellType();
  int outlet_BC = outletCellType();
  // int symmetry_celltypeval = -3;
  
  CCVariable<Stencil7>& A = vars->pressCoeff;
  constCCVariable<int>& cellType = constvars->cellType;
  
  //__________________________________
  //  intrusion Boundary Conditions
  if(d_intrusionBC){
    for(CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter;
      if(cellType[c] == wall_BC){
        A[c].p = 0.0;
        A[c].e = 0.0;
        A[c].w = 0.0;
        A[c].n = 0.0;
        A[c].s = 0.0;
        A[c].t = 0.0;
        A[c].b = 0.0;
        vars->pressNonlinearSrc[c] = 0.0;
        vars->pressLinearSrc[c] = -1.0;
      }
    }
  }

  //__________________________________
  //  Pressure, outlet, and wall BC
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    
    IntVector offset = patch->faceDirection(face);
    
    CellIterator iter = patch->getFaceIterator__New(face, Patch::InteriorFaceCells);
    
    //face:       -x +x -y +y -z +z
    //Stencil 7   w, e, s, n, b, t;
    for(;!iter.done(); iter++){
      IntVector c = *iter;
      IntVector adj = c + offset;
      
      if( cellType[adj] == pressure_BC ||
          cellType[adj] == outlet_BC){
        // dirichlet_BC
        A[c].p = A[c].p +  A[c][face];
        A[c][face] = 0.0;
      }

      if( cellType[adj] == wall_BC){
        // Neumann zero gradient BC
        A[c].p = A[c].p - A[c][face];
        A[c][face] = 0.0;
      }
    }
  }
  
  //__________________________________
  //  Inlets
  // This assumes that all inlets have Neumann (zero gradient) pressure BCs
  for (int ii = 0; ii < d_numInlets; ii++) {
    
    for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
      Patch::FaceType face = *itr;

      IntVector offset = patch->faceDirection(face);

      CellIterator iter = patch->getFaceIterator__New(face, Patch::InteriorFaceCells);

      //face:       -x +x -y +y -z +z
      //Stencil 7   w, e, s, n, b, t;
      for(;!iter.done(); iter++){
        IntVector c = *iter;
        IntVector adj = c + offset;
        
        if( cellType[adj] == d_flowInlets[ii]->d_cellTypeID){
          // Neumann zero gradient BC
          
          A[c].p = A[c].p -  A[c][face];
          A[c][face] = 0.0;
        }
      }
    }
  }
}

//****************************************************************************
// Actually compute the scalar bcs
//****************************************************************************
void 
BoundaryCondition::scalarBC(const Patch* patch,
                            ArchesVariables* vars,
                            ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // Get the wall boundary and flow field codes
  int wall_celltypeval = wallCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;


  //fortran call
  fort_bcscalar(idxLo, idxHi,
                vars->scalarCoeff[Arches::AE],
                vars->scalarCoeff[Arches::AW],
                vars->scalarCoeff[Arches::AN],
                vars->scalarCoeff[Arches::AS],
                vars->scalarCoeff[Arches::AT],
                vars->scalarCoeff[Arches::AB],
                vars->scalarNonlinearSrc, vars->scalarLinearSrc,
                vars->scalarConvectCoeff[Arches::AE],
                vars->scalarConvectCoeff[Arches::AW],
                vars->scalarConvectCoeff[Arches::AN],
                vars->scalarConvectCoeff[Arches::AS],
                vars->scalarConvectCoeff[Arches::AT],
                vars->scalarConvectCoeff[Arches::AB],
                vars->scalarDiffusionCoeff[Arches::AE],
                vars->scalarDiffusionCoeff[Arches::AW],
                vars->scalarDiffusionCoeff[Arches::AN], 
                vars->scalarDiffusionCoeff[Arches::AS],
                vars->scalarDiffusionCoeff[Arches::AT],
                vars->scalarDiffusionCoeff[Arches::AB],
                constvars->cellType, wall_celltypeval,
                xminus, xplus, yminus, yplus, zminus, zplus);
}

//______________________________________________________________________
void 
BoundaryCondition::scalarBC__new(const Patch* patch,
                                 ArchesVariables* vars,
                                 ArchesConstVariables* constvars)
{
  //This will be removed once the new boundary condition stuff is online:
  // Like the old code, this only takes care of wall bc's. 
  // Also, like the old code, it only allows for wall in the x-direction

  // Get the wall boundary and flow field codes
  int wall = wallCellType();
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
 
  for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
    IntVector curr = *iter;

    if (constvars->cellType[curr] == wall){
      //interior intrusions
      vars->scalarTotCoef[curr].e = 0.0;
      vars->scalarTotCoef[curr].w = 0.0;
      vars->scalarTotCoef[curr].n = 0.0;
      vars->scalarTotCoef[curr].s = 0.0;
      vars->scalarTotCoef[curr].t = 0.0;
      vars->scalarTotCoef[curr].b = 0.0;
      vars->scalarNonlinearSrc[curr] = 0.0;
      vars->scalarLinearSrc[curr] = -1.0;
 
      vars->scalarConvCoef[curr].e = 0.0;
      vars->scalarConvCoef[curr].w = 0.0;
      vars->scalarConvCoef[curr].n = 0.0;
      vars->scalarConvCoef[curr].s = 0.0;
      vars->scalarConvCoef[curr].t = 0.0;
      vars->scalarConvCoef[curr].b = 0.0;

      vars->scalarDiffCoef[curr].e = 0.0;
      vars->scalarDiffCoef[curr].w = 0.0;
      vars->scalarDiffCoef[curr].n = 0.0;
      vars->scalarDiffCoef[curr].s = 0.0;
      vars->scalarDiffCoef[curr].t = 0.0;
      vars->scalarDiffCoef[curr].b = 0.0;
    }

    if (xminus){
      //domani boundary bc's 
      if (constvars->cellType[curr - IntVector(1,0,0)] == wall){

        vars->scalarTotCoef[curr].w = 0.0;
        vars->scalarDiffCoef[curr].w = 0.0;
        vars->scalarConvCoef[curr].w = 0.0;

      }

    }
 
  }
}


//****************************************************************************
void
BoundaryCondition::intrusionTemperatureBC(const Patch* patch,
                                          constCCVariable<int>& cellType,
                                          CCVariable<double>& temperature)
{
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell = IntVector(colX, colY, colZ);          
        if (cellType[currCell]==d_intrusionBC->d_cellTypeID){
          temperature[currCell] = d_intrusionBC->d_temperature;
        }
      }
    }
  }
}
//______________________________________________________________________
//
void
BoundaryCondition::mmWallTemperatureBC(const Patch* patch,
                                       constCCVariable<int>& cellType,
                                       constCCVariable<double> solidTemp,
                                       CCVariable<double>& temperature,
                                       bool d_energyEx)
{
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

        IntVector currCell = IntVector(colX, colY, colZ);

        if (cellType[currCell]==d_mmWallID) {

          if (d_energyEx) {
            if (d_fixTemp){
              temperature[currCell] = 298.0;
            }else{
              temperature[currCell] = solidTemp[currCell];
            }
          }else{
            temperature[currCell] = 298.0;
          }  //d_energyEx
        }  // wall
      }  // x
    }  // y
  }  // z
}

//______________________________________________________________________
//
void 
BoundaryCondition::intrusionMomExchangeBC(const Patch* patch,
                                          int index, CellInformation* cellinfo,
                                          ArchesVariables* vars,
                                          ArchesConstVariables* constvars)
{
  // Call the fortran routines
  switch(index) {
  case 1:
    intrusionuVelMomExBC(patch, cellinfo, vars, constvars);
    break;
  case 2:
    intrusionvVelMomExBC(patch, cellinfo, vars, constvars);
    break;
  case 3:
    intrusionwVelMomExBC(patch, cellinfo, vars, constvars);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;

  }
}
//______________________________________________________________________
//
void 
BoundaryCondition::intrusionuVelMomExBC(const Patch* patch,
                                        CellInformation* cellinfo,
                                        ArchesVariables* vars,
                                        ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcvelocity_momex(idxLoU, idxHiU,
                          vars->uVelocityCoeff[Arches::AN],
                          vars->uVelocityCoeff[Arches::AS],
                          vars->uVelocityCoeff[Arches::AT],
                          vars->uVelocityCoeff[Arches::AB],
                          vars->uVelLinearSrc,
                          cellinfo->sewu, cellinfo->sns, cellinfo->stb,
                          cellinfo->yy, cellinfo->yv, cellinfo->zz,
                          cellinfo->zw, Viscos, constvars->cellType, 
                          d_intrusionBC->d_cellTypeID, ioff, joff, koff);
}

//______________________________________________________________________
//
void 
BoundaryCondition::intrusionvVelMomExBC(const Patch* patch,
                                        CellInformation* cellinfo,
                                        ArchesVariables* vars,
                                        ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCYFORTLowIndex();
  IntVector idxHiU = patch->getSFCYFORTHighIndex();
  int ioff = 0;
  int joff = 1;
  int koff = 0;
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcvelocity_momex(idxLoU, idxHiU,
                          vars->vVelocityCoeff[Arches::AT],
                          vars->vVelocityCoeff[Arches::AB],
                          vars->vVelocityCoeff[Arches::AE],
                          vars->vVelocityCoeff[Arches::AW],
                          vars->vVelLinearSrc,
                          cellinfo->snsv, cellinfo->stb, cellinfo->sew,
                          cellinfo->zz, cellinfo->zw, cellinfo->xx,
                          cellinfo->xu, Viscos, constvars->cellType, 
                          d_intrusionBC->d_cellTypeID, ioff, joff, koff);
}
//______________________________________________________________________
//
void 
BoundaryCondition::intrusionwVelMomExBC(const Patch* patch,
                                        CellInformation* cellinfo,
                                        ArchesVariables* vars,
                                        ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLoU = patch->getSFCZFORTLowIndex();
  IntVector idxHiU = patch->getSFCZFORTHighIndex();
  int ioff = 0;
  int joff = 0;
  int koff = 1;
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcvelocity_momex(idxLoU, idxHiU,
                          vars->wVelocityCoeff[Arches::AE],
                          vars->wVelocityCoeff[Arches::AW],
                          vars->wVelocityCoeff[Arches::AN],
                          vars->wVelocityCoeff[Arches::AS],
                          vars->wVelLinearSrc,
                          cellinfo->stbw, cellinfo->sew, cellinfo->sns,
                          cellinfo->xx, cellinfo->xu, cellinfo->yy,
                          cellinfo->yv, Viscos, constvars->cellType, 
                          d_intrusionBC->d_cellTypeID, ioff, joff, koff);
}
//______________________________________________________________________
//
void 
BoundaryCondition::intrusionEnergyExBC(const Patch* patch,
                                       CellInformation* cellinfo,
                                       ArchesVariables* vars,
                                       ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  double Viscos = d_physicalConsts->getMolecularViscosity();
  fort_mmbcenthalpy_energyex(idxLo, idxHi,
                             vars->scalarNonlinearSrc,
                             constvars->temperature,
                             constvars->cp,
                             cellinfo->sew, cellinfo->sns, cellinfo->stb,
                             cellinfo->xx, cellinfo->xu,
                             cellinfo->yy, cellinfo->yv,
                             cellinfo->zz, cellinfo->zw,
                             Viscos,
                             constvars->cellType, 
                             d_intrusionBC->d_cellTypeID);
}

//______________________________________________________________________
// applies multimaterial bc's for scalars and pressure
void
BoundaryCondition::intrusionScalarBC( const Patch* patch,
                                      CellInformation*,
                                      ArchesVariables* vars,
                                      ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  //fortran call
  fort_mmwallbc(idxLo, idxHi,
                vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW],
                vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS],
                vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB],
                vars->scalarNonlinearSrc, vars->scalarLinearSrc,
                constvars->cellType, d_intrusionBC->d_cellTypeID);
}
//______________________________________________________________________
//
void
BoundaryCondition::intrusionEnthalpyBC( const Patch* patch, 
                                        double delta_t,
                                        CellInformation* cellinfo,
                                        ArchesVariables* vars,
                                        ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  //fortran call
  fort_mmwallbc_trans(idxLo, idxHi,
                      vars->scalarCoeff[Arches::AE],
                      vars->scalarCoeff[Arches::AW],
                      vars->scalarCoeff[Arches::AN],
                      vars->scalarCoeff[Arches::AS],
                      vars->scalarCoeff[Arches::AT],
                      vars->scalarCoeff[Arches::AB],
                      vars->scalarNonlinearSrc,
                      vars->scalarLinearSrc,
                      constvars->old_enthalpy,
                      constvars->old_density,
                      constvars->cellType, d_intrusionBC->d_cellTypeID,
                      cellinfo->sew, cellinfo->sns, cellinfo->stb, 
                      delta_t);
}


//______________________________________________________________________
// compute multimaterial wall bc
void 
BoundaryCondition::mmvelocityBC(const Patch* patch,
                                CellInformation*,
                                ArchesVariables* vars,
                                ArchesConstVariables* constvars)
{
  //__________________________________
  //    X dir
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  int ioff = 1;
  int joff = 0;
  int koff = 0;

  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->uVelocityCoeff[Arches::AE],
                    vars->uVelocityCoeff[Arches::AW],
                    vars->uVelocityCoeff[Arches::AN],
                    vars->uVelocityCoeff[Arches::AS],
                    vars->uVelocityCoeff[Arches::AT],
                    vars->uVelocityCoeff[Arches::AB],
                    vars->uVelNonlinearSrc, vars->uVelLinearSrc,
                    constvars->cellType, d_mmWallID, ioff, joff, koff);
  //__________________________________
  //    Y dir
  idxLoU = patch->getSFCYFORTLowIndex();
  idxHiU = patch->getSFCYFORTHighIndex();
  
  ioff = 0;
  joff = 1;
  koff = 0;
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->vVelocityCoeff[Arches::AN],
                    vars->vVelocityCoeff[Arches::AS],
                    vars->vVelocityCoeff[Arches::AT],
                    vars->vVelocityCoeff[Arches::AB],
                    vars->vVelocityCoeff[Arches::AE],
                    vars->vVelocityCoeff[Arches::AW],
                    vars->vVelNonlinearSrc, vars->vVelLinearSrc,
                    constvars->cellType, d_mmWallID, ioff, joff, koff);

  //__________________________________
  //    Z dir
  idxLoU = patch->getSFCZFORTLowIndex();
  idxHiU = patch->getSFCZFORTHighIndex();

  ioff = 0;
  joff = 0;
  koff = 1;
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->wVelocityCoeff[Arches::AT],
                    vars->wVelocityCoeff[Arches::AB],
                    vars->wVelocityCoeff[Arches::AE],
                    vars->wVelocityCoeff[Arches::AW],
                    vars->wVelocityCoeff[Arches::AN],
                    vars->wVelocityCoeff[Arches::AS],
                    vars->wVelNonlinearSrc, vars->wVelLinearSrc,
                    constvars->cellType, d_mmWallID, ioff, joff, koff);
}

//______________________________________________________________________
//
void 
BoundaryCondition::mmpressureBC(DataWarehouse* new_dw,
                                const Patch* patch,
                                ArchesVariables* vars,
                                ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  IntVector domLong = vars->pressLinearSrc.getFortLowIndex();
  IntVector domHing = vars->pressLinearSrc.getFortHighIndex();
  ASSERTEQ(domLong,
           vars->pressCoeff.getWindow()->getLowIndex());
  ASSERTEQ(domHing+IntVector(1,1,1),
           vars->pressCoeff.getWindow()->getHighIndex());
  ASSERTEQ(domLong, vars->pressNonlinearSrc.getWindow()->getLowIndex());
  ASSERTEQ(domHing+IntVector(1,1,1), vars->pressNonlinearSrc.getWindow()->getHighIndex());


  //__________________________________
  // Move stencil7 data into separate CCVariable<double> arrays
  // so fortran code can deal with it.  This sucks --Todd
  CCVariable<double>AP, AE, AW, AN, AS, AT, AB;
  
  string direction = "copyInto";
  CellIterator iter = patch->getExtraCellIterator__New();
  copy_stencil7<CCVariable<Stencil7>, CCVariable<double> >(new_dw, patch, direction, iter,
                vars->pressCoeff, AP, AE, AW, AN, AS, AT, AB);
                
  //fortran call
  fort_mmwallbc(idxLo, idxHi,
                AE, AW, AN, AS, AT, AB,
                vars->pressNonlinearSrc, vars->pressLinearSrc,
                constvars->cellType, d_mmWallID);
                
  //__________________________________
  //  This sucks --Todd
  direction = "out";
  copy_stencil7<CCVariable<Stencil7>, CCVariable<double> >(new_dw, patch, direction, iter,
                vars->pressCoeff, AP, AE, AW, AN, AS, AT, AB);
}

//______________________________________________________________________
// applies multimaterial bc's for scalars and pressure
void
BoundaryCondition::mmscalarWallBC( const Patch* patch,
                                   CellInformation*,
                                   ArchesVariables* vars,
                                   ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  //fortran call
  fort_mmwallbc(idxLo, idxHi,
                      vars->scalarConvectCoeff[Arches::AE], vars->scalarConvectCoeff[Arches::AW],
                      vars->scalarConvectCoeff[Arches::AN], vars->scalarConvectCoeff[Arches::AS],
                      vars->scalarConvectCoeff[Arches::AT], vars->scalarConvectCoeff[Arches::AB],
                      vars->scalarNonlinearSrc, vars->scalarLinearSrc,
                      constvars->cellType, d_mmWallID);
  fort_mmwallbc(idxLo, idxHi,
                      vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW],
                      vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS],
                      vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB],
                      vars->scalarNonlinearSrc, vars->scalarLinearSrc,
                      constvars->cellType, d_mmWallID);
}

//______________________________________________________________________
// New intrusion scalar BC
void 
BoundaryCondition::mmscalarWallBC__new( const Patch* patch,
                                        CellInformation*,
                                        ArchesVariables* vars,
                                        ArchesConstVariables* constvars)
{
  // **NOTE**
  // Why is there a special d_mmWallID? This isn't consistent with how wallid is handled

  // Get the wall boundary and flow field codes
  for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
    IntVector curr = *iter;

    if (constvars->cellType[curr] == d_mmWallID) {
      vars->scalarConvCoef[curr].e = 0.0;
      vars->scalarConvCoef[curr].w = 0.0;
      vars->scalarConvCoef[curr].n = 0.0;
      vars->scalarConvCoef[curr].s = 0.0;
      vars->scalarConvCoef[curr].t = 0.0;
      vars->scalarConvCoef[curr].b = 0.0;

      vars->scalarTotCoef[curr].e = 0.0;
      vars->scalarTotCoef[curr].w = 0.0;
      vars->scalarTotCoef[curr].n = 0.0;
      vars->scalarTotCoef[curr].s = 0.0;
      vars->scalarTotCoef[curr].t = 0.0;
      vars->scalarTotCoef[curr].b = 0.0;
  
      vars->scalarNonlinearSrc[curr] = 0.0;
      vars->scalarLinearSrc[curr] = -1.0;
    }
    else {
      if (constvars->cellType[curr + IntVector(1,0,0)]==d_mmWallID){
        vars->scalarConvCoef[curr].e = 0.0;
        vars->scalarTotCoef[curr].e = 0.0;
      }
      if (constvars->cellType[curr - IntVector(1,0,0)]==d_mmWallID){
        vars->scalarConvCoef[curr].w = 0.0;
        vars->scalarTotCoef[curr].w = 0.0;
      }
      if (constvars->cellType[curr + IntVector(0,1,0)]==d_mmWallID){
        vars->scalarConvCoef[curr].n = 0.0;
        vars->scalarTotCoef[curr].n = 0.0;
      }
      if (constvars->cellType[curr - IntVector(0,1,0)]==d_mmWallID){
        vars->scalarConvCoef[curr].s = 0.0;
        vars->scalarTotCoef[curr].s = 0.0;
      }
       if (constvars->cellType[curr + IntVector(0,0,1)]==d_mmWallID){
        vars->scalarConvCoef[curr].t = 0.0;
        vars->scalarTotCoef[curr].t = 0.0;
      } 
       if (constvars->cellType[curr - IntVector(0,0,1)]==d_mmWallID){
        vars->scalarConvCoef[curr].b = 0.0;
        vars->scalarTotCoef[curr].b = 0.0;
      } 
    }

  }
}
 

//______________________________________________________________________
// applies multimaterial bc's for enthalpy
void
BoundaryCondition::mmEnthalpyWallBC( const Patch* patch,
                                     CellInformation*,
                                     ArchesVariables* vars,
                                     ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  //fortran call
  fort_mmwallbc(idxLo, idxHi,
                vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW],
                vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS],
                vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB],
                vars->scalarNonlinearSrc, vars->scalarLinearSrc,
                constvars->cellType, d_mmWallID);
}


//****************************************************************************
// constructor for BoundaryCondition::WallBdry
//****************************************************************************
BoundaryCondition::WallBdry::WallBdry(int cellID):
  d_cellTypeID(cellID)
{
}


//****************************************************************************
// Problem Setup for BoundaryCondition::WallBdry
//****************************************************************************
void 
BoundaryCondition::WallBdry::problemSetup(ProblemSpecP& params)
{
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
}

//****************************************************************************
// constructor for BoundaryCondition::WallBdry
//****************************************************************************
BoundaryCondition::IntrusionBdry::IntrusionBdry(int cellID):
  d_cellTypeID(cellID)
{
}

//****************************************************************************
// Problem Setup for BoundaryCondition::WallBdry
//****************************************************************************
void 
BoundaryCondition::IntrusionBdry::problemSetup(ProblemSpecP& params)
{
  if (params->findBlock("temperature"))
    params->require("temperature", d_temperature);
  else
    d_temperature = 300;
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);

}




//****************************************************************************
// constructor for BoundaryCondition::FlowInlet
//****************************************************************************
BoundaryCondition::FlowInlet::FlowInlet(int cellID, 
                                        bool calcVariance,
                                        bool reactingScalarSolve):
  d_cellTypeID(cellID), d_calcVariance(calcVariance), 
  d_reactingScalarSolve(reactingScalarSolve)
{
  flowRate = 0.0;
  inletVel = 0.0;
  fcr = 0.0;
  fsr = 0.0;
  d_prefill_index = 0;
  d_ramping_inlet_flowrate = false;
  d_prefill = false;
  // add cellId to distinguish different inlets
  std::stringstream stream_cellID;
  stream_cellID << d_cellTypeID;
  d_area_label = VarLabel::create("flowarea"+stream_cellID.str(),
   ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription()); 
  d_flowRate_label = VarLabel::create("flowRate"+stream_cellID.str(),
   ReductionVariable<double, Reductions::Min<double> >::getTypeDescription()); 
}

BoundaryCondition::FlowInlet::FlowInlet():
  d_cellTypeID(0), d_calcVariance(0), d_reactingScalarSolve(0),
  d_area_label(0), d_flowRate_label(0)
{
  flowRate = 0.0;
  inletVel = 0.0;
  fcr = 0.0;
  fsr = 0.0;
  d_prefill_index = 0;
  d_ramping_inlet_flowrate = false;
  d_prefill = false;
}

BoundaryCondition::FlowInlet::FlowInlet( const FlowInlet& copy ) :
  d_cellTypeID (copy.d_cellTypeID),
  d_calcVariance (copy.d_calcVariance),
  d_reactingScalarSolve (copy.d_reactingScalarSolve),
  flowRate(copy.flowRate),
  inletVel(copy.inletVel),
  fcr(copy.fcr),
  fsr(copy.fsr),
  d_prefill_index(copy.d_prefill_index),
  d_ramping_inlet_flowrate(copy.d_ramping_inlet_flowrate),
  streamMixturefraction(copy.streamMixturefraction),
  calcStream(copy.calcStream),
  d_area_label(copy.d_area_label),
  d_flowRate_label(copy.d_flowRate_label)
{
  for (vector<GeometryPieceP>::const_iterator it = copy.d_geomPiece.begin();
       it != copy.d_geomPiece.end(); ++it)
    d_geomPiece.push_back((*it)->clone());
  
  if (d_prefill)
    for (vector<GeometryPieceP>::const_iterator it = copy.d_prefillGeomPiece.begin();
       it != copy.d_prefillGeomPiece.end(); ++it)
      d_prefillGeomPiece.push_back((*it)->clone());

  d_area_label->addReference();
  d_flowRate_label->addReference();
}

BoundaryCondition::FlowInlet& BoundaryCondition::FlowInlet::operator=(const FlowInlet& copy)
{
  // remove reference from the old label
  VarLabel::destroy(d_area_label);
  d_area_label = copy.d_area_label;
  d_area_label->addReference();
  VarLabel::destroy(d_flowRate_label);
  d_flowRate_label = copy.d_flowRate_label;
  d_flowRate_label->addReference();

  d_cellTypeID = copy.d_cellTypeID;
  d_calcVariance = copy.d_calcVariance;
  d_reactingScalarSolve = copy.d_reactingScalarSolve;
  flowRate = copy.flowRate;
  inletVel = copy.inletVel;
  fcr = copy.fcr;
  fsr = copy.fsr;
  d_prefill_index = copy.d_prefill_index;
  d_ramping_inlet_flowrate = copy.d_ramping_inlet_flowrate;
  streamMixturefraction = copy.streamMixturefraction;
  calcStream = copy.calcStream;
  d_geomPiece = copy.d_geomPiece;
  if (d_prefill)
    d_prefillGeomPiece = copy.d_prefillGeomPiece;

  return *this;
}


BoundaryCondition::FlowInlet::~FlowInlet()
{
  VarLabel::destroy(d_area_label);
  VarLabel::destroy(d_flowRate_label);
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowInlet
//****************************************************************************
void 
BoundaryCondition::FlowInlet::problemSetup(ProblemSpecP& params)
{
  params->getWithDefault("Flow_rate", flowRate,0.0);
  params->getWithDefault("InletVelocity", inletVel,0.0);
// ramping function is the same for all inlets where ramping is on  
  params->getWithDefault("ramping_inlet_flowrate",
                         d_ramping_inlet_flowrate, false);
  // This parameter only needs to be set for fuel inlets for which
  // mixture fraction > 0, if there is an air inlet, and air has some CO2,
  // this air CO2 will be counted in the balance automatically
  // params->getWithDefault("CarbonMassFractionInFuel", fcr, 0.0);
  params->getWithDefault("SulfurMassFractionInFuel", fsr, 0.0);
  // check to see if this will work
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPieceP> pieces;
  //  GeometryPieceFactory::create(geom_obj_ps, pieces);
  //  if(pieces.size() == 0){
  //    throw ParameterNotFound("No piece specified in geom_object");
  //  } else if(pieces.size() > 1){
  //    d_geomPiece = scinew UnionGeometryPiece(pieces);
  //  } else {
  //    d_geomPiece = pieces[0];
  //  }
  //}

  double mixfrac;
  params->require("mixture_fraction", mixfrac);
  streamMixturefraction.d_mixVars.push_back(mixfrac);
  if (d_calcVariance){
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  }
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
  }
 
  //Prefill a geometry object with the flow inlet condition 
  d_prefill = params->findBlock("Prefill"); 
  if (params->findBlock("Prefill")){
    ProblemSpecP db_prefill = params->findBlock("Prefill");
    std::string prefill_direction = "null"; 
    db_prefill->getAttribute("direction",prefill_direction); 
    //get the direction
    if (prefill_direction == "X") {
      d_prefill_index = 1;
    }
    else if (prefill_direction == "Y") {
      d_prefill_index = 2;
    }
    else if (prefill_direction == "Z") {
      d_prefill_index = 3;
    }
    else {
      throw InvalidValue("Wrong prefill direction. Please add the direction attribute to the <Prefill> tag.", __FILE__, __LINE__);
    }
    //get the object
    ProblemSpecP prefillGeomObjPS = db_prefill->findBlock("geom_object");
    if (prefillGeomObjPS)
    {
      GeometryPieceFactory::create(prefillGeomObjPS, d_prefillGeomPiece); 
    } 
    else 
    { 
      throw ProblemSetupException("Error! Must specify a geom_object in <Prefill> block.",__FILE__,__LINE__); 
    }
  }
}


//****************************************************************************
// constructor for BoundaryCondition::PressureInlet
//****************************************************************************
BoundaryCondition::PressureInlet::PressureInlet(int cellID, 
                                                bool calcVariance,
                                                bool reactingScalarSolve):
  d_cellTypeID(cellID), d_calcVariance(calcVariance),
  d_reactingScalarSolve(reactingScalarSolve)
{
}

//****************************************************************************
// Problem Setup for BoundaryCondition::PressureInlet
//****************************************************************************
void 
BoundaryCondition::PressureInlet::problemSetup(ProblemSpecP& params)
{
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the pressure inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPieceP> pieces;
  //  GeometryPieceFactory::create(geom_obj_ps, pieces);
  //  if(pieces.size() == 0){
  //    throw ParameterNotFound("No piece specified in geom_object");
  //  } else if(pieces.size() > 1){
  //    d_geomPiece = scinew UnionGeometryPiece(pieces);
  //  } else {
  //    d_geomPiece = pieces[0];
  //  }
  //}
  double mixfrac;
  params->require("mixture_fraction", mixfrac);
  streamMixturefraction.d_mixVars.push_back(mixfrac);
  if (d_calcVariance){
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  }
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
  }
}

//****************************************************************************
// constructor for BoundaryCondition::FlowOutlet
//****************************************************************************
BoundaryCondition::FlowOutlet::FlowOutlet(int cellID, 
                                          bool calcVariance,
                                          bool reactingScalarSolve):
  d_cellTypeID(cellID), d_calcVariance(calcVariance),
  d_reactingScalarSolve(reactingScalarSolve)
{
}

//****************************************************************************
// Problem Setup for BoundaryCondition::FlowOutlet
//****************************************************************************
void 
BoundaryCondition::FlowOutlet::problemSetup(ProblemSpecP& params)
{
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  // loop thru all the inlet geometry objects
  //for (ProblemSpecP geom_obj_ps = params->findBlock("geom_object");
  //     geom_obj_ps != 0; 
  //     geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
  //  vector<GeometryPieceP> pieces;
  //  GeometryPieceFactory::create(geom_obj_ps, pieces);
  //  if(pieces.size() == 0){
  //    throw ParameterNotFound("No piece specified in geom_object");
  //  } else if(pieces.size() > 1){
  //    d_geomPiece = scinew UnionGeometryPiece(pieces);
  //  } else {
  //    d_geomPiece = pieces[0];
  //  }
  //}
  double mixfrac;
  params->require("mixture_fraction", mixfrac);
  streamMixturefraction.d_mixVars.push_back(mixfrac);
  if (d_calcVariance)
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
  }
}
//****************************************************************************
// Constructor Setup for BoundaryCondition::BCSourceInfo
//****************************************************************************
BoundaryCondition::BCSourceInfo::BCSourceInfo(bool calcVariance, bool reactingScalarSolve):
d_calcVariance(calcVariance), d_reactingScalarSolve(reactingScalarSolve)
{
  //initialize some variables
  area_x = 0.0;
  area_y = 0.0;
  area_z = 0.0;
  umom_flux = 0.0;
  vmom_flux = 0.0;
  wmom_flux = 0.0;
  f_flux = 0.0;
  h_flux = 0.0;
  totalMassFlux = 0.0;
  totalVelocity = 0.0;
  summed_area = 0.0;
  computedArea = false;
   total_area_label = VarLabel::create("flowarea",
                                        ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());


}
//****************************************************************************
// Destructor Setup for BoundaryCondition::BCSourceInfo
//****************************************************************************
BoundaryCondition::BCSourceInfo::~BCSourceInfo(){
                VarLabel::destroy(total_area_label);
}
//****************************************************************************
// Problem Setup for BoundaryCondition::BCSourceInfo
//****************************************************************************
void 
BoundaryCondition::BCSourceInfo::problemSetup(ProblemSpecP& params)
{
  //You can pick MassFlux or Velocity
  if (ProblemSpecP massfluxchild = params->findBlock("MassFlux")){
    doAreaCalc = true;        
    massfluxchild->require("massflux_value", totalMassFlux); //value of the vector

    massfluxchild->getAttribute("type",velocityType); //relative or absolute
    if (velocityType == "relative"){
      massfluxchild->getAttribute("relation",velocityRelation); // choose "point" or "axis"

      if (velocityRelation == "point"){
        massfluxchild->require("point_location", point);
      }
      else if (velocityRelation == "axis"){
        massfluxchild->require("axis_start",axisStart); //Starting and ending point of the axis
        massfluxchild->require("axis_end",  axisEnd);
      }
    }else if (velocityType == "absolute"){
      massfluxchild->require("normals",normal); //since it is absolute, we need tell it what faces and how to scale it. ie v_{face,i} = n_i*V
      proc0cout << "normal =" << normal<<endl;
    }else{
      throw ParameterNotFound(" Must specify an absolute or relative attribute for the <Velocity> or <MassFlux>.",__FILE__,__LINE__); 
    }

  }else if (ProblemSpecP velchild = params->findBlock("Velocity")){
    velchild->require("velocity_value", totalVelocity); //value of the vector

    velchild->getAttribute("type",velocityType); //relative or absolute
    if (velocityType == "relative"){
      velchild->getAttribute("relation",velocityRelation); // choose "point" or "axis"

      if (velocityRelation == "point"){
        velchild->require("point_location", point);        

      }else if (velocityRelation == "axis"){
        velchild->require("axis_start",axisStart); //Starting and ending point of the axis
        velchild->require("axis_end",  axisEnd);
      }
    }
    else if (velocityType == "absolute"){
      velchild->require("normals",normal); //since it is absolute, we need tell it what faces and how to scale it. ie v_{face,i} = n_i*V
      proc0cout << "normal =" << normal<<endl;
    }else{
      throw ParameterNotFound(" Must specify an absolute or relative attribute for the <Velocity> or <MassFlux>.",__FILE__,__LINE__); 
    }
  }else{
    throw ParameterNotFound(" Please enter a MassFlux or Velocity for the <IntrusionWithBCSource> block!",__FILE__,__LINE__);
  }
                                                  

  // Get the scalar information
  if (ProblemSpecP mfchild = params->findBlock("MixtureFraction")){
    mfchild->require("inlet_value",mixfrac_inlet);
  }
  //for getting the inlet properties
  //hard coded for now!
  d_calcVariance = false;
  d_reactingScalarSolve = false;
  streamMixturefraction.d_mixVars.push_back(mixfrac_inlet);
  if (d_calcVariance){
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  }
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
  }

  // Get the geometry piece(s)
  if (ProblemSpecP geomObjPS = params->findBlock("geom_object")){
    GeometryPieceFactory::create(geomObjPS, d_geomPiece);
  }else{
    throw ParameterNotFound(" Must specify a geometry piece for BCSource.\nPlease add a <geom_object> to the <IntrusionWithBCSource> block in the inputfile.  Stopping...",__FILE__,__LINE__);
  }
}

//-------------------------------------------------------------
// Schedule to compute the boudary source inlet areas
//-------------------------------------------------------------
void 
BoundaryCondition::sched_computeInletAreaBCSource(SchedulerP& sched, 
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::computeInletAreaBCSource",
                          this, 
                          &BoundaryCondition::computeInletAreaBCSource);

  int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();        
  for (int bp = 0; bp < nofBoundaryPieces; bp++){
    tsk->computes(d_sourceBoundaryInfo[bp]->total_area_label);
  }
  sched->addTask(tsk, patches, matls);
}
//****************************************************************************
// Compute surface area of inlet for BoundaryCondition::BCSourceInfo
//****************************************************************************
void
BoundaryCondition::computeInletAreaBCSource(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset*,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{

  //Loop through current patch...compute area        
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Box patchInteriorBox = patch->getBox();
        
    int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();

    //assuming a uniform mesh!
    Vector dx = patch->dCell();

    for (int bp = 0; bp < nofBoundaryPieces; bp++){
      
      // Zero out the area here...total area will be put in total_area member
      d_sourceBoundaryInfo[bp]->area_x = 0.0;
      d_sourceBoundaryInfo[bp]->area_y = 0.0;
      d_sourceBoundaryInfo[bp]->area_z = 0.0;
    
      // The main loop for computing the source term
      //**get the geometry piece for this boundary source block**
      // we could have more than one geometry piece per block.
      int nofGeomPieces = (int)d_sourceBoundaryInfo[bp]->d_geomPiece.size();
    
      for (int gp = 0; gp < nofGeomPieces; gp++){
        
        GeometryPieceP piece = d_sourceBoundaryInfo[bp]->d_geomPiece[gp];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchInteriorBox);
  
        if (!(b.degenerate()) && !(d_sourceBoundaryInfo[bp]->computedArea)){
          
          //iterator over cells and see if a boundary source needs adding.
          for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
          
            Point p = patch->cellPosition(*iter);
            Point p_xp = patch->cellPosition(*iter + IntVector(1,0,0));
            Point p_xm = patch->cellPosition(*iter - IntVector(1,0,0));
            Point p_yp = patch->cellPosition(*iter + IntVector(0,1,0));
            Point p_ym = patch->cellPosition(*iter - IntVector(0,1,0));
            Point p_zp = patch->cellPosition(*iter + IntVector(0,0,1));
            Point p_zm = patch->cellPosition(*iter - IntVector(0,0,1));
          
            if (piece->inside(p)){
                       
               //Now check neighbors
               // x+
               if (!(piece->inside(p_xp)))
                       d_sourceBoundaryInfo[bp]->area_x += dx.y()*dx.z();
               // x-
               if (!(piece->inside(p_xm)))
                       d_sourceBoundaryInfo[bp]->area_x += dx.y()*dx.z();
               // y+        
               if (!(piece->inside(p_yp)))
                       d_sourceBoundaryInfo[bp]->area_y += dx.x()*dx.z();
               // y-
               if (!(piece->inside(p_ym)))
                       d_sourceBoundaryInfo[bp]->area_y += dx.x()*dx.z();
               // z+
               if (!(piece->inside(p_zp)))
                       d_sourceBoundaryInfo[bp]->area_z += dx.x()*dx.y();
               // z-
               if (!(piece->inside(p_zm)))
                       d_sourceBoundaryInfo[bp]->area_z += dx.x()*dx.y();
            }
          }  // cell iterator
        }else{
          new_dw->put(sum_vartype(0), d_sourceBoundaryInfo[bp]->total_area_label);
        }        
      }  // geom pieces

      double inlet_area;
      inlet_area = d_sourceBoundaryInfo[bp]->area_y + d_sourceBoundaryInfo[bp]->area_z;

      new_dw->put(sum_vartype(inlet_area), d_sourceBoundaryInfo[bp]->total_area_label);        

    }  // boundary pieces
  }  // patches
}
// *------------------------------------------------*
// Schedule the compute of the boundary source term
// *------------------------------------------------*
void 
BoundaryCondition::sched_computeMomSourceTerm(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::computeMomSourceTerm", this, 
                          &BoundaryCondition::computeMomSourceTerm);

  tsk->modifies(d_lab->d_umomBoundarySrcLabel);
  tsk->modifies(d_lab->d_vmomBoundarySrcLabel);
  tsk->modifies(d_lab->d_wmomBoundarySrcLabel);
  tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, Ghost::AroundCells, 2);

  int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();
  for (int bp = 0; bp < nofBoundaryPieces; bp++){
    tsk->requires(Task::NewDW, d_sourceBoundaryInfo[bp]->total_area_label);
  }
  sched->addTask(tsk, patches, matls);         
          
}
// *------------------------------------------------*
// Carry out the compute of the boundary source term
// *------------------------------------------------*
void 
BoundaryCondition::computeMomSourceTerm(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw )
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Box patchInteriorBox = patch->getBox();
    
    SFCXVariable<double> umomSource;
    SFCYVariable<double> vmomSource;
    SFCZVariable<double> wmomSource;
    constCCVariable<int> cellType;

    new_dw->getModifiable(umomSource, d_lab->d_umomBoundarySrcLabel, indx, patch);
    new_dw->getModifiable(vmomSource, d_lab->d_vmomBoundarySrcLabel, indx, patch);
    new_dw->getModifiable(wmomSource, d_lab->d_wmomBoundarySrcLabel, indx, patch);
    old_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch,  Ghost::AroundCells, 2);

    umomSource.initialize(0.0);
    vmomSource.initialize(0.0);
    wmomSource.initialize(0.0);

    //assuming a uniform mesh!
    Vector dx = patch->dCell();

    IntVector idxLo = patch->getCellLowIndex__New();
    IntVector idxHi = patch->getCellHighIndex__New();

    IntVector idxLos = patch->getSFCYLowIndex__New();
    IntVector idxHis = patch->getSFCYHighIndex__New();

    IntVector idxLof = patch->getFortranCellLowIndex__New();
    IntVector idxHif = patch->getFortranCellHighIndex__New();

    IntVector idxLoe = patch->getExtraCellLowIndex__New();
    IntVector idxHie = patch->getExtraCellHighIndex__New();

    int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();

    for (int bp = 0; bp < nofBoundaryPieces; bp++){
      // The main loop for computing the source term
      //**get the geometry piece for this boundary source block**
      // we could have more than one geometry piece per block.
      int nofGeomPieces = (int)d_sourceBoundaryInfo[bp]->d_geomPiece.size();
    
      sum_vartype total_area;
      double area;

      // We only need to populate the area once
      if (!(d_sourceBoundaryInfo[bp]->computedArea)) {
        new_dw->get(total_area, d_sourceBoundaryInfo[bp]->total_area_label);
        area = total_area;
        d_sourceBoundaryInfo[bp]->summed_area = total_area;
        d_sourceBoundaryInfo[bp]->computedArea = true;
      }
                      
      d_sourceBoundaryInfo[bp]->totalVelocity = d_sourceBoundaryInfo[bp]->totalMassFlux/(d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->summed_area);


      for (int gp = 0; gp < nofGeomPieces; gp++){

        GeometryPieceP piece = d_sourceBoundaryInfo[bp]->d_geomPiece[gp];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchInteriorBox);
        
        for (CellIterator iter=patch->getCellIterator__New();!iter.done(); iter++){
          Point p = patch->cellPosition(*iter);
          Point p_xp = patch->cellPosition(*iter + IntVector(1,0,0));
          Point p_xm = patch->cellPosition(*iter - IntVector(1,0,0));
          Point p_yp = patch->cellPosition(*iter + IntVector(0,1,0));
          Point p_ym = patch->cellPosition(*iter - IntVector(0,1,0));
          Point p_ymm = patch->cellPosition(*iter - IntVector(0,2,0));
          Point p_zp = patch->cellPosition(*iter + IntVector(0,0,1));
          Point p_zm = patch->cellPosition(*iter - IntVector(0,0,1));
          Point p_zmm = patch->cellPosition(*iter - IntVector(0,0,2));
        
//          if (cellType[*iter] == d_flowfieldCellTypeVal){
          if (!(piece->inside(p))){
            
            //this is a nasty embedded set of if's....will fix in the future.
            // x+
            if ((piece->inside(p_xp))){
              if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){                
              }
              else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                }
                else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                }
              }
            }
            // x-
            if ((piece->inside(p_xm))){
              if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){                
              }
              else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                }
                else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                }
              }
            }
        
            // y+ is a wall
//            if (cellType[*iter + IntVector(0,1,0)] == d_intrusionBC->d_cellTypeID){
            if ((piece->inside(p_yp))){                
              if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){                
              }
              else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                  //hard coding for jennifer for now.
                  double y = (p.y()-dx.y()/2.0) - d_sourceBoundaryInfo[bp]->axisStart[1];                
                  double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
                  double theta = atan(z/y);

                  double y_comp = d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta);
                
                  vmomSource[*iter] = d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*y_comp*dx.x()*dx.z();
                }
                else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                }

              }

            }        
            // y- is a wall
            IntVector testi = *iter + IntVector(0,1,0);
            IntVector testi2 = *iter;
//            if (cellType[*iter - IntVector(0,1,0)] == d_intrusionBC->d_cellTypeID && 
//                            testi.y() < idxHi.y()  ){
            if (piece->inside(p_ym) && testi.y() < idxHi.y()){
                      //cout << "------for y- -------" << endl;
                      //cout << "  cell type - 1  =  " << cellType[*iter-IntVector(0,1,0)] << endl;
                      //cout << "  cell type + 0  =  " << cellType[*iter] << endl;                                                
              if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){                
              }
              else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                  //hard coding for jennifer for now.
                  double y = (p.y()-dx.y()/2.0) - d_sourceBoundaryInfo[bp]->axisStart[1];                
                  double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
                  double theta = atan(z/y);

                  double y_comp = -1.0*d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta);
                  vmomSource[*iter+IntVector(0,1,0)] = -1.0*d_sourceBoundaryInfo[bp]->calcStream.d_density*
                                                                                            y_comp*y_comp*dx.x()*dx.z();

                }
                else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                }

              }

            }

            //because we are adding the source term to the j+1 face, 
            // we have to do an additional check for patch boundaries
            bool yminus = patch->getBCType(Patch::yminus);
            if (yminus && testi2.y() == idxLo.y()){
//                    if (cellType[*iter - IntVector(0,2,0)] == d_intrusionBC->d_cellTypeID && 
//                            cellType[*iter - IntVector(0,1,0)] != d_intrusionBC->d_cellTypeID){
              if (piece->inside(p_ymm) && !(piece->inside(p_ym))){        
            
                if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){                
                }
                else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                  if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                          //hard coding for jennifer for now.
                          double y = (p.y()-dx.y()/2.0) - d_sourceBoundaryInfo[bp]->axisStart[1];                
                          double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
                          double theta = atan(z/y);

                          double y_comp = -1.0*d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta);
                          vmomSource[*iter] = -1.0*d_sourceBoundaryInfo[bp]->calcStream.d_density*
                                                                                            y_comp*y_comp*dx.x()*dx.z();

                  }
                  else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                  }
                }
              }
            }
            // z+ is a wall
//            if (cellType[*iter + IntVector(0,0,1)] == d_intrusionBC->d_cellTypeID){
            if (piece->inside(p_zp)){

              if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){
              }
              else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                  //hard coding for jennifer for now.
                  double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];
                  double z = (p.z()-dx.z()/2.0) - d_sourceBoundaryInfo[bp]->axisStart[2];
                  double theta = atan(z/y);

                  double z_comp = d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta);
 
                  wmomSource[*iter] = d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*z_comp*dx.x()*dx.y();

                }
                else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                }
              }
            }
            // z- is a wall
            testi = *iter + IntVector(0,0,1);
//            if (cellType[*iter - IntVector(0,0,1)] == d_intrusionBC->d_cellTypeID && 
//                            testi.z() < idxHi.z()){
            if (piece->inside(p_zm) && testi.z() < idxHi.z()){

              if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){                
              }
              else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                  //hard coding for jennifer for now.
                  double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];                
                  double z = (p.z()-dx.z()/2.0) - d_sourceBoundaryInfo[bp]->axisStart[2];
                  double theta = atan(z/y);

                  double z_comp = d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta);
                  wmomSource[*iter+IntVector(0,0,1)] = -1.0*d_sourceBoundaryInfo[bp]->calcStream.d_density*
                                                                                            z_comp*z_comp*dx.x()*dx.y();

                }
                else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                }
              }
            }
            //because we are adding the source term to the j+1 face, 
            // we have to do an additional check for patch boundaries
            bool zminus = patch->getBCType(Patch::zminus);
            if (zminus && testi2.z() == idxLo.z()){
//              if (cellType[*iter - IntVector(0,0,2)] == d_intrusionBC->d_cellTypeID && 
//                      cellType[*iter - IntVector(0,0,1)] != d_intrusionBC->d_cellTypeID){
              if (piece->inside(p_zmm) && !(piece->inside(p_zm))){         
                
                if (d_sourceBoundaryInfo[bp]->velocityType == "absolute"){                
                }
                else if (d_sourceBoundaryInfo[bp]->velocityType == "relative"){
                  if (d_sourceBoundaryInfo[bp]->velocityRelation == "axis"){
                  //hard coding for jennifer for now.
                    double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];                
                    double z = (p.z()-dx.z()/2.0) - d_sourceBoundaryInfo[bp]->axisStart[2];
                    double theta = atan(z/y);

                    double z_comp = d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta);
                    wmomSource[*iter] = -1.0*d_sourceBoundaryInfo[bp]->calcStream.d_density*
                                                                                            z_comp*z_comp*dx.x()*dx.y();


                  }else if (d_sourceBoundaryInfo[bp]->velocityRelation == "point"){
                  }
                }
              }
            }
          }
        }
      } // end Geometry Pieces loop
    } // end of Boundary Pieces loop 
  }
}
// *--------------------------------------------------------*
// Schedule the compute of the boundary source term-scalar
// *--------------------------------------------------------*
void 
BoundaryCondition::sched_computeScalarSourceTerm(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls
                                                                                                  )
{
  Task* tsk = scinew Task("BoundaryCondition::computeScalarSourceTerm", this, 
                          &BoundaryCondition::computeScalarSourceTerm);

  tsk->modifies(d_lab->d_scalarBoundarySrcLabel);
  tsk->modifies(d_lab->d_enthalpyBoundarySrcLabel);
  tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, Ghost::AroundCells, 1);

  sched->addTask(tsk, patches, matls);           
}
// *--------------------------------------------------------*
// Perform the compute of the boundary source term-scalar
// This includes: Mixture fraction, Enthalpy
// *--------------------------------------------------------*
void 
BoundaryCondition::computeScalarSourceTerm(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Box patchInteriorBox = patch->getBox();

    //assuming a uniform mesh!
    Vector dx = patch->dCell();

    CCVariable<double> scalarBoundarySrc;
    CCVariable<double> enthalpyBoundarySrc;
    constCCVariable<int> cellType;

    new_dw->getModifiable(scalarBoundarySrc,   d_lab->d_scalarBoundarySrcLabel,   indx, patch);
    new_dw->getModifiable(enthalpyBoundarySrc, d_lab->d_enthalpyBoundarySrcLabel, indx, patch);
    old_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::AroundCells, 1);

    scalarBoundarySrc.initialize(0.0);
    enthalpyBoundarySrc.initialize(0.0);

    IntVector idxLo = patch->getFortranCellLowIndex__New();
    IntVector idxHi = patch->getFortranCellHighIndex__New();

    int nofBoundaryPieces = (int)d_sourceBoundaryInfo.size();

    for (int bp = 0; bp < nofBoundaryPieces; bp++){
      // The main loop for computing the source term
      //**get the geometry piece for this boundary source block**
      // we could have more than one geometry piece per block.
      int nofGeomPieces = (int)d_sourceBoundaryInfo[bp]->d_geomPiece.size();
    
      for (int gp = 0; gp < nofGeomPieces; gp++){
        
        GeometryPieceP piece = d_sourceBoundaryInfo[bp]->d_geomPiece[gp];
        Box geomBox = piece->getBoundingBox();
        Box b = geomBox.intersect(patchInteriorBox);
        
        //iterator over all patch cells and see if a boundary source needs adding.
        for (CellIterator iter=patch->getCellIterator__New(); 
          !iter.done(); iter++){
        
          Point p = patch->cellPosition(*iter);
          Point p_xp = patch->cellPosition(*iter + IntVector(1,0,0));
          Point p_xm = patch->cellPosition(*iter - IntVector(1,0,0));
          Point p_yp = patch->cellPosition(*iter + IntVector(0,1,0));
          Point p_ym = patch->cellPosition(*iter - IntVector(0,1,0));
          Point p_zp = patch->cellPosition(*iter + IntVector(0,0,1));
          Point p_zm = patch->cellPosition(*iter - IntVector(0,0,1));
      
          //Here we could have used pcell instead of using the geometric intersection.
          // However, the geometry is set using the intrusion mechanism which does
          // the same thing we are doing here.  Note that for face centered variables (ie, mom. terms)
          // the pcell must be used.        
          if (!(piece->inside(p))){ 

            //Now check neighbors
            // x+
            if ((piece->inside(p_xp))){                 
              // source term = \int \rho u \phi \cdot dS        
//              scalarBoundarySrc[*iter] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
//                                                                 d_sourceBoundaryInfo[bp]->normal[0]*        
//                                                                 d_sourceBoundaryInfo[bp]->mixfrac_inlet*
//                                                                 dx.y()*dx.z();
//              if (d_enthalpySolve)
//                      enthalpyBoundarySrc[*iter] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
//                                                                      d_sourceBoundaryInfo[bp]->normal[0]*
//                                                                      d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
//                                                                      dx.y()*dx.z();                                                           
            }
            // x-
            if ((piece->inside(p_xm))){
//            scalarBoundarySrc[*iter] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
//                                                               d_sourceBoundaryInfo[bp]->normal[0]*        
//                                                               d_sourceBoundaryInfo[bp]->mixfrac_inlet*
//                                                               dx.y()*dx.z();                                                        
//            if (d_enthalpySolve)
//                    enthalpyBoundarySrc[*iter] = d_sourceBoundaryInfo[bp]->calcStream.d_density*d_sourceBoundaryInfo[bp]->totalVelocity*
//                                                                    d_sourceBoundaryInfo[bp]->normal[0]*
//                                                                    d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
//                                                                    dx.y()*dx.z();
            }
            // y+
            if ((piece->inside(p_yp))){                
              //hard coding for jennifer for now.
              double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];                
              double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
              double theta = atan(z/y);
              double y_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta));                        
                                      
              scalarBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
                                                                 d_sourceBoundaryInfo[bp]->mixfrac_inlet*
                                                                 dx.x()*dx.z();                
                                                                 
              if (d_enthalpySolve)
                      enthalpyBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
                                                                               d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
                                                                               dx.x()*dx.z();                                                                                                   
            }
            // y-
            if ((piece->inside(p_ym))){                
              //hard coding for jennifer for now.
              double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];                
              double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
              double theta = atan(z/y);
              double y_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*cos(theta));        

              scalarBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
                                                                 d_sourceBoundaryInfo[bp]->mixfrac_inlet*
                                                                 dx.x()*dx.z();
              if (d_enthalpySolve)
                      enthalpyBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*y_comp*
                                                                               d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
                                                                               dx.x()*dx.z();
            }
            // z+
            if ((piece->inside(p_zp))){
              double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];                
              double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
              double theta = atan(z/y);
              double z_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta));
              scalarBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
                                                                 d_sourceBoundaryInfo[bp]->mixfrac_inlet*
                                                                 dx.x()*dx.y();                                
              if (d_enthalpySolve)
                      enthalpyBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
                                                                               d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
                                                                               dx.x()*dx.y();                                                                                                                                   
            }
            // z-
            if ((piece->inside(p_zm))){
              double y = p.y() - d_sourceBoundaryInfo[bp]->axisStart[1];                
              double z = p.z() - d_sourceBoundaryInfo[bp]->axisStart[2];
              double theta = atan(z/y);
              double z_comp = Abs(d_sourceBoundaryInfo[bp]->totalVelocity*sin(theta));


              scalarBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
                                                                 d_sourceBoundaryInfo[bp]->mixfrac_inlet*
                                                                 dx.x()*dx.y();
                                                                 
              if (d_enthalpySolve)        
                      enthalpyBoundarySrc[*iter] += d_sourceBoundaryInfo[bp]->calcStream.d_density*z_comp*
                                                                               d_sourceBoundaryInfo[bp]->calcStream.d_enthalpy*
                                                                               dx.x()*dx.y();                                                                                                   
                                                                                                 
            }                                 
          }
        }
      } // end Geometry Pieces loop
    } // end Boundary Pieces loop
  } // end patch loop
        
}

//______________________________________________________________________
//
void
BoundaryCondition::calculateIntrusionVel(const Patch* patch,
                                         int index,
                                         CellInformation* ,
                                         ArchesVariables* vars,
                                         ArchesConstVariables* constvars)
{
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;

  switch(index) {

  case Arches::XDIR:

    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_intrusion_computevel(vars->uVelRhoHat,
                              ioff, joff, koff,
                              constvars->cellType,
                              idxLoU, idxHiU,
                              d_intrusionBC->d_cellTypeID);

    break;

  case Arches::YDIR:

    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_intrusion_computevel(vars->vVelRhoHat,
                              ioff, joff, koff,
                              constvars->cellType,
                              idxLoU, idxHiU,
                              d_intrusionBC->d_cellTypeID);

    break;

  case Arches::ZDIR:

    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();

    ioff = 0; joff = 0; koff = 1;
    
    fort_intrusion_computevel(vars->wVelRhoHat,
                       ioff, joff, koff,
                       constvars->cellType,
                       idxLoU, idxHiU,
                       d_intrusionBC->d_cellTypeID);

    break;

  default:
    
    throw InvalidValue("Invalid index in Source::calcVelSrc", __FILE__, __LINE__);
    
  }
  
}
//______________________________________________________________________
//
void
BoundaryCondition::calculateVelocityPred_mm(const Patch* patch,
                                            double delta_t,
                                            CellInformation* cellinfo,
                                            ArchesVariables* vars,
                                            ArchesConstVariables* constvars)
{
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;

  //__________________________________
  idxLoU = patch->getSFCXFORTLowIndex();
  idxHiU = patch->getSFCXFORTHighIndex();
  ioff = 1; joff = 0; koff = 0;

  fort_mm_computevel(
                    vars->uVelRhoHat,
                    constvars->pressure,
                    constvars->density,
                    constvars->voidFraction,
                    cellinfo->dxpw,
                    delta_t, 
                    ioff, joff, koff,
                    constvars->cellType,
                    idxLoU, idxHiU,
                    d_mmWallID);

  //__________________________________
  idxLoU = patch->getSFCYFORTLowIndex();
  idxHiU = patch->getSFCYFORTHighIndex();
  ioff = 0; joff = 1; koff = 0;

  fort_mm_computevel(
                    vars->vVelRhoHat,
                    constvars->pressure,
                    constvars->density,
                    constvars->voidFraction,
                    cellinfo->dyps,
                    delta_t, 
                    ioff, joff, koff,
                    constvars->cellType,
                    idxLoU, idxHiU,
                    d_mmWallID);

  //__________________________________
  idxLoU = patch->getSFCZFORTLowIndex();
  idxHiU = patch->getSFCZFORTHighIndex();

  ioff = 0; joff = 0; koff = 1;
  fort_mm_computevel(
                    vars->wVelRhoHat,
                    constvars->pressure,
                    constvars->density,
                    constvars->voidFraction,
                    cellinfo->dzpb,
                    delta_t, 
                    ioff, joff, koff,
                    constvars->cellType,
                    idxLoU, idxHiU,
                    d_mmWallID);
}
//______________________________________________________________________
//
void 
BoundaryCondition::calculateVelRhoHat_mm(const Patch* patch,
                                         double delta_t,
                                         CellInformation* cellinfo,
                                         ArchesVariables* vars,
                                         ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  int ioff, joff, koff;
  //__________________________________
  //    X dir
  idxLo = patch->getSFCXFORTLowIndex();
  idxHi = patch->getSFCXFORTHighIndex();
  ioff = 1; joff = 0; koff = 0;

  fort_mm_explicit_vel(idxLo, idxHi, 
                       vars->uVelRhoHat,
                       constvars->uVelocity,
                       vars->uVelocityCoeff[Arches::AE], 
                       vars->uVelocityCoeff[Arches::AW], 
                       vars->uVelocityCoeff[Arches::AN], 
                       vars->uVelocityCoeff[Arches::AS], 
                       vars->uVelocityCoeff[Arches::AT], 
                       vars->uVelocityCoeff[Arches::AB], 
                       vars->uVelocityCoeff[Arches::AP], 
                       vars->uVelNonlinearSrc,
                       constvars->new_density,
                       cellinfo->sewu, cellinfo->sns, cellinfo->stb,
                       delta_t, ioff, joff, koff,
                       constvars->cellType,
                       d_mmWallID);
  //__________________________________
  //    Y dir
  idxLo = patch->getSFCYFORTLowIndex();
  idxHi = patch->getSFCYFORTHighIndex();
  ioff = 0; joff = 1; koff = 0;

  fort_mm_explicit_vel(idxLo, idxHi, 
                       vars->vVelRhoHat,
                       constvars->vVelocity,
                       vars->vVelocityCoeff[Arches::AE], 
                       vars->vVelocityCoeff[Arches::AW], 
                       vars->vVelocityCoeff[Arches::AN], 
                       vars->vVelocityCoeff[Arches::AS], 
                       vars->vVelocityCoeff[Arches::AT], 
                       vars->vVelocityCoeff[Arches::AB], 
                       vars->vVelocityCoeff[Arches::AP], 
                       vars->vVelNonlinearSrc,
                       constvars->new_density,
                       cellinfo->sew, cellinfo->snsv, cellinfo->stb,
                       delta_t, ioff, joff, koff,
                       constvars->cellType,
                       d_mmWallID);

  //__________________________________
  //     Z dir 
  idxLo = patch->getSFCZFORTLowIndex();
  idxHi = patch->getSFCZFORTHighIndex();
  ioff = 0; joff = 0; koff = 1;

  fort_mm_explicit_vel(idxLo, idxHi, 
                       vars->wVelRhoHat,
                       constvars->wVelocity,
                       vars->wVelocityCoeff[Arches::AE], 
                       vars->wVelocityCoeff[Arches::AW], 
                       vars->wVelocityCoeff[Arches::AN], 
                       vars->wVelocityCoeff[Arches::AS], 
                       vars->wVelocityCoeff[Arches::AT], 
                       vars->wVelocityCoeff[Arches::AB], 
                       vars->wVelocityCoeff[Arches::AP], 
                       vars->wVelNonlinearSrc,
                       constvars->new_density,
                       cellinfo->sew, cellinfo->sns, cellinfo->stbw,
                       delta_t, ioff, joff, koff,
                       constvars->cellType,
                       d_mmWallID);
}

//****************************************************************************
// Scalar Solve for Multimaterial
//****************************************************************************
void 
BoundaryCondition::scalarLisolve_mm(const Patch* patch,
                                    double delta_t,
                                    ArchesVariables* vars,
                                    ArchesConstVariables* constvars,
                                    CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();


  fort_mm_explicit(idxLo, idxHi, vars->scalar, constvars->old_scalar,
                   constvars->scalarCoeff[Arches::AE], 
                   constvars->scalarCoeff[Arches::AW], 
                   constvars->scalarCoeff[Arches::AN], 
                   constvars->scalarCoeff[Arches::AS], 
                   constvars->scalarCoeff[Arches::AT], 
                   constvars->scalarCoeff[Arches::AB], 
                   constvars->scalarCoeff[Arches::AP], 
                   constvars->scalarNonlinearSrc, constvars->density_guess,
                   cellinfo->sew, cellinfo->sns, cellinfo->stb, 
                   delta_t,
                   constvars->cellType, d_mmWallID);

}

//****************************************************************************
// Enthalpy Solve for Multimaterial
//****************************************************************************
void 
BoundaryCondition::enthalpyLisolve_mm(const Patch* patch,
                                      double delta_t,
                                      ArchesVariables* vars,
                                      ArchesConstVariables* constvars,
                                      CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  fort_mm_explicit_oldvalue(idxLo, idxHi, vars->enthalpy, constvars->old_enthalpy,
                            constvars->scalarCoeff[Arches::AE], 
                            constvars->scalarCoeff[Arches::AW], 
                            constvars->scalarCoeff[Arches::AN], 
                            constvars->scalarCoeff[Arches::AS], 
                            constvars->scalarCoeff[Arches::AT], 
                            constvars->scalarCoeff[Arches::AB], 
                            constvars->scalarCoeff[Arches::AP], 
                            constvars->scalarNonlinearSrc, constvars->density_guess,
                            cellinfo->sew, cellinfo->sns, cellinfo->stb, 
                            delta_t,
                            constvars->cellType, d_mmWallID);

}

//****************************************************************************
// Set zero gradient for scalar for outlet and pressure BC
//****************************************************************************
void 
BoundaryCondition::scalarOutletPressureBC(const Patch* patch,
                                          ArchesVariables* vars,
                                          ArchesConstVariables* constvars)
                                          
{  
  int outlet_celltypeval = outletCellType();
  int pressure_celltypeval = pressureCellType();
  
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
  
  for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    IntVector offset = patch->faceDirection(face);
    
    for(CellIterator iter=patch->getFaceIterator__New(face, MEC); !iter.done();iter++) {
      IntVector c = *iter;
      IntVector intCell = c - offset;    // interiorCell
      
      if ((constvars->cellType[c] == outlet_celltypeval)||
          (constvars->cellType[c] == pressure_celltypeval)){
        vars->scalar[c]= vars->scalar[intCell];
      }
    }
  }
}

//****************************************************************************
// Set the inlet rho hat velocity BC
//****************************************************************************
void 
BoundaryCondition::velRhoHatInletBC(const Patch* patch,
                                    ArchesVariables* vars,
                                    ArchesConstVariables* constvars,
                                    double time_shift)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  double current_time = time + time_shift;
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  // stores cell type info for the patch with the ghost cell type
  for (int indx = 0; indx < d_numInlets; indx++) {
    // Get a copy of the current flow inlet
    FlowInlet* fi = d_flowInlets[indx];
    
    // assign flowType the value that corresponds to flow
    //CellTypeInfo flowType = FLOW;
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    fort_inlbcs(vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat,
                idxLo, idxHi, constvars->new_density, constvars->cellType, 
                fi->d_cellTypeID, current_time,
                xminus, xplus, yminus, yplus, zminus, zplus,
          fi->d_ramping_inlet_flowrate);
    
  }
}
//****************************************************************************
// Set hat velocity at the outlet
// Tangential bc's are not needed to be set for hat velocities
// Commented them out to avoid confusion
//****************************************************************************
void 
BoundaryCondition::velRhoHatOutletPressureBC(const Patch* patch,
                                             ArchesVariables* vars,
                                             ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  int outlet_celltypeval = outletCellType();
  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  int sign = 0;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if ((constvars->cellType[xminusCell] == outlet_celltypeval)||
            (constvars->cellType[xminusCell] == pressure_celltypeval)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
            vars->uVelRhoHat[currCell] = 0.0;
          else {
          if (constvars->cellType[xminusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
          if (sign * constvars->old_uVelocity[currCell] < -1.0e-10)
            vars->uVelRhoHat[currCell] = vars->uVelRhoHat[xplusCell];
          else
            vars->uVelRhoHat[currCell] = 0.0;
          }
          vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];
 /*         if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
          if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if ((constvars->cellType[xplusCell] == outlet_celltypeval)||
            (constvars->cellType[xplusCell] == pressure_celltypeval)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
            vars->uVelRhoHat[xplusCell] = 0.0;
          else {
          if (constvars->cellType[xplusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
          if (sign * constvars->old_uVelocity[xplusCell] > 1.0e-10)
            vars->uVelRhoHat[xplusCell] = vars->uVelRhoHat[currCell];
          else
            vars->uVelRhoHat[xplusCell] = 0.0;
          }
          vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];
   /*       if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
          if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if ((constvars->cellType[yminusCell] == outlet_celltypeval)||
            (constvars->cellType[yminusCell] == pressure_celltypeval)) {
     /*     if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vars->vVelRhoHat[currCell] = 0.0;
          else {
          if (constvars->cellType[yminusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
          if (sign * constvars->old_vVelocity[currCell] < -1.0e-10)
            vars->vVelRhoHat[currCell] = vars->vVelRhoHat[yplusCell];
          else
            vars->vVelRhoHat[currCell] = 0.0;
          }
          vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];
     /*     if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if ((constvars->cellType[yplusCell] == outlet_celltypeval)||
            (constvars->cellType[yplusCell] == pressure_celltypeval)) {
      /*    if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];*/
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vars->vVelRhoHat[yplusCell] = 0.0;
          else {
          if (constvars->cellType[yplusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
          if (sign * constvars->old_vVelocity[yplusCell] > 1.0e-10)
            vars->vVelRhoHat[yplusCell] = vars->vVelRhoHat[currCell];
          else
            vars->vVelRhoHat[yplusCell] = 0.0;
          }
          vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];
       /*   if (!(zminus && (colZ == idxLo.z())))
            vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];*/
        }
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        IntVector zplusCell(colX, colY, colZ+1);
        if ((constvars->cellType[zminusCell] == outlet_celltypeval)||
            (constvars->cellType[zminusCell] == pressure_celltypeval)) {
      /*    if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vars->wVelRhoHat[currCell] = 0.0;
          else {
          if (constvars->cellType[zminusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
          if (sign * constvars->old_wVelocity[currCell] < -1.0e-10)
            vars->wVelRhoHat[currCell] = vars->wVelRhoHat[zplusCell];
          else
            vars->wVelRhoHat[currCell] = 0.0;
          }
          vars->wVelRhoHat[zminusCell] = vars->wVelRhoHat[currCell];
        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if ((constvars->cellType[zplusCell] == outlet_celltypeval)||
            (constvars->cellType[zplusCell] == pressure_celltypeval)) {
       /*   if (!(xminus && (colX == idxLo.x())))
            vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
          if (!(yminus && (colY == idxLo.y())))
            vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];*/
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vars->wVelRhoHat[zplusCell] = 0.0;
          else {
          if (constvars->cellType[zplusCell] == outlet_celltypeval)
            sign = 1;
          else
            sign = -1;
          if (sign * constvars->old_wVelocity[zplusCell] > 1.0e-10)
            vars->wVelRhoHat[zplusCell] = vars->wVelRhoHat[currCell];
          else
            vars->wVelRhoHat[zplusCell] = 0.0;
          }
          vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];
        }
      }
    }
  }
}
//****************************************************************************
// Set zero gradient for tangent velocity on outlet and pressure bc
//****************************************************************************
void 
BoundaryCondition::velocityOutletPressureTangentBC(const Patch* patch,
                                                   ArchesVariables* vars,
                                                   ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  int outlet_celltypeval = outletCellType();
  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  for (int index = 1; index <= Arches::NDIM; ++index) {
    if (xminus) {
      int colX = idxLo.x();
      int maxY = idxHi.y();
      if (yplus){
        maxY++;
      }
      int maxZ = idxHi.z();
      if (zplus){
         maxZ++;
      }

      for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
        for (int colY = idxLo.y(); colY <= maxY; colY ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector xminusyminusCell(colX-1, colY-1, colZ);
          IntVector xminuszminusCell(colX-1, colY, colZ-1);
          if ((constvars->cellType[xminusCell] == pressure_celltypeval)||
              (constvars->cellType[xminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
               if (!(yplus && (colY == maxY)))
                 vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
                  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
            }
          }
          else {
            if ((constvars->cellType[xminusyminusCell] == pressure_celltypeval)
                ||(constvars->cellType[xminusyminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
                 if (!(zplus && (colZ == maxZ)))
                   vars->vVelRhoHat[xminusCell] = vars->vVelRhoHat[currCell];
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
            if ((constvars->cellType[xminuszminusCell] == pressure_celltypeval)
                ||(constvars->cellType[xminuszminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
                 if (!(yplus && (colY == maxY)))
                   vars->wVelRhoHat[xminusCell] = vars->wVelRhoHat[currCell];
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
          }
        }
      }
    }
    if (xplus) {
      int colX = idxHi.x();
      int maxY = idxHi.y();
      if (yplus) maxY++;
      int maxZ = idxHi.z();
      if (zplus) maxZ++;
      for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
        for (int colY = idxLo.y(); colY <= maxY; colY ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);
          IntVector xplusyminusCell(colX+1, colY-1, colZ);
          IntVector xpluszminusCell(colX+1, colY, colZ-1);
          if ((constvars->cellType[xplusCell] == pressure_celltypeval)||
              (constvars->cellType[xplusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
             break;
             case Arches::YDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
               if (!(yplus && (colY == maxY)))
                 vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
                  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
            }
          }
          else {
            if ((constvars->cellType[xplusyminusCell] == pressure_celltypeval)
                ||(constvars->cellType[xplusyminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
                 if (!(zplus && (colZ == maxZ)))
                   vars->vVelRhoHat[xplusCell] = vars->vVelRhoHat[currCell];
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
            if ((constvars->cellType[xpluszminusCell] == pressure_celltypeval)
                ||(constvars->cellType[xpluszminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
                 if (!(yplus && (colY == maxY)))
                   vars->wVelRhoHat[xplusCell] = vars->wVelRhoHat[currCell];
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
          }
        }
      }
    }
    if (yminus) {
      int colY = idxLo.y();
      int maxX = idxHi.x();
      if (xplus) maxX++;
      int maxZ = idxHi.z();
      if (zplus) maxZ++;
      for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
        for (int colX = idxLo.x(); colX <= maxX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector yminusxminusCell(colX-1, colY-1, colZ);
          IntVector yminuszminusCell(colX, colY-1, colZ-1);
          if ((constvars->cellType[yminusCell] == pressure_celltypeval)||
              (constvars->cellType[yminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
               if (!(xplus && (colX == maxX)))
                 vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
                  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
            }
          }
          else {
            if ((constvars->cellType[yminusxminusCell] == pressure_celltypeval)
                ||(constvars->cellType[yminusxminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
                 if (!(zplus && (colZ == maxZ)))
                   vars->uVelRhoHat[yminusCell] = vars->uVelRhoHat[currCell];
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
            if ((constvars->cellType[yminuszminusCell] == pressure_celltypeval)
                ||(constvars->cellType[yminuszminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
                 if (!(xplus && (colX == maxX)))
                   vars->wVelRhoHat[yminusCell] = vars->wVelRhoHat[currCell];
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
          }  
        }
      }
    }
    if (yplus) {
      int colY = idxHi.y();
      int maxX = idxHi.x();
      if (xplus) maxX++;
      int maxZ = idxHi.z();
      if (zplus) maxZ++;
      for (int colZ = idxLo.z(); colZ <= maxZ; colZ ++) {
        for (int colX = idxLo.x(); colX <= maxX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector yplusCell(colX, colY+1, colZ);
          IntVector yplusxminusCell(colX-1, colY+1, colZ);
          IntVector ypluszminusCell(colX, colY+1, colZ-1);
          if ((constvars->cellType[yplusCell] == pressure_celltypeval)||
              (constvars->cellType[yplusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(zplus && (colZ == maxZ)))
                 vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
             break;
             case Arches::ZDIR:
               if (!(xplus && (colX == maxX)))
                 vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];
             break;
             default:
                  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
            }
          }
          else {
            if ((constvars->cellType[yplusxminusCell] == pressure_celltypeval)
                ||(constvars->cellType[yplusxminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
                 if (!(zplus && (colZ == maxZ)))
                   vars->uVelRhoHat[yplusCell] = vars->uVelRhoHat[currCell];
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
            if ((constvars->cellType[ypluszminusCell] == pressure_celltypeval)
                ||(constvars->cellType[ypluszminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
                 if (!(xplus && (colX == maxX)))
                   vars->wVelRhoHat[yplusCell] = vars->wVelRhoHat[currCell];
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
          }
        }
      }
    }
    if (zminus) {
      int colZ = idxLo.z();
      int maxX = idxHi.x();
      if (xplus) maxX++;
      int maxY = idxHi.y();
      if (yplus) maxY++;
      for (int colY = idxLo.y(); colY <= maxY; colY ++) {
        for (int colX = idxLo.x(); colX <= maxX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          IntVector zminusxminusCell(colX-1, colY, colZ-1);
          IntVector zminusyminusCell(colX, colY-1, colZ-1);
          if ((constvars->cellType[zminusCell] == pressure_celltypeval)||
              (constvars->cellType[zminusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(yplus && (colY == maxY)))
                 vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
               if (!(xplus && (colX == maxX)))
                 vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
             break;
             default:
                  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
            }
          }
          else {
            if ((constvars->cellType[zminusxminusCell] == pressure_celltypeval)
                ||(constvars->cellType[zminusxminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
                 if (!(yplus && (colY == maxY)))
                   vars->uVelRhoHat[zminusCell] = vars->uVelRhoHat[currCell];
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
            if ((constvars->cellType[zminusyminusCell] == pressure_celltypeval)
                ||(constvars->cellType[zminusyminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
                 if (!(xplus && (colX == maxX)))
                   vars->vVelRhoHat[zminusCell] = vars->vVelRhoHat[currCell];
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
          }
        }
      }
    }
    if (zplus) {
      int colZ = idxHi.z();
      int maxX = idxHi.x();
      if (xplus) maxX++;
      int maxY = idxHi.y();
      if (yplus) maxY++;
      for (int colY = idxLo.y(); colY <= maxY; colY ++) {
        for (int colX = idxLo.x(); colX <= maxX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector zplusCell(colX, colY, colZ+1);
          IntVector zplusxminusCell(colX-1, colY, colZ+1);
          IntVector zplusyminusCell(colX, colY-1, colZ+1);
          if ((constvars->cellType[zplusCell] == pressure_celltypeval)||
              (constvars->cellType[zplusCell] == outlet_celltypeval)) {
            switch (index) {
             case Arches::XDIR:
               if (!(yplus && (colY == maxY)))
                 vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
             break;
             case Arches::YDIR:
               if (!(xplus && (colX == maxX)))
                 vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];
             break;
             case Arches::ZDIR:
             break;
             default:
                  throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
            }
          }
          else {
            if ((constvars->cellType[zplusxminusCell] == pressure_celltypeval)
                ||(constvars->cellType[zplusxminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
                 if (!(yplus && (colY == maxY)))
                   vars->uVelRhoHat[zplusCell] = vars->uVelRhoHat[currCell];
               break;
               case Arches::YDIR:
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
            if ((constvars->cellType[zplusyminusCell] == pressure_celltypeval)
                ||(constvars->cellType[zplusyminusCell] == outlet_celltypeval)) {
              switch (index) {
               case Arches::XDIR:
               break;
               case Arches::YDIR:
                 if (!(xplus && (colX == maxX)))
                   vars->vVelRhoHat[zplusCell] = vars->vVelRhoHat[currCell];
               break;
               case Arches::ZDIR:
               break;
               default:
                    throw InvalidValue("Invalid index in velocityPressureBC", __FILE__, __LINE__);
              }
            }
          }
        }
      }
    }
  }
}
//****************************************************************************
// Add pressure gradient to outlet velocity
//****************************************************************************
void 
BoundaryCondition::addPresGradVelocityOutletPressureBC(const Patch* patch,
                                                       CellInformation* cellinfo,
                                                       const double delta_t,
                                                       ArchesVariables* vars,
                                                       ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  int outlet_celltypeval = outletCellType();
  int pressure_celltypeval = pressureCellType();

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
  //__________________________________
  //
  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        if ((constvars->cellType[xminusCell] == outlet_celltypeval)||
            (constvars->cellType[xminusCell] == pressure_celltypeval)) {
           double avdenlow = 0.5 * (constvars->density[currCell] +
                                    constvars->density[xminusCell]);

           vars->uVelRhoHat[currCell] -= 2.0*delta_t*constvars->pressure[currCell]/
                                (cellinfo->dxpw[colX] * avdenlow);

           vars->uVelRhoHat[xminusCell] = vars->uVelRhoHat[currCell];

        }
      }
    }
  }
  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if ((constvars->cellType[xplusCell] == outlet_celltypeval)||
            (constvars->cellType[xplusCell] == pressure_celltypeval)) {
           double avden = 0.5 * (constvars->density[xplusCell] +
                                 constvars->density[currCell]);

           vars->uVelRhoHat[xplusCell] += 2.0*delta_t*constvars->pressure[currCell]/
                                (cellinfo->dxpw[colX+1] * avden);

           vars->uVelRhoHat[xplusplusCell] = vars->uVelRhoHat[xplusCell];
        }
      }
    }
  }
  //__________________________________
  //
  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        if ((constvars->cellType[yminusCell] == outlet_celltypeval)||
            (constvars->cellType[yminusCell] == pressure_celltypeval)) {
           double avdenlow = 0.5 * (constvars->density[currCell] +
                                    constvars->density[yminusCell]);

           vars->vVelRhoHat[currCell] -= 2.0*delta_t*constvars->pressure[currCell]/
                                (cellinfo->dyps[colY] * avdenlow);

           vars->vVelRhoHat[yminusCell] = vars->vVelRhoHat[currCell];

        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if ((constvars->cellType[yplusCell] == outlet_celltypeval)||
            (constvars->cellType[yplusCell] == pressure_celltypeval)) {
           double avden = 0.5 * (constvars->density[yplusCell] +
                                 constvars->density[currCell]);

           vars->vVelRhoHat[yplusCell] += 2.0*delta_t*constvars->pressure[currCell]/
                                (cellinfo->dyps[colY+1] * avden);

           vars->vVelRhoHat[yplusplusCell] = vars->vVelRhoHat[yplusCell];

        }
      }
    }
  }
  //__________________________________
  //
  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        if ((constvars->cellType[zminusCell] == outlet_celltypeval)||
            (constvars->cellType[zminusCell] == pressure_celltypeval)) {
           double avdenlow = 0.5 * (constvars->density[currCell] +
                                    constvars->density[zminusCell]);

           vars->wVelRhoHat[currCell] -= 2.0*delta_t*constvars->pressure[currCell]/
                                (cellinfo->dzpb[colZ] * avdenlow);

           vars->wVelRhoHat[zminusCell] = vars->wVelRhoHat[currCell];

        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if ((constvars->cellType[zplusCell] == outlet_celltypeval)||
            (constvars->cellType[zplusCell] == pressure_celltypeval)) {
           double avden = 0.5 * (constvars->density[zplusCell] +
                                 constvars->density[currCell]);

           vars->wVelRhoHat[zplusCell] += 2.0*delta_t*constvars->pressure[currCell]/
                                (cellinfo->dzpb[colZ+1] * avden);

           vars->wVelRhoHat[zplusplusCell] = vars->wVelRhoHat[zplusCell];

        }
      }
    }
  }

}
//****************************************************************************
// Schedule computation of mass balance for the outlet velocity correction
//****************************************************************************
void
BoundaryCondition::sched_getFlowINOUT(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls,
                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "BoundaryCondition::getFlowINOUT" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &BoundaryCondition::getFlowINOUT,
                          timelabels);
  
  
  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  tsk->requires(Task::NewDW, d_lab->d_filterdrhodtLabel,gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,    gac, 1);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gn, 0);

  tsk->computes(timelabels->flowIN);
  tsk->computes(timelabels->flowOUT);
  tsk->computes(timelabels->denAccum);
  tsk->computes(timelabels->floutbc);
  tsk->computes(timelabels->areaOUT);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Get mass balance for the outlet velocity correction
//****************************************************************************
void 
BoundaryCondition::getFlowINOUT(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constCCVariable<double> filterdrhodt;
    constCCVariable<double> density;
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
  
    new_dw->get(filterdrhodt, d_lab->d_filterdrhodtLabel, indx, patch, gn, 0);
    new_dw->get(cellType,     d_lab->d_cellTypeLabel,     indx, patch, gac,1);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, indx, patch)){ 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }else{ 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(density, d_lab->d_densityCPLabel,       indx, patch, gn, 0);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);

    // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getFortranCellLowIndex__New();
    IntVector idxHi = patch->getFortranCellHighIndex__New();

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    double flowIN = 0.0;
    double flowOUT = 0.0;
    double denAccum = 0.0;
    double floutbc = 0.0;
    double areaOUT = 0.0;
    double varIN  = 0.0;
    double varOUT  = 0.0;

    for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
        for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
          IntVector currCell(ii,jj,kk);
          denAccum += filterdrhodt[currCell];
        }
      }
    }

    if (xminus||xplus||yminus||yplus||zminus||zplus) {

      bool doing_balance = false;
      for (int indx = 0; indx < d_numInlets; indx++) {

        // Get a copy of the current flow inlet
        // assign flowType the value that corresponds to flow
        //CellTypeInfo flowType = FLOW;
        FlowInlet* fi = d_flowInlets[indx];
        double fout = 0.0;
        fort_inlpresbcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
                           density, cellType, fi->d_cellTypeID,
                           flowIN, fout, cellinfo->sew, cellinfo->sns,
                           cellinfo->stb, xminus, xplus, yminus, yplus,
                           zminus, zplus, doing_balance,
                           density, varIN, varOUT);
        if (fout > 0.0){
          throw InvalidValue("Flow comming out of inlet", __FILE__, __LINE__);
        }
      } 

      if (d_pressureBoundary) {
        int pressure_celltypeval = d_pressureBC->d_cellTypeID;
        fort_inlpresbcinout(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
                            density, cellType, pressure_celltypeval,
                            flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
                            cellinfo->stb, xminus, xplus, yminus, yplus,
                            zminus, zplus, doing_balance,
                            density, varIN, varOUT);
      }
      if (d_outletBoundary) {
        int outlet_celltypeval = d_outletBC->d_cellTypeID;
        if (xminus) {
          int colX = idxLo.x();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector xminusCell(colX-1, colY, colZ);

              if (cellType[xminusCell] == outlet_celltypeval) {
                double avdenlow = 0.5 * (density[currCell] + density[xminusCell]);
                floutbc -= avdenlow*uVelocity[currCell] *
                               cellinfo->sns[colY] * cellinfo->stb[colZ];
                areaOUT += cellinfo->sns[colY] * cellinfo->stb[colZ];
              }
            }
          }
        }
        if (xplus) {
          int colX = idxHi.x();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector xplusCell(colX+1, colY, colZ);

              if (cellType[xplusCell] == outlet_celltypeval) {
                double avden = 0.5 * (density[xplusCell] +
                                       density[currCell]);
                floutbc += avden*uVelocity[xplusCell] *
                            cellinfo->sns[colY] * cellinfo->stb[colZ];
                areaOUT += cellinfo->sns[colY] * cellinfo->stb[colZ];
              }
            }
          }
        }
        if (yminus) {
          int colY = idxLo.y();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector yminusCell(colX, colY-1, colZ);

              if (cellType[yminusCell] == outlet_celltypeval) {
                 double avdenlow = 0.5 * (density[currCell] +
                                          density[yminusCell]);
                 flowOUT -= Min(0.0,avdenlow*vVelocity[currCell] *
                          cellinfo->sew[colX] * cellinfo->stb[colZ]);
                 flowIN += Max(0.0,avdenlow*vVelocity[currCell] *
                          cellinfo->sew[colX] * cellinfo->stb[colZ]);
                 areaOUT += cellinfo->sew[colX] * cellinfo->stb[colZ];
              }
            }
          }
        }
        if (yplus) {
          int colY = idxHi.y();
          for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector yplusCell(colX, colY+1, colZ);

              if (cellType[yplusCell] == outlet_celltypeval) {
                double avden = 0.5 * (density[yplusCell] +
                                      density[currCell]);
                flowOUT += Max(0.0,avden*vVelocity[yplusCell] *
                              cellinfo->sew[colX] * cellinfo->stb[colZ]);
                flowIN -= Min(0.0,avden*vVelocity[yplusCell] *
                              cellinfo->sew[colX] * cellinfo->stb[colZ]);
                areaOUT += cellinfo->sew[colX] * cellinfo->stb[colZ];
              }
            }
          }
        }
        if (zminus) {
          int colZ = idxLo.z();
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector zminusCell(colX, colY, colZ-1);

              if (cellType[zminusCell] == outlet_celltypeval) {
                double avdenlow = 0.5 * (density[currCell] +
                                         density[zminusCell]);
                flowOUT -= Min(0.0,avdenlow*wVelocity[currCell] *
                              cellinfo->sew[colX] * cellinfo->sns[colY]);
                flowIN += Max(0.0,avdenlow*wVelocity[currCell] *
                              cellinfo->sew[colX] * cellinfo->sns[colY]);
                areaOUT += cellinfo->sew[colX] * cellinfo->sns[colY];
              }
            }
          }
        }
        if (zplus) {
          int colZ = idxHi.z();
          for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
            for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
              IntVector currCell(colX, colY, colZ);
              IntVector zplusCell(colX, colY, colZ+1);

              if (cellType[zplusCell] == outlet_celltypeval) {
                double avden = 0.5 * (density[zplusCell] +
                                       density[currCell]);
                flowOUT += Max(0.0,avden*wVelocity[zplusCell] *
                               cellinfo->sew[colX] * cellinfo->sns[colY]);
                flowIN -= Min(0.0,avden*wVelocity[zplusCell] *
                               cellinfo->sew[colX] * cellinfo->sns[colY]);
                areaOUT += cellinfo->sew[colX] * cellinfo->sns[colY];
              }
            }
          }
        } // zplus
      }
    }

  new_dw->put(sum_vartype(flowIN), timelabels->flowIN);
  new_dw->put(sum_vartype(flowOUT), timelabels->flowOUT);
  new_dw->put(sum_vartype(denAccum), timelabels->denAccum);
  new_dw->put(sum_vartype(floutbc), timelabels->floutbc);
  new_dw->put(sum_vartype(areaOUT), timelabels->areaOUT);
  }
}

//****************************************************************************
// Schedule mass balance computation
// Does not perform any velocity correction
// Named for historical reasons
//****************************************************************************
void BoundaryCondition::sched_correctVelocityOutletBC(SchedulerP& sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* matls,
                                                      const TimeIntegratorLabel* timelabels)
{
  string taskname =  "BoundaryCondition::correctVelocityOutletBC" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &BoundaryCondition::correctVelocityOutletBC,
                          timelabels);
  
  tsk->requires(Task::NewDW, timelabels->flowIN);
  tsk->requires(Task::NewDW, timelabels->flowOUT);
  tsk->requires(Task::NewDW, timelabels->denAccum);
  tsk->requires(Task::NewDW, timelabels->floutbc);
  tsk->requires(Task::NewDW, timelabels->areaOUT);

  if (timelabels->integrator_last_step){
    tsk->computes(d_lab->d_uvwoutLabel);
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Correct outlet velocity
//****************************************************************************
void 
BoundaryCondition::correctVelocityOutletBC(const ProcessorGroup*,
                                           const PatchSubset* ,
                                           const MaterialSubset*,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw,
                                           const TimeIntegratorLabel* timelabels)
{
  sum_vartype sum_totalFlowIN, sum_totalFlowOUT, sum_netflowOutbc,
              sum_totalAreaOUT, sum_denAccum;
  double totalFlowIN, totalFlowOUT, netFlowOUT_outbc, totalAreaOUT, denAccum;

  new_dw->get(sum_totalFlowIN,  timelabels->flowIN);
  new_dw->get(sum_totalFlowOUT, timelabels->flowOUT);
  new_dw->get(sum_denAccum,     timelabels->denAccum);
  new_dw->get(sum_netflowOutbc, timelabels->floutbc);
  new_dw->get(sum_totalAreaOUT, timelabels->areaOUT);
                        
  totalFlowIN      = sum_totalFlowIN;
  totalFlowOUT     = sum_totalFlowOUT;
  netFlowOUT_outbc = sum_netflowOutbc;
  totalAreaOUT     = sum_totalAreaOUT;
  denAccum         = sum_denAccum;
  double uvwcorr = 0.0;

  d_overallMB = fabs((totalFlowIN - denAccum - totalFlowOUT - 
                       netFlowOUT_outbc)/(totalFlowIN+1.e-20));

  if (d_outletBoundary) {
    if (totalAreaOUT > 0.0) {
      uvwcorr = (totalFlowIN - denAccum - totalFlowOUT - netFlowOUT_outbc)/
                 totalAreaOUT;
    }else {
       throw ProblemSetupException("ERROR: The specified outlet has zero area", __FILE__, __LINE__);
    }
  }else{
    uvwcorr = 0.0;
  }


  if (d_overallMB > 0.0){
    proc0cout << "Overall Mass Balance " << log10(d_overallMB/1.e-7+1.e-20)<<endl;
  }
  proc0cout << "Total flow in               " << totalFlowIN << endl;
  proc0cout << "Total flow out              " << totalFlowOUT<< endl;
  proc0cout << "Total flow out BC           " << netFlowOUT_outbc << endl;
  proc0cout << "Total Area out              " << totalAreaOUT << endl;
  proc0cout << "Overall velocity correction " << uvwcorr << endl;
  proc0cout << "Density accumulation        " << denAccum << endl;

  
  if (timelabels->integrator_last_step){
    new_dw->put(delt_vartype(uvwcorr), d_lab->d_uvwoutLabel);
  }
}
//****************************************************************************
// Schedule init inlet bcs
//****************************************************************************
void 
BoundaryCondition::sched_initInletBC(SchedulerP& sched, 
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::initInletBC",this,
                          &BoundaryCondition::initInletBC);

  // This task requires cellTypeVariable and areaLabel for inlet boundary
  // Also densityIN, [u,v,w] velocityIN, scalarIN
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,  Ghost::None, 0);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None, 0);
    
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);
  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  tsk->computes(d_lab->d_densityOldOldLabel);

//#ifdef divergenceconstraint
    tsk->computes(d_lab->d_divConstraintLabel);
//#endif
  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually initialize inlet BCs
//****************************************************************************
void 
BoundaryCondition::initInletBC(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse*,
                               DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    constCCVariable<int> cellType;
    constCCVariable<double> density;
    CCVariable<double> density_oldold;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;

    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(uVelRhoHat, d_lab->d_uVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(vVelRhoHat, d_lab->d_vVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(wVelRhoHat, d_lab->d_wVelRhoHatLabel, indx, patch);
    
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::None, 0);
    new_dw->get(density, d_lab->d_densityCPLabel, indx, patch, Ghost::None, 0);
    new_dw->allocateAndPut(density_oldold, d_lab->d_densityOldOldLabel, indx, patch);

    IntVector idxLo = patch->getFortranCellLowIndex__New();
    IntVector idxHi = patch->getFortranCellHighIndex__New();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    if (d_inletBoundary) {
      double current_time = 0.0; 
      for (int indx = 0; indx < d_numInlets; indx++) {
        // Get a copy of the current flow inlet
        FlowInlet* fi = d_flowInlets[indx];
    
        fort_inlbcs(uVelocity, vVelocity, wVelocity,
                    idxLo, idxHi, density, cellType, 
                    fi->d_cellTypeID, current_time,
                    xminus, xplus, yminus, yplus, zminus, zplus,
                    fi->d_ramping_inlet_flowrate);
      }
    }  
    
    density_oldold.copyData(density); // copy old into new
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 

//#ifdef divergenceconstraint    
    CCVariable<double> divergence;
    new_dw->allocateAndPut(divergence,
                             d_lab->d_divConstraintLabel, indx, patch);
    divergence.initialize(0.0);
//#endif

  }
}
//****************************************************************************
// Schedule computation of mixture fraction flow rate
//****************************************************************************
void
BoundaryCondition::sched_getScalarFlowRate(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::getScalarFlowRate", this,
                          &BoundaryCondition::getScalarFlowRate);
  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel,      gn, 0);
  if (d_carbon_balance){
    tsk->requires(Task::NewDW, d_lab->d_co2INLabel, gn, 0);
  }
  if (d_sulfur_balance){
    tsk->requires(Task::NewDW, d_lab->d_so2INLabel, gn, 0);
  }
  if (d_enthalpySolve){
    tsk->requires(Task::NewDW, d_lab->d_enthalpySPLabel, gn, 0);
  }
  
  tsk->computes(d_lab->d_scalarFlowRateLabel);
  if (d_carbon_balance){
    tsk->computes(d_lab->d_CO2FlowRateLabel);
  }
  if (d_carbon_balance_es){
    tsk->computes(d_lab->d_CO2FlowRateESLabel);
  }
  if (d_sulfur_balance){
    tsk->computes(d_lab->d_SO2FlowRateLabel);
  }
  if (d_sulfur_balance_es){
    tsk->computes(d_lab->d_SO2FlowRateESLabel);
  }                  
  if (d_enthalpySolve){
    tsk->computes(d_lab->d_enthalpyFlowRateLabel);
  }

  for (BoundaryCondition::SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){

    //this will need to change when the new table stuff is ready
    const VarLabel* temp1 = VarLabel::find(iter->first);
    tsk->requires(Task::NewDW, temp1, gn, 0); // Species needed for calculation

    tsk->computes(iter->second.flowRateLabel);
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Get mixture fraction flow rate
//****************************************************************************
void 
BoundaryCondition::getScalarFlowRate(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse*,
                                     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    ArchesConstVariables constVars;

    constCCVariable<double> scalar;
    constCCVariable<double> co2;
    constCCVariable<double> co2_es;
    constCCVariable<double> so2_es;
    constCCVariable<double> so2;
    constCCVariable<double> enthalpy;
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(constVars.cellType, d_lab->d_cellTypeLabel, indx, patch, gac,1);

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, indx, patch)){ 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    }else{ 
      throw VariableNotFoundInGrid("cellInformation"," ", __FILE__, __LINE__);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constVars.density,   d_lab->d_densityCPLabel,     indx, patch, gn, 0);
    new_dw->get(constVars.uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(constVars.vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(constVars.wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(scalar,              d_lab->d_scalarSPLabel,      indx, patch, gn, 0);

    double scalarIN = 0.0;
    double scalarOUT = 0.0;
    getVariableFlowRate(patch, cellinfo, &constVars, scalar,
                        &scalarIN, &scalarOUT); 

    new_dw->put(sum_vartype(scalarOUT-scalarIN), d_lab->d_scalarFlowRateLabel);

    double co2IN = 0.0;
    double co2OUT = 0.0;
    if (d_carbon_balance) {
      new_dw->get(co2, d_lab->d_co2INLabel, indx, patch, gn, 0);
      getVariableFlowRate(patch, cellinfo, &constVars, co2,
                        &co2IN, &co2OUT); 
      new_dw->put(sum_vartype(co2OUT-co2IN), d_lab->d_CO2FlowRateLabel);
    }
    co2IN = 0.0;
    co2OUT = 0.0;
    if (d_carbon_balance_es) {

      std::vector<ExtraScalarSolver*>::iterator iss; 
      for (iss=d_extraScalars->begin(); iss!=d_extraScalars->end(); ++iss){
        bool checkCarbonBalance = (*iss)->doCarbonBalance();
        if (checkCarbonBalance){         
          const VarLabel* templabel = (*iss)->getScalarLabel();
                  
          new_dw->get(co2_es, templabel, indx, patch, gn, 0);
          getVariableFlowRate(patch, cellinfo, &constVars, co2_es,
                  &co2IN, &co2OUT); 
          new_dw->put(sum_vartype(co2OUT-co2IN), d_lab->d_CO2FlowRateESLabel);
        }
      }
    }

    // --- new efficiency calculator --- 
    for (BoundaryCondition::SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){
      double IN = 0.0;
      double OUT = 0.0;

      constCCVariable<double> species;
      const VarLabel* temp = VarLabel::find(iter->first);

      new_dw->get(species, temp, indx, patch, gn, 0);  // this is like co2 scalar

      getVariableFlowRate(patch, cellinfo, &constVars, species, &IN, &OUT);
      new_dw->put(sum_vartype(OUT-IN), iter->second.flowRateLabel);

    }

    double so2IN = 0.0;
    double so2OUT = 0.0;
    if (d_sulfur_balance) {
      new_dw->get(so2, d_lab->d_so2INLabel, indx, patch, gn, 0);
      getVariableFlowRate(patch, cellinfo, &constVars, so2, &so2IN, &so2OUT); 
      new_dw->put(sum_vartype(so2OUT-so2IN), d_lab->d_SO2FlowRateLabel);
    } 
        
    so2IN = 0.0;
    so2OUT = 0.0; 
    if (d_sulfur_balance_es) {

      std::vector<ExtraScalarSolver*>::iterator iss; 
      for (iss=d_extraScalars->begin(); iss!=d_extraScalars->end(); ++iss){
        bool checkSulfurBalance = (*iss)->doSulfurBalance();
        if (checkSulfurBalance){         
          
          const VarLabel* templabel = (*iss)->getScalarLabel();
                  
          new_dw->get(so2_es, templabel, indx, patch, gn, 0);
          getVariableFlowRate(patch, cellinfo, &constVars, so2_es,
                  &so2IN, &so2OUT); 
          new_dw->put(sum_vartype(so2OUT-so2IN), d_lab->d_SO2FlowRateESLabel);
        }
      }
    }
          
    double enthalpyIN = 0.0;
    double enthalpyOUT = 0.0;
    if (d_enthalpySolve) {
      new_dw->get(enthalpy, d_lab->d_enthalpySPLabel, indx, patch, gn, 0);
      getVariableFlowRate(patch, cellinfo, &constVars, enthalpy,
                        &enthalpyIN, &enthalpyOUT); 
      new_dw->put(sum_vartype(enthalpyOUT-enthalpyIN), d_lab->d_enthalpyFlowRateLabel);
    }
  }
}

//****************************************************************************
// Schedule scalar efficiency computation
//****************************************************************************
void BoundaryCondition::sched_getScalarEfficiency(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::getScalarEfficiency", this,
                          &BoundaryCondition::getScalarEfficiency);
  
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::OldDW, d_flowInlets[ii]->d_flowRate_label);
    tsk->computes(d_flowInlets[ii]->d_flowRate_label);
  }

  tsk->requires(Task::NewDW, d_lab->d_scalarFlowRateLabel);
  tsk->computes(d_lab->d_scalarEfficiencyLabel);

  if (d_carbon_balance) {
    tsk->requires(Task::NewDW, d_lab->d_CO2FlowRateLabel);
    tsk->computes(d_lab->d_carbonEfficiencyLabel);
  }
  if (d_carbon_balance_es) {
    tsk->requires(Task::NewDW, d_lab->d_CO2FlowRateESLabel);
    tsk->computes(d_lab->d_carbonEfficiencyESLabel);
  }
  if (d_sulfur_balance) {
    tsk->requires(Task::NewDW, d_lab->d_SO2FlowRateLabel);
    tsk->computes(d_lab->d_sulfurEfficiencyLabel);
  }
  if (d_sulfur_balance_es) {
    tsk->requires(Task::NewDW, d_lab->d_SO2FlowRateESLabel);
    tsk->computes(d_lab->d_sulfurEfficiencyESLabel);
  }
  if (d_enthalpySolve) {
    tsk->requires(Task::NewDW, d_lab->d_enthalpyFlowRateLabel);
    tsk->requires(Task::NewDW, d_lab->d_totalRadSrcLabel);
    tsk->computes(d_lab->d_normTotalRadSrcLabel);
    tsk->computes(d_lab->d_enthalpyEfficiencyLabel);
  }

  for ( EfficiencyMap::iterator iter = d_effVars.begin(); iter != d_effVars.end(); iter++){
    tsk->computes(iter->second.label);
  }

  for ( SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){
    tsk->requires(Task::NewDW, iter->second.flowRateLabel);
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Correct outlet velocity
//****************************************************************************
void 
BoundaryCondition::getScalarEfficiency(const ProcessorGroup*,
                                       const PatchSubset* ,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
    sum_vartype sum_scalarFlowRate, sum_CO2FlowRate, sum_SO2FlowRate, sum_enthalpyFlowRate;
    sum_vartype sum_CO2FlowRateES, sum_SO2FlowRateES;
    sum_vartype sum_totalRadSrc;
    delt_vartype flowRate;
    double scalarFlowRate = 0.0;
    double CO2FlowRate = 0.0;
    double CO2FlowRateES = 0.0;
    double SO2FlowRate = 0.0;
    double SO2FlowRateES = 0.0;
    double enthalpyFlowRate = 0.0;
    double totalFlowRate = 0.0;
    double totalCarbonFlowRate = 0.0;
    double totalCarbonFlowRateES = 0.0;
    double totalSulfurFlowRate = 0.0;
    double totalSulfurFlowRateES = 0.0;
    double totalEnthalpyFlowRate = 0.0;
    double scalarEfficiency = 0.0;
    double carbonEfficiency = 0.0;
    double carbonEfficiencyES = 0.0;
    double sulfurEfficiency = 0.0;
    double sulfurEfficiencyES = 0.0;
    double enthalpyEfficiency = 0.0;
    double totalRadSrc = 0.0;
    double normTotalRadSrc = 0.0;

    new_dw->get(sum_scalarFlowRate, d_lab->d_scalarFlowRateLabel);
    scalarFlowRate = sum_scalarFlowRate;
    if (d_carbon_balance) {
      new_dw->get(sum_CO2FlowRate, d_lab->d_CO2FlowRateLabel);
      CO2FlowRate = sum_CO2FlowRate;
    }
    if (d_carbon_balance_es) {
      new_dw->get(sum_CO2FlowRateES, d_lab->d_CO2FlowRateESLabel);
      CO2FlowRateES = sum_CO2FlowRateES;
    }

    if (d_sulfur_balance) {
      new_dw->get(sum_SO2FlowRate, d_lab->d_SO2FlowRateLabel);
      SO2FlowRate = sum_SO2FlowRate;
    }
    if (d_sulfur_balance_es) {
      new_dw->get(sum_SO2FlowRateES, d_lab->d_SO2FlowRateESLabel);
      SO2FlowRateES = sum_SO2FlowRateES;
    }
    if (d_enthalpySolve) {
      new_dw->get(sum_enthalpyFlowRate, d_lab->d_enthalpyFlowRateLabel);
      enthalpyFlowRate = sum_enthalpyFlowRate;
      new_dw->get(sum_totalRadSrc, d_lab->d_totalRadSrcLabel);
      totalRadSrc = sum_totalRadSrc;
    }
    for (int indx = 0; indx < d_numInlets; indx++) {
      FlowInlet* fi = d_flowInlets[indx];
      old_dw->get(flowRate, d_flowInlets[indx]->d_flowRate_label);
      d_flowInlets[indx]->flowRate = flowRate;
      fi->flowRate = flowRate;
      new_dw->put(flowRate, d_flowInlets[indx]->d_flowRate_label);
      double scalarValue = fi->streamMixturefraction.d_mixVars[0];
      if (scalarValue > 0.0)
          totalFlowRate += fi->flowRate;
      if ((d_carbon_balance)&&(scalarValue > 0.0))
            totalCarbonFlowRate += fi->flowRate * fi->fcr;
      if ((d_carbon_balance_es)&&(scalarValue > 0.0))
            totalCarbonFlowRateES += fi->flowRate * fi->fcr;
      if ((d_sulfur_balance)&&(scalarValue > 0.0))
            totalSulfurFlowRate += fi->flowRate * fi->fsr;
      if ((d_sulfur_balance_es)&&(scalarValue > 0.0))
            totalSulfurFlowRateES += fi->flowRate * fi->fsr;
      if ((d_enthalpySolve)&&(scalarValue > 0.0))
            totalEnthalpyFlowRate += fi->flowRate * fi->calcStream.getEnthalpy();
    }
    if (totalFlowRate > 0.0){
      scalarEfficiency = scalarFlowRate / totalFlowRate;
    }else{ 
      proc0cout << "WARNING! No mixture fraction in the domain."<<endl;
    }
    new_dw->put(delt_vartype(scalarEfficiency), d_lab->d_scalarEfficiencyLabel);

    if (d_carbon_balance) {
      if (totalCarbonFlowRate > 0.0)
        carbonEfficiency = CO2FlowRate * 12.0/44.0 /totalCarbonFlowRate;
      else 
        throw InvalidValue("No carbon in the domain", __FILE__, __LINE__);
      new_dw->put(delt_vartype(carbonEfficiency), d_lab->d_carbonEfficiencyLabel);
    }
    if (d_carbon_balance_es) {
      if (totalCarbonFlowRateES > 0.0)
        carbonEfficiencyES = CO2FlowRateES * 12.0/44.0 /totalCarbonFlowRateES;
      else 
        throw InvalidValue("No carbon in the domain from ExtraScalar", __FILE__, __LINE__);
      new_dw->put(delt_vartype(carbonEfficiencyES), d_lab->d_carbonEfficiencyESLabel);
    }

    if (d_sulfur_balance) {
      if (totalSulfurFlowRate > 0.0)
        sulfurEfficiency = SO2FlowRate * 32.0/64.0 /totalSulfurFlowRate;
      else 
        throw InvalidValue("No sulfur in the domain", __FILE__, __LINE__);
      new_dw->put(delt_vartype(sulfurEfficiency), d_lab->d_sulfurEfficiencyLabel);
    }

    if (d_sulfur_balance_es) {
      if (totalSulfurFlowRateES > 0.0)
                sulfurEfficiencyES = SO2FlowRateES * 32.0/64.0 /totalSulfurFlowRateES;
      else 
        throw InvalidValue("No sulfur in the domain from ExtraScalar", __FILE__, __LINE__);
      new_dw->put(delt_vartype(sulfurEfficiencyES), d_lab->d_sulfurEfficiencyESLabel);
    }

    // new efficiency calculation
    for ( EfficiencyMap::iterator iter = d_effVars.begin(); iter != d_effVars.end(); iter++){
      double comp_eff = 0.0;
      double new_totalFlowRate = 0.0;

      // loop over all inlets and get the stuff coming into the domain
      for (int indx = 0; indx < d_numInlets; indx++) {
        FlowInlet* fi = d_flowInlets[indx];
        
        for (std::vector<GeometryPieceP>::iterator gi_iter = fi->d_geomPiece.begin();
            gi_iter != fi->d_geomPiece.end(); gi_iter++) {

          GeometryPieceP inlet = *gi_iter;
          std::string inlet_name = inlet->getName();

          // check if this geometry/inlet a part of this efficiency calculation
          std::vector<std::string>::iterator name_iter = find( iter->second.which_inlets.begin(), 
              iter->second.which_inlets.end(), inlet_name );

          if (name_iter != iter->second.which_inlets.end())
            new_totalFlowRate += fi->flowRate * iter->second.fuel_ratio;
        }
      }

      // loop over all species to get the stuff leaving the domain
      double flowRates = 0.0;
      for ( vector<std::string>::iterator sp_iter = iter->second.species.begin(); 
            sp_iter != iter->second.species.end(); sp_iter++ ){

        SpeciesEffMap::iterator sem_iter = d_speciesEffInfo.find(*sp_iter);
        sum_vartype my_sum_var;
        double species_flow_rate; 
        new_dw->get(my_sum_var, sem_iter->second.flowRateLabel);
        species_flow_rate = my_sum_var; 
        flowRates += species_flow_rate * sem_iter->second.molWeightRatio; 

      }

      comp_eff = flowRates / new_totalFlowRate; 

      new_dw->put(delt_vartype(comp_eff), iter->second.label);
    }

    if (d_enthalpySolve) {
      if (totalEnthalpyFlowRate < 0.0) {
        enthalpyEfficiency = enthalpyFlowRate/totalEnthalpyFlowRate;
        normTotalRadSrc = totalRadSrc/totalEnthalpyFlowRate;
        enthalpyEfficiency -= normTotalRadSrc;
      }else{ 
        proc0cout << "No enthalpy in the domain"<< endl;
      }
      new_dw->put(delt_vartype(enthalpyEfficiency), d_lab->d_enthalpyEfficiencyLabel);
      new_dw->put(delt_vartype(normTotalRadSrc), d_lab->d_normTotalRadSrcLabel);
    }
 
}
//****************************************************************************
// Get boundary flow rate for a given variable
//****************************************************************************
void 
BoundaryCondition::getVariableFlowRate(const Patch* patch,
                                       CellInformation* cellinfo,
                                       ArchesConstVariables* constvars,
                                       constCCVariable<double> balance_var,
                                       double* varIN, double* varOUT) 
{
    // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getFortranCellLowIndex__New();
    IntVector idxHi = patch->getFortranCellHighIndex__New();

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    if (xminus||xplus||yminus||yplus||zminus||zplus) {

      double flowIN = 0.0;
      double flowOUT = 0.0;
      bool doing_balance = true;
      for (int indx = 0; indx < d_numInlets; indx++) {

        // Get a copy of the current flow inlet
        // assign flowType the value that corresponds to flow
        //CellTypeInfo flowType = FLOW;
        FlowInlet* fi = d_flowInlets[indx];
        double varIN_inlet = 0.0;
        double varOUT_inlet = 0.0;
        fort_inlpresbcinout(constvars->uVelocity, constvars->vVelocity,
                           constvars->wVelocity, idxLo, idxHi,
                           constvars->density, constvars->cellType,
                           fi->d_cellTypeID,
                           flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
                           cellinfo->stb, xminus, xplus, yminus, yplus,
                           zminus, zplus, doing_balance,
                           balance_var, varIN_inlet, varOUT_inlet);

        if (varOUT_inlet > 0.0)
          throw InvalidValue("Balance variable comming out of inlet", __FILE__, __LINE__);

        // Count balance variable comming through the air inlet
        //double scalarValue = fi->streamMixturefraction.d_mixVars[0];
        //if (scalarValue == 0.0)
          *varIN += varIN_inlet;
      } 

      if (d_pressureBoundary) {
        double varIN_bc = 0.0;
        double varOUT_bc = 0.0;
        int pressure_celltypeval = d_pressureBC->d_cellTypeID;
        fort_inlpresbcinout(constvars->uVelocity, constvars->vVelocity,
                           constvars->wVelocity, idxLo, idxHi,
                           constvars->density, constvars->cellType,
                           pressure_celltypeval,
                           flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
                           cellinfo->stb, xminus, xplus, yminus, yplus,
                           zminus, zplus, doing_balance,
                           balance_var, varIN_bc, varOUT_bc);
        *varIN += varIN_bc;
        *varOUT += varOUT_bc;
      }
      if (d_outletBoundary) {
        double varIN_bc = 0.0;
        double varOUT_bc = 0.0;
        int outlet_celltypeval = d_outletBC->d_cellTypeID;
        fort_inlpresbcinout(constvars->uVelocity, constvars->vVelocity,
                           constvars->wVelocity, idxLo, idxHi,
                           constvars->density, constvars->cellType,
                           outlet_celltypeval,
                           flowIN, flowOUT, cellinfo->sew, cellinfo->sns,
                           cellinfo->stb, xminus, xplus, yminus, yplus,
                           zminus, zplus, doing_balance,
                           balance_var, varIN_bc, varOUT_bc);
        *varIN += varIN_bc;
        *varOUT += varOUT_bc;
      }
    }  
}

//****************************************************************************
// schedule copy of inlet flow rates for nosolve
//****************************************************************************
void BoundaryCondition::sched_setInletFlowRates(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  string taskname =  "BoundaryCondition::setInletFlowRates";
  Task* tsk = scinew Task(taskname, this,
                          &BoundaryCondition::setInletFlowRates);
  
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::OldDW, d_flowInlets[ii]->d_flowRate_label);
    tsk->computes(d_flowInlets[ii]->d_flowRate_label);
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// copy inlet flow rates for nosolve
//****************************************************************************
void 
BoundaryCondition::setInletFlowRates(const ProcessorGroup*,
                                     const PatchSubset* ,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  delt_vartype flowRate;
  for (int indx = 0; indx < d_numInlets; indx++) {
    FlowInlet* fi = d_flowInlets[indx];
    old_dw->get(flowRate, d_flowInlets[indx]->d_flowRate_label);
    d_flowInlets[indx]->flowRate = flowRate;
    fi->flowRate = flowRate;
    new_dw->put(flowRate, d_flowInlets[indx]->d_flowRate_label);
  }
}

//****************************************************************************
//Actually calculate the mms velocity BC
//****************************************************************************
void 
BoundaryCondition::mmsvelocityBC(const Patch* patch,
                                 CellInformation* cellinfo,
                                 ArchesVariables* vars,
                                 ArchesConstVariables* constvars,
                                 double time_shift,
                                 double dt)
{
  mmsuVelocityBC(patch, cellinfo, vars, constvars, time_shift, dt);

  mmsvVelocityBC(patch, cellinfo, vars, constvars, time_shift, dt);

  mmswVelocityBC(patch, cellinfo, vars, constvars, time_shift, dt);
}


//****************************************************************************
// call fortran routine to calculate the MMS U Velocity BC
// Sets the uncorrected velocity values (velRhoHat)!  These should not be
// corrected after the projection so that the values persist to 
// the next time step.
//****************************************************************************
void 
BoundaryCondition::mmsuVelocityBC(const Patch* patch,
                                  CellInformation* cellinfo,
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars, 
                                  double time_shift,
                                  double dt)
{
  int wall_celltypeval = wallCellType();

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // Check to see if patch borders a wall
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  double time=d_lab->d_sharedState->getElapsedTime();
  double current_time = time + time_shift;
  
  if (xminus) {
    
    int colX = idxLo.x();
    double pi = acos(-1.0);

    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        
        if (constvars->cellType[xminusCell] == wall_celltypeval){
          // Directly set the hat velocity

          if (d_mms == "constantMMS"){
            vars->uVelRhoHat[currCell] = cu;
            vars->uVelRhoHat[xminusCell] = cu;
          }
          else if (d_mms == "almgrenMMS"){
            vars->uVelRhoHat[currCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time))
                                                 * sin(2.0*pi*( cellinfo->yy[colY] - current_time))
                                                 * exp(-2.0*d_viscosity*current_time);
                                                 
            vars->uVelRhoHat[xminusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX-1] - current_time))
                                                   * sin(2.0*pi*( cellinfo->yy[colY] - current_time))
                                                   * exp(-2.0*d_viscosity*current_time);

          }
        }
      }
    }
  }

  if (xplus) {
  
    double pi = acos(-1.0);

    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2,colY,colZ);
        
        if (constvars->cellType[xplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->uVelRhoHat[xplusCell] = cu;
          }
          else if (d_mms == "almgrenMMS"){
            vars->uVelRhoHat[xplusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX+1] - current_time))
                                                  * sin(2.0*pi*( cellinfo->yy[colY] - current_time))
                                                  * exp(-2.0*d_viscosity*current_time);
          }
        }
      }
    }
  }

  if (yminus) {
    int colY = idxLo.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        
        if (constvars->cellType[yminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->uVelRhoHat[yminusCell] = cu;
          }
          else if (d_mms == "almgrenMMS"){
            vars->uVelRhoHat[yminusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time ))
                                                   * sin(2.0*pi*( cellinfo->yy[colY-1] - current_time ))
                                                   * exp(-2.0*d_viscosity*current_time);
          }
        }
      }
    }
  }

  if (yplus) {
    int colY = idxHi.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);

        if (constvars->cellType[yplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->uVelRhoHat[yplusCell] = cu;
          }
          else if (d_mms == "almgrenMMS"){
            
            vars->uVelRhoHat[yplusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time ))
                                                  * sin(2.0*pi*( cellinfo->yy[colY+1] - current_time ))
                                                  * exp(-2.0*d_viscosity*current_time);
          }
        }
      }
    }
  }


  if (zminus) {
    int colZ = idxLo.z();
    double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        
        if (constvars->cellType[zminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->uVelRhoHat[zminusCell] = cu;
          }
          else if (d_mms == "almgrenMMS"){
            
            vars->uVelRhoHat[zminusCell] = 1 - amp * cos(2.0*pi*( cellinfo->xu[colX] - current_time ))
                                                   * sin(2.0*pi*( cellinfo->yy[colY] - current_time ))
                                                   * exp(-2.0*d_viscosity*current_time);
          }            
        }
      }
    }
  }

  if (zplus) {
    int colZ = idxHi.z();
    //double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        
        if (constvars->cellType[zplusCell] == wall_celltypeval){
          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->uVelRhoHat[zplusCell] = cu;
          }
          else if (d_mms == "almgrenMMS"){
            double pi = acos(-1.0);
            vars->uVelRhoHat[zplusCell] = 1 - amp * cos(2.0*pi* ( cellinfo->xu[colX] - current_time ))
                                                  * sin(2.0*pi* ( cellinfo->yy[colY] - current_time ))
                                                  * exp(-2.0*d_viscosity*current_time);
          }            
        }
      }
    }
  }
}
  
//****************************************************************************
// call fortran routine to calculate the MMS V Velocity BC
//****************************************************************************
void 
BoundaryCondition::mmsvVelocityBC(const Patch* patch,
                                  CellInformation* cellinfo,
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars,
                                  double time_shift, 
                                  double dt) 
{
  int wall_celltypeval = wallCellType();

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // Check to see if patch borders a wall
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  double time=d_lab->d_sharedState->getElapsedTime();
  double current_time=time + time_shift;
  
  if (xminus) {
    int colX = idxLo.x();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        
        if (constvars->cellType[xminusCell] == wall_celltypeval){
          
          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->vVelRhoHat[xminusCell] = cv;
          }
          else if (d_mms == "almgrenMMS"){
            vars->vVelRhoHat[xminusCell] = 1 + amp * sin(2.0*pi* ( cellinfo->xx[colX-1] - current_time ))
                                                   * cos(2.0*pi* ( cellinfo->yv[colY] - current_time ))
                                                   * exp(-2.0*d_viscosity*current_time);
          }
          
        }
      }
    }
  }
  
  if (xplus) {
    int colX = idxHi.x();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        
        if (constvars->cellType[xplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->vVelRhoHat[xplusCell] = cv;
          }
          else if (d_mms == "almgrenMMS"){
            vars->vVelRhoHat[xplusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX+1] - current_time ))
                                                 * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
                                                 * exp(-2.0*d_viscosity*current_time);
          }
          
        }
      }
    }
  }
  
  if (yminus) {
    int colY = idxLo.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        
        if (constvars->cellType[yminusCell] == wall_celltypeval){
          
          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->vVelRhoHat[yminusCell] = cv;
            vars->vVelRhoHat[currCell]   = cv;
          }
          else if (d_mms == "almgrenMMS"){
            vars->vVelRhoHat[yminusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
                                                  * cos(2.0*pi*( cellinfo->yv[colY-1] - current_time ))
                                                  * exp(-2.0*d_viscosity*current_time);

            vars->vVelRhoHat[currCell]   = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
                                                  * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
                                                  * exp(-2.0*d_viscosity*current_time);

          }
          
        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        
        if (constvars->cellType[yplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->vVelRhoHat[yplusCell] = cv;
          }
          else if (d_mms == "almgrenMMS"){
            vars->vVelRhoHat[yplusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
                                                 * cos(2.0*pi*( cellinfo->yv[colY+1] - current_time ))
                                                 * exp(-2.0*d_viscosity*current_time);
          }
          
        }
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        
        if (constvars->cellType[zminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->vVelRhoHat[zminusCell] = cv;
          }
          else if (d_mms == "almgrenMMS"){
            vars->vVelRhoHat[zminusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
                                                  * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
                                                  * exp(-2.0*d_viscosity*current_time);
          }            
        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        
        if (constvars->cellType[zplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->vVelRhoHat[zplusCell] = cv;
          }
          else if (d_mms == "almgrenMMS"){
            vars->vVelRhoHat[zplusCell] = 1 + amp * sin(2.0*pi*( cellinfo->xx[colX] - current_time ))
                                                 * cos(2.0*pi*( cellinfo->yv[colY] - current_time ))
                                                 * exp(-2.0*d_viscosity*current_time);
          }            
        }
      }
    }
  }
}

//****************************************************************************
// call fortran routine to calculate the MMS W Velocity BC
//****************************************************************************
void 
BoundaryCondition::mmswVelocityBC(const Patch* patch,
                                  CellInformation* cellinfo,
                                  ArchesVariables* vars,
                                  ArchesConstVariables* constvars,
                                  double time_shift, 
                                  double dt) 
{
  int wall_celltypeval = wallCellType();

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // Check to see if patch borders a wall
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  //double time=d_lab->d_sharedState->getElapsedTime();
  //double current_time = time + time_shift;

  //currently only supporting sinemms in x-y plane
  
  if (xminus) {
    int colX = idxLo.x();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        
        if (constvars->cellType[xminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->wVelRhoHat[xminusCell] = cw;
          }
          else if (d_mms == "almgrenMMS"){
            vars->wVelRhoHat[xminusCell] = 0.0;
          }
        }
      }
    }
  }
  
  if (xplus) {
    int colX = idxHi.x();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        
        if (constvars->cellType[xplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->wVelRhoHat[xplusCell] = cw;
          }
          else if (d_mms == "almgrenMMS"){
            vars->wVelRhoHat[xplusCell] = 0.0;
          }
        }
      }
    }
  }
  
  if (yminus) {
    int colY = idxLo.y();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        
        if (constvars->cellType[yminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->wVelRhoHat[yminusCell] = cw;
          }
          else if (d_mms == "almgrenMMS"){
            vars->wVelRhoHat[yminusCell] = 0.0;
          }
          
        }
      }
    }
  }
  if (yplus) {
    int colY = idxHi.y();
    //double pi = acos(-1.0);
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        
        if (constvars->cellType[yplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->wVelRhoHat[yplusCell] = cw;
          }
          else if (d_mms == "almgrenMMS"){
            vars->wVelRhoHat[yplusCell] = 0.0;
          }
          
        }
      }
    }
  }
  if (zminus) {
    int colZ = idxLo.z();
    //double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        
        if (constvars->cellType[zminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->wVelRhoHat[currCell]   = cw;
            vars->wVelRhoHat[zminusCell] = cw;
          }
          else if (d_mms == "almgrenMMS"){
            vars->wVelRhoHat[currCell]   = 0.0;
            vars->wVelRhoHat[zminusCell] = 0.0;
          }            
        }
      }
    }
  }
  if (zplus) {
    int colZ = idxHi.z();
    //double pi = acos(-1.0);
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        
        if (constvars->cellType[zplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->wVelRhoHat[zplusCell] = cw;
          }
          else if (d_mms == "almgrenMMS"){
            vars->wVelRhoHat[zplusCell] = 0.0;
          }            
        }
      }  // X
    }  // Y
  }  // zplus
}

//****************************************************************************
// Actually compute the MMS scalar bcs
//****************************************************************************
void 
BoundaryCondition::mmsscalarBC(const Patch* patch,
                               CellInformation* cellinfo,
                               ArchesVariables* vars,
                               ArchesConstVariables* constvars,
                               double time_shift,
                               double dt)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // Get the wall boundary and flow field codes
  int wall_celltypeval = wallCellType();

  //double time=d_lab->d_sharedState->getElapsedTime();
  //double current_time = time + time_shift;

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

        if (constvars->cellType[xminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->scalar[xminusCell] = phi0;
          }
          else if (d_mms == "almgrenMMS"){
          }
          
        }
      }
    }
  }


  if (xplus) {
    int colX = idxHi.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {

        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);

        if (constvars->cellType[xplusCell] == wall_celltypeval){

          // Directly set the hat velocity
           if (d_mms == "constantMMS"){
            vars->scalar[xplusCell] = phi0;
          }
          else if (d_mms == "almgrenMMS"){
          }
          
        }
      }
    }
  }


  if (yminus) {
    int colY = idxLo.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        
        if (constvars->cellType[yminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->scalar[yminusCell] = phi0;
          }
          else if (d_mms == "almgrenMMS"){
          }
          
        }
      }
    }
  }


  if (yplus) {
    int colY = idxHi.y();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        
        if (constvars->cellType[yplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->scalar[yplusCell] = phi0;
          }
          else if (d_mms == "almgrenMMS"){
          }
          
        }
      }
    }
  }



  if (zminus) {
    int colZ = idxLo.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        
        if (constvars->cellType[zminusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->scalar[zminusCell] = phi0;
          }
          else if (d_mms == "almgrenMMS"){
          }            
        }
      }
    }
  }


  if (zplus) {
    int colZ = idxHi.z();
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        
        if (constvars->cellType[zplusCell] == wall_celltypeval){

          // Directly set the hat velocity
          if (d_mms == "constantMMS"){
            vars->scalar[zplusCell] = phi0;
          }
          else if (d_mms == "almgrenMMS"){
          }            
        }
      }
    }
  }
}

//****************************************************************************
// Schedule  prefill
//****************************************************************************
void 
BoundaryCondition::sched_Prefill(SchedulerP& sched, 
                                 const PatchSet* patches,
                                 const MaterialSet* matls)
{
  Task* tsk = scinew Task("BoundaryCondition::Prefill",this,
                          &BoundaryCondition::Prefill);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,Ghost::None, 0);
  
  for (int ii = 0; ii < d_numInlets; ii++) {
    tsk->requires(Task::NewDW, d_flowInlets[ii]->d_area_label);
    tsk->requires(Task::NewDW, d_flowInlets[ii]->d_flowRate_label);
  }

  if (d_enthalpySolve) {
    tsk->modifies(d_lab->d_enthalpySPLabel);
  }
  if (d_reactingScalarSolve) {
    tsk->modifies(d_lab->d_reactscalarSPLabel);
  }
    
  // This task computes new density, uVelocity, vVelocity and wVelocity, scalars
  tsk->modifies(d_lab->d_densityCPLabel);
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);
  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);
  //tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 1);

  tsk->modifies(d_lab->d_scalarSPLabel);

  if (d_calcExtraScalars){
    for (int i=0; i < static_cast<int>(d_extraScalars->size()); i++) {
      tsk->modifies(d_extraScalars->at(i)->getScalarLabel());
    }
  }

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actually set flat profile at flow inlet boundary
//****************************************************************************
void 
BoundaryCondition::Prefill(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<double> density;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uVelRhoHat;
    SFCYVariable<double> vVelRhoHat;
    SFCZVariable<double> wVelRhoHat;
    CCVariable<double> scalar;
    CCVariable<double> reactscalar;
    CCVariable<double> enthalpy;
    constCCVariable<int> cellType;

    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(uVelRhoHat, d_lab->d_uVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(vVelRhoHat, d_lab->d_vVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(wVelRhoHat, d_lab->d_wVelRhoHatLabel, indx, patch);
    
    new_dw->getModifiable(density, d_lab->d_densityCPLabel, indx, patch);
    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, indx, patch);
    if (d_reactingScalarSolve)
      new_dw->getModifiable(reactscalar, d_lab->d_reactscalarSPLabel, indx, patch);
    
    if (d_enthalpySolve){ 
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel, indx, patch);
    }
    
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::None, 0);

    // loop thru the flow inlets to set all the components of velocity and density
    if (d_inletBoundary) {

      double time = 0.0;
      double ramping_factor;
      Vector dx = patch->dCell();
      Box patchInteriorBox = patch->getBox();

      for (int indx = 0; indx < d_numInlets; indx++) {

        sum_vartype area_var;
        new_dw->get(area_var, d_flowInlets[indx]->d_area_label);
        double area = area_var;
        delt_vartype flow_r;
        new_dw->get(flow_r, d_flowInlets[indx]->d_flowRate_label);
        double flow_rate = flow_r;
        FlowInlet* fi = d_flowInlets[indx];
        // This call does something magical behind the scenes. 
        fort_get_ramping_factor(fi->d_ramping_inlet_flowrate,
                                time, ramping_factor);

        if (fi->d_prefill) {

          int nofGeomPieces = (int)fi->d_prefillGeomPiece.size();

          for (int ii = 0; ii < nofGeomPieces; ii++) {

            GeometryPieceP  piece = fi->d_prefillGeomPiece[ii];
            Box geomBox = piece->getBoundingBox();
            Box b = geomBox.intersect(patchInteriorBox);

            if (!(b.degenerate())) {

              for (CellIterator iter = patch->getCellCenterIterator(b); !iter.done(); iter++) {

                Point p = patch->cellPosition(*iter);
                // Do Velocities
                if (piece->inside(p) && cellType[*iter] == d_flowfieldCellTypeVal) {
                  if (fi->d_prefill_index == 1) {
                    Point p_fx(p.x()-dx.x()/2.,p.y(),p.z()); // minus x-face
                    if (piece->inside(p_fx))
                      uVelocity[*iter] = flow_rate / (fi->calcStream.d_density * area);
                  }
                  if (fi->d_prefill_index == 2) {
                    Point p_fy(p.x(),p.y()-dx.y()/2.,p.z()); // minus y-face
                    if (piece->inside(p_fy))
                      vVelocity[*iter] = flow_rate / (fi->calcStream.d_density * area);
                  }
                  if (fi->d_prefill_index == 3) {
                    Point p_fz(p.x(),p.y(),p.z()-dx.z()/2.); // minus z-face
                    if (piece->inside(p_fz))
                      wVelocity[*iter] = flow_rate / (fi->calcStream.d_density * area);
                  }

                  // Now do scalars
                  density[*iter] = fi->calcStream.d_density;
                  scalar[*iter] = fi->streamMixturefraction.d_mixVars[0];
                  if (d_enthalpySolve)
                    enthalpy[*iter] = fi->calcStream.d_enthalpy;
                  if (d_reactingScalarSolve)
                    reactscalar[*iter] = fi->streamMixturefraction.d_rxnVars[0];
                
                  //Now handle extra scalars
                  if (d_calcExtraScalars) {
                    for ( int i=0; i < static_cast<int>(d_extraScalars->size()); i++ ) {

                      CCVariable<double> extra_scalar;
                      new_dw->getModifiable(extra_scalar, d_extraScalars->at(i)->getScalarLabel(), indx, patch); // get the current extra scalar
                      string extra_scalar_name = d_extraScalars->at(i)->getScalarName(); // actual name of the extra scalar
                      int BC_ID = fi->d_cellTypeID; 

                      for (int j=0; j < static_cast<int>(d_extraScalarBCs.size()); j++) {
                        if ((d_extraScalarBCs[j]->d_scalar_name == extra_scalar_name)&&
                           (d_extraScalarBCs[j]->d_BC_ID) == BC_ID) {
                         extra_scalar[*iter] = d_extraScalarBCs[j]->d_scalarBC_value; //finally set the extra scalar's value
                        }
                      }
                    } 
                  } // end extra scalars
                } // Cell Iter loop 
  
              } // cell iterator loop
            }
          }  // geom iter
        }  // prefill
      }  // inlets loop
    }
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 

  }
}

void BoundaryCondition::insertIntoEffMap ( std::string name, double fuel_ratio, double air_ratio, vector<std::string> species, vector<std::string> which_inlets ) {

  EfficiencyMap::iterator iter = d_effVars.find( name ); 
  if ( iter == d_effVars.end() ){
    EfficiencyInfo info;

    const VarLabel* tempLabel; 
    tempLabel = VarLabel::create(name, min_vartype::getTypeDescription());

    info.label = tempLabel;
    info.fuel_ratio = fuel_ratio; 
    info.air_ratio  = air_ratio; 
    info.species = species; 
    info.which_inlets = which_inlets;

    iter = d_effVars.insert( std::make_pair( name, info ) ).first; 
  } else {
    // Each scalar name must be unique
    throw InvalidValue("Found two scalars in the ScalarEfficiency section that are identical! Please choose unique names.",__FILE__,__LINE__);

  }
}

void BoundaryCondition::insertIntoSpeciesMap ( std::string name, double mol_ratio )
{
  SpeciesEffMap::iterator iter = d_speciesEffInfo.find( name ); 
  // we have a problem here...what if the species is needed twice for two different components?
  if ( iter == d_speciesEffInfo.end() ){

    SpeciesEfficiencyInfo info; 

    std::string modName = name;
    modName += "_flowrate"; 
    info.flowRateLabel = VarLabel::create(modName, sum_vartype::getTypeDescription());
    info.molWeightRatio = mol_ratio; 

    iter = d_speciesEffInfo.insert( std::make_pair( name, info ) ).first; 

  }
}

void 
BoundaryCondition::sched_bcdummySolve( SchedulerP& sched, 
                                     const PatchSet* patches, 
                                     const MaterialSet* matls )
{
  Task* tsk = scinew Task( "BoundaryCondition::bcdummySolve",this, &BoundaryCondition::bcdummySolve);
 
  for ( EfficiencyMap::iterator iter = d_effVars.begin(); iter != d_effVars.end(); iter++){
    tsk->computes(iter->second.label);
  }

  for ( SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){
    tsk->computes(iter->second.flowRateLabel);
  }

  sched->addTask(tsk, patches, matls);
}
void 
BoundaryCondition::bcdummySolve( const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {
    int archIndex = 0; // only one arches material

    for ( EfficiencyMap::iterator iter = d_effVars.begin(); iter != d_effVars.end(); iter++){
      new_dw->put(delt_vartype(0.0),iter->second.label);
    }

    for ( SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){
      new_dw->put(delt_vartype(0.0),iter->second.flowRateLabel);
    }
  }
}


