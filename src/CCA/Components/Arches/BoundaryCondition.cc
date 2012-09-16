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


//----- BoundaryCondition.cc ----------------------------------------------

#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/IntrusionBC.h>

#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
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
#include <Core/IO/UintahZlibUtil.h>

#include <iostream>
#include <sstream>
#include <stdlib.h>


using namespace std;
using namespace Uintah;

#include <CCA/Components/Arches/fortran/celltypeInit_fort.h>
#include <CCA/Components/Arches/fortran/areain_fort.h>
#include <CCA/Components/Arches/fortran/profscalar_fort.h>
#include <CCA/Components/Arches/fortran/inlbcs_fort.h>
#include <CCA/Components/Arches/fortran/inlpresbcinout_fort.h>
#include <CCA/Components/Arches/fortran/bcscalar_fort.h>
#include <CCA/Components/Arches/fortran/bcuvel_fort.h>
#include <CCA/Components/Arches/fortran/bcvvel_fort.h>
#include <CCA/Components/Arches/fortran/bcwvel_fort.h>
#include <CCA/Components/Arches/fortran/mmbcvelocity_fort.h>
#include <CCA/Components/Arches/fortran/mmcelltypeinit_fort.h>
#include <CCA/Components/Arches/fortran/mmwallbc_fort.h>
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
  _using_new_intrusion  = false; 
  d_calcEnergyExchange  = false;

  // x-direction
  index_map[0][0] = 0;
  index_map[0][1] = 1; 
  index_map[0][2] = 2; 
  // y-direction 
  index_map[1][0] = 1;
  index_map[1][1] = 2; 
  index_map[1][2] = 0; 
  // z-direction
  index_map[2][0] = 1;
  index_map[2][1] = 2; 
  index_map[2][2] = 0; 
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
  for ( EfficiencyMap::iterator iter = d_effVars.begin(); iter != d_effVars.end(); iter++){
    VarLabel::destroy(iter->second.label);
  }

  for ( SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){
    VarLabel::destroy(iter->second.flowRateLabel);
  }

  delete d_newBC; 
  for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

    VarLabel::destroy( bc_iter->second.total_area_label ); 
    
    if (bc_iter->second.type ==  TURBULENT_INLET ){
      delete bc_iter->second.TurbIn;
    }
  }

  if (_using_new_intrusion) { 
    delete _intrusionBC; 
  } 
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
BoundaryCondition::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db_params = params; 
  ProblemSpecP db = params->findBlock("BoundaryConditions");
  d_flowfieldCellTypeVal = -1;
  d_numInlets = 0;
  int total_cellTypes = 100;

  d_newBC = scinew BoundaryCondition_new( d_lab ); // need to declare a new boundary condition here 
                                                   // while transition to new code is taking place
  if(db.get_rep()==0)
  {
    proc0cout << "No Boundary Conditions Specified \n";
    d_inletBoundary = false;
    d_wallBoundary = false;
    d_pressureBoundary = false;
    d_outletBoundary = false;
    d_use_new_bcs = false; 
    _using_new_intrusion = false;
  }
  else
  {

     // new bc:                                                 
     d_use_new_bcs = false; 
     if ( db->findBlock("use_new_bcs") ) { 
       d_use_new_bcs = true; 
     }
     if ( d_use_new_bcs ) { 
       setupBCs( db_params );
     }

    if ( d_use_new_bcs ) { 

      for (ProblemSpecP prefill_db = db->findBlock("Prefill"); prefill_db != 0; 
            prefill_db = prefill_db->findNextBlock("Prefill") ) { 

        std::string which_boundary = "null"; 
        prefill_db->getAttribute( "bc", which_boundary ); 

        if ( which_boundary == "null" ) { 
          throw ProblemSetupException("Error: Must specify an associated boundary for the prefill attribute.",__FILE__,__LINE__); 
        } 

        ProblemSpecP geometry_db = prefill_db->findBlock("geom_object"); 
        std::vector<GeometryPieceP> geometry; 
        if ( geometry_db ) { 
          GeometryPieceFactory::create( geometry_db, geometry ); 
        } else { 
          throw ProblemSetupException("Error: Must specify a geom_object in <Prefill> block.",__FILE__,__LINE__); 
        } 

        d_prefill_map.insert(make_pair(which_boundary, geometry));

      } 
    }

    if ( db->findBlock("intrusions") ){ 

      _intrusionBC = scinew IntrusionBC( d_lab, d_MAlab, d_props, BoundaryCondition::INTRUSION ); 
      ProblemSpecP db_new_intrusion = db->findBlock("intrusions"); 
      _using_new_intrusion = true; 

      _intrusionBC->problemSetup( db_new_intrusion ); 

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

    //-------------------------------------------------------------------
    // Flow Inlets:
    //
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
        string bc_type = "flow_inlet"; 
        d_props->computeInletProperties(
            d_flowInlets[d_numInlets]->streamMixturefraction,
            d_flowInlets[d_numInlets]->calcStream, bc_type);
        double f = d_flowInlets[d_numInlets]->streamMixturefraction.d_mixVars[0];
        if (f > 0.0){
          d_flowInlets[d_numInlets]->fcr = d_props->getCarbonContent(f);
          
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
      d_wallBdry = scinew WallBdry(WALL);
      d_wallBdry->problemSetup(wall_db);
      //++total_cellTypes;
    }
    else {
      proc0cout << "Wall boundary not specified"<<endl;
      d_wallBoundary = false;
    }

    if (ProblemSpecP press_db = db->findBlock("PressureBC")) {
      d_pressureBoundary = true;
      d_pressureBC = scinew PressureInlet(PRESSURE, d_calcVariance,
          d_reactingScalarSolve);
      d_pressureBC->problemSetup(press_db);
      // compute density and other dependent properties
      d_pressureBC->streamMixturefraction.d_initEnthalpy=true;
      d_pressureBC->streamMixturefraction.d_scalarDisp=0.0;
      string bc_type = "pressure"; 
      d_props->computeInletProperties(d_pressureBC->streamMixturefraction, 
          d_pressureBC->calcStream, bc_type);

      //++total_cellTypes;
    }
    else {
      proc0cout << "Pressure boundary not specified"<< endl;
      d_pressureBoundary = false;
    }

    if (ProblemSpecP outlet_db = db->findBlock("OutletBC")) {
      d_outletBoundary = true;
      d_outletBC = scinew FlowOutlet(OUTLET, d_calcVariance,
          d_reactingScalarSolve);
      d_outletBC->problemSetup(outlet_db);
      // compute density and other dependent properties
      d_outletBC->streamMixturefraction.d_initEnthalpy=true;
      d_outletBC->streamMixturefraction.d_scalarDisp=0.0;
      string bc_type = "outlet"; 
      d_props->computeInletProperties(d_outletBC->streamMixturefraction, 
          d_outletBC->calcStream, bc_type);
      //++total_cellTypes;
    }
    else {
      proc0cout << "Outlet boundary not specified"<<endl;
      d_outletBoundary = false;
    }
  }

  // if multimaterial then add an id for multimaterial wall
  // trying to reduce all interior walls to type:INTRUSION
  d_mmWallID = INTRUSION;
//  if ( d_MAlab ){
//    d_mmWallID = INTRUSION; 
//  }

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
    } else if (d_mms == "almgrenMMS") {
      ProblemSpecP db_whichmms = db_mmsblock->findBlock("almgrenMMS");
      db_whichmms->getWithDefault("amplitude",amp,0.0);
      db_whichmms->require("viscosity",d_viscosity);
    } else {
      throw InvalidValue("current MMS "
                         "not supported: " + d_mms, __FILE__, __LINE__);
    }
  }

  //look for velocity file input information... 
  ProblemSpecP db_root = db_params->getRootNode();
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions"); 

  if ( db_bc ) { 

    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != 0; 
          db_face = db_face->findNextBlock("Face") ){
      
      for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != 0; 
          db_BCType = db_BCType->findNextBlock("BCType") ){

        std::string name; 
        std::string type; 
        db_BCType->getAttribute("label", name);
        db_BCType->getAttribute("var", type); 

        if ( type == "VelocityFileInput" ){ 

          std::string file_name;
          db_BCType->require("inputfile", file_name); 

          gzFile file = gzopen( file_name.c_str(), "r" ); 
          int total_variables;
          // name of variable, filename to open
          std::map<std::string, std::string> input_files;

          if ( file == NULL ) { 
            proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << endl;
            throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
          }

          total_variables = getInt( file ); 
          for ( int i = 0; i < total_variables; i++ ){
            std::string varname  = getString( file );
            std::string which_file  = getString( file ); 
            input_files.insert( make_pair( varname, which_file)); 
          }
          gzclose( file ); 

          if ( total_variables == 0 ){ 
            throw ProblemSetupException("Error: Number of variables in reference file is zero! See file: " + file_name, __FILE__, __LINE__);
          } 

          std::map<string, string>::iterator iter = input_files.find( "uvel" ); 
          _u_input = readInputFile__NEW( iter->second ); 

          iter = input_files.find( "vvel" ); 
          _v_input = readInputFile__NEW( iter->second ); 

          iter = input_files.find( "wvel" ); 
          _w_input = readInputFile__NEW( iter->second ); 

        } 
      }
    }
  }
 // for ( CellToValue::iterator i=_u_input.begin(); i!=_u_input.end(); i++){ 
 //   cout << " FOR U, celliter = " << i->first << " with value = " << i->second << endl;
 // } 

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

      if ( _using_new_intrusion ) {
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
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
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
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);
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
    CCVariable<double> enthalpy;

    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::None, 0);
    new_dw->getModifiable(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch);
    new_dw->getModifiable(uVelRhoHat, d_lab->d_uVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(vVelRhoHat, d_lab->d_vVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(wVelRhoHat, d_lab->d_wVelRhoHatLabel, indx, patch);
    new_dw->getModifiable(density, d_lab->d_densityCPLabel, indx, patch);
    new_dw->getModifiable(scalar, d_lab->d_scalarSPLabel, indx, patch);
    if (d_enthalpySolve){ 
      new_dw->getModifiable(enthalpy, d_lab->d_enthalpySPLabel, indx, patch);
    }

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;

    // loop thru the flow inlets to set all the components of velocity and density
    if (d_inletBoundary) {

      //double time = 0.0; 

      for (int indx = 0; indx < d_numInlets; indx++) {

        sum_vartype area_var;
        new_dw->get(area_var, d_flowInlets[indx]->d_area_label);

        double area = area_var;
        double actual_flow_rate;
      
        // Get a copy of the current flow inlet
        // check if given patch intersects with the inlet boundary of type index
        FlowInlet* fi = d_flowInlets[indx];

        proc0cout << "Actual area for inlet: " << fi->d_inlet_name << " = " << area << endl;
        proc0cout << endl;

        switch ( fi->d_inletVelType ){
          case FlowInlet::VEL_FLAT_PROFILE:

            setFlatProfV( patch, 
                          uVelocity, vVelocity, wVelocity, 
                          cellType, area, fi->d_cellTypeID, 
                          fi->flowRate, fi->inletVel, fi->calcStream.d_density, 
                          xminus, xplus, 
                          yminus, yplus, 
                          zminus, zplus, 
                          actual_flow_rate ); 
                          
            d_flowInlets[indx]->flowRate = actual_flow_rate;
            new_dw->put(delt_vartype(actual_flow_rate),
                        d_flowInlets[indx]->d_flowRate_label);

            break; 
          case FlowInlet::VEL_FUNCTION:

            break; 
          case FlowInlet::VEL_VECTOR:

            break; 
          case FlowInlet::VEL_FILE_INPUT:

            break; 
        }

        switch ( fi->d_inletScalarType ){
          case FlowInlet::SCALAR_FLAT_PROFILE:

            setFlatProfS( patch, 
                          density, 
                          fi->calcStream.d_density, cellType, area, fi->d_cellTypeID, 
                          xminus, xplus, 
                          yminus, yplus, 
                          zminus, zplus ); 

            setFlatProfS( patch, 
                          scalar, 
                          fi->streamMixturefraction.d_mixVars[0], cellType, area, fi->d_cellTypeID, 
                          xminus, xplus, 
                          yminus, yplus, 
                          zminus, zplus );  

            if (d_enthalpySolve) {

              setFlatProfS( patch, 
                            enthalpy, 
                            fi->calcStream.d_enthalpy, cellType, area, fi->d_cellTypeID, 
                            xminus, xplus, 
                            yminus, yplus, 
                            zminus, zplus ); 
            }

            break; 
          case FlowInlet::SCALAR_FUNCTION:

            break; 
          case FlowInlet::SCALAR_FILE_INPUT:

            break; 
        }

        
        //fort_profv(uVelocity, vVelocity, wVelocity, idxLo, idxHi,
        //           cellType, area, fi->d_cellTypeID, fi->flowRate, fi->inletVel,
        //           fi->calcStream.d_density,
        //           xminus, xplus, yminus, yplus, zminus, zplus, time,
        //           fi->d_ramping_inlet_flowrate, actual_flow_rate);


        //fort_profscalar(idxLo, idxHi, density, cellType,
        //                fi->calcStream.d_density, fi->d_cellTypeID,
        //                xminus, xplus, yminus, yplus, zminus, zplus);
        //if (d_enthalpySolve){
        //  fort_profscalar(idxLo, idxHi, enthalpy, cellType,
        //                  fi->calcStream.d_enthalpy, fi->d_cellTypeID,
        //                  xminus, xplus, yminus, yplus, zminus, zplus);
        //}
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

    //if (d_inletBoundary) {
    //  for (int ii = 0; ii < d_numInlets; ii++) {
    //    double scalarValue = 
    //           d_flowInlets[ii]->streamMixturefraction.d_mixVars[0];
    //    fort_profscalar(idxLo, idxHi, scalar, cellType,
    //                    scalarValue, d_flowInlets[ii]->d_cellTypeID,
    //                    xminus, xplus, yminus, yplus, zminus, zplus);
    //  }
    // }

    if (d_pressureBoundary) {
      double scalarValue = 
             d_pressureBC->streamMixturefraction.d_mixVars[0];
      fort_profscalar(idxLo, idxHi, scalar, cellType, scalarValue,
                      d_pressureBC->d_cellTypeID,
                      xminus, xplus, yminus, yplus, zminus, zplus);
    }

    if (d_outletBoundary) {
      double scalarValue = 
             d_outletBC->streamMixturefraction.d_mixVars[0];
      fort_profscalar(idxLo, idxHi, scalar, cellType, scalarValue,
                      d_outletBC->d_cellTypeID,
                      xminus, xplus, yminus, yplus, zminus, zplus);
    }
    uVelRhoHat.copyData(uVelocity); 
    vVelRhoHat.copyData(vVelocity); 
    wVelRhoHat.copyData(wVelocity); 

  }
}

// set a velocity profile for a boundary 
// should replace the klunky fortran routines
void 
BoundaryCondition::setFlatProfV( const Patch* patch, 
                             SFCXVariable<double>& u, SFCYVariable<double>& v, SFCZVariable<double>& w, 
                             const CCVariable<int>& cellType, const double area, const int inlet_type, 
                             const double flow_rate, const double inlet_vel, const double density, 
                             const bool xminus, const bool xplus, 
                             const bool yminus, const bool yplus, 
                             const bool zminus, const bool zplus, 
                             double& actual_flow_rate ) 
{
  vector<Patch::FaceType>::const_iterator fiter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  double ave_normal_vel = 0; 
  double tiny = 1.0e-16; 

  // for normal flow inlet BC: 
  if ( flow_rate < tiny ) { //ie, it is zero
    ave_normal_vel   = inlet_vel;
    actual_flow_rate = ave_normal_vel * density * area; 
  } else { 
    if ( area < tiny ) {
      ave_normal_vel   = 0.0;
      actual_flow_rate = 0.0; 
    } else {
      ave_normal_vel = flow_rate / ( area * density ); 
      actual_flow_rate = flow_rate; 
    }
  }

  for (fiter = bf.begin(); fiter !=bf.end(); fiter++){
    Patch::FaceType face = *fiter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);
    switch (face) {
      case Patch::xminus:
        for ( CellIterator citer = patch->getFaceIterator( face, Patch::InteriorFaceCells );
              !citer.done(); citer++ ){

          IntVector c  = *citer; 
          IntVector cb = *citer + insideCellDir; 

          if ( cellType[cb] == inlet_type ) { 
            u[c]  = ave_normal_vel; 
            u[cb] = ave_normal_vel; 
            v[cb] = 0.0; 
            w[cb] = 0.0;
          }
        }
        break;
      case Patch::xplus:
        for ( CellIterator citer = patch->getFaceIterator( face, Patch::InteriorFaceCells );
              !citer.done(); citer++ ){

          IntVector c  = *citer; 
          IntVector cb = *citer + insideCellDir; 
          IntVector cbb= *citer + insideCellDir + insideCellDir; 
          
          if ( cellType[cb] == inlet_type ) { 
            u[cb] = ave_normal_vel; 
            u[cbb]= ave_normal_vel; 
            v[cb] = 0.0; 
            w[cb] = 0.0;
          }
        }
        break; 
      case Patch::yminus:
        for ( CellIterator citer = patch->getFaceIterator( face, Patch::InteriorFaceCells );
              !citer.done(); citer++ ){

          IntVector c  = *citer; 
          IntVector cb = *citer + insideCellDir; 

          if ( cellType[cb] == inlet_type ) { 
            v[c]  = ave_normal_vel; 
            v[cb] = ave_normal_vel; 
            u[cb] = 0.0; 
            w[cb] = 0.0;
          }
        }
        break; 
      case Patch::yplus:
        for ( CellIterator citer = patch->getFaceIterator( face, Patch::InteriorFaceCells );
              !citer.done(); citer++ ){

          IntVector c  = *citer; 
          IntVector cb = *citer + insideCellDir; 
          IntVector cbb= *citer + insideCellDir + insideCellDir; 

          if ( cellType[cb] == inlet_type ) { 
            v[cb] = ave_normal_vel; 
            v[cbb]= ave_normal_vel;
            u[cb] = 0.0; 
            w[cb] = 0.0;
          }
        }
        break; 
      case Patch::zminus: 
        for ( CellIterator citer = patch->getFaceIterator( face, Patch::InteriorFaceCells );
              !citer.done(); citer++ ){

          IntVector c  = *citer; 
          IntVector cb = *citer + insideCellDir; 

          if ( cellType[c] == inlet_type ) { 
            w[c]  = ave_normal_vel; 
            w[cb] = ave_normal_vel; 
            u[cb] = 0.0; 
            v[cb] = 0.0;
          }
        }
        break; 
      case Patch::zplus:
        for ( CellIterator citer = patch->getFaceIterator( face, Patch::InteriorFaceCells );
              !citer.done(); citer++ ){

          IntVector c  = *citer; 
          IntVector cb = *citer + insideCellDir; 
          IntVector cbb= *citer + insideCellDir + insideCellDir; 

          if ( cellType[cb] == inlet_type ) { 
            w[cb] = ave_normal_vel; 
            w[cbb]= ave_normal_vel; 
            u[cb] = 0.0; 
            v[cb] = 0.0;
          }
        }
        break; 
      default: 
        throw InvalidValue("Error: Face type for setFlatProfV not recognized.",__FILE__,__LINE__); 
        break; 
    }
  }
}

// set a scalar profile for a boundary 
// should replace the klunky fortran routines
void 
BoundaryCondition::setFlatProfS( const Patch* patch, 
                             CCVariable<double>& scalar, 
                             double set_point, 
                             const CCVariable<int>& cellType, const double area, const int check_type, 
                             const bool xminus, const bool xplus, 
                             const bool yminus, const bool yplus, 
                             const bool zminus, const bool zplus )
{
  vector<Patch::FaceType>::const_iterator fiter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for (fiter = bf.begin(); fiter !=bf.end(); fiter++){
    Patch::FaceType face = *fiter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);

    for ( CellIterator citer = patch->getFaceIterator( face, Patch::InteriorFaceCells );
          !citer.done(); citer++ ){

      IntVector cb = *citer + insideCellDir; 

      if (cellType[cb] == check_type ) 
        scalar[cb] = set_point; 

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
  IntVector idxLo = patch->getSFCXFORTLowIndex__Old();
  IntVector idxHi = patch->getSFCXFORTHighIndex__Old();
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
  idxLo = patch->getSFCYFORTLowIndex__Old();
  idxHi = patch->getSFCYFORTHighIndex__Old();

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
  idxLo = patch->getSFCZFORTLowIndex__Old();
  idxHi = patch->getSFCZFORTHighIndex__Old();

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
                              const int matl_index, 
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
  if(_using_new_intrusion){
    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
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


  if ( !d_use_new_bcs ) { 
    //__________________________________
    //  Pressure, outlet, and wall BC
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    
    for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
      Patch::FaceType face = *itr;
      
      IntVector offset = patch->faceDirection(face);
      
      CellIterator iter = patch->getFaceIterator(face, Patch::InteriorFaceCells);
      
      //face:       -x +x -y +y -z +z
      //Stencil 7   w, e, s, n, b, t;
      for(;!iter.done(); iter++){
        IntVector c = *iter;
        IntVector adj = c + offset;
        
        if( cellType[adj] == pressure_BC ||
            cellType[adj] == outlet_BC){
          // dirichlet_BC
          A[c].p = A[c].p -  A[c][face];
          A[c][face] = 0.0;
        }

        if( cellType[adj] == wall_BC){
          // Neumann zero gradient BC
          A[c].p = A[c].p + A[c][face];
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

        CellIterator iter = patch->getFaceIterator(face, Patch::InteriorFaceCells);

        //face:       -x +x -y +y -z +z
        //Stencil 7   w, e, s, n, b, t;
        for(;!iter.done(); iter++){
          IntVector c = *iter;
          IntVector adj = c + offset;
          
          if( cellType[adj] == d_flowInlets[ii]->d_cellTypeID){
            // Neumann zero gradient BC
            
            A[c].p = A[c].p +  A[c][face];
            A[c][face] = 0.0;
          }
        }
      }
    }
  } else { 

    std::vector<BC_TYPE> add_types; 
    add_types.push_back( OUTLET ); 
    add_types.push_back( PRESSURE ); 
    int sign = -1; 

    zeroStencilDirection( patch, matl_index, sign, A, add_types ); 

    std::vector<BC_TYPE> sub_types; 
    sub_types.push_back( WALL ); 
    sub_types.push_back( MASSFLOW_INLET ); 
    sub_types.push_back( VELOCITY_INLET ); 
    sub_types.push_back( VELOCITY_FILE ); 
    sub_types.push_back( MASSFLOW_FILE ); 
    sub_types.push_back( SWIRL );
    sub_types.push_back( TURBULENT_INLET ) ;
    sign = 1;

    zeroStencilDirection( patch, matl_index, sign, A, sub_types ); 

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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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
 
  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
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
      //domain boundary bc's 
      if (constvars->cellType[curr - IntVector(1,0,0)] == wall){

        vars->scalarTotCoef[curr].w = 0.0;
        vars->scalarDiffCoef[curr].w = 0.0;
        vars->scalarConvCoef[curr].w = 0.0;

      }

    }
 
  }
}

void
BoundaryCondition::mmWallTemperatureBC(const Patch* patch,
                                       constCCVariable<int>& cellType,
                                       constCCVariable<double> solidTemp,
                                       CCVariable<double>& temperature,
                                       bool d_energyEx)
{
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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

void 
BoundaryCondition::sched_setIntrusionTemperature( SchedulerP& sched, 
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls) 
{ 
  if ( _using_new_intrusion ){ 
    // Interface to new intrusions
    _intrusionBC->sched_setIntrusionT( sched, patches, matls ); 
  }
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
  IntVector idxLoU = patch->getSFCXFORTLowIndex__Old();
  IntVector idxHiU = patch->getSFCXFORTHighIndex__Old();
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
  idxLoU = patch->getSFCYFORTLowIndex__Old();
  idxHiU = patch->getSFCYFORTHighIndex__Old();
  
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
  idxLoU = patch->getSFCZFORTLowIndex__Old();
  idxHiU = patch->getSFCZFORTHighIndex__Old();

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

  for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 

    IntVector c = *iter; 

    if ( constvars->cellType[c] == d_mmWallID ){ 

      const double constant = 1.0; 
      const double value    = 0.0; 

      fix_value( vars->pressCoeff, vars->pressNonlinearSrc,  
          vars->pressLinearSrc, value, constant, c ); 

    } else { 

      if ( constvars->cellType[ c + IntVector(1,0,0) ] == d_mmWallID ){ 
        vars->pressCoeff[c].e = 0.0; 
      } 
      if ( constvars->cellType[ c - IntVector(1,0,0) ] == d_mmWallID ){ 
        vars->pressCoeff[c].w = 0.0; 
      } 
      if ( constvars->cellType[ c + IntVector(0,1,0) ] == d_mmWallID ){ 
        vars->pressCoeff[c].n = 0.0; 
      } 
      if ( constvars->cellType[ c - IntVector(0,1,0) ] == d_mmWallID ){ 
        vars->pressCoeff[c].s = 0.0; 
      } 
      if ( constvars->cellType[ c + IntVector(0,0,1) ] == d_mmWallID ){ 
        vars->pressCoeff[c].t = 0.0; 
      } 
      if ( constvars->cellType[ c - IntVector(0,0,1) ] == d_mmWallID ){ 
        vars->pressCoeff[c].b = 0.0; 
      } 

    } 
  } 
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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
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
// applies multimaterial bc's for enthalpy
void
BoundaryCondition::mmEnthalpyWallBC( const Patch* patch,
                                     CellInformation*,
                                     ArchesVariables* vars,
                                     ArchesConstVariables* constvars)
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
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
  d_flowRate_label(copy.d_flowRate_label),
  d_inlet_name(copy.d_inlet_name)
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

  // ---- Velocity inlet information ---- 
  std::string input_type; 
  params->getWithDefault("velocity_type", input_type, "flat");

  if ( input_type == "flat" ) {

    if ( params->findBlock( "Flow_rate" ) && params->findBlock("InletVelocity") ) 
      throw InvalidValue("Error: Flow_rate and InletVelocity cannot both be specified in FlowInlet BC", __FILE__, __LINE__); 

    if ( params->findBlock( "Flow_rate" ) ) {
      params->getWithDefault("Flow_rate", flowRate,0.0);
      inletVel = 0.0; 
    } else if ( params->findBlock( "InletVelocity" ) ) { 
      params->getWithDefault("InletVelocity", inletVel,0.0);
      flowRate = 0.0; 
    } 

    d_inletVelType = FlowInlet::VEL_FLAT_PROFILE; 

  } else if ( input_type == "function" ) {
      
    throw InvalidValue("Error: velocity_type = function not yet supported. ", __FILE__, __LINE__); 

  } else if ( input_type == "vector" ) {

    throw InvalidValue("Error: velocity_type = vector not yet supported. ", __FILE__, __LINE__); 

  } else if ( input_type == "file" ) {

    std::string filename; 
    params->require("vel_filename", filename); 

    d_inletVelType = FlowInlet::VEL_FILE_INPUT; 

  } else 
      throw InvalidValue("Error: FlowInlet velocity_type not recognized.", __FILE__, __LINE__); 

  // ---- swirl ------------
  params->getWithDefault("swirl_no", swirl_no, 0.0 ); 
  params->getWithDefault("swirl_cent", swirl_cent, Vector(0,0,0) );
  do_swirl = false; 
  if ( swirl_no > 0.0 ) { 
    do_swirl = true; 
  } 

  // ---- Scalar inlet information --- 
  params->getWithDefault("scalar_type", input_type, "flat");
  if ( input_type == "flat" ) {

    double mixfrac;
    double heatloss; 
    double mixfrac2; 
    mixfrac2 = 0.0; 

    params->require("mixture_fraction", mixfrac);
    streamMixturefraction.d_mixVars.push_back(mixfrac);
    streamMixturefraction.d_has_second_mixfrac = false; 

    if (params->findBlock("mixture_fraction_2")){
      params->require("mixture_fraction_2", mixfrac2); 
      streamMixturefraction.d_f2 = mixfrac2; 
      streamMixturefraction.d_has_second_mixfrac = true;
    }

    params->getWithDefault("heat_loss", heatloss, 0); 
    streamMixturefraction.d_heatloss = heatloss; 

    d_inletScalarType = FlowInlet::SCALAR_FLAT_PROFILE;

  } else if ( input_type == "function" ) {
      
    throw InvalidValue("Error: scalar_type = function not yet supported. ", __FILE__, __LINE__); 

  } else if ( input_type == "file" ) {

    std::string filename; 
    params->require("scalar_filename", filename); 

    d_inletScalarType = FlowInlet::SCALAR_FILE_INPUT; 

  } else 
      throw InvalidValue("Error: FlowInlet scalar_type not recognized.", __FILE__, __LINE__); 

  // check to see if this will work
  ProblemSpecP geomObjPS = params->findBlock("geom_object");
  params->getWithDefault("name",d_inlet_name,"not named"); 
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


  if (d_calcVariance){
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  }
 
  //Prefill a geometry object with the flow inlet condition 
  d_prefill = params->findBlock("Prefill"); 
  if (params->findBlock("Prefill") ){

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

  if (params->findBlock("mixture_fraction_2")){
    double mixfrac2; 
    params->require("mixture_fraction_2", mixfrac2); 
    streamMixturefraction.d_f2 = mixfrac2; 
    streamMixturefraction.d_has_second_mixfrac = true;
  } else { 
    streamMixturefraction.d_has_second_mixfrac = false; 
    streamMixturefraction.d_f2 = 0.0; 
  }

  double heatloss; 
  params->getWithDefault("heat_loss", heatloss, 0); 
  streamMixturefraction.d_heatloss = heatloss; 

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

  if (params->findBlock("mixture_fraction_2")){
    double mixfrac2; 
    params->require("mixture_fraction_2", mixfrac2); 
    streamMixturefraction.d_f2 = mixfrac2; 
    streamMixturefraction.d_has_second_mixfrac = true;
  } else { 
    streamMixturefraction.d_has_second_mixfrac = false; 
    streamMixturefraction.d_f2 = 0.0; 
  }

  double heatloss; 
  params->getWithDefault("heat_loss", heatloss, 0); 
  streamMixturefraction.d_heatloss = heatloss; 

  if (d_calcVariance)
    streamMixturefraction.d_mixVarVariance.push_back(0.0);
  double reactscalar;
  if (d_reactingScalarSolve) {
    params->require("reacting_scalar", reactscalar);
    streamMixturefraction.d_rxnVars.push_back(reactscalar);
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
  idxLoU = patch->getSFCXFORTLowIndex__Old();
  idxHiU = patch->getSFCXFORTHighIndex__Old();
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
  idxLoU = patch->getSFCYFORTLowIndex__Old();
  idxHiU = patch->getSFCYFORTHighIndex__Old();
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
  idxLoU = patch->getSFCZFORTLowIndex__Old();
  idxHiU = patch->getSFCZFORTHighIndex__Old();

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
  idxLo = patch->getSFCXFORTLowIndex__Old();
  idxHi = patch->getSFCXFORTHighIndex__Old();
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
  idxLo = patch->getSFCYFORTLowIndex__Old();
  idxHi = patch->getSFCYFORTHighIndex__Old();
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
  idxLo = patch->getSFCZFORTLowIndex__Old();
  idxHi = patch->getSFCZFORTHighIndex__Old();
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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();


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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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
    
    for(CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done();iter++) {
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
                                    const int matl_index, 
                                    double time_shift)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  double current_time = time + time_shift;
  Vector Dx = patch->dCell(); 

  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  // stores cell type info for the patch with the ghost cell type
  
  if ( !d_use_new_bcs ) { 
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

      double cent_y = fi->swirl_cent[1];
      double cent_z = fi->swirl_cent[2]; 
      double dy = Dx.y(); 
      double dz = Dx.z(); 

      fort_inlbcs(vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat,
                  idxLo, idxHi, constvars->new_density, constvars->cellType, 
                  fi->d_cellTypeID, current_time,
                  xminus, xplus, yminus, yplus, zminus, zplus,
                  fi->d_ramping_inlet_flowrate, dy, dz, 
                  fi->do_swirl, cent_y, cent_z, fi->swirl_no );

    }
  } else { 

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
          bc_iter != d_bc_information.end(); bc_iter++){

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ){

        //get the face
        Patch::FaceType face = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face); 

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++){

          double bc_value = 0;
          Vector bc_v_value(0,0,0); 
          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false;

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ){ 
            foundIterator = 
              getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind); 
          } else { 
            foundIterator = 
              getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 
          } 

          if ( foundIterator ) {

            bound_ptr.reset(); 

            if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == MASSFLOW_INLET ) { 

              setVel__NEW( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, constvars->new_density, bound_ptr, bc_iter->second.velocity ); 
 
            } else if (bc_iter->second.type == TURBULENT_INLET) {

              setTurbInlet( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, constvars->new_density, bound_ptr, bc_iter->second.TurbIn );
            
            } else if ( bc_iter->second.type == SWIRL ) { 

              if ( face == Patch::xminus || face == Patch::xplus ) { 

                setSwirl( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, 
                    constvars->new_density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no, bc_iter->second.swirl_cent ); 

              } else if ( face == Patch::yminus || face == Patch::yplus ){ 

               setSwirl( patch, face, vars->vVelRhoHat, vars->wVelRhoHat, vars->uVelRhoHat, 
                    constvars->new_density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no, bc_iter->second.swirl_cent  ); 

              } else if ( face == Patch::zminus || face == Patch::zplus ){ 

                setSwirl( patch, face, vars->wVelRhoHat, vars->uVelRhoHat, vars->vVelRhoHat, 
                    constvars->new_density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no, bc_iter->second.swirl_cent  ); 

              } 

            } else if ( bc_iter->second.type == VELOCITY_FILE ) {

              setVelFromExtraValue__NEW( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, constvars->new_density, bound_ptr, bc_iter->second.velocity ); 
              //enthalpy? 

            }

          }
        }
      }
    }
  } 
}

//****************************************************************************
// Set hat velocity at the outlet
//****************************************************************************
void 
BoundaryCondition::velRhoHatOutletPressureBC( const Patch* patch,
                                              SFCXVariable<double>& uvel, 
                                              SFCYVariable<double>& vvel, 
                                              SFCZVariable<double>& wvel, 
                                              constSFCXVariable<double>& old_uvel, 
                                              constSFCYVariable<double>& old_vvel, 
                                              constSFCZVariable<double>& old_wvel, 
                                              constCCVariable<int>& cellType )  
{
  // Get the low and high index for the patch
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  int outlet_type = outletCellType();
  int pressure_type = pressureCellType();

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
        if ((cellType[xminusCell] == outlet_type)||
            (cellType[xminusCell] == pressure_type)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
            uvel[currCell] = 0.0;
          else {
          if (cellType[xminusCell] == outlet_type)
            sign = 1;
          else
            sign = -1;
          if (sign * old_uvel[currCell] < -1.0e-10)
            uvel[currCell] = uvel[xplusCell];
          else
            uvel[currCell] = 0.0;
          }
          uvel[xminusCell] = uvel[currCell];
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
        if ((cellType[xplusCell] == outlet_type)||
            (cellType[xplusCell] == pressure_type)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
            uvel[xplusCell] = 0.0;
          else {
          if (cellType[xplusCell] == outlet_type)
            sign = 1;
          else
            sign = -1;
          if (sign * old_uvel[xplusCell] > 1.0e-10)
            uvel[xplusCell] = uvel[currCell];
          else
            uvel[xplusCell] = 0.0;
          }
          uvel[xplusplusCell] = uvel[xplusCell];
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
        if ((cellType[yminusCell] == outlet_type)||
            (cellType[yminusCell] == pressure_type)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vvel[currCell] = 0.0;
          else {
          if (cellType[yminusCell] == outlet_type)
            sign = 1;
          else
            sign = -1;
          if (sign * old_vvel[currCell] < -1.0e-10)
            vvel[currCell] = vvel[yplusCell];
          else
            vvel[currCell] = 0.0;
          }
          vvel[yminusCell] = vvel[currCell];
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
        if ((cellType[yplusCell] == outlet_type)||
            (cellType[yplusCell] == pressure_type)) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vvel[yplusCell] = 0.0;
          else {
          if (cellType[yplusCell] == outlet_type)
            sign = 1;
          else
            sign = -1;
          if (sign * old_vvel[yplusCell] > 1.0e-10)
            vvel[yplusCell] = vvel[currCell];
          else
            vvel[yplusCell] = 0.0;
          }
          vvel[yplusplusCell] = vvel[yplusCell];
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
        if ((cellType[zminusCell] == outlet_type)||
            (cellType[zminusCell] == pressure_type)) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            wvel[currCell] = 0.0;
          else {
          if (cellType[zminusCell] == outlet_type)
            sign = 1;
          else
            sign = -1;
          if (sign * old_wvel[currCell] < -1.0e-10)
            wvel[currCell] = wvel[zplusCell];
          else
            wvel[currCell] = 0.0;
          }
          wvel[zminusCell] = wvel[currCell];
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
        if ((cellType[zplusCell] == outlet_type)||
            (cellType[zplusCell] == pressure_type)) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            wvel[zplusCell] = 0.0;
          else {
          if (cellType[zplusCell] == outlet_type)
            sign = 1;
          else
            sign = -1;
          if (sign * old_wvel[zplusCell] > 1.0e-10)
            wvel[zplusCell] = wvel[currCell];
          else
            wvel[zplusCell] = 0.0;
          }
          wvel[zplusplusCell] = wvel[zplusCell];
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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
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
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(density, d_lab->d_densityCPLabel,       indx, patch, gn, 0);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);

    // Get the low and high index for the patch and the variables
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

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
        int outlet_celltypeval = BoundaryCondition::OUTLET;
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
    Vector Dx = patch->dCell(); 
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

    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
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
    
        double cent_y = fi->swirl_cent[1];
        double cent_z = fi->swirl_cent[2]; 
        double dy = Dx.y(); 
        double dz = Dx.z(); 

        fort_inlbcs( uVelocity, vVelocity, wVelocity, 
                    idxLo, idxHi, density, cellType, 
                    fi->d_cellTypeID, current_time,
                    xminus, xplus, yminus, yplus, zminus, zplus,
                    fi->d_ramping_inlet_flowrate, dy, dz, 
                    fi->do_swirl, cent_y, cent_z, fi->swirl_no );
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
  
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gn, 0);

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

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(constVars.cellType, d_lab->d_cellTypeLabel, indx, patch, gac,1);

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    new_dw->get(constVars.density,   d_lab->d_densityCPLabel,     indx, patch, gn, 0);
    new_dw->get(constVars.uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(constVars.vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gn, 0);
    new_dw->get(constVars.wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gn, 0);

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
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

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
    //int archIndex = 0; // only one arches material

    for ( EfficiencyMap::iterator iter = d_effVars.begin(); iter != d_effVars.end(); iter++){
      new_dw->put(delt_vartype(0.0),iter->second.label);
    }

    for ( SpeciesEffMap::iterator iter = d_speciesEffInfo.begin(); iter != d_speciesEffInfo.end(); iter++){
      new_dw->put(delt_vartype(0.0),iter->second.flowRateLabel);
    }
  }
}
void BoundaryCondition::sched_setAreaFraction(SchedulerP& sched, 
                                     const PatchSet* patches, 
                                     const MaterialSet* matls )
{
  Task* tsk = scinew Task( "BoundaryCondition::setAreaFraction",this, &BoundaryCondition::setAreaFraction);

  tsk->modifies(d_lab->d_areaFractionLabel); 
#ifdef WASATCH_IN_ARCHES
  tsk->modifies(d_lab->d_areaFractionFXLabel); 
  tsk->modifies(d_lab->d_areaFractionFYLabel); 
  tsk->modifies(d_lab->d_areaFractionFZLabel); 
#endif
  tsk->modifies(d_lab->d_volFractionLabel); 
  tsk->requires( Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells, 1 ); 
 
  sched->addTask(tsk, patches, matls);
}
void 
BoundaryCondition::setAreaFraction( const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<Vector>   areaFraction; 
#ifdef WASATCH_IN_ARCHES
    SFCXVariable<double> areaFractionFX; 
    SFCYVariable<double> areaFractionFY; 
    SFCZVariable<double> areaFractionFZ; 
#endif
    CCVariable<double>   volFraction; 
    constCCVariable<int> cellType; 

    new_dw->get( cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::AroundCells, 1 ); 
    new_dw->getModifiable( areaFraction, d_lab->d_areaFractionLabel, indx, patch );  
#ifdef WASATCH_IN_ARCHES
    new_dw->getModifiable( areaFractionFX, d_lab->d_areaFractionFXLabel, indx, patch );  
    new_dw->getModifiable( areaFractionFY, d_lab->d_areaFractionFYLabel, indx, patch );  
    new_dw->getModifiable( areaFractionFZ, d_lab->d_areaFractionFZLabel, indx, patch );  
#endif 
    new_dw->getModifiable( volFraction, d_lab->d_volFractionLabel, indx, patch );  

    int flowType = -1; 
    if (d_MAlab)
      d_newBC->setAreaFraction( patch, areaFraction, volFraction, cellType, d_mmWallID, flowType ); 

    if (d_wallBdry) 
      d_newBC->setAreaFraction( patch, areaFraction, volFraction, cellType, d_wallBdry->d_cellTypeID, flowType ); 

    d_newBC->setAreaFraction( patch, areaFraction, volFraction, cellType, WALL, flowType ); 
    d_newBC->setAreaFraction( patch, areaFraction, volFraction, cellType, INTRUSION, flowType ); 
    d_newBC->setAreaFraction( patch, areaFraction, volFraction, cellType, MMWALL, flowType ); 

#ifdef WASATCH_IN_ARCHES
    //copy for wasatch-arches: 
    for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      areaFractionFX[c] = areaFraction[c].x(); 
      areaFractionFY[c] = areaFraction[c].y(); 
      areaFractionFZ[c] = areaFraction[c].z(); 
    }
#endif 
  }
}

//-------------------------------------------------------------
// New Domain BCs
//
void 
BoundaryCondition::setupBCs( ProblemSpecP& db )
{
 
  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions"); 
  int bc_type_index = 0; 

  //Map types to strings:
  d_bc_type_to_string.insert( std::make_pair( TURBULENT_INLET, "TurbulentInlet" ) );                            
  d_bc_type_to_string.insert( std::make_pair( VELOCITY_INLET , "VelocityInlet" ) );
  d_bc_type_to_string.insert( std::make_pair( MASSFLOW_INLET , "MassFlowInlet" ) );
  d_bc_type_to_string.insert( std::make_pair( VELOCITY_FILE  , "VelocityFileInput" ) );
  d_bc_type_to_string.insert( std::make_pair( PRESSURE       , "PressureBC" ) );
  d_bc_type_to_string.insert( std::make_pair( OUTLET         , "OutletBC" ) );
  d_bc_type_to_string.insert( std::make_pair( SWIRL          , "Swirl" ) ); 
  d_bc_type_to_string.insert( std::make_pair( WALL           , "WallBC" ) );

  // Now actually look for the boundary types
  if ( db_bc ) { 
    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != 0; 
          db_face = db_face->findNextBlock("Face") ){
      
      for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != 0; 
          db_BCType = db_BCType->findNextBlock("BCType") ){

        std::string name; 
        std::string type; 
        bool found_bc = false; 
        BCInfo my_info; 
        db_BCType->getAttribute("label", name);
        db_BCType->getAttribute("var", type); 
        my_info.name = name;
        std::stringstream color; 
        color << bc_type_index; 

        if ( type == "VelocityInlet" ){

          my_info.type = VELOCITY_INLET; 
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          db_BCType->require("value", my_info.velocity);
          found_bc = true; 

          //old: remove when this is cleaned up: 
          d_inletBoundary = true; 
          
        } else if ( type == "TurbulentInlet" ) {
          
          my_info.type = TURBULENT_INLET;
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          db_BCType->require("inputfile", my_info.filename);
          db_BCType->require("value", my_info.velocity);
          found_bc = true; 
         
          my_info.TurbIn = scinew DigitalFilterInlet( );
          my_info.TurbIn->problemSetup( db_BCType );
   
          //old: remove when this is cleaned up: 
          d_inletBoundary = true;
        
        } else if ( type == "MassFlowInlet" ){

          my_info.type = MASSFLOW_INLET;
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          my_info.velocity = Vector(0,0,0); 
          my_info.mass_flow_rate = 0.0;
          found_bc = true; 

          // note that the mass flow rate is in the BCstruct value 

          //old: remove when this is cleaned up: 
          d_inletBoundary = true; 

        } else if ( type == "VelocityFileInput" ){ 

          my_info.type = VELOCITY_FILE; 
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          db_BCType->require("inputfile", my_info.filename); 
          my_info.velocity = Vector(0,0,0); 
          found_bc = true; 

          //old: remove when this is cleaned up: 
          d_inletBoundary = true; 

        } else if ( type == "Swirl" ){ 

          my_info.type = SWIRL; 
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          db_BCType->require("swirl_no", my_info.swirl_no);
          db_BCType->require("swirl_centroid", my_info.swirl_cent); 

          // note that the mass flow rate is in the BCstruct value 

          found_bc = true; 

          //old: remove when this is cleaned up: 
          d_inletBoundary = true; 

        } else if ( type == "PressureBC" ){

          my_info.type = PRESSURE; 
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          my_info.velocity = Vector(0,0,0); 
          found_bc = true; 

          //old: remove when this is cleaned up: 
          d_pressureBoundary = true; 

        } else if ( type == "OutletBC" ){ 

          my_info.type = OUTLET; 
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          my_info.velocity = Vector(0,0,0); 
          found_bc = true; 

          //old: remove when this is cleaned up: 
          d_outletBoundary = true; 

        } else if ( type == "WallBC" ){

          my_info.type = WALL;
          my_info.total_area_label = VarLabel::create( "bc_area"+color.str()+name, ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());
          my_info.velocity = Vector(0,0,0); 
          found_bc = true; 

        }

        if ( found_bc ) {
          d_bc_information.insert( std::make_pair(bc_type_index, my_info));
          bc_type_index++; 
        }

      }
    }
  }
}

//-------------------------------------------------------------
// Set the cell Type
//
void 
BoundaryCondition::sched_cellTypeInit__NEW(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::cellTypeInit__NEW",
                          this, &BoundaryCondition::cellTypeInit__NEW);

  tsk->computes(d_lab->d_cellTypeLabel);

  sched->addTask(tsk, patches, matls);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void 
BoundaryCondition::cellTypeInit__NEW(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matl_index = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<int> cellType;
    new_dw->allocateAndPut(cellType, d_lab->d_cellTypeLabel, matl_index, patch);
    cellType.initialize(999);

    for ( CellIterator iter=patch->getCellIterator(); !iter.done(); iter++ ){

      // initialize all cells in the interior as flow
      // intrusions will be dealt with later
      cellType[*iter] = -1; 

    }

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

      for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++){

        //get the face
        Patch::FaceType face = *bf_iter;

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++){

          double bc_value = 0;
          Vector bc_v_value(0,0,0); 
          
          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false; 

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ){ 
            foundIterator = 
              getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind); 
          } else { 
            foundIterator = 
              getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 
          } 

          if ( foundIterator ) {

            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

              cellType[*bound_ptr] = bc_iter->second.type;

            }
          }
        }
      }
    }
  }
}

//-------------------------------------------------------------
// Compute BC Areas
//
void 
BoundaryCondition::sched_computeBCArea__NEW(SchedulerP& sched,
                                      const LevelP& level, 
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  IntVector lo, hi;
  level->findInteriorCellIndexRange(lo,hi);

  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::computeBCArea__NEW",
                          this, &BoundaryCondition::computeBCArea__NEW, lo, hi);

  for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

    BCInfo the_info = bc_iter->second; 
    tsk->computes( the_info.total_area_label ); 

  }

  sched->addTask(tsk, patches, matls);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void 
BoundaryCondition::computeBCArea__NEW(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw, 
                                const IntVector lo, 
                                const IntVector hi )
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matl_index = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    Vector Dx = patch->dCell(); 

    for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
          bc_iter != d_bc_information.end(); bc_iter++){

      double area = 0; 

      for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++){

        //get the face
        Patch::FaceType face = *bf_iter;

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++){

          double bc_value = 0;
          Vector bc_v_value(0,0,0); 
          
          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false; 

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ){ 
            foundIterator = 
              getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind); 
          } else { 
            foundIterator = 
              getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 
          } 

          double dx_1 = 0.0;
          double dx_2 = 0.0; 
          IntVector shift; 
          shift = IntVector(0,0,0);

          if ( foundIterator ) {

            switch (face) {
              case Patch::xminus:
                dx_1 = Dx.y();
                dx_2 = Dx.z(); 
                shift = IntVector( 1, 0, 0);
                break;
              case Patch::xplus:
                dx_1 = Dx.y();
                dx_2 = Dx.z(); 
                shift = IntVector( 1, 0, 0);
                break;
              case Patch::yminus:
                dx_1 = Dx.x();
                dx_2 = Dx.z(); 
                shift = IntVector( 0, 1, 0);
                break;
              case Patch::yplus:
                dx_1 = Dx.x();
                dx_2 = Dx.z(); 
                shift = IntVector( 0, 1, 0);
                break;
              case Patch::zminus: 
                dx_1 = Dx.y();
                dx_2 = Dx.x(); 
                shift = IntVector( 0, 0, 1);
                break;
              case Patch::zplus:
                dx_1 = Dx.y();
                dx_2 = Dx.x(); 
                shift = IntVector( 0, 0, 1);
                break;
              default:
                break;
            }

            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

              IntVector c = *bound_ptr; 

              // "if" needed to ensure that extra cell contributions aren't added
              if ( c.x() >= lo.x() - shift.x() && c.x() < hi.x() + shift.x() ){ 
                if ( c.y() >= lo.y() - shift.y() && c.y() < hi.y() + shift.y() ){ 
                  if ( c.z() >= lo.z() - shift.z() && c.z() < hi.z() + shift.z() ){ 

                    area += dx_1*dx_2;
                  }
                }
              }
            }
          }
        }
      }

      new_dw->put( sum_vartype(area), bc_iter->second.total_area_label );
  
    }
  }
}

//--------------------------------------------------------------------------------
// Compute velocities from mass flow rates for bc's
//
void 
BoundaryCondition::sched_setupBCInletVelocities__NEW(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::setupBCInletVelocities__NEW",
                          this, &BoundaryCondition::setupBCInletVelocities__NEW);

  for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

    BCInfo the_info = bc_iter->second; 
    tsk->requires( Task::NewDW, the_info.total_area_label ); 

  }

  tsk->requires( Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 0 ); 

  sched->addTask(tsk, patches, matls);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void 
BoundaryCondition::setupBCInletVelocities__NEW(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matl_index = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
//    Vector Dx = patch->dCell(); 
    constCCVariable<double> density; 
    new_dw->get( density, d_lab->d_densityCPLabel, matl_index, patch, Ghost::None, 0 ); 

    proc0cout << "\nBoundary condition summary for inlets: \n";

    for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
          bc_iter != d_bc_information.end(); bc_iter++){

      sum_vartype area_var;
      new_dw->get( area_var, bc_iter->second.total_area_label );
      double area = area_var; 

      for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++){

        //get the face
        Patch::FaceType face = *bf_iter;

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++){

          double bc_value = 0;
          Vector bc_v_value(0,0,0); 
          int norm = getNormal( face ); 
          
          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false; 

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ){ 
            foundIterator = 
              getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind); 
          } else { 
            foundIterator = 
              getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 
          } 

          if ( foundIterator ) {

            // Notice: 
            // In the case of mass flow inlets, we are going to assume the density is constant across the inlet
            // so as to compute the average velocity.  As a result, we will just use the first iterator: 
            bound_ptr.reset(); 
            if ( (bc_iter->second).type == MASSFLOW_INLET ) {
              (bc_iter->second).mass_flow_rate = bc_value; 
              (bc_iter->second).velocity[norm] = (bc_iter->second).mass_flow_rate / 
                                               ( area * density[*bound_ptr] );
            } else if ( (bc_iter->second).type == SWIRL ) { 
                (bc_iter->second).mass_flow_rate = bc_value; 
                (bc_iter->second).velocity[norm] = (bc_iter->second).mass_flow_rate / 
                                                 ( area * density[*bound_ptr] ); 
            } 

            switch ( bc_iter->second.type ) {

              case ( VELOCITY_INLET ): 
                bc_iter->second.mass_flow_rate = bc_iter->second.velocity[norm] * area * density[*bound_ptr];
                break;
              case (TURBULENT_INLET):
                bc_iter->second.mass_flow_rate = bc_iter->second.velocity[norm] * area * density[*bound_ptr];
                break;
              case ( MASSFLOW_INLET ): 
                bc_iter->second.mass_flow_rate = bc_value; 
                bc_iter->second.velocity[norm] = bc_iter->second.mass_flow_rate / 
                                                 ( area * density[*bound_ptr] );
                break;
              case ( SWIRL ):
                bc_iter->second.mass_flow_rate = bc_value; 
                bc_iter->second.velocity[norm] = bc_iter->second.mass_flow_rate / 
                                                 ( area * density[*bound_ptr] ); 
                break; 

              default: 
                break; 
            }

          }
        }
      }
    }
    proc0cout << endl;
  }
}
//--------------------------------------------------------------------------------
// Apply the boundary conditions
//
void 
BoundaryCondition::sched_setInitProfile__NEW(SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::setInitProfile__NEW",
                          this, &BoundaryCondition::setInitProfile__NEW);

  // Momentum
  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);

  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  // Density
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 0); 

  MixingRxnModel* mixingTable = d_props->getMixRxnModel(); 
  MixingRxnModel::VarMap iv_vars = mixingTable->getIVVars(); 

  for ( MixingRxnModel::VarMap::iterator i = iv_vars.begin(); i != iv_vars.end(); i++ ){ 

    tsk->requires( Task::NewDW, i->second, Ghost::AroundCells, 0 ); 

  }

  // Energy
  if ( d_enthalpySolve ){ 

    tsk->modifies( d_lab->d_enthalpySPLabel ); 

  }

  sched->addTask(tsk, patches, matls);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void 
BoundaryCondition::setInitProfile__NEW(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matl_index = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    SFCXVariable<double> uVelocity; 
    SFCYVariable<double> vVelocity; 
    SFCZVariable<double> wVelocity; 
    SFCXVariable<double> uRhoHat; 
    SFCYVariable<double> vRhoHat; 
    SFCZVariable<double> wRhoHat; 
    CCVariable<double> enthalpy; 
    constCCVariable<double> density; 

    new_dw->getModifiable( uVelocity, d_lab->d_uVelocitySPBCLabel, matl_index, patch ); 
    new_dw->getModifiable( vVelocity, d_lab->d_vVelocitySPBCLabel, matl_index, patch ); 
    new_dw->getModifiable( wVelocity, d_lab->d_wVelocitySPBCLabel, matl_index, patch ); 
    new_dw->getModifiable( uRhoHat, d_lab->d_uVelRhoHatLabel, matl_index, patch ); 
    new_dw->getModifiable( vRhoHat, d_lab->d_vVelRhoHatLabel, matl_index, patch ); 
    new_dw->getModifiable( wRhoHat, d_lab->d_wVelRhoHatLabel, matl_index, patch ); 
    if ( d_enthalpySolve )
      new_dw->getModifiable( enthalpy, d_lab->d_enthalpySPLabel, matl_index, patch ); 
    new_dw->get( density, d_lab->d_densityCPLabel, matl_index, patch, Ghost::None, 0 ); 

    MixingRxnModel* mixingTable = d_props->getMixRxnModel(); 
    MixingRxnModel::VarMap iv_vars = mixingTable->getIVVars(); 

    // Get the independent variable information for table lookup
    BoundaryCondition::HelperMap ivGridVarMap; 
    BoundaryCondition::HelperVec allIndepVarNames = mixingTable->getAllIndepVars(); 

    for ( MixingRxnModel::VarMap::iterator i = iv_vars.begin(); i != iv_vars.end(); i++ ){ 
      constCCVariable<double> variable; 
      new_dw->get( variable, i->second, matl_index, patch, Ghost::None, 0 );
      ivGridVarMap.insert( make_pair( i->first, variable));
    }

    for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
          bc_iter != d_bc_information.end(); bc_iter++){

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ){

        //get the face
        Patch::FaceType face = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face); 

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++){

          double bc_value = 0;
          Vector bc_v_value(0,0,0); 
          
          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false; 

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ){ 
            foundIterator = 
              getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind); 
          } else { 
            foundIterator = 
              getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 
          } 

          if ( foundIterator ) {

            bound_ptr.reset(); 

            if ( bc_iter->second.type != VELOCITY_FILE ) { 
              
              if ( bc_iter->second.type != TURBULENT_INLET ) {
              //---- set velocities
                setVel__NEW( patch, face, uVelocity, vVelocity, wVelocity, density, bound_ptr, bc_iter->second.velocity ); 
              } else {
#if 1
                bc_iter->second.TurbIn->findOffsetVector(  patch, face, bound_ptr );
                //testign which int vector is returned for each inlet patch
#endif
                setTurbInlet( patch, face, uVelocity, vVelocity, wVelocity, density, bound_ptr, bc_iter->second.TurbIn );
              }
              
              //---- set the enthalpy
              if ( d_enthalpySolve ) 
                setEnthalpy__NEW( patch, face, enthalpy, ivGridVarMap, allIndepVarNames, bound_ptr ); 
            } else {
              //---- set velocities
              setVelFromInput__NEW( patch, face, uVelocity, vVelocity, wVelocity, bound_ptr, bc_iter->second.filename ); 
              //---- set the enthalpy
              if ( d_enthalpySolve ) 
                setEnthalpyFromInput__NEW( patch, face, enthalpy, ivGridVarMap, allIndepVarNames, bound_ptr ); 
            }

          }
        }
      }
    }

    uRhoHat.copyData( uVelocity ); 
    vRhoHat.copyData( vVelocity ); 
    wRhoHat.copyData( wVelocity ); 

  }
}

void BoundaryCondition::setEnthalpy__NEW( const Patch* patch, const Patch::FaceType& face, 
    CCVariable<double>& enthalpy, BoundaryCondition::HelperMap ivGridVarMap, BoundaryCondition::HelperVec allIndepVarNames, 
    Iterator bound_ptr)
{
  //get the face direction
  IntVector insideCellDir = patch->faceDirection(face);
  MixingRxnModel* mixingTable = d_props->getMixRxnModel(); 

  for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

    IntVector c = *bound_ptr; 
    IntVector ci = *bound_ptr - insideCellDir; 

    std::vector<double> iv; 
    double hl = 0.0;
    for ( BoundaryCondition::HelperVec::iterator ivnames_iter = allIndepVarNames.begin(); 
          ivnames_iter != allIndepVarNames.end(); ivnames_iter++ ){ 

      BoundaryCondition::HelperMap::iterator which_var = ivGridVarMap.find( *ivnames_iter ); 
      double value = ( (which_var->second)[c] + (which_var->second)[ci] ) / 2.0; 
      iv.push_back( value );  

      if ( *ivnames_iter == "heat_loss" || *ivnames_iter == "HeatLoss" )
        hl = value; 
      
    }

    double h_a = mixingTable->getTableValue( iv, "adiabaticenthalpy" ); 
    double h_s = mixingTable->getTableValue( iv, "sensibleenthalpy" );

    // actually set the enthalpy on this boundary
    enthalpy[c] = h_a - hl * h_s;

  }
}

void BoundaryCondition::setEnthalpyFromInput__NEW( const Patch* patch, const Patch::FaceType& face, 
    CCVariable<double>& enthalpy, BoundaryCondition::HelperMap ivGridVarMap, BoundaryCondition::HelperVec allIndepVarNames, 
    Iterator bound_ptr ) 
{
}

template<class d0T, class d1T, class d2T>
void BoundaryCondition::setSwirl( const Patch* patch, const Patch::FaceType& face, 
        d0T& uVel, d1T& vVel, d2T& wVel,
        constCCVariable<double>& density, 
        Iterator bound_ptr, Vector value, 
        double swrl_no, Vector swrl_cent )
{

 //get the face direction
 IntVector insideCellDir = patch->faceDirection(face);
 Vector Dx = patch->dCell(); 
 Vector mDx; //mapped dx 
 int dir = 0; 

 //remap the dx's and vector values
 for (int i = 0; i < 3; i++ ){ 
  if ( insideCellDir[i] != 0 ) { 
    dir = i; 
  } 
 }
 Vector bc_values;
 for (int i = 0; i < 3; i++ ){ 
   int index = index_map[dir][i]; 
   bc_values[i] = value[index];
   mDx[i] = Dx[index];
 }

 for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

   IntVector c  = *bound_ptr; 
   IntVector cp = *bound_ptr - insideCellDir; 

   uVel[c]  = bc_values.x();
   uVel[cp] = bc_values.x() * density[c] / ( 0.5 * ( density[c] + density[cp] )); 

   double ave_u = (uVel[c] + uVel[cp])/2.0;

   Point p = patch->cellPosition(c); 
   vector<double> my_p; 
   my_p.push_back(p.x());
   my_p.push_back(p.y());
   my_p.push_back(p.z());

   double y = my_p[index_map[dir][1]] - swrl_cent[index_map[dir][1]];
   double z = my_p[index_map[dir][2]] + mDx.z()/2.0 - swrl_cent[index_map[dir][2]];

   double denom = pow(y,2) + pow(z,2); 
   denom = pow(denom,0.5); 

   vVel[c] = -1.0 * z * swrl_no * ave_u /denom; 

   y = my_p[index_map[dir][1]] + mDx.y()/2.0 - swrl_cent[index_map[dir][1]];
   z = my_p[index_map[dir][2]] - swrl_cent[index_map[dir][2]]; 

   denom = pow(y,2) + pow(z,2); 
   denom = pow(denom,0.5); 

   wVel[c] = y * swrl_no * ave_u / denom;

 }
}

void BoundaryCondition::setTurbInlet( const Patch* patch, const Patch::FaceType& face, 
                                    SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                                    constCCVariable<double>& density, 
                                    Iterator bound_ptr, DigitalFilterInlet * TurbInlet )
{
  IntVector insideCellDir = patch->faceDirection(face);
    
  int j, k;
  int ts = d_lab->d_sharedState->getCurrentTopLevelTimeStep();
  double elapTime = d_lab->d_sharedState->getElapsedTime();
  int t = TurbInlet->getTimeIndex( ts , elapTime);

  IntVector	shiftVec;
  shiftVec = TurbInlet->getOffsetVector( );
  
  switch ( face ) {
    case Patch::xminus:
      for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
        
        IntVector c  = *bound_ptr; 
        IntVector cp = *bound_ptr - insideCellDir;
        
        vector<double> velVal (3);
        j = c.y() - shiftVec.y();
        k = c.z() - shiftVec.z();
        
        velVal = TurbInlet->getVelocityVector(t , j, k);
        
        uVel[c]  = velVal[0];
        uVel[cp] = velVal[0]; 
        vVel[c]  = velVal[1]; 
        wVel[c]  = velVal[2];
      }
      break;
    case Patch::xplus:
      for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
        
        IntVector c  = *bound_ptr; 
        IntVector cp = *bound_ptr + insideCellDir;  
        
        vector<double> velVal (3);
        j = c.y() - shiftVec.y();
        k = c.z() - shiftVec.z();
        
        velVal = TurbInlet->getVelocityVector(t , j, k);
        
        uVel[cp] = velVal[0];
        uVel[c]  = velVal[0]; 
        vVel[c]  = velVal[1]; 
        wVel[c]  = velVal[2]; 
      }
      break;
    case Patch::yminus:
      for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
        
        IntVector c  = *bound_ptr; 
        IntVector cp = *bound_ptr - insideCellDir;  
        
        vector<double> velVal (3);
        j = c.x() - shiftVec.x();
        k = c.z() - shiftVec.z();
        
        velVal = TurbInlet->getVelocityVector(t , j, k);
        
        uVel[c]  = velVal[0]; 
        vVel[c]  = velVal[1]; 
        vVel[cp] = velVal[1];
        wVel[c]  = velVal[2]; 
      }
      
      break;
    case Patch::yplus:
      for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
        
        IntVector c  = *bound_ptr; 
        IntVector cp = *bound_ptr + insideCellDir;  
        
        vector<double> velVal (3);
        j = c.x() - shiftVec.x();
        k = c.z() - shiftVec.z();

        velVal = TurbInlet->getVelocityVector(t , j, k);
        
        uVel[c]  = velVal[0]; 
        vVel[c]  = velVal[1]; 
        vVel[cp] = velVal[1];
        wVel[c]  = velVal[2]; 
      }
      break;
    case Patch::zminus:
      for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
        
        IntVector c  = *bound_ptr; 
        IntVector cp = *bound_ptr - insideCellDir;  
        
        vector<double> velVal (3);
        j = c.x() - shiftVec.x();
        k = c.y() - shiftVec.y();

        velVal = TurbInlet->getVelocityVector(t , j, k);
        
        uVel[c]  = velVal[0]; 
        vVel[c]  = velVal[1]; 
        wVel[c]  = velVal[2];
        wVel[cp] = velVal[2]; 
      }
      break;
    case Patch::zplus:
      for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
        
        IntVector c  = *bound_ptr; 
        IntVector cp = *bound_ptr + insideCellDir;  
        
        vector<double> velVal (3);
        j = c.x() - shiftVec.x();
        k = c.y() - shiftVec.y();

        velVal = TurbInlet->getVelocityVector(t , j, k);
        
        uVel[c]  = velVal[0]; 
        vVel[c]  = velVal[1]; 
        wVel[c]  = velVal[2];
        wVel[cp] = velVal[2]; 
      }
      break;
    default:
      break;
      
  }
  
//  cout << "Inlet Timestep is " << t << endl;
}


void BoundaryCondition::setVel__NEW( const Patch* patch, const Patch::FaceType& face, 
        SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
        constCCVariable<double>& density, 
        Iterator bound_ptr, Vector value )
{

 //get the face direction
 IntVector insideCellDir = patch->faceDirection(face);

 switch ( face ) {

   case Patch::xminus :

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       uVel[c]  = value.x();
       uVel[cp] = value.x(); 

       vVel[c] = value.y(); 
       wVel[c] = value.z(); 
     }

     break; 
   case Patch::xplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr + insideCellDir; 
       IntVector cm = *bound_ptr - insideCellDir; 

       uVel[cp]  = value.x();
       uVel[c]   = value.x(); 

       vVel[c] = value.y(); 
       wVel[c] = value.z(); 
     }
     break; 
   case Patch::yminus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       vVel[c] = value.y();
       vVel[cp] = value.y(); 

       uVel[c] = value.x(); 
       wVel[c] = value.z(); 

     }
     break; 
   case Patch::yplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr + insideCellDir; 
       IntVector cm = *bound_ptr - insideCellDir; 

       vVel[cp] = value.y();
       vVel[c] = value.y(); 

       uVel[c] = value.x(); 
       wVel[c] = value.z(); 

     }
     break; 
   case Patch::zminus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       wVel[c] = value.z();
       wVel[cp] = value.z(); 

       uVel[c] = value.x(); 
       vVel[c] = value.y(); 

     }
     break; 
   case Patch::zplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr + insideCellDir; 
       IntVector cm = *bound_ptr - insideCellDir; 

       wVel[cp] = value.z();
       wVel[c] = value.z(); 

       uVel[c] = value.x(); 
       vVel[c] = value.y(); 

     }
     break; 
   default:

     break;

 }
}

void BoundaryCondition::setVelFromExtraValue__NEW( const Patch* patch, const Patch::FaceType& face, 
        SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
        constCCVariable<double>& density, 
        Iterator bound_ptr, Vector value )
{

 //get the face direction
 IntVector insideCellDir = patch->faceDirection(face);

 switch ( face ) {

   case Patch::xminus :

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       uVel[cp] = uVel[c] * density[c] / ( 0.5 * ( density[c] + density[cp] )); 

     }

     break; 
   case Patch::xplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       uVel[cp] = uVel[c] * density[c] / ( 0.5 * ( density[c] + density[cp] )); 

     }
     break; 
   case Patch::yminus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       vVel[cp] = vVel[c] * density[c] / ( 0.5 * ( density[c] + density[cp] )); 

     }
     break; 
   case Patch::yplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       vVel[cp] = vVel[c] * density[c] / ( 0.5 * ( density[c] + density[cp] )); 

     }
     break; 
   case Patch::zminus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       wVel[cp] = wVel[c] * density[c] / ( 0.5 * ( density[c] + density[cp] )); 

     }
     break; 
   case Patch::zplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       IntVector c  = *bound_ptr; 
       IntVector cp = *bound_ptr - insideCellDir; 

       wVel[cp] = wVel[c] * density[c] / ( 0.5 * ( density[c] + density[cp] )); 

     }
     break; 
   default:

     break;

 }
}

void BoundaryCondition::setVelFromInput__NEW( const Patch* patch, const Patch::FaceType& face, 
        SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
        Iterator bound_ptr, std::string file_name )
{

 //get the face direction
 IntVector insideCellDir = patch->faceDirection(face);

 switch ( face ) {

   case Patch::xminus:

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){
        
       CellToValue::iterator u_iter = _u_input.find( *bound_ptr ); 
       CellToValue::iterator v_iter = _v_input.find( *bound_ptr ); 
       CellToValue::iterator w_iter = _w_input.find( *bound_ptr ); 

       if ( u_iter != _u_input.end() ){ 
        uVel[ *bound_ptr ] = u_iter->second; 
        uVel[ *bound_ptr - insideCellDir ] = u_iter->second; 
       }

       if ( v_iter != _v_input.end() ) 
        vVel[ *bound_ptr ] = v_iter->second; 
       if ( w_iter != _w_input.end() )
        wVel[ *bound_ptr ] = w_iter->second; 

     }

     break; 
   case Patch::xplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       CellToValue::iterator u_iter = _u_input.find( *bound_ptr ); 
       CellToValue::iterator v_iter = _v_input.find( *bound_ptr ); 
       CellToValue::iterator w_iter = _w_input.find( *bound_ptr ); 

       if ( u_iter != _u_input.end() ){ 
        uVel[ *bound_ptr ] = u_iter->second; 
        uVel[ *bound_ptr - insideCellDir ] = u_iter->second; 
       }

       if ( v_iter != _v_input.end() )
        vVel[ *bound_ptr ] = v_iter->second; 
       if ( w_iter != _w_input.end() ) 
        wVel[ *bound_ptr ] = w_iter->second; 

     }
     break; 
   case Patch::yminus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       CellToValue::iterator u_iter = _u_input.find( *bound_ptr ); 
       CellToValue::iterator v_iter = _v_input.find( *bound_ptr ); 
       CellToValue::iterator w_iter = _w_input.find( *bound_ptr ); 

       if ( v_iter != _v_input.end()) { 
       vVel[ *bound_ptr ] = v_iter->second; 
       vVel[ *bound_ptr - insideCellDir ] = v_iter->second; 
       }

       if ( u_iter != _u_input.end() ) 
       uVel[ *bound_ptr ] = u_iter->second;
       if ( w_iter != _w_input.end() ) 
       wVel[ *bound_ptr ] = w_iter->second; 

     }
     break; 
   case Patch::yplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       CellToValue::iterator u_iter = _u_input.find( *bound_ptr ); 
       CellToValue::iterator v_iter = _v_input.find( *bound_ptr ); 
       CellToValue::iterator w_iter = _w_input.find( *bound_ptr ); 

       if ( v_iter != _v_input.end() ) {
       vVel[ *bound_ptr ] = v_iter->second; 
       vVel[ *bound_ptr - insideCellDir ] = v_iter->second; 
       }

       if ( u_iter != _u_input.end() )
       uVel[ *bound_ptr ] = u_iter->second;
       if ( w_iter != _w_input.end() ) 
       wVel[ *bound_ptr ] = w_iter->second; 

     }
     break; 
   case Patch::zminus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       CellToValue::iterator u_iter = _u_input.find( *bound_ptr ); 
       CellToValue::iterator v_iter = _v_input.find( *bound_ptr ); 
       CellToValue::iterator w_iter = _w_input.find( *bound_ptr ); 

       if ( w_iter != _w_input.end() ) { 
       wVel[ *bound_ptr ] = w_iter->second; 
       wVel[ *bound_ptr - insideCellDir ] = w_iter->second;
       }

       if ( u_iter != _u_input.end() ) 
       uVel[ *bound_ptr ] = u_iter->second;
       if ( v_iter != _v_input.end() ) 
       vVel[ *bound_ptr ] = v_iter->second; 

     }
     break; 
   case Patch::zplus: 

     for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

       CellToValue::iterator u_iter = _u_input.find( *bound_ptr ); 
       CellToValue::iterator v_iter = _v_input.find( *bound_ptr ); 
       CellToValue::iterator w_iter = _w_input.find( *bound_ptr ); 

       if ( w_iter != _w_input.end() ) { 
       wVel[ *bound_ptr ] = w_iter->second; 
       wVel[ *bound_ptr - insideCellDir ] = w_iter->second; 
       }

       if ( u_iter != _u_input.end() ) 
       uVel[ *bound_ptr ] = u_iter->second;
       if ( v_iter != _v_input.end() ) 
       vVel[ *bound_ptr ] = v_iter->second; 
     }
     break; 
   default:

     break;

 }
}

std::map<IntVector, double>
BoundaryCondition::readInputFile__NEW( std::string file_name )
{

  gzFile file = gzopen( file_name.c_str(), "r" ); 
  if ( file == NULL ) { 
    proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << endl;
    throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
  }

  std::string variable = getString( file ); 
  int         num_points = getInt( file ); 
  std::map<IntVector, double> result; 

  for ( int i = 0; i < num_points; i++ ) {
    int I = getInt( file ); 
    int J = getInt( file ); 
    int K = getInt( file ); 
    double v = getDouble( file ); 

    IntVector C(I,J,K);

    result.insert( make_pair( C, v )); 

  }

  gzclose( file ); 
  return result; 
}

void 
BoundaryCondition::velocityOutletPressureBC__NEW( const Patch* patch, 
                                                  int  matl_index, 
                                                  SFCXVariable<double>& uvel, 
                                                  SFCYVariable<double>& vvel, 
                                                  SFCZVariable<double>& wvel, 
                                                  constSFCXVariable<double>& old_uvel, 
                                                  constSFCYVariable<double>& old_vvel, 
                                                  constSFCZVariable<double>& old_wvel ) 
{
  vector<Patch::FaceType>::const_iterator bf_iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  // This business is to get the outlet/pressure bcs to behave like the 
  // original arches outlet/pressure bc
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  for ( BCInfoMap::iterator bc_iter = d_bc_information.begin(); 
        bc_iter != d_bc_information.end(); bc_iter++){

    if ( bc_iter->second.type == OUTLET || bc_iter->second.type == PRESSURE ) { 

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ){

        //get the face
        Patch::FaceType face = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face); 

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++){

          double bc_value = 0;
          //int norm = getNormal( face ); 
          
          string bc_kind = "NotSet";
          Iterator bound_ptr;

          //ALWAYS a double so no need to check for vectors
          bool foundIterator = 
            getIteratorBCValueBCKind( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind); 

          if ( foundIterator ) {

            bound_ptr.reset();
            double negsmall = -1.0E-10;
            double possmall =  1.0E-10;
            double zero     = 0.0E0; 
            double sign        = 1.0;

            if ( bc_iter->second.type == PRESSURE ) { 
              sign = -1.0; 
            }

            switch (face) {

              case Patch::xminus:

                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                  IntVector c   = *bound_ptr; 
                  IntVector cp  = *bound_ptr - insideCellDir; 
                  IntVector cpp = cp - insideCellDir; 

                  if ( (zminus && (c.z() == idxLo.z())) ||
                       (zplus  && (c.z() == idxHi.z())) ||
                       (yminus && (c.y() == idxLo.y())) ||
                       (yplus  && (c.y() == idxHi.y())) ){ 

                    uvel[cp] = zero; 

                  } else {

                    if ( sign * old_uvel[cp] < negsmall ) { 
                      uvel[cp] = uvel[cpp]; 
                    } else {
                      uvel[cp] = zero; 
                    }
                    uvel[c] = uvel[cp]; 
                  }
                }
                break; 

              case Patch::xplus:

                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                  IntVector c   = *bound_ptr; 
                  IntVector cp  = *bound_ptr - insideCellDir; 
                  IntVector cm  = c + insideCellDir; 

                  if ( (zminus && (c.z() == idxLo.z())) ||
                       (zplus  && (c.z() == idxHi.z())) ||
                       (yminus && (c.y() == idxLo.y())) ||
                       (yplus  && (c.y() == idxHi.y())) ){ 

                    uvel[c] = zero; 

                  } else {

                    if ( sign * old_uvel[c] > possmall ) { 
                      uvel[c] = uvel[cp]; 
                    } else {
                      uvel[c] = zero; 
                    }
                    uvel[cm] = uvel[c]; 
                  }

                }
                break; 

              case Patch::yminus:

                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                  IntVector c   = *bound_ptr; 
                  IntVector cp  = *bound_ptr - insideCellDir; 
                  IntVector cpp = cp - insideCellDir; 

                  if ( (zminus && (c.z() == idxLo.z())) ||
                       (zplus  && (c.z() == idxHi.z())) ||
                       (xminus && (c.x() == idxLo.x())) ||
                       (xplus  && (c.x() == idxHi.x())) ){ 

                    vvel[cp] = zero; 

                  } else {

                    if ( sign * old_vvel[cp] < negsmall ) { 
                      vvel[cp] = vvel[cpp]; 
                    } else {
                      vvel[cp] = zero; 
                    }
                    vvel[c] = vvel[cp]; 
                  }
                }
                break; 

              case Patch::yplus: 

                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                  IntVector c   = *bound_ptr; 
                  IntVector cp  = *bound_ptr - insideCellDir; 
                  IntVector cm  = c + insideCellDir; 

                  if ( (zminus && (c.z() == idxLo.z())) ||
                       (zplus  && (c.z() == idxHi.z())) ||
                       (xminus && (c.x() == idxLo.x())) ||
                       (xplus  && (c.x() == idxHi.x())) ){ 

                    vvel[c] = zero; 

                  } else {

                    if ( sign * old_vvel[c] > possmall ) { 
                      vvel[c] = vvel[cp]; 
                    } else {
                      vvel[c] = zero; 
                    }
                    vvel[cm] = vvel[c]; 
                  }
                }
                break; 

              case Patch::zminus: 

                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                  IntVector c   = *bound_ptr; 
                  IntVector cp  = *bound_ptr - insideCellDir; 
                  IntVector cpp = cp - insideCellDir; 

                  if ( (xminus && (c.x() == idxLo.x())) ||
                       (xplus  && (c.x() == idxHi.x())) ||
                       (yminus && (c.y() == idxLo.y())) ||
                       (yplus  && (c.y() == idxHi.y())) ){ 

                    wvel[cp] = zero; 

                  } else {

                    if ( sign * old_wvel[cp] < negsmall ) { 
                      wvel[cp] = wvel[cpp]; 
                    } else {
                      wvel[cp] = zero; 
                    }
                    wvel[c] = wvel[cp]; 
                  }
                }
                break; 

              case Patch::zplus:

                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){

                  IntVector c   = *bound_ptr; 
                  IntVector cp  = *bound_ptr - insideCellDir; 
                  IntVector cm  = c + insideCellDir; 

                  if ( (xminus && (c.x() == idxLo.x())) ||
                       (xplus  && (c.x() == idxHi.x())) ||
                       (yminus && (c.y() == idxLo.y())) ||
                       (yplus  && (c.y() == idxHi.y())) ){ 

                    wvel[c] = zero; 

                  } else {

                    if ( sign * old_wvel[c] > possmall ) { 
                      wvel[c] = wvel[cp]; 
                    } else {
                      wvel[c] = zero; 
                    }
                    wvel[cm] = wvel[c]; 
                  }
                }
                break; 

              default: 
                throw InvalidValue("Error: Face type not recognized: " + face, __FILE__, __LINE__); 
                break; 
            }
          }
        }
      }
    }
  }
}

// Bandaid until the face-centered scalar eqn is implemented 
void 
BoundaryCondition::sched_setPrefill__NEW( SchedulerP& sched, const PatchSet* patches, const MaterialSet* matls )
{ 
  Task* tsk = scinew Task("BoundaryCondition::setPrefill__NEW", this, &BoundaryCondition::setPrefill__NEW); 

  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);

  tsk->requires( Task::NewDW, d_lab->d_cellTypeLabel, Ghost::None, 0 ); 

  sched->addTask(tsk, patches, matls);

} 

void BoundaryCondition::setPrefill__NEW( const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse* new_dw )
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matl_index = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Box patchInteriorBox = patch->getBox();

    SFCXVariable<double> uVelocity; 
    SFCYVariable<double> vVelocity; 
    SFCZVariable<double> wVelocity; 
    constCCVariable<int> cellType; 

    new_dw->getModifiable( uVelocity, d_lab->d_uVelocitySPBCLabel, matl_index, patch ); 
    new_dw->getModifiable( vVelocity, d_lab->d_vVelocitySPBCLabel, matl_index, patch ); 
    new_dw->getModifiable( wVelocity, d_lab->d_wVelocitySPBCLabel, matl_index, patch ); 
    new_dw->get( cellType, d_lab->d_cellTypeLabel, matl_index, patch, Ghost::None, 0 ); 

    for ( std::map<std::string, std::vector<GeometryPieceP> >::iterator iter = d_prefill_map.begin(); 
          iter != d_prefill_map.end(); iter++ ) { 

      BCInfoMap::iterator ifound_bc = d_bc_information.end(); 
      BCInfoMap::iterator ibc_info; 

      // search for the boundary condition to match this prefill instruction 
      for ( ibc_info = d_bc_information.begin(); ibc_info != d_bc_information.end(); ibc_info++ ){ 

        if ( ibc_info->second.name == iter->first ){ 

          ifound_bc = ibc_info; 

        } 
      } 

      if ( ifound_bc == d_bc_information.end() ) { 

        throw InvalidValue("Error: Unable to match prefill bc name with actual boundary condition. ", __FILE__, __LINE__); 

      } else { 

        // Prefill matched with boundary condition.  Now set all cells inside the 
        // geometry piece with the velocity as specified by the matching boundary. 
        int nofGeomPieces = (int)iter->second.size(); 

        for (int i = 0; i < nofGeomPieces; i++) {

          GeometryPieceP piece = iter->second[i]; 
          Box geomBox = piece->getBoundingBox(); 
          Box box = geomBox.intersect( patchInteriorBox ); 
          Vector velocity = ifound_bc->second.velocity; 

          if ( !(box.degenerate()) ){ 

            for ( CellIterator icell = patch->getCellCenterIterator(box); !icell.done(); icell++ ) { 

              Point p = patch->cellPosition( *icell ); 
              if ( piece->inside( p ) && cellType[*icell] == -1 ) { 

                uVelocity[*icell] = velocity[0];
                vVelocity[*icell] = velocity[1]; 
                wVelocity[*icell] = velocity[2]; 

              } 
            } 
          } 
        }
      } 
    } 
  }
}
void 
BoundaryCondition::setHattedIntrusionVelocity( const Patch* p, 
                                               SFCXVariable<double>& u, 
                                               SFCYVariable<double>& v, 
                                               SFCZVariable<double>& w, 
                                               constCCVariable<double>& density ) 
{ 
  if ( _using_new_intrusion ) { 
    _intrusionBC->setHattedVelocity( p, u, v, w, density );
  } 
} 
void
BoundaryCondition::sched_setupNewIntrusionCellType( SchedulerP& sched, const PatchSet* patches, const MaterialSet* matls, const bool doing_restart )
{
  if ( _using_new_intrusion ) { 
    _intrusionBC->sched_setCellType( sched, patches, matls, doing_restart ); 
  }
}


void
BoundaryCondition::sched_setupNewIntrusions( SchedulerP& sched, const PatchSet* patches, const MaterialSet* matls )
{

  if ( _using_new_intrusion ) { 
    _intrusionBC->sched_computeBCArea( sched, patches, matls ); 
    _intrusionBC->sched_computeProperties( sched, patches, matls ); 
    _intrusionBC->sched_setIntrusionVelocities( sched, patches, matls );  
  }

}

void 
BoundaryCondition::sched_setIntrusionDensity( SchedulerP& sched, const PatchSet* patches, const MaterialSet* matls )
{ 
  Task* tsk = scinew Task( "BoundaryCondition::setIntrusionDensity", 
                           this, &BoundaryCondition::setIntrusionDensity); 
  tsk->modifies( d_lab->d_densityCPLabel ); 
  sched->addTask( tsk, patches, matls ); 

} 

void 
BoundaryCondition::setIntrusionDensity( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    if ( _using_new_intrusion ){ 
      const Patch* patch = patches->get(p);
      int archIndex = 0; // only one arches material
      int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

      CCVariable<double> density; 
      new_dw->getModifiable( density, d_lab->d_densityCPLabel, indx, patch ); 

      _intrusionBC->setDensity( patch, density ); 
    }
  }
}

