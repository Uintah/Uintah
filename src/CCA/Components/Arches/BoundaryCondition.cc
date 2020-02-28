/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/Arches/Filter.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryFunctors.h>

#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>

#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/MaterialManager.h>
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

#include <CCA/Components/Arches/fortran/mmbcvelocity_fort.h>
#include <CCA/Components/Arches/fortran/mm_computevel_fort.h>
#include <CCA/Components/Arches/fortran/mm_explicit_vel_fort.h>

//****************************************************************************
// Constructor for BoundaryCondition
//****************************************************************************
BoundaryCondition::BoundaryCondition(const ArchesLabel* label,
                                     const MPMArchesLabel* MAlb,
                                     PhysicalConstants* phys_const,
                                     Properties* props,
                                     TableLookup* table_lookup ) :
  d_lab(label), d_MAlab(MAlb),
  d_physicalConsts(phys_const),
  d_props(props),
  d_table_lookup(table_lookup)
{

  MM_CUTOFF_VOID_FRAC = 0.5;
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
  index_map[2][0] = 2;
  index_map[2][1] = 0;
  index_map[2][2] = 1;

  d_check_inlet_obstructions = false;

  d_radiation_temperature_label = VarLabel::create("radiation_temperature", CCVariable<double>::getTypeDescription());

  d_temperature_label = VarLabel::find("temperature");

}


//****************************************************************************
// Destructor
//****************************************************************************
BoundaryCondition::~BoundaryCondition()
{

  delete d_newBC;

  for ( auto i = d_bc_information.begin(); i != d_bc_information.end(); i++ ){
    for ( auto j = (i->second).begin(); j != (i->second).end(); j++ ){

      if ( j->second.type == TURBULENT_INLET ){
        delete j->second.TurbIn;
      }

    }
  }

  if (_using_new_intrusion) {
    for ( auto i = _intrusionBC.begin(); i != _intrusionBC.end(); i++ ){
      delete i->second;
    }
  }

  VarLabel::destroy(d_radiation_temperature_label);

}

//****************************************************************************
// Problem Setup
//****************************************************************************
void
BoundaryCondition::problemSetup( const ProblemSpecP& params,
                                 GridP& grid )
{

  ProblemSpecP db_params = params;
  ProblemSpecP db = params->findBlock("BoundaryConditions");

  m_arches_spec = db_params;

  d_newBC = scinew BoundaryCondition_new( d_lab->d_materialManager->getMaterial( "Arches", 0)->getDWIndex() );

  if ( db != nullptr ) {

    if ( db->findBlock("check_for_inlet_obstructions") ) {
      d_check_inlet_obstructions = true;
    }

    //setupBCs( db_params );

    if ( db->findBlock("intrusions") ) {

      for ( int i = 0; i < grid->numLevels(); i++ ){
        _intrusionBC.insert(std::make_pair(i, scinew IntrusionBC( d_lab, d_MAlab, d_props,
                                                                  d_table_lookup,
                                                                  BoundaryCondition::INTRUSION )));
        ProblemSpecP db_new_intrusion = db->findBlock("intrusions");
        if (i == (grid->numLevels() - 1)){  //  Only create intrusions on the finest level.
                                            //  In the future, we may want to create intrusions on all levels,
                                            //  if so, we will need to resolve problems with redundant intrusion
                                            //  names in the infrastructure.
          _intrusionBC[i]->problemSetup( db_new_intrusion, i );
        }

      }
      _using_new_intrusion = true;

    }

    d_no_corner_recirc = false;
    if ( db->findBlock("suppress_corner_recirculation" )) {
      d_no_corner_recirc = true;
    }

    d_ignore_invalid_celltype = false;
    if ( db->findBlock("ignore_invalid_celltype")) {
      d_ignore_invalid_celltype = true;
    }

    // if multimaterial then add an id for multimaterial wall
    // trying to reduce all interior walls to type:INTRUSION
    d_mmWallID = INTRUSION;

    //look for velocity file input information...
    ProblemSpecP db_root = db_params->getRootNode();
    ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");

    if ( db_bc ) {

      for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ) {

        std::string face_name = "NA";
        db_face->getAttribute("name", face_name );

        for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ) {

          std::string name;
          std::string type;
          db_BCType->getAttribute("label", name);
          db_BCType->getAttribute("var", type);

          if ( type == "VelocityFileInput" ) {

            if ( std::find( d_all_v_inlet_names.begin(), d_all_v_inlet_names.end(), name ) != d_all_v_inlet_names.end() )
              throw ProblemSetupException("Error: You have two VelocityFileInput specs with the same label: "+name, __FILE__, __LINE__);
            else
              d_all_v_inlet_names.push_back(name);

            if ( face_name == "NA" ) {
              //require that the face be named:
              throw ProblemSetupException("Error: For BCType VelocityFileInput, the <Face> must have a name attribute.", __FILE__, __LINE__);
            }

            std::string default_type;
            Vector default_value(9999,9999,9999);
            db_BCType->findBlock("default")->getAttribute("type",default_type);
            db_BCType->findBlock("default")->getAttribute("velvalue",default_value);

            if ( !db_BCType->findBlock("default")->findAttribute("velvalue") ) {
              throw ProblemSetupException("Error: The default for velocity handoff files must be specified using the \'velvalue\' attribute.", __FILE__, __LINE__);
            }

            std::string file_name;
            db_BCType->require("value", file_name);
            Vector rel_xyz;
            db_BCType->require("relative_xyz", rel_xyz);

            BoundaryCondition::FFInfo u_info;
            readInputFile( file_name, u_info, 0 );
            u_info.relative_xyz = rel_xyz;
            u_info.default_type = default_type;
            u_info.default_value = default_value[0];

            if ( default_type == "Neumann" && default_value[0] != 0.0 ) {
              throw ProblemSetupException("Error: Sorry.  I currently cannot support non-zero Neumann default for handoff velocity at this time.", __FILE__, __LINE__);
            }

            FaceToInput::iterator check_iter = _u_input.find(face_name);

            if ( check_iter == _u_input.end() ) {
              _u_input.insert(make_pair(face_name,u_info));
            } else {
              throw ProblemSetupException("Error: Two <Face> speficiations in the input file have the same name attribute. This is not allowed.", __FILE__, __LINE__);
            }

            BoundaryCondition::FFInfo v_info;
            readInputFile( file_name, v_info, 1 );
            v_info.relative_xyz = rel_xyz;
            v_info.default_type = default_type;
            v_info.default_value = default_value[1];

            if ( default_type == "Neumann" && default_value[1] != 0.0 ) {
              throw ProblemSetupException("Error: Sorry.  I currently cannot support non-zero Neumann default for handoff velocity at this time.", __FILE__, __LINE__);
            }

            check_iter = _v_input.find(face_name);

            if ( check_iter == _v_input.end() ) {
              _v_input.insert(make_pair(face_name,v_info));
            } else {
              throw ProblemSetupException("Error: Two <Face> speficiations in the input file have the same name attribute. This is not allowed.", __FILE__, __LINE__);
            }

            BoundaryCondition::FFInfo w_info;
            readInputFile( file_name, w_info, 2 );
            w_info.relative_xyz = rel_xyz;
            w_info.default_type = default_type;
            w_info.default_value = default_value[2];

            if ( default_type == "Neumann" && default_value[2] != 0.0 ) {
              throw ProblemSetupException("Error: Sorry.  I currently cannot support non-zero Neumann default for handoff velocity at this time.", __FILE__, __LINE__);
            }

            check_iter = _w_input.find(face_name);

            if ( check_iter == _w_input.end() ) {
              _w_input.insert(make_pair(face_name,w_info));
            } else {
              throw ProblemSetupException("Error: Two <Face> speficiations in the input file have the same name attribute. This is not allowed.", __FILE__, __LINE__);
            }
          }
        }
      }
    }
  } else {
    //band-aid
    //throw ProblemSetupException("Error: Please insert a <BoundaryConditions/> in your <ARCHES> node of the UPS.", __FILE__, __LINE__);
    m_has_boundaries = false;
  }
}

void
BoundaryCondition::set_bc_information(const LevelP& level){
  if ( m_has_boundaries ){
    setupBCs(m_arches_spec, level);
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
  if (whichWay == "copyInto") {

    new_dw->allocateTemporary(AP,patch);
    new_dw->allocateTemporary(AE,patch);
    new_dw->allocateTemporary(AW,patch);
    new_dw->allocateTemporary(AN,patch);
    new_dw->allocateTemporary(AS,patch);
    new_dw->allocateTemporary(AT,patch);
    new_dw->allocateTemporary(AB,patch);

    for(; !iter.done(); iter++) {
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
    for(; !iter.done(); iter++) {
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

//______________________________________________________________________
//  Set the boundary conditions on the pressure stencil.
// This will change when we move to the UCF based boundary conditions

void
BoundaryCondition::pressureBC(const Patch* patch,
                              const int matl_index,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars)
{

  CCVariable<Stencil7>& A = vars->pressCoeff;

  std::vector<BC_TYPE> add_types;
  add_types.push_back( OUTLET );
  add_types.push_back( PRESSURE );
  add_types.push_back( NEUTRAL_OUTLET );
  int sign = -1;

  zeroStencilDirection( patch, matl_index, sign, A, add_types );

  std::vector<BC_TYPE> sub_types;
  sub_types.push_back( WALL );
  sub_types.push_back( INTRUSION );
  sub_types.push_back( MASSFLOW_INLET );
  sub_types.push_back( PARTMASSFLOW_INLET );
  sub_types.push_back( VELOCITY_INLET );
  sub_types.push_back( VELOCITY_FILE );
  sub_types.push_back( MASSFLOW_FILE );
  sub_types.push_back( SWIRL );
  sub_types.push_back( TURBULENT_INLET );
  sub_types.push_back( STABL );
  sign = 1;

  zeroStencilDirection( patch, matl_index, sign, A, sub_types );

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

  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {

        IntVector currCell = IntVector(colX, colY, colZ);

        if (cellType[currCell]==d_mmWallID) {

          if (d_energyEx) {
            if (d_fixTemp) {
              temperature[currCell] = 298.0;
            }else{
              temperature[currCell] = solidTemp[currCell];
            }
          }else{
            temperature[currCell] = 298.0;
          } //d_energyEx
        } // wall
      } // x
    } // y
  } // z
}

void
BoundaryCondition::sched_setIntrusionTemperature( SchedulerP& sched,
                                                  const LevelP& level,
                                                  const MaterialSet* matls)
{
  if ( _using_new_intrusion ) {
    const int ilvl = level->getID();
    _intrusionBC[ilvl]->sched_setIntrusionT( sched, level, matls );
  }
}

void
BoundaryCondition::sched_computeAlphaG( SchedulerP& sched,
                                        const LevelP& level,
                                        const MaterialSet* matls,
                                        const bool carry_forward )
{
  if ( _using_new_intrusion ) {
    const int ilvl = level->getID();
    //if carry_forward = true, we are simply moving the computed value forward in time. 
    _intrusionBC[ilvl]->sched_setAlphaG( sched, level, matls, carry_forward );
  }
}

//______________________________________________________________________
// compute multimaterial wall bc
void
BoundaryCondition::wallVelocityBC(const Patch* patch,
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

  int boundary_type = BoundaryCondition::INTRUSION;

  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->uVelocityCoeff[Arches::AE],
                    vars->uVelocityCoeff[Arches::AW],
                    vars->uVelocityCoeff[Arches::AN],
                    vars->uVelocityCoeff[Arches::AS],
                    vars->uVelocityCoeff[Arches::AT],
                    vars->uVelocityCoeff[Arches::AB],
                    vars->uVelNonlinearSrc, vars->uVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);

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
                    constvars->cellType, boundary_type, ioff, joff, koff);

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
                    constvars->cellType, boundary_type, ioff, joff, koff);

  //__________________________________
  //    X dir
  ioff = 1;
  joff = 0;
  koff = 0;

  idxLoU = patch->getSFCXFORTLowIndex__Old();
  idxHiU = patch->getSFCXFORTHighIndex__Old();
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->uVelocityConvectCoeff[Arches::AE],
                    vars->uVelocityConvectCoeff[Arches::AW],
                    vars->uVelocityConvectCoeff[Arches::AN],
                    vars->uVelocityConvectCoeff[Arches::AS],
                    vars->uVelocityConvectCoeff[Arches::AT],
                    vars->uVelocityConvectCoeff[Arches::AB],
                    vars->uVelNonlinearSrc, vars->uVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);

  //__________________________________
  //    Y dir
  idxLoU = patch->getSFCYFORTLowIndex__Old();
  idxHiU = patch->getSFCYFORTHighIndex__Old();

  ioff = 0;
  joff = 1;
  koff = 0;
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->vVelocityConvectCoeff[Arches::AN],
                    vars->vVelocityConvectCoeff[Arches::AS],
                    vars->vVelocityConvectCoeff[Arches::AT],
                    vars->vVelocityConvectCoeff[Arches::AB],
                    vars->vVelocityConvectCoeff[Arches::AE],
                    vars->vVelocityConvectCoeff[Arches::AW],
                    vars->vVelNonlinearSrc, vars->vVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);

  //__________________________________
  //    Z dir
  idxLoU = patch->getSFCZFORTLowIndex__Old();
  idxHiU = patch->getSFCZFORTHighIndex__Old();

  ioff = 0;
  joff = 0;
  koff = 1;
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->wVelocityConvectCoeff[Arches::AT],
                    vars->wVelocityConvectCoeff[Arches::AB],
                    vars->wVelocityConvectCoeff[Arches::AE],
                    vars->wVelocityConvectCoeff[Arches::AW],
                    vars->wVelocityConvectCoeff[Arches::AN],
                    vars->wVelocityConvectCoeff[Arches::AS],
                    vars->wVelNonlinearSrc, vars->wVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);

  boundary_type = BoundaryCondition::WALL;

  //__________________________________
  //    X dir
  idxLoU = patch->getSFCXFORTLowIndex__Old();
  idxHiU = patch->getSFCXFORTHighIndex__Old();
  ioff = 1;
  joff = 0;
  koff = 0;

  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->uVelocityCoeff[Arches::AE],
                    vars->uVelocityCoeff[Arches::AW],
                    vars->uVelocityCoeff[Arches::AN],
                    vars->uVelocityCoeff[Arches::AS],
                    vars->uVelocityCoeff[Arches::AT],
                    vars->uVelocityCoeff[Arches::AB],
                    vars->uVelNonlinearSrc, vars->uVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);

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
                    constvars->cellType, boundary_type, ioff, joff, koff);

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
                    constvars->cellType, boundary_type, ioff, joff, koff);

  //__________________________________
  //    X dir
  ioff = 1;
  joff = 0;
  koff = 0;

  idxLoU = patch->getSFCXFORTLowIndex__Old();
  idxHiU = patch->getSFCXFORTHighIndex__Old();
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->uVelocityConvectCoeff[Arches::AE],
                    vars->uVelocityConvectCoeff[Arches::AW],
                    vars->uVelocityConvectCoeff[Arches::AN],
                    vars->uVelocityConvectCoeff[Arches::AS],
                    vars->uVelocityConvectCoeff[Arches::AT],
                    vars->uVelocityConvectCoeff[Arches::AB],
                    vars->uVelNonlinearSrc, vars->uVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);

  //__________________________________
  //    Y dir
  idxLoU = patch->getSFCYFORTLowIndex__Old();
  idxHiU = patch->getSFCYFORTHighIndex__Old();

  ioff = 0;
  joff = 1;
  koff = 0;
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->vVelocityConvectCoeff[Arches::AN],
                    vars->vVelocityConvectCoeff[Arches::AS],
                    vars->vVelocityConvectCoeff[Arches::AT],
                    vars->vVelocityConvectCoeff[Arches::AB],
                    vars->vVelocityConvectCoeff[Arches::AE],
                    vars->vVelocityConvectCoeff[Arches::AW],
                    vars->vVelNonlinearSrc, vars->vVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);

  //__________________________________
  //    Z dir
  idxLoU = patch->getSFCZFORTLowIndex__Old();
  idxHiU = patch->getSFCZFORTHighIndex__Old();

  ioff = 0;
  joff = 0;
  koff = 1;
  fort_mmbcvelocity(idxLoU, idxHiU,
                    vars->wVelocityConvectCoeff[Arches::AT],
                    vars->wVelocityConvectCoeff[Arches::AB],
                    vars->wVelocityConvectCoeff[Arches::AE],
                    vars->wVelocityConvectCoeff[Arches::AW],
                    vars->wVelocityConvectCoeff[Arches::AN],
                    vars->wVelocityConvectCoeff[Arches::AS],
                    vars->wVelNonlinearSrc, vars->wVelLinearSrc,
                    constvars->cellType, boundary_type, ioff, joff, koff);
}

//______________________________________________________________________
//
void
BoundaryCondition::mmpressureBC(DataWarehouse* new_dw,
                                const Patch* patch,
                                ArchesVariables* vars,
                                ArchesConstVariables* constvars)
{

  for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ) {

    IntVector c = *iter;

    if ( constvars->cellType[c] == INTRUSION ) {

      const double constant = 1.0;
      const double value    = 0.0;

      fix_value( vars->pressCoeff, vars->pressNonlinearSrc,
                 vars->pressLinearSrc, value, constant, c );

    } else {

      if ( constvars->cellType[ c + IntVector(1,0,0) ] == INTRUSION ) {
        vars->pressCoeff[c].e = 0.0;
      }
      if ( constvars->cellType[ c - IntVector(1,0,0) ] == INTRUSION ) {
        vars->pressCoeff[c].w = 0.0;
      }
      if ( constvars->cellType[ c + IntVector(0,1,0) ] == INTRUSION ) {
        vars->pressCoeff[c].n = 0.0;
      }
      if ( constvars->cellType[ c - IntVector(0,1,0) ] == INTRUSION ) {
        vars->pressCoeff[c].s = 0.0;
      }
      if ( constvars->cellType[ c + IntVector(0,0,1) ] == INTRUSION ) {
        vars->pressCoeff[c].t = 0.0;
      }
      if ( constvars->cellType[ c - IntVector(0,0,1) ] == INTRUSION ) {
        vars->pressCoeff[c].b = 0.0;
      }

    }
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
// Set the inlet rho hat velocity BC
//****************************************************************************
void
BoundaryCondition::velRhoHatInletBC(const Patch* patch,
                                    ArchesVariables* vars,
                                    ArchesConstVariables* constvars,
                                    const int matl_index,
                                    const int timeStep,
                                    const double simTime,
                                    double time_shift)
{
  //double simTime = d_lab->d_materialManager->getElapsedSimTime();
  //double current_time = simTime + time_shift;
  // Get the low and high index for the patch and the variables
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  // stores cell type info for the patch with the ghost cell type

  vector<Patch::FaceType>::const_iterator bf_iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  const Level* level = patch->getLevel();
  const int ilvl = level->getID();

  for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
        bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

    for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ) {

      //get the face
      Patch::FaceType face = *bf_iter;
      IntVector insideCellDir = patch->faceDirection(face);

      //get the number of children
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

      for (int child = 0; child < numChildren; child++) {

        double bc_value = 0;
        Vector bc_v_value(0,0,0);
        std::string bc_s_value = "NA";
        string bc_kind = "NotSet";
        Iterator bound_ptr;
        bool foundIterator = false;

        if ( bc_iter->second.type == VELOCITY_INLET ||
             bc_iter->second.type == TURBULENT_INLET ||
             bc_iter->second.type == STABL ) {
          foundIterator =
                  getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind);
        } else if ( bc_iter->second.type == VELOCITY_FILE ) {
          foundIterator =
                  getIteratorBCValue<std::string>( patch, face, child, bc_iter->second.name, matl_index, bc_s_value, bound_ptr);
        } else {
          foundIterator =
                  getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);
        }

        if ( foundIterator ) {

          bound_ptr.reset();

          if ( bc_iter->second.type == VELOCITY_INLET
            || bc_iter->second.type == MASSFLOW_INLET ) {

            setVel( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, constvars->new_density, bound_ptr, bc_iter->second.velocity );

          } else if ( bc_iter->second.type == STABL ) {

            setStABL( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, &bc_iter->second, bound_ptr );

          } else if (bc_iter->second.type == TURBULENT_INLET) {

            setTurbInlet( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, constvars->new_density, bound_ptr, bc_iter->second.TurbIn,
                          timeStep, simTime );

          } else if ( bc_iter->second.type == SWIRL ) {

            if ( face == Patch::xminus || face == Patch::xplus ) {

              setSwirl( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat,
                        constvars->new_density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no, bc_iter->second.swirl_cent );

            } else if ( face == Patch::yminus || face == Patch::yplus ) {

              setSwirl( patch, face, vars->vVelRhoHat, vars->wVelRhoHat, vars->uVelRhoHat,
                        constvars->new_density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no, bc_iter->second.swirl_cent  );

            } else if ( face == Patch::zminus || face == Patch::zplus ) {

              setSwirl( patch, face, vars->wVelRhoHat, vars->uVelRhoHat, vars->vVelRhoHat,
                        constvars->new_density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no, bc_iter->second.swirl_cent  );

            }

          } else if ( bc_iter->second.type == VELOCITY_FILE ) {

            setVelFromExtraValue( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, constvars->new_density, bound_ptr, bc_iter->second.velocity );

          } else if ( bc_iter->second.type == WALL ) {

            setVel( patch, face, vars->uVelRhoHat, vars->vVelRhoHat, vars->wVelRhoHat, constvars->new_density, bound_ptr, bc_iter->second.velocity );

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

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  int sign = 0;

  if (xminus) {
    int colX = idxLo.x();
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        if ( (cellType[xminusCell] == OUTLET ) ||
             (cellType[xminusCell] == PRESSURE ) ||
             (cellType[xminusCell] == NEUTRAL_OUTLET )) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
            uvel[currCell] = 0.0;
          else {
            if (cellType[xminusCell] == OUTLET || cellType[xminusCell] == NEUTRAL_OUTLET )
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xplusCell(colX+1, colY, colZ);
        IntVector xplusplusCell(colX+2, colY, colZ);
        if ((cellType[xplusCell] == OUTLET )||
            (cellType[xplusCell] == PRESSURE ) ||
            (cellType[xplusCell] == NEUTRAL_OUTLET )) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y())))
            uvel[xplusCell] = 0.0;
          else {
            if (cellType[xplusCell] == OUTLET || cellType[xplusCell] == NEUTRAL_OUTLET )
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        if ((cellType[yminusCell] == OUTLET )||
            (cellType[yminusCell] == PRESSURE ) ||
            (cellType[yminusCell] == NEUTRAL_OUTLET )) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vvel[currCell] = 0.0;
          else {
            if (cellType[yminusCell] == OUTLET || cellType[yminusCell] == NEUTRAL_OUTLET )
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yplusCell(colX, colY+1, colZ);
        IntVector yplusplusCell(colX, colY+2, colZ);
        if ((cellType[yplusCell] == OUTLET )||
            (cellType[yplusCell] == PRESSURE ) ||
            (cellType[yplusCell] == NEUTRAL_OUTLET )) {
          if ((zminus && (colZ == idxLo.z()))||(zplus && (colZ == idxHi.z()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            vvel[yplusCell] = 0.0;
          else {
            if (cellType[yplusCell] == OUTLET || cellType[yplusCell] == NEUTRAL_OUTLET )
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
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);
        IntVector zplusCell(colX, colY, colZ+1);
        if ((cellType[zminusCell] == OUTLET) ||
            (cellType[zminusCell] == PRESSURE ) ||
            (cellType[zminusCell] == NEUTRAL_OUTLET )) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            wvel[currCell] = 0.0;
          else {
            if (cellType[zminusCell] == OUTLET || cellType[zminusCell] == NEUTRAL_OUTLET )
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
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zplusCell(colX, colY, colZ+1);
        IntVector zplusplusCell(colX, colY, colZ+2);
        if ((cellType[zplusCell] == OUTLET )||
            (cellType[zplusCell] == PRESSURE ) ||
            (cellType[zplusCell] == NEUTRAL_OUTLET )) {
          if ((yminus && (colY == idxLo.y()))||(yplus && (colY == idxHi.y()))
              ||(xminus && (colX == idxLo.x()))||(xplus && (colX == idxHi.x())))
            wvel[zplusCell] = 0.0;
          else {
            if (cellType[zplusCell] == OUTLET || cellType[zplusCell] == NEUTRAL_OUTLET )
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

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  for (int index = 1; index <= 3; ++index) {
    if (xminus) {
      int colX = idxLo.x();
      int maxY = idxHi.y();
      if (yplus) {
        maxY++;
      }
      int maxZ = idxHi.z();
      if (zplus) {
        maxZ++;
      }

      for (int colZ = idxLo.z(); colZ <= maxZ; colZ++) {
        for (int colY = idxLo.y(); colY <= maxY; colY++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector xminusyminusCell(colX-1, colY-1, colZ);
          IntVector xminuszminusCell(colX-1, colY, colZ-1);

          if ((constvars->cellType[xminusCell] == PRESSURE )||
              (constvars->cellType[xminusCell] == OUTLET )  ||
              (constvars->cellType[xminusCell] == NEUTRAL_OUTLET )) {

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
          } else {

            if ( (constvars->cellType[xminusyminusCell] == PRESSURE ) ||
                 (constvars->cellType[xminusyminusCell] == OUTLET ) ||
                 (constvars->cellType[xminusyminusCell] == NEUTRAL_OUTLET )) {

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
            if ( (constvars->cellType[xminuszminusCell] == PRESSURE ) ||
                 (constvars->cellType[xminuszminusCell] == OUTLET ) ||
                 (constvars->cellType[xminuszminusCell] == NEUTRAL_OUTLET )) {
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
      for (int colZ = idxLo.z(); colZ <= maxZ; colZ++) {
        for (int colY = idxLo.y(); colY <= maxY; colY++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);
          IntVector xplusyminusCell(colX+1, colY-1, colZ);
          IntVector xpluszminusCell(colX+1, colY, colZ-1);

          if ( (constvars->cellType[xplusCell] == PRESSURE ) ||
               (constvars->cellType[xplusCell] == OUTLET ) ||
               (constvars->cellType[xplusCell] == NEUTRAL_OUTLET )) {

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
          } else {

            if ( (constvars->cellType[xplusyminusCell] == PRESSURE ) ||
                 (constvars->cellType[xplusyminusCell] == OUTLET ) ||
                 (constvars->cellType[xplusyminusCell] == NEUTRAL_OUTLET )) {

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

            if ( (constvars->cellType[xpluszminusCell] == PRESSURE ) ||
                 (constvars->cellType[xpluszminusCell] == OUTLET ) ||
                 (constvars->cellType[xpluszminusCell] == NEUTRAL_OUTLET )) {

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
      for (int colZ = idxLo.z(); colZ <= maxZ; colZ++) {
        for (int colX = idxLo.x(); colX <= maxX; colX++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector yminusxminusCell(colX-1, colY-1, colZ);
          IntVector yminuszminusCell(colX, colY-1, colZ-1);

          if ( (constvars->cellType[yminusCell] == PRESSURE ) ||
               (constvars->cellType[yminusCell] == OUTLET ) ||
               (constvars->cellType[yminusCell] == NEUTRAL_OUTLET )) {

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
          } else {

            if ( (constvars->cellType[yminusxminusCell] == PRESSURE ) ||
                 (constvars->cellType[yminusxminusCell] == OUTLET ) ||
                 (constvars->cellType[yminusxminusCell] == NEUTRAL_OUTLET )) {

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

            if ( (constvars->cellType[yminuszminusCell] == PRESSURE ) ||
                 (constvars->cellType[yminuszminusCell] == OUTLET ) ||
                 (constvars->cellType[yminuszminusCell] == NEUTRAL_OUTLET )) {

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

      for (int colZ = idxLo.z(); colZ <= maxZ; colZ++) {
        for (int colX = idxLo.x(); colX <= maxX; colX++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yplusCell(colX, colY+1, colZ);
          IntVector yplusxminusCell(colX-1, colY+1, colZ);
          IntVector ypluszminusCell(colX, colY+1, colZ-1);

          if ( (constvars->cellType[yplusCell] == PRESSURE ) ||
               (constvars->cellType[yplusCell] == OUTLET ) ||
               (constvars->cellType[yplusCell] == NEUTRAL_OUTLET )) {

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
          } else {

            if ( (constvars->cellType[yplusxminusCell] == PRESSURE ) ||
                 (constvars->cellType[yplusxminusCell] == OUTLET ) ||
                 (constvars->cellType[yplusxminusCell] == NEUTRAL_OUTLET )) {

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

            if ( (constvars->cellType[ypluszminusCell] == PRESSURE ) ||
                 (constvars->cellType[ypluszminusCell] == OUTLET ) ||
                 (constvars->cellType[ypluszminusCell] == NEUTRAL_OUTLET )) {

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

      for (int colY = idxLo.y(); colY <= maxY; colY++) {
        for (int colX = idxLo.x(); colX <= maxX; colX++) {

          IntVector currCell(colX, colY, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          IntVector zminusxminusCell(colX-1, colY, colZ-1);
          IntVector zminusyminusCell(colX, colY-1, colZ-1);

          if ( (constvars->cellType[zminusCell] == PRESSURE ) ||
               (constvars->cellType[zminusCell] == OUTLET ) ||
               (constvars->cellType[zminusCell] == NEUTRAL_OUTLET )) {

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
            if ( (constvars->cellType[zminusxminusCell] == PRESSURE ) ||
                 (constvars->cellType[zminusxminusCell] == OUTLET ) ||
                 (constvars->cellType[zminusxminusCell] == NEUTRAL_OUTLET )) {

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
            if ( (constvars->cellType[zminusyminusCell] == PRESSURE ) ||
                 (constvars->cellType[zminusyminusCell] == OUTLET ) ||
                 (constvars->cellType[zminusyminusCell] == NEUTRAL_OUTLET )) {
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
      for (int colY = idxLo.y(); colY <= maxY; colY++) {
        for (int colX = idxLo.x(); colX <= maxX; colX++) {
          IntVector currCell(colX, colY, colZ);
          IntVector zplusCell(colX, colY, colZ+1);
          IntVector zplusxminusCell(colX-1, colY, colZ+1);
          IntVector zplusyminusCell(colX, colY-1, colZ+1);
          if ( (constvars->cellType[zplusCell] == PRESSURE ) ||
               (constvars->cellType[zplusCell] == OUTLET ) ||
               (constvars->cellType[zplusCell] == NEUTRAL_OUTLET )) {
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
            if ( (constvars->cellType[zplusxminusCell] == PRESSURE ) ||
                 (constvars->cellType[zplusxminusCell] == OUTLET ) ||
                 (constvars->cellType[zplusxminusCell] == NEUTRAL_OUTLET )) {
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
            if ( (constvars->cellType[zplusyminusCell] == PRESSURE ) ||
                 (constvars->cellType[zplusyminusCell] == OUTLET ) ||
                 (constvars->cellType[zplusyminusCell] == NEUTRAL_OUTLET )) {
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
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
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
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
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
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
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX++) {
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

void BoundaryCondition::sched_setAreaFraction( SchedulerP& sched,
                                               const LevelP& level,
                                               const MaterialSet* matls,
                                               const int timesubstep,
                                               const bool reinitialize )
{

  Task* tsk = scinew Task( "BoundaryCondition::setAreaFraction",this, &BoundaryCondition::setAreaFraction, timesubstep, reinitialize );

  if ( timesubstep == 0 ) {

    tsk->computes( d_lab->d_areaFractionLabel );
    tsk->computes( d_lab->d_volFractionLabel );
    tsk->computes(d_lab->d_areaFractionFXLabel);
    tsk->computes(d_lab->d_areaFractionFYLabel);
    tsk->computes(d_lab->d_areaFractionFZLabel);

  } else {

    //only in cases where geometry moves.
    tsk->modifies( d_lab->d_areaFractionLabel );
    tsk->modifies( d_lab->d_volFractionLabel);
    tsk->modifies(d_lab->d_areaFractionFXLabel);
    tsk->modifies(d_lab->d_areaFractionFYLabel);
    tsk->modifies(d_lab->d_areaFractionFZLabel);

  }

  if ( !reinitialize ) {

    tsk->requires( Task::OldDW, d_lab->d_areaFractionLabel, Ghost::None, 0 );
    tsk->requires( Task::OldDW, d_lab->d_volFractionLabel, Ghost::None, 0 );
    tsk->requires( Task::OldDW, d_lab->d_areaFractionFXLabel, Ghost::None, 0 );
    tsk->requires( Task::OldDW, d_lab->d_areaFractionFYLabel, Ghost::None, 0 );
    tsk->requires( Task::OldDW, d_lab->d_areaFractionFZLabel, Ghost::None, 0 );

  }

  tsk->requires( Task::NewDW, d_lab->d_cellTypeLabel, Ghost::AroundCells, 1 );

  sched->addTask(tsk, level->eachPatch(), matls);

}
void
BoundaryCondition::setAreaFraction( const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const int timesubstep,
                                    const bool reinitialize )
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<Vector>   areaFraction;
    SFCXVariable<double> areaFractionFX;
    SFCYVariable<double> areaFractionFY;
    SFCZVariable<double> areaFractionFZ;
    CCVariable<double>   volFraction;
    constCCVariable<int> cellType;

    new_dw->get( cellType, d_lab->d_cellTypeLabel, indx, patch, Ghost::AroundCells, 1 );

    if ( timesubstep == 0 ) {

      new_dw->allocateAndPut( areaFraction, d_lab->d_areaFractionLabel, indx, patch );
      new_dw->allocateAndPut( volFraction,  d_lab->d_volFractionLabel, indx, patch );
      volFraction.initialize(1.0);

      for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++) {
        areaFraction[*iter] = Vector(1.0,1.0,1.0);
      }

      new_dw->allocateAndPut( areaFractionFX, d_lab->d_areaFractionFXLabel, indx, patch );
      new_dw->allocateAndPut( areaFractionFY, d_lab->d_areaFractionFYLabel, indx, patch );
      new_dw->allocateAndPut( areaFractionFZ, d_lab->d_areaFractionFZLabel, indx, patch );
      areaFractionFX.initialize(1.0);
      areaFractionFY.initialize(1.0);
      areaFractionFZ.initialize(1.0);

    } else {

      new_dw->getModifiable( areaFraction, d_lab->d_areaFractionLabel, indx, patch );
      new_dw->getModifiable( volFraction, d_lab->d_volFractionLabel, indx, patch );

      new_dw->getModifiable( areaFractionFX, d_lab->d_areaFractionFXLabel, indx, patch );
      new_dw->getModifiable( areaFractionFY, d_lab->d_areaFractionFYLabel, indx, patch );
      new_dw->getModifiable( areaFractionFZ, d_lab->d_areaFractionFZLabel, indx, patch );

    }

    if ( !reinitialize ) {

      constCCVariable<double> old_vol_frac;
      constCCVariable<Vector> old_area_frac;
      constCCVariable<double> old_filter_vol;
      old_dw->get( old_area_frac, d_lab->d_areaFractionLabel, indx, patch, Ghost::None, 0 );
      old_dw->get( old_vol_frac,  d_lab->d_volFractionLabel, indx, patch, Ghost::None, 0 );

      areaFraction.copyData( old_area_frac );
      volFraction.copyData( old_vol_frac );

      constSFCXVariable<double> old_Fx;
      constSFCYVariable<double> old_Fy;
      constSFCZVariable<double> old_Fz;
      old_dw->get( old_Fx, d_lab->d_areaFractionFXLabel, indx, patch, Ghost::None, 0 );
      old_dw->get( old_Fy, d_lab->d_areaFractionFYLabel, indx, patch, Ghost::None, 0 );
      old_dw->get( old_Fz, d_lab->d_areaFractionFZLabel, indx, patch, Ghost::None, 0 );
      areaFractionFX.copyData( old_Fx );
      areaFractionFY.copyData( old_Fy );
      areaFractionFZ.copyData( old_Fz );

    } else {

      int flowType = -1;

      vector<int> wall_type;

      if (d_MAlab)
        wall_type.push_back( d_mmWallID );

      wall_type.push_back( WALL );
      wall_type.push_back( MMWALL );
      wall_type.push_back( INTRUSION );

      d_newBC->setAreaFraction( patch, areaFraction, volFraction, cellType, wall_type, flowType );

      for (CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        areaFractionFX[c] = areaFraction[c].x();
        areaFractionFY[c] = areaFraction[c].y();
        areaFractionFZ[c] = areaFraction[c].z();
      }
    }
  }
}

//-------------------------------------------------------------
// New Domain BCs
//
void
BoundaryCondition::setupBCs( ProblemSpecP db, const LevelP& level )
{

  const int ilvl = level->getID();

  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");
  Vector grav;
  unsigned int dir_grav = 999;
  if ( db_root->findBlock("PhysicalConstants") ) {
    db_root->findBlock("PhysicalConstants")->require("gravity",grav);
    if ( grav.x() != 0 ) {
      dir_grav = 0;
    }
    else if ( grav.y() != 0 ) {
      dir_grav = 1;
    }
    else if ( grav.z() != 0 ) {
      dir_grav = 2;
    }
  }
  int bc_type_index = 0;

  // Now actually look for the boundary types
  if ( db_bc ) {
    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ) {

      string which_face;
      int v_index=999;
      if ( db_face->getAttribute("side", which_face)) {
        db_face->getAttribute("side",which_face);
      }
      else if ( db_face->getAttribute( "circle", which_face)) {
        db_face->getAttribute("circle",which_face);
      }
      else if ( db_face->getAttribute( "rectangle", which_face)) {
        db_face->getAttribute("rectangle",which_face);
      }
      else if ( db_face->getAttribute( "rectangulus", which_face)) {
        db_face->getAttribute("rectangulus",which_face);
      }
      else if ( db_face->getAttribute( "annulus", which_face)) {
        db_face->getAttribute("annulus",which_face);
      }
      else if ( db_face->getAttribute( "ellipse", which_face)) {
        db_face->getAttribute("ellipse",which_face);
      }

      std::string faceName;
      db_face->getAttribute("name",faceName);

      //avoid the "or" in case I want to add more logic
      //re: the face normal.
      double v_sign;
      if ( which_face =="x-") {
        v_index = 0;
        v_sign=-1;
      }
      else if ( which_face =="x+") {
        v_index = 0;
        v_sign=1;
      }
      else if ( which_face =="y-") {
        v_index = 1;
        v_sign=-1;
      }
      else if ( which_face =="y+") {
        v_index = 1;
        v_sign=1;
      }
      else if ( which_face =="z-") {
        v_index = 2;
        v_sign=-1;
      }
      else if ( which_face =="z+") {
        v_index = 2;
        v_sign=1;
      }
      else {
        throw InvalidValue("Error: Could not identify the boundary face direction.", __FILE__, __LINE__);
      }

      Patch::FaceType the_face = getFaceTypeFromUPS(db_face);

      int numberOfMomentumBCs = 0;
      for ( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ) {

        std::string name;
        std::string type;
        bool found_bc = false;
        BCInfo my_info;
        db_BCType->getAttribute("label", name);
        db_BCType->getAttribute("var", type);
        my_info.name = name;
        my_info.faceName=faceName;
        my_info.lHasPartMassFlow=false;
        my_info.face = the_face;
        std::stringstream color;
        color << bc_type_index;

        if ( type == "VelocityInlet" ) {

          my_info.type = VELOCITY_INLET;
          db_BCType->require("value", my_info.velocity);
          found_bc = true;

        } else if ( type == "TurbulentInlet" ) {

          my_info.type = TURBULENT_INLET;
          db_BCType->require("inputfile", my_info.filename);
          db_BCType->require("value", my_info.velocity);
          found_bc = true;

          my_info.TurbIn = scinew DigitalFilterInlet( );
          my_info.TurbIn->problemSetup( db_BCType );

        } else if ( type == "MassFlowInlet" ) {

          my_info.type = MASSFLOW_INLET;
          my_info.velocity = Vector(0,0,0);
          my_info.mass_flow_rate = 0.0;
          found_bc = true;

          //allows for non-normal mass flow spec.
          Vector defaultNormalVector= Vector(0,0,0); defaultNormalVector[v_index]=-1*v_sign; // defaul normal vector
          db_BCType->getWithDefault( "massflow_unitVector", my_info.unitVector, defaultNormalVector);

          if (-v_sign*my_info.unitVector[v_index] <= 0.0) {   // defaul normal vector
            throw InvalidValue("massflow_unitVector Error: Normal component cannot be zero or be oriented in direction outward of domain. ", __FILE__, __LINE__);
          }

          double tolly=1e-5;
          double mag_squared=my_info.unitVector.length2();
          if ( mag_squared < 1-tolly || mag_squared > 1.0 +tolly){    // defaul normal vector
             proc0cout << "massflow_unitVector WARNING: unit vector nas non-unity magnitude, normalizing . \n";
             my_info.unitVector.safe_normalize();
          }


          // note that the mass flow rate is in the BCstruct value

          //compute the density:
          typedef std::vector<std::string> StringVec;
          MixingRxnModel* mixingTable = d_table_lookup->get_table();
          StringVec iv_var_names = mixingTable->getAllIndepVars();
          vector<double> iv;

          for ( StringVec::iterator iv_iter = iv_var_names.begin(); iv_iter != iv_var_names.end(); iv_iter++) {

            string curr_iv = *iv_iter;

            for ( ProblemSpecP db_BCType2 = db_face->findBlock("BCType"); db_BCType2 != nullptr; db_BCType2 = db_BCType2->findNextBlock("BCType") ) {

              string curr_var;
              db_BCType2->getAttribute("label",curr_var);

              if ( curr_var == curr_iv ) {
                string type;
                db_BCType2->getAttribute("var",type);
                if ( type != "Dirichlet") {
                  throw InvalidValue("Error: Cannot compute property values for MassFlowInlet because not all IVs are of type Dirichlet: "+curr_var, __FILE__, __LINE__);
                } else {
                  double value;
                  db_BCType2->require("value",value);
                  iv.push_back(value);
                }
              }
            }
          }

          double density = mixingTable->getTableValue(iv,"density");
          my_info.density = density;


        } else if ( type == "VelocityFileInput" ) {

          my_info.type = VELOCITY_FILE;
          db_BCType->require("value", my_info.filename);
          my_info.velocity = Vector(0,0,0);
          found_bc = true;

        } else if ( type == "Swirl" ) {

          my_info.type = SWIRL;
          my_info.velocity = Vector(0,0,0);
          my_info.mass_flow_rate = 0.0;
          db_BCType->require("swirl_no", my_info.swirl_no);

          my_info.swirl_no *= 3./2.;
          // swirl number definition as equation 5.14 from Combustion Aerodynamics
          // J.M. BEER and N.A. CHIGIER 1983 pag 107
          // assuming:
          // constant axial velocity
          // constant density
          // constant tangential velocity
          //
          std::string str_vec; // This block sets the default centroid to the origin unless otherwise specified by swirl_cent
          bool Luse_origin =   db_face->getAttribute("origin", str_vec);
          if( Luse_origin ) {
            std::stringstream ss;
            ss << str_vec;
            Vector origin;
            ss >> origin[0] >> origin[1] >> origin[2];
            db_BCType->getWithDefault("swirl_centroid", my_info.swirl_cent,origin);
          }else{
            db_BCType->require("swirl_centroid", my_info.swirl_cent);
          }

          //compute the density:
          typedef std::vector<std::string> StringVec;
          MixingRxnModel* mixingTable = d_table_lookup->get_table();
          StringVec iv_var_names = mixingTable->getAllIndepVars();
          vector<double> iv;

          for ( StringVec::iterator iv_iter = iv_var_names.begin(); iv_iter != iv_var_names.end(); iv_iter++) {

            string curr_iv = *iv_iter;

            for ( ProblemSpecP db_BCType2 = db_face->findBlock("BCType"); db_BCType2 != nullptr; db_BCType2 = db_BCType2->findNextBlock("BCType") ) {

              string curr_var;
              db_BCType2->getAttribute("label",curr_var);

              if ( curr_var == curr_iv ) {
                string type;
                db_BCType2->getAttribute("var",type);
                if ( type != "Dirichlet") {
                  throw InvalidValue("Error: Cannot compute property values for MassFlowInlet because not all IVs are of type Dirichlet: "+curr_var, __FILE__, __LINE__);
                } else {
                  double value;
                  db_BCType2->require("value",value);
                  iv.push_back(value);
                }
              }
            }
          }

          double density = mixingTable->getTableValue(iv,"density");
          my_info.density = density;


          // note that the mass flow rate is in the BCstruct value

          found_bc = true;

        } else if ( type == "StABL" ) {

          my_info.type = STABL;
          db_BCType->require("roughness",my_info.zo);
          db_BCType->require("freestream_h",my_info.zh);
          db_BCType->require("value",my_info.velocity); // Using <value> as the infinite velocity
          db_BCType->getWithDefault("k",my_info.k,0.41);

          my_info.kappa = pow( my_info.k / log( my_info.zh / my_info.zo ), 2.0);
          my_info.ustar = pow( (my_info.kappa * pow(my_info.velocity[v_index],2.0)), 0.5 );

          if ( dir_grav < 3 ) {
            my_info.dir_gravity = dir_grav;
          } else {
            throw InvalidValue("Error: You must have a gravity direction specified to use the StABL BC.", __FILE__, __LINE__);
          }

          found_bc = true;

        } else if ( type == "PressureBC" ) {

          my_info.type = PRESSURE;
          my_info.velocity = Vector(0,0,0);
          found_bc = true;

        } else if ( type == "OutletBC" ) {

          my_info.type = OUTLET;
          my_info.velocity = Vector(0,0,0);
          found_bc = true;

        } else if ( type == "NOutletBC" ) {

          my_info.type = NEUTRAL_OUTLET;
          my_info.velocity = Vector(0,0,0);
          found_bc = true;

        } else if ( type == "WallBC" ) {

          my_info.type = WALL;
          my_info.velocity = Vector(0,0,0);
          my_info.mass_flow_rate = 0.0;
          found_bc = true;

        }


        if (found_bc) {
          for ( ProblemSpecP db_BCType1 = db_face->findBlock("BCType"); db_BCType1 != nullptr; db_BCType1 = db_BCType1->findNextBlock("BCType") ) {

            std::string type1;
            db_BCType1->getAttribute("var", type1);


            if ( type1 == "PartMassFlowInlet" ) {
              db_BCType1->getAttribute("label", my_info.partName);
              db_BCType1->require("value",my_info.mass_flow_rate);
              my_info.lHasPartMassFlow=true;
              my_info.partVelocity = Vector(0,0,0);
              //------------------For non-orthoganol flows ------------//
              Vector defaultNormalVector= Vector(0,0,0); defaultNormalVector[v_index]=-1*v_sign; // defaul normal vector
              db_BCType->getWithDefault( "massflow_unitVector", my_info.unitVector, defaultNormalVector);

              if (-v_sign*my_info.unitVector[v_index] <= 0.0) {   // defaul normal vector
                throw InvalidValue("massflow_unitVector Error: Normal component cannot be zero or be oriented in direction outward of domain. ", __FILE__, __LINE__);
              }

              double tolly=1e-5;
              double mag_squared=my_info.unitVector.length2();
              if ( mag_squared < 1-tolly || mag_squared > 1.0 +tolly){    // defaul normal vector
                 proc0cout << "massflow_unitVector WARNING: unit vector nas non-unity magnitude, normalizing . \n";
                 my_info.unitVector.safe_normalize();
              }
              //-------------------------------------------------------//

              int qn_total;
              qn_total=ArchesCore::get_num_env(db_face,ArchesCore::DQMOM_METHOD);

              double MassParticleDensity=0;  // (kg/ m^3)
              my_info.vWeights = std::vector<double>(qn_total) ;
              my_info.vVelScalingConst = std::vector<std::vector<double > > (qn_total,std::vector<double> (3,0.0));
              my_info.vVelLabels = std::vector<std::vector<std::string > > (qn_total,std::vector<std::string> (3));


              //// convert #/m^3  --->  kg/m^3, we need weight, diameter, and particle density at inlet
              for (int qn=0; qn< qn_total; qn++){
                // get weight BC
                double weightScalingConstant = ArchesCore::get_scaling_constant(db_BCType,"weight",qn);
                double weight;
                for ( ProblemSpecP db_BCType2 = db_face->findBlock("BCType"); db_BCType2 != nullptr; db_BCType2 = db_BCType2->findNextBlock("BCType") ) {
                  std::string tempLabelName;
                  db_BCType2->getAttribute("label",tempLabelName);

                  std::string weightNode=ArchesCore::append_qn_env ("w",qn);
                  if (tempLabelName == weightNode){
                    db_BCType2->require("value",weight);  // read in bc for weights
                    break;
                  }

                  if( db_BCType2 == nullptr ){
                    throw ProblemSetupException("Arches was unable to find weight boundary condition", __FILE__, __LINE__);
                  }
                }

                my_info.vWeights[qn]=weight;

                // get radius BC (radius particle model will dominate BC specification)
                double diameter;
                std::string sizeLabelName =ArchesCore::parse_for_particle_role_to_label(db_BCType,ArchesCore::P_SIZE);
                if (ArchesCore::get_model_value(db_BCType,sizeLabelName,qn,diameter)== false){
                  double diameterScalingConstant = ArchesCore::get_scaling_constant(db_BCType,sizeLabelName,qn);

                  for ( ProblemSpecP db_BCType2 = db_face->findBlock("BCType"); db_BCType2 != nullptr; db_BCType2 = db_BCType2->findNextBlock("BCType") ) {
                    std::string tempLabelName;
                    db_BCType2->getAttribute("label",tempLabelName);

                    std::string sizeNode=ArchesCore::append_qn_env(sizeLabelName,qn);
                    if( tempLabelName == sizeNode ){
                      db_BCType2->require( "value", diameter );
                      break;
                    }
                    if( db_BCType2 == nullptr ){
                      throw ProblemSetupException("Arches was unable to find length model or boundary condition", __FILE__, __LINE__);
                    }
                  }

                  diameter = diameter/weight*diameterScalingConstant;
                }

                ArchesCore::PARTICLE_ROLE vel_enums[] = {ArchesCore::P_XVEL,
                                                         ArchesCore::P_YVEL,
                                                         ArchesCore::P_ZVEL};
                for(unsigned int i = 0; i<3; i++) {
                  std::string velLabelName =ArchesCore::parse_for_particle_role_to_label(db_BCType,vel_enums[i]);
                  my_info.vVelScalingConst[qn][i] =  ArchesCore::get_scaling_constant(db_BCType,velLabelName,qn);
                  my_info.vVelLabels[qn][i] = ArchesCore::append_qn_env(velLabelName,qn);
                }

                // compute actual particle density (#/m^3)
                weight=weight*weightScalingConstant;

                // get particle density
                double density = ArchesCore::get_inlet_particle_density(db_face);

                MassParticleDensity+=weight*M_PI*diameter*diameter*diameter/6.0*density;  // (kg/ m^3)
              }
              my_info.partDensity = MassParticleDensity;
              // note that the mass flow rate is in the BCstruct value
              break; // exit bcType spec  loop
            }
          }
        }

        if ( found_bc ) {
          numberOfMomentumBCs++;
          if (numberOfMomentumBCs > 1){
            throw ProblemSetupException("Arches found multiple gas-momentum boundary conditions.  I don't know which one to apply.", __FILE__, __LINE__);
          }

          auto i_bc_information = d_bc_information.find(ilvl);
          if ( i_bc_information == d_bc_information.end()){
            BCInfoMap tmp;
            tmp.insert( std::make_pair( bc_type_index, my_info));
            d_bc_information.insert(std::make_pair(ilvl, tmp));
          } else {
            (i_bc_information->second).insert( std::make_pair( bc_type_index, my_info));
          }
          //d_bc_information.insert( std::make_pair(bc_type_index, my_info));
          bc_type_index++;

        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
BoundaryCondition::prune_per_patch_bcinfo( SchedulerP& sched,
                                           const LevelP& level,
                                           WBCHelper* bcHelper )
{
  BndMapT& bc_map = bcHelper->get_for_edit_boundary_information();
  std::vector<int> matches_found;
  const int ilvl = level->getID();
  for ( auto j = d_bc_information[ilvl].begin(); j != d_bc_information[ilvl].end(); j++ ){

    std::string check_name =  j->second.faceName;
    bool match_found = false;

    for (auto i = bc_map.begin(); i != bc_map.end(); i++ ){
      std::string face_name = i->second.name;
      if ( check_name == face_name ){
        match_found = true;
        break;
      }
    }
    if ( ! match_found ){
      matches_found.push_back(j->first);
    }
  }

  if ( d_bc_information[ilvl].size() > 0 ){

    int my_size = d_bc_information[ilvl].size();

    for (int i = 0; i < my_size; i++){

      auto it = std::find(matches_found.begin(), matches_found.end(), i );

      if ( it != matches_found.end() ){
        //info not on this patch:
        if ( d_bc_information[ilvl][*it].type == TURBULENT_INLET ){
          delete d_bc_information[ilvl][*it].TurbIn;
        }
        d_bc_information[ilvl].erase(*it);
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
BoundaryCondition::sched_cellTypeInit(SchedulerP& sched,
                                      const LevelP& level,
                                      const MaterialSet* matls)
{
  IntVector lo, hi;
  level->findInteriorCellIndexRange(lo,hi);

  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::cellTypeInit",
                          this, &BoundaryCondition::cellTypeInit, lo, hi);

  tsk->computes(d_lab->d_cellTypeLabel);

  sched->addTask(tsk, level->eachPatch(), matls);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void
BoundaryCondition::cellTypeInit(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw,
                                IntVector lo, IntVector hi)
{

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    const Level* level = getLevel(patches);
    const int ilvl = level->getID();
    int archIndex = 0;
    int matl_index = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<int> cellType;
    new_dw->allocateAndPut(cellType, d_lab->d_cellTypeLabel, matl_index, patch);
    cellType.initialize(999);

    //going to put "walls" in the corners, even though they aren't accessed:
    for ( CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++ ) {

      IntVector c = *iter;

      bool is_corner = is_corner_cell(patch, c, lo, hi);

      if ( is_corner ) {
        cellType[c] = -1;
      }

    }

    IntVector periodic = level->getPeriodicBoundaries();

    for ( CellIterator iter=patch->getCellIterator(); !iter.done(); iter++ ) {

      // initialize all cells in the interior as flow
      // intrusions will be dealt with later
      cellType[*iter] = -1;

    }

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
          bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

      for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++) {

        //get the face
        Patch::FaceType face = *bf_iter;

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++) {

          double bc_value = 0;
          Vector bc_v_value(0,0,0);
          std::string bc_s_value = "NA";

          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false;

          if ( bc_iter->second.type == VELOCITY_INLET
               || bc_iter->second.type == TURBULENT_INLET
               || bc_iter->second.type == STABL ) {
            foundIterator =
                    getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind);
          } else if ( bc_iter->second.type == VELOCITY_FILE ) {
            foundIterator =
                    getIteratorBCValue<std::string>( patch, face, child, bc_iter->second.name, matl_index, bc_s_value, bound_ptr);
          } else {
            foundIterator =
                    getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);
          }

          if ( foundIterator ) {

            IntVector shift;
            shift = IntVector(0,0,0);

            switch (face) {
            case Patch::xminus:
              shift = IntVector( 1, 0, 0);
              break;
            case Patch::xplus:
              shift = IntVector( 1, 0, 0);
              break;
            case Patch::yminus:
              shift = IntVector( 0, 1, 0);
              break;
            case Patch::yplus:
              shift = IntVector( 0, 1, 0);
              break;
            case Patch::zminus:
              shift = IntVector( 0, 0, 1);
              break;
            case Patch::zplus:
              shift = IntVector( 0, 0, 1);
              break;
            default:
              break;
            }

            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

              IntVector c = *bound_ptr;
              BC_TYPE my_type = bc_iter->second.type;

              if ( my_type == OUTLET ||
                   my_type == TURBULENT_INLET ||
                   my_type == VELOCITY_INLET ||
                   my_type == MASSFLOW_INLET ||
                   my_type == NEUTRAL_OUTLET ) {

                // "if" needed to ensure that extra cell contributions aren't added
                if ( c.x() >= lo.x() - shift.x() && c.x() < hi.x() + shift.x() ) {
                  if ( c.y() >= lo.y() - shift.y() && c.y() < hi.y() + shift.y() ) {
                    if ( c.z() >= lo.z() - shift.z() && c.z() < hi.z() + shift.z() ) {

                      cellType[c] = my_type;

                    }
                  }
                }

              } else {

                cellType[c] = my_type;

              }

            }
          }
        }
      }
    }

    //Now, for this patch, check to make sure you have valid cell types
    //specified everywhere.
    if ( !d_ignore_invalid_celltype ) {
      for ( CellIterator iter=patch->getExtraCellIterator(); !iter.done(); iter++ ) {

        IntVector c = *iter;

        if ( cellType[c] == 999 ) {

          throw InvalidValue("Error: Patch found with an invalid cell type.", __FILE__, __LINE__);

        }
      }
    }
  }
}

//--------------------------------------------------------------------------------
// Compute velocities from mass flow rates for bc's
//
void
BoundaryCondition::sched_setupBCInletVelocities(SchedulerP& sched,
                                                const LevelP& level,
                                                const MaterialSet* matls,
                                                bool doing_restart,
                                                bool doing_regrid = 0)
{
  // cell type initialization
  if ( doing_restart ) {
// This Task helps the infrastructure to find variables in the NEWDW on a restart.
    Task* tskh = scinew Task("BoundaryCondition::setupBCInletVelocitiesHack",
        this, &BoundaryCondition::setupBCInletVelocitiesHack );
    //tskh->computes(  d_lab->d_volFractionLabel );
    tskh->computes(  d_lab->d_densityCPLabel );
    sched->addTask(tskh, level->eachPatch(), matls);
  }

  Task* tsk = scinew Task("BoundaryCondition::setupBCInletVelocities",
                          this, &BoundaryCondition::setupBCInletVelocities,doing_regrid);

  if(doing_regrid){
    tsk->requires( Task::OldDW, d_lab->d_volFractionLabel, Ghost::None, 0 );
    tsk->requires( Task::OldDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 0 );
  }else{
    tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, Ghost::None, 0 );
    tsk->requires( Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 0 );
    tsk->modifies(d_lab->d_volFractionLabel ); // to create task dependancy??  Needs further testing
    tsk->modifies(d_lab->d_densityCPLabel );   // see sched_checkBCs for more info
  }

  sched->addTask(tsk, level->eachPatch(), matls);
}



// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void
BoundaryCondition::setupBCInletVelocities(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset*,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw,
                                          bool doing_regrid )
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    const LevelP level = patch->getLevelP();
    const int ilvl = level->getID();
    int archIndex = 0;
    int matl_index = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    constCCVariable<double> density;
    constCCVariable<double> volFraction;

    if( doing_regrid ){
      old_dw->get( density, d_lab->d_densityCPLabel, matl_index, patch, Ghost::None, 0 );
      old_dw->get( volFraction, d_lab->d_volFractionLabel, matl_index, patch, Ghost::None, 0 );
    }
    else {
      new_dw->get( density, d_lab->d_densityCPLabel, matl_index, patch, Ghost::None, 0 );
      new_dw->get( volFraction, d_lab->d_volFractionLabel, matl_index, patch, Ghost::None, 0 );
    }

    proc0cout << "\n Domain boundary condition summary: \n";

    for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
          bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

      double area = 0.0;

      const BndMapT& my_map =(*m_bcHelper)[level->getID()]->get_boundary_information();
      for (auto iter = my_map.begin(); iter != my_map.end(); iter++ ) {
        BndSpec a_spec = iter->second;
        if ( a_spec.name == bc_iter->second.faceName ){
          area = a_spec.area;
        }
      }

      for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++) {

        //get the face
        Patch::FaceType face = *bf_iter;

        if ( bc_iter->second.face == face ){

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        //get the face direction
        IntVector insideCellDir = patch->faceDirection(face);

        for (int child = 0; child < numChildren; child++) {

          double bc_value = 0;
          Vector bc_v_value(0,0,0);
          std::string bc_s_value = "NA";
          int norm = getNormal( face );

          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false;

          if ( bc_iter->second.type == VELOCITY_INLET || bc_iter->second.type == TURBULENT_INLET ) {
            foundIterator =
                    getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind);
          } else if ( bc_iter->second.type == VELOCITY_FILE ) {
            foundIterator =
                    getIteratorBCValue<std::string>( patch, face, child, bc_iter->second.name, matl_index, bc_s_value, bound_ptr);
          } else {
            foundIterator =
                    getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);
          }

          double small = 1e-10;

          if ( foundIterator ) {


            if (bc_iter->second.lHasPartMassFlow)
            {
              double pm = -1.0*insideCellDir[norm];

              if ( bc_iter->second.partDensity < 1e-200 &&  bc_iter->second.mass_flow_rate > 1e-300 ) {
                throw ProblemSetupException("Arches was unable to satisfy the specified mass flow inlet of particles.  Did you specify reasonable particle density and weights? ", __FILE__, __LINE__);
              }
              bc_iter->second.partVelocity[norm] = 0.0;
              if ( bc_iter->second.mass_flow_rate > 1.e-300 ){
                bc_iter->second.partVelocity[norm] = pm*bc_iter->second.mass_flow_rate /
                  (area * bc_iter->second.partDensity);
              }
              //------------------For non-orthoganol flows-------------//
              int nDim=3; // three dimensional space
              for (int ix=0 ; ix<nDim; ix++){
                bc_iter->second.partVelocity[ix]=bc_iter->second.partVelocity[norm]*bc_iter->second.unitVector[ix]/bc_iter->second.unitVector[norm];
              }
              //-------------------------------------------------------//

              std::string Ubc_kind = "Dirichlet"; // this must be specified for setting uintah BC

              int qn_total =  bc_iter->second.vWeights.size();
              for (int qn=0; qn< qn_total; qn++){

                for(unsigned int i = 0; i < 3; i++) {
                  double uintahVal = bc_iter->second.partVelocity[i]*bc_iter->second.vWeights[qn]*bc_iter->second.vVelScalingConst[qn][i]; // use weighted scaled boundary condition
                  patch->possiblyAddBC(face, child, bc_iter->second.partName, matl_index,uintahVal, Ubc_kind,bc_iter->second.vVelLabels[qn][i],bc_iter->second.faceName );
                }
              }
            }

            // Notice:
            // In the case of mass flow inlets, we are going to assume the density is constant across the inlet
            // so as to compute the average velocity.  As a result, we will just use the first iterator:
            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

              IntVector c = *bound_ptr;

              switch (bc_iter->second.type) {

                case (VELOCITY_INLET) :
                  bc_iter->second.mass_flow_rate = bc_iter->second.velocity[norm] * area * density[*bound_ptr];
                  if (d_check_inlet_obstructions) {
                    if (volFraction[c - insideCellDir] < small) {
                      std::cout << "WARNING: Intrusion blocking a velocity inlet. " << std::endl;
                    }
                  }
                  break;

                case (TURBULENT_INLET) :
                  bc_iter->second.mass_flow_rate = bc_iter->second.velocity[norm] * area * density[*bound_ptr];
                  break;

                case (MASSFLOW_INLET) : {
                  bc_iter->second.mass_flow_rate = bc_value;
                  double pm = -1.0 * insideCellDir[norm];
                  if (bc_iter->second.density > 0.0) {
                    bc_iter->second.velocity[norm] = pm * bc_iter->second.mass_flow_rate / (area * bc_iter->second.density);
                  }

                  int nDim=3; // three dimensional space
                  for (int ix=0 ; ix<nDim; ix++){
                    bc_iter->second.velocity[ix]=bc_iter->second.velocity[norm]*bc_iter->second.unitVector[ix]/bc_iter->second.unitVector[norm];
                  }

                  if (d_check_inlet_obstructions) {
                    if (volFraction[c - insideCellDir] < small) {
                      std::stringstream msg;
                      msg << " Error: Intrusion blocking mass flow inlet on boundary cell: " << c << std::endl;
                      throw InvalidValue(msg.str(), __FILE__, __LINE__);
                    }
                  }

                  break;
                }

                case (SWIRL) :
                {
                  double pm = -1.0 * insideCellDir[norm];
                  bc_iter->second.mass_flow_rate = bc_value;
                  bc_iter->second.velocity[norm] = pm * bc_iter->second.mass_flow_rate / (area * bc_iter->second.density);
                  break;
                }
                case (STABL) :
                  bc_iter->second.mass_flow_rate = 0.0;
                  break;
                default :
                  break;

              }
            }
          }
        }
        }
      }

      proc0cout << "  ----> BC Label: " << bc_iter->second.name << std::endl;
      if ( area < 1.e-16 ){
        proc0cout << "            area: Area not computed because this BC type did not require it. Please use <force_area_calc> in <Arches><BoundaryCondition> to see the value." << std::endl;
      } else {
        proc0cout << "            area: " << area << std::endl;
      }
      // proc0cout << "           m_dot: " << bc_iter->second.mass_flow_rate << std::endl;
      // proc0cout << "               U: " << bc_iter->second.velocity[0] << ", " << bc_iter->second.velocity[1] << ", " << bc_iter->second.velocity[2] << std::endl;
      // proc0cout << "   Particle_vel : " << bc_iter->second.partVelocity[0]<< ", " <<bc_iter->second.partVelocity[1]<< ", " <<bc_iter->second.partVelocity[2]<< std::endl;

    }
    proc0cout << std::endl;
  }
}
//--------------------------------------------------------------------------------
// Apply the boundary conditions
//
void
BoundaryCondition::sched_setInitProfile(SchedulerP& sched,
                                             const LevelP& level,
                                             const MaterialSet* matls)
{
  // cell type initialization
  Task* tsk = scinew Task("BoundaryCondition::setInitProfile",
                          this, &BoundaryCondition::setInitProfile);

  tsk->requires(Task::NewDW, d_lab->d_timeStepLabel);
  tsk->requires(Task::NewDW, d_lab->d_simulationTimeLabel);

  tsk->modifies(d_lab->d_uVelocitySPBCLabel);
  tsk->modifies(d_lab->d_vVelocitySPBCLabel);
  tsk->modifies(d_lab->d_wVelocitySPBCLabel);

  tsk->modifies(d_lab->d_uVelRhoHatLabel);
  tsk->modifies(d_lab->d_vVelRhoHatLabel);
  tsk->modifies(d_lab->d_wVelRhoHatLabel);

  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_lab->d_volFractionLabel, Ghost::None, 0);


  MixingRxnModel* mixingTable = d_table_lookup->get_table();
  MixingRxnModel::VarMap iv_vars = mixingTable->getIVVars();

  for ( MixingRxnModel::VarMap::iterator i = iv_vars.begin(); i != iv_vars.end(); i++ ) {

    tsk->requires( Task::NewDW, i->second, Ghost::AroundCells, 0 );

  }

  sched->addTask(tsk, level->eachPatch(), matls);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void
BoundaryCondition::setInitProfile(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse*,
                                       DataWarehouse* new_dw)
{
  timeStep_vartype timeStep;
  new_dw->get(timeStep, d_lab->d_timeStepLabel );

  simTime_vartype simTime;
  new_dw->get(simTime, d_lab->d_simulationTimeLabel );

  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    const LevelP level = patch->getLevelP();
    const int ilvl = level->getID();
    int archIndex = 0;
    int matl_index = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    vector<Patch::FaceType>::const_iterator bf_iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    SFCXVariable<double> uRhoHat;
    SFCYVariable<double> vRhoHat;
    SFCZVariable<double> wRhoHat;
    constCCVariable<double> density;
    constCCVariable<double> volFraction;

    new_dw->getModifiable( uVelocity, d_lab->d_uVelocitySPBCLabel, matl_index, patch );
    new_dw->getModifiable( vVelocity, d_lab->d_vVelocitySPBCLabel, matl_index, patch );
    new_dw->getModifiable( wVelocity, d_lab->d_wVelocitySPBCLabel, matl_index, patch );
    new_dw->getModifiable( uRhoHat, d_lab->d_uVelRhoHatLabel, matl_index, patch );
    new_dw->getModifiable( vRhoHat, d_lab->d_vVelRhoHatLabel, matl_index, patch );
    new_dw->getModifiable( wRhoHat, d_lab->d_wVelRhoHatLabel, matl_index, patch );
    new_dw->get( density, d_lab->d_densityCPLabel, matl_index, patch, Ghost::None, 0 );
    new_dw->get( volFraction, d_lab->d_volFractionLabel, matl_index, patch, Ghost::None, 0 );

    MixingRxnModel* mixingTable = d_table_lookup->get_table();
    MixingRxnModel::VarMap iv_vars = mixingTable->getIVVars();

    // Get the independent variable information for table lookup
    BoundaryCondition::HelperMap ivGridVarMap;
    BoundaryCondition::HelperVec allIndepVarNames = mixingTable->getAllIndepVars();

    for ( MixingRxnModel::VarMap::iterator i = iv_vars.begin(); i != iv_vars.end(); i++ ) {
      constCCVariable<double> variable;
      new_dw->get( variable, i->second, matl_index, patch, Ghost::None, 0 );
      ivGridVarMap.insert( make_pair( i->first, variable));
    }

    for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
          bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ) {

        //get the face
        Patch::FaceType face = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face);

        if ( bc_iter->second.face == face ){

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++) {

          double bc_value = 0;
          Vector bc_v_value(0,0,0);
          std::string bc_s_value = "NA";

          string bc_kind = "NotSet";
          Iterator bound_ptr;
          bool foundIterator = false;
          string face_name;
          getBCKind( patch, face, child, bc_iter->second.name, matl_index, bc_kind, face_name );

          if ( bc_iter->second.type == VELOCITY_INLET ||
               bc_iter->second.type == TURBULENT_INLET ||
               bc_iter->second.type == STABL ) {
            foundIterator =
                    getIteratorBCValueBCKind<Vector>( patch, face, child, bc_iter->second.name, matl_index, bc_v_value, bound_ptr, bc_kind);
          } else if ( bc_iter->second.type == VELOCITY_FILE ) {
            foundIterator =
                    getIteratorBCValue<std::string>( patch, face, child, bc_iter->second.name, matl_index, bc_s_value, bound_ptr);
          } else {
            foundIterator =
                    getIteratorBCValueBCKind<double>( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);
          }

          if ( foundIterator ) {

            bound_ptr.reset();

            if ( bc_iter->second.type != VELOCITY_FILE ) {
              if ( bc_iter->second.type == SWIRL   ) {
                if ( face == Patch::xminus || face == Patch::xplus ) {

                  setSwirl( patch, face, uVelocity, vVelocity, wVelocity,
                            density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no,
                            bc_iter->second.swirl_cent );

                } else if ( face == Patch::yminus || face == Patch::yplus ) {

                  setSwirl( patch, face, vVelocity, wVelocity, uVelocity,
                            density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no,
                            bc_iter->second.swirl_cent );

                } else if ( face == Patch::zminus || face == Patch::zplus ) {

                  setSwirl( patch, face, wVelocity, uVelocity, vVelocity,
                            density, bound_ptr, bc_iter->second.velocity, bc_iter->second.swirl_no,
                            bc_iter->second.swirl_cent );

                }
              } else if ( bc_iter->second.type == MASSFLOW_INLET || bc_iter->second.type == VELOCITY_INLET ){

                setVel( patch, face, uVelocity, vVelocity, wVelocity, density, bound_ptr, bc_iter->second.velocity );

              } else if ( bc_iter->second.type == STABL ) {

                setStABL( patch, face, uVelocity, vVelocity, wVelocity, &bc_iter->second, bound_ptr );

              } else if ( bc_iter->second.type == TURBULENT_INLET ){

                setTurbInlet( patch, face, uVelocity, vVelocity, wVelocity, density, bound_ptr, bc_iter->second.TurbIn, timeStep, simTime );

              } else if ( bc_iter->second.type == WALL ) {

                setVel( patch, face, uVelocity, vVelocity, wVelocity, density, bound_ptr, bc_iter->second.velocity );

              }

            } else {

              //---- set velocities
              setVelFromInput( patch, face, face_name, uVelocity, vVelocity, wVelocity, bound_ptr, bc_iter->second.filename );

            }
            }
          }
        }
      }
    }

    uRhoHat.copyData( uVelocity );
    vRhoHat.copyData( vVelocity );
    wRhoHat.copyData( wVelocity );

  } //patch iterator
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
  IntVector outsideCell(0,0,0);
  Vector Dx = patch->dCell();

  int idim=-99; //normal direction
  int jdim=-99; //1st tangential
  int kdim=-99; //2nd tangential

  double sign_v = -1.;
  double sign_w = 1.;
  if ( face == Patch::xminus || face == Patch::xplus ){
    idim = 0; jdim = 1; kdim = 2;
    if ( face == Patch::xplus ){
      outsideCell[0] = 1;
      sign_v = 1;
      sign_w = -1;
    }
  } else if ( face == Patch::yminus || face == Patch::yplus ){
    idim = 1; jdim = 2; kdim = 0;
    if ( face == Patch::yplus ){
      outsideCell[1] = 1;
      sign_v = 1;
      sign_w = -1;
    }
  } else if ( face == Patch::zminus || face == Patch::zplus ){
    idim = 2; jdim = 0; kdim = 1;
    if ( face == Patch::zplus ){
      outsideCell[2] = 1;
      sign_v = 1;
      sign_w = -1;
    }
  }

  const double noise = 1e-10; //avoid divide by zero

  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {

    IntVector c  = *bound_ptr;
    IntVector cp = *bound_ptr - insideCellDir;
    Point p = patch->cellPosition(c);
    Vector pp; pp[0] = p.x(); pp[1] = p.y(); pp[2] = p.z();
    Vector pt; //translated point
    pt[idim] = 99; //error check...
    pt[jdim] = pp[jdim] - swrl_cent[jdim];
    pt[kdim] = pp[kdim] - Dx[kdim]/2. - swrl_cent[kdim];

    double denom = pt[jdim]*pt[jdim] + pt[kdim]*pt[kdim]; denom = std::pow(denom,0.5);
    denom = (denom > 1e-16) ? denom : denom+noise;

    uVel[c] = value[idim];
    uVel[cp] = value[idim];
    uVel[c+outsideCell] = value[idim];  //sets the extra cell in the + direction (that isn't used) for vis.

    double swirl_condition = sign_v * pt[kdim] * swrl_no * value[idim] / denom;
    vVel[c] = 2.0*swirl_condition - vVel[cp];

    pt[jdim] = pp[jdim] - Dx[jdim]/2. - swrl_cent[jdim];
    pt[kdim] = pp[kdim] - swrl_cent[kdim];

    denom = pt[jdim]*pt[jdim] + pt[kdim]*pt[kdim]; denom = std::pow(denom,0.5);
    denom = (denom > 1e-16) ? denom : denom+noise;

    swirl_condition = sign_w * pt[jdim] * swrl_no * value[idim] / denom;
    wVel[c] = 2.0*swirl_condition - wVel[cp];

  }
}

void BoundaryCondition::setTurbInlet( const Patch* patch, const Patch::FaceType& face,
                                      SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                                      constCCVariable<double>& density,
                                      Iterator bound_ptr, DigitalFilterInlet * TurbInlet,
                                      const int timeStep,
                                      const double simTime )
{
  IntVector insideCellDir = patch->faceDirection(face);

  int j, k;
  // int timeStep = d_lab->d_materialManager->getCurrentTopLevelTimeStep();
  // double simTime = d_lab->d_materialManager->getElapsedSimTime();
  int t = TurbInlet->getTimeIndex( timeStep, simTime);

  IntVector shiftVec;
  shiftVec = TurbInlet->getOffsetVector( );

  switch ( face ) {
  case Patch::xminus:
    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      vector<double> velVal (3);
      j = c.y() - shiftVec.y();
      k = c.z() - shiftVec.z();

      velVal = TurbInlet->getVelocityVector(t, j, k);

      uVel[c]  = velVal[0];
      uVel[cp] = velVal[0];
      vVel[c]  = velVal[1];
      wVel[c]  = velVal[2];
    }
    break;
  case Patch::xplus:
    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr + insideCellDir;

      vector<double> velVal (3);
      j = c.y() - shiftVec.y();
      k = c.z() - shiftVec.z();

      velVal = TurbInlet->getVelocityVector(t, j, k);

      uVel[cp] = velVal[0];
      uVel[c]  = velVal[0];
      vVel[c]  = velVal[1];
      wVel[c]  = velVal[2];
    }
    break;
  case Patch::yminus:
    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      vector<double> velVal (3);
      j = c.x() - shiftVec.x();
      k = c.z() - shiftVec.z();

      velVal = TurbInlet->getVelocityVector(t, j, k);

      uVel[c]  = velVal[0];
      vVel[c]  = velVal[1];
      vVel[cp] = velVal[1];
      wVel[c]  = velVal[2];
    }

    break;
  case Patch::yplus:
    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr + insideCellDir;

      vector<double> velVal (3);
      j = c.x() - shiftVec.x();
      k = c.z() - shiftVec.z();

      velVal = TurbInlet->getVelocityVector(t, j, k);

      uVel[c]  = velVal[0];
      vVel[c]  = velVal[1];
      vVel[cp] = velVal[1];
      wVel[c]  = velVal[2];
    }
    break;
  case Patch::zminus:
    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      vector<double> velVal (3);
      j = c.x() - shiftVec.x();
      k = c.y() - shiftVec.y();

      velVal = TurbInlet->getVelocityVector(t, j, k);

      uVel[c]  = velVal[0];
      vVel[c]  = velVal[1];
      wVel[c]  = velVal[2];
      wVel[cp] = velVal[2];
    }
    break;
  case Patch::zplus:
    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr + insideCellDir;

      vector<double> velVal (3);
      j = c.x() - shiftVec.x();
      k = c.y() - shiftVec.y();

      velVal = TurbInlet->getVelocityVector(t, j, k);

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

void BoundaryCondition::setStABL( const Patch* patch, const Patch::FaceType& face,
                                  SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                                  BCInfo* bcinfo,
                                  Iterator bound_ptr  )
{

  IntVector insideCellDir = patch->faceDirection(face);
  Vector Dx = patch->dCell();

  switch ( face ) {
  case Patch::xminus:
    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      Point p = patch->getCellPosition(c);
      double vel = 0;
      if ( p(bcinfo->dir_gravity) > 0.0 ) {
        vel = bcinfo->ustar / bcinfo->k * log( p(bcinfo->dir_gravity) / bcinfo->zo );
      } else {
        vel = bcinfo->ustar / bcinfo->k * log( ( p(bcinfo->dir_gravity)+Dx[0] ) / bcinfo->zo );
      }

      uVel[c]  = vel;
      uVel[cp] = vel;

      vVel[c] = bcinfo->velocity[1];
      wVel[c] = bcinfo->velocity[2];
    }

    break;
  case Patch::xplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      Point p = patch->getCellPosition(c);
      double vel = 0;
      if ( p(bcinfo->dir_gravity) > 0.0 ) {
        vel = bcinfo->ustar / bcinfo->k * log( p(bcinfo->dir_gravity) / bcinfo->zo );
      } else {
        vel = bcinfo->ustar / bcinfo->k * log( ( p(bcinfo->dir_gravity)+Dx[0] ) / bcinfo->zo );
      }

      uVel[c]  = -vel;
      uVel[cp] = -vel;

      vVel[c] = bcinfo->velocity[1];
      wVel[c] = bcinfo->velocity[2];

    }
    break;
  case Patch::yminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      Point p = patch->getCellPosition(c);
      double vel = 0;
      if ( p(bcinfo->dir_gravity) > 0.0 ) {
        vel = bcinfo->ustar / bcinfo->k * log( p(bcinfo->dir_gravity) / bcinfo->zo );
      } else {
        vel = bcinfo->ustar / bcinfo->k * log( ( p(bcinfo->dir_gravity)+Dx[1] ) / bcinfo->zo );
      }

      vVel[c]  = vel;
      vVel[cp] = vel;

      uVel[c] = bcinfo->velocity[0];
      wVel[c] = bcinfo->velocity[2];


    }
    break;
  case Patch::yplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      Point p = patch->getCellPosition(c);
      double vel = 0;
      if ( p(bcinfo->dir_gravity) > 0.0 ) {
        vel = bcinfo->ustar / bcinfo->k * log( p(bcinfo->dir_gravity) / bcinfo->zo );
      } else {
        vel = bcinfo->ustar / bcinfo->k * log( ( p(bcinfo->dir_gravity)+Dx[1] ) / bcinfo->zo );
      }

      vVel[c]  = -vel;
      vVel[cp] = -vel;

      uVel[c] = bcinfo->velocity[0];
      wVel[c] = bcinfo->velocity[2];

    }
    break;
  case Patch::zminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      Point p = patch->getCellPosition(c);
      double vel = 0;
      if ( p(bcinfo->dir_gravity) > 0.0 ) {
        vel = bcinfo->ustar / bcinfo->k * log( p(bcinfo->dir_gravity) / bcinfo->zo );
      } else {
        vel = bcinfo->ustar / bcinfo->k * log( ( p(bcinfo->dir_gravity)+Dx[2] ) / bcinfo->zo );
      }

      wVel[c]  = vel;
      wVel[cp] = vel;

      uVel[c] = bcinfo->velocity[0];
      vVel[c] = bcinfo->velocity[1];


    }
    break;
  case Patch::zplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      Point p = patch->getCellPosition(c);
      double vel = 0;
      if ( p(bcinfo->dir_gravity) > 0.0 ) {
        vel = bcinfo->ustar / bcinfo->k * log( p(bcinfo->dir_gravity) / bcinfo->zo );
      } else {
        vel = bcinfo->ustar / bcinfo->k * log( ( p(bcinfo->dir_gravity)+Dx[2] ) / bcinfo->zo );
      }

      wVel[c]  = -vel;
      wVel[cp] = -vel;

      uVel[c] = bcinfo->velocity[0];
      vVel[c] = bcinfo->velocity[1];

    }
    break;
  default:

    break;

  }
}

void BoundaryCondition::setVel( const Patch* patch, const Patch::FaceType& face,
                                SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                                constCCVariable<double>& density,
                                Iterator bound_ptr, Vector value )
{

  //get the face direction
  IntVector insideCellDir = patch->faceDirection(face);

  switch ( face ) {

  case Patch::xminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cin = *bound_ptr - insideCellDir;

      uVel[c]  = value.x();
      uVel[cin] = value.x();

      vVel[c] = 2.*value.y() - vVel[cin];
      wVel[c] = 2.*value.z() - wVel[cin];

    }

    break;
  case Patch::xplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cec = *bound_ptr + insideCellDir;
      IntVector cin = *bound_ptr - insideCellDir;

      uVel[cec]  = value.x();
      uVel[c]   = value.x();

      vVel[c] = 2.0*value.y() - vVel[cin];
      wVel[c] = 2.0*value.z() - wVel[cin];

    }
    break;
  case Patch::yminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cin = *bound_ptr - insideCellDir;

      vVel[c] = value.y();
      vVel[cin] = value.y();

      uVel[c] = 2.0 * value.x() - uVel[cin];
      wVel[c] = 2.0 * value.z() - wVel[cin];

    }
    break;
  case Patch::yplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cec = *bound_ptr + insideCellDir;
      IntVector cin = *bound_ptr - insideCellDir;

      vVel[cec] = value.y();
      vVel[c] = value.y();

      uVel[c] = 2.0 * value.x() - uVel[cin];
      wVel[c] = 2.0 * value.z() - wVel[cin];

    }
    break;
  case Patch::zminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cin = *bound_ptr - insideCellDir;

      wVel[c] = value.z();
      wVel[cin] = value.z();

      uVel[c] = 2.0 * value.x() - uVel[cin];
      vVel[c] = 2.0 * value.y() - vVel[cin];

    }
    break;
  case Patch::zplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cec = *bound_ptr + insideCellDir;
      IntVector cin = *bound_ptr - insideCellDir;

      wVel[cec] = value.z();
      wVel[c] = value.z();

      uVel[c] = 2.0 * value.x() - uVel[cin];
      vVel[c] = 2.0 * value.y() - vVel[cin];

    }
    break;
  default:

    break;

  }
}

void BoundaryCondition::setVelFromExtraValue( const Patch* patch, const Patch::FaceType& face,
                                                   SFCXVariable<double>& uVel, SFCYVariable<double>& vVel, SFCZVariable<double>& wVel,
                                                   constCCVariable<double>& density,
                                                   Iterator bound_ptr, Vector value )
{

  //get the face direction
  IntVector insideCellDir = patch->faceDirection(face);

  switch ( face ) {

  case Patch::xminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      uVel[cp] = uVel[c];

    }

    break;
  case Patch::xplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      uVel[cp] = uVel[c];

    }
    break;
  case Patch::yminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      vVel[cp] = vVel[c];

    }
    break;
  case Patch::yplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      vVel[cp] = vVel[c];

    }
    break;
  case Patch::zminus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      wVel[cp] = wVel[c];

    }
    break;
  case Patch::zplus:

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector c  = *bound_ptr;
      IntVector cp = *bound_ptr - insideCellDir;

      wVel[cp] = wVel[c];

    }
    break;
  default:

    break;

  }
}

void BoundaryCondition::setVelFromInput( const Patch* patch, const Patch::FaceType& face,
                                              string face_name,
                                              SFCXVariable<double>& uVel, SFCYVariable<double>& vVel,
                                              SFCZVariable<double>& wVel,
                                              Iterator bound_ptr, std::string file_name )
{

  //get the face direction
  IntVector insideCellDir = patch->faceDirection(face);
  FaceToInput::iterator fu_iter = _u_input.find( face_name );
  FaceToInput::iterator fv_iter = _v_input.find( face_name );
  FaceToInput::iterator fw_iter = _w_input.find( face_name );

  if ( face == Patch::xminus || face == Patch::xplus ) {

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector rel_ijk = *bound_ptr - fu_iter->second.relative_ijk;
      CellToValue::iterator u_iter = fu_iter->second.values.find( rel_ijk );
      CellToValue::iterator v_iter = fv_iter->second.values.find( rel_ijk );
      CellToValue::iterator w_iter = fw_iter->second.values.find( rel_ijk );

      if ( u_iter != fu_iter->second.values.end() ) {
        uVel[ *bound_ptr ] = u_iter->second;
        uVel[ *bound_ptr - insideCellDir ] = u_iter->second;
        vVel[ *bound_ptr ] = v_iter->second;
        wVel[ *bound_ptr ] = w_iter->second;
      } else if ( fu_iter->second.default_type == "Neumann" ) {
        uVel[ *bound_ptr ] = uVel[*bound_ptr + insideCellDir];
        uVel[ *bound_ptr - insideCellDir ] = uVel[*bound_ptr];
        vVel[ *bound_ptr ] = vVel[*bound_ptr + insideCellDir];
        wVel[ *bound_ptr ] = wVel[*bound_ptr + insideCellDir];
      } else if ( fu_iter->second.default_type == "Dirichlet" ) {
        uVel[ *bound_ptr ] = fu_iter->second.default_value;
        uVel[ *bound_ptr - insideCellDir ] = fu_iter->second.default_value;
        vVel[ *bound_ptr ] = fv_iter->second.default_value;
        wVel[ *bound_ptr ] = fw_iter->second.default_value;
      }
    }

  } else if ( face == Patch::yminus || face == Patch::yplus ) {

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector rel_ijk = *bound_ptr - fv_iter->second.relative_ijk;
      CellToValue::iterator v_iter = fv_iter->second.values.find( rel_ijk );
      CellToValue::iterator w_iter = fw_iter->second.values.find( rel_ijk );
      CellToValue::iterator u_iter = fu_iter->second.values.find( rel_ijk );

      if ( v_iter != fv_iter->second.values.end() ) {
        vVel[ *bound_ptr ] = v_iter->second;
        vVel[ *bound_ptr - insideCellDir ] = v_iter->second;
        wVel[ *bound_ptr ] = w_iter->second;
        uVel[ *bound_ptr ] = u_iter->second;
      } else if ( fv_iter->second.default_type == "Neumann" ) {
        vVel[ *bound_ptr ] = vVel[*bound_ptr + insideCellDir];
        vVel[ *bound_ptr - insideCellDir ] = vVel[*bound_ptr];
        wVel[ *bound_ptr ] = wVel[*bound_ptr + insideCellDir];
        uVel[ *bound_ptr ] = uVel[*bound_ptr + insideCellDir];
      } else if ( fv_iter->second.default_type == "Dirichlet" ) {
        vVel[ *bound_ptr ] = fv_iter->second.default_value;
        vVel[ *bound_ptr - insideCellDir ] = fv_iter->second.default_value;
        wVel[ *bound_ptr ] = fw_iter->second.default_value;
        uVel[ *bound_ptr ] = fu_iter->second.default_value;
      }
    }

  } else if ( face == Patch::zminus || face == Patch::zplus ) {

    for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {

      IntVector rel_ijk = *bound_ptr - fw_iter->second.relative_ijk;
      CellToValue::iterator w_iter = fw_iter->second.values.find( rel_ijk );
      CellToValue::iterator u_iter = fu_iter->second.values.find( rel_ijk );
      CellToValue::iterator v_iter = fv_iter->second.values.find( rel_ijk );

      if ( w_iter != fw_iter->second.values.end() ) {
        wVel[ *bound_ptr ] = w_iter->second;
        wVel[ *bound_ptr - insideCellDir ] = w_iter->second;
        uVel[ *bound_ptr ] = u_iter->second;
        vVel[ *bound_ptr ] = v_iter->second;
      } else if ( fw_iter->second.default_type == "Neumann" ) {
        wVel[ *bound_ptr ] = wVel[*bound_ptr + insideCellDir];
        wVel[ *bound_ptr - insideCellDir ] = wVel[*bound_ptr];
        uVel[ *bound_ptr ] = uVel[*bound_ptr + insideCellDir];
        vVel[ *bound_ptr ] = vVel[*bound_ptr + insideCellDir];
      } else if ( fw_iter->second.default_type == "Dirichlet" ) {
        wVel[ *bound_ptr ] = fw_iter->second.default_value;
        wVel[ *bound_ptr - insideCellDir ] = fw_iter->second.default_value;
        uVel[ *bound_ptr ] = fu_iter->second.default_value;
        vVel[ *bound_ptr ] = fv_iter->second.default_value;
      }
    }

  }
}

void
BoundaryCondition::readInputFile( std::string file_name, BoundaryCondition::FFInfo& struct_result, const int index )
{

  gzFile file = gzopen( file_name.c_str(), "r" );
  if ( file == nullptr ) {
    proc0cout << "Error opening file: " << file_name << " for boundary conditions. Errno: " << errno << endl;
    throw ProblemSetupException("Unable to open the given input file: " + file_name, __FILE__, __LINE__);
  }

  struct_result.name = getString( file );

  struct_result.dx = getDouble( file );
  struct_result.dy = getDouble( file );

  int num_points = getInt( file );

  std::map<IntVector, double> values;

  int is=0; int js=0; int ks = 0;
  if ( index == 0 ){
      is = 1;
  } else if ( index == 1 ){
      js = 1;
  } else {
      ks = 1;
  }

  for ( int i = 0; i < num_points; i++ ) {
    int I = getInt( file );
    int J = getInt( file );
    int K = getInt( file );
    Vector v;
    v[0] = getDouble( file );
    v[1] = getDouble( file );
    v[2] = getDouble( file );

    IntVector C(I,J,K);

    values.insert( make_pair( C, v[index] ));

    IntVector C2(I-is, J-js, K-ks);

    values.insert( make_pair( C2, v[index] ));

  }

  struct_result.values = values;

  gzclose( file );

}

void
BoundaryCondition::velocityOutletPressureBC( const Patch* patch,
                                             int matl_index,
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
  const Level* level = patch->getLevel();
  const int ilvl = level->getID();

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

  for ( BCInfoMap::iterator bc_iter = d_bc_information[ilvl].begin();
        bc_iter != d_bc_information[ilvl].end(); bc_iter++) {

    if ( bc_iter->second.type == OUTLET || bc_iter->second.type == PRESSURE ) {

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ) {

        //get the face
        Patch::FaceType face = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face);

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++) {

          double bc_value = 0;
          //int norm = getNormal( face );

          string bc_kind = "NotSet";
          Iterator bound_ptr;

          //ALWAYS a double so no need to check for vectors
          bool foundIterator =
                  getIteratorBCValueBCKind( patch, face, child, bc_iter->second.name, matl_index, bc_value, bound_ptr, bc_kind);

          if ( foundIterator ) {

            bound_ptr.reset();
            double sign        = 1.0;

            if ( bc_iter->second.type == PRESSURE ) {
              sign = -1.0;
            }

            switch (face) {

            case Patch::xminus:

              if ( d_no_corner_recirc ) {

                outletPressureMinus( insideCellDir, bound_ptr, idxLo, idxHi,
                                     2, 1, sign, uvel, old_uvel,
                                     zminus, zplus, yminus, yplus );

              } else {

                outletPressureMinus( insideCellDir, bound_ptr, sign, uvel, old_uvel );

              }
              break;

            case Patch::xplus:

              if ( d_no_corner_recirc ) {

                outletPressurePlus(  insideCellDir, bound_ptr, idxLo, idxHi,
                                     2, 1, sign, uvel, old_uvel,
                                     zminus, zplus, yminus, yplus );

              } else {

                outletPressurePlus( insideCellDir, bound_ptr, sign, uvel, old_uvel );

              }
              break;

            case Patch::yminus:

              if ( d_no_corner_recirc ) {

                outletPressureMinus( insideCellDir, bound_ptr, idxLo, idxHi,
                                     2, 0, sign, vvel, old_vvel,
                                     zminus, zplus, xminus, xplus );

              } else {

                outletPressureMinus( insideCellDir, bound_ptr, sign, vvel, old_vvel );

              }
              break;

            case Patch::yplus:

              if ( d_no_corner_recirc ) {

                outletPressurePlus(  insideCellDir, bound_ptr, idxLo, idxHi,
                                     2, 0, sign, vvel, old_vvel,
                                     zminus, zplus, xminus, xplus );

              } else {

                outletPressurePlus( insideCellDir, bound_ptr, sign, vvel, old_vvel );

              }
              break;

            case Patch::zminus:

              if ( d_no_corner_recirc ) {

                outletPressureMinus( insideCellDir, bound_ptr, idxLo, idxHi,
                                     0, 1, sign, wvel, old_wvel,
                                     xminus, xplus, yminus, yplus );

              } else {

                outletPressureMinus( insideCellDir, bound_ptr, sign, wvel, old_wvel );

              }
              break;

            case Patch::zplus:

              if ( d_no_corner_recirc ) {

                outletPressurePlus(  insideCellDir, bound_ptr, idxLo, idxHi,
                                     0, 1, sign, wvel, old_wvel,
                                     xminus, xplus, yminus, yplus );

              } else {

                outletPressurePlus( insideCellDir, bound_ptr, sign, wvel, old_wvel );

              }
              break;

            default:
              std::stringstream msg;
              msg << "Error: Face type not recognized: " << face << std::endl;
              throw InvalidValue(msg.str(), __FILE__, __LINE__);
              break;
            }
          }
        }
      }
    } else if ( bc_iter->second.type == NEUTRAL_OUTLET ) {

      for ( bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++ ) {

        //get the face
        Patch::FaceType face = *bf_iter;
        IntVector insideCellDir = patch->faceDirection(face);

        //get the number of children
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl_index); //assumed one material

        for (int child = 0; child < numChildren; child++) {

          double bc_value = 0;
          //int norm = getNormal( face );

          string bc_kind = "NotSet";
          Iterator bound_ptr;

          //ALWAYS a double so no need to check for vectors
          bool foundIterator =
                  getIteratorBCValueBCKind( patch, face, child, bc_iter->second.name,
                                            matl_index, bc_value, bound_ptr, bc_kind );

          if ( foundIterator ) {

            bound_ptr.reset();
            switch (face) {

            case Patch::xminus:

              neutralOutleMinus( insideCellDir, bound_ptr, uvel, old_uvel );
              break;

            case Patch::xplus:

              neutralOutletPlus( insideCellDir, bound_ptr, uvel, old_uvel );
              break;

            case Patch::yminus:

              neutralOutleMinus( insideCellDir, bound_ptr, vvel, old_vvel );
              break;

            case Patch::yplus:

              neutralOutletPlus( insideCellDir, bound_ptr, vvel, old_vvel );
              break;

            case Patch::zminus:

              neutralOutleMinus( insideCellDir, bound_ptr, wvel, old_wvel );
              break;

            case Patch::zplus:

              neutralOutletPlus( insideCellDir, bound_ptr, wvel, old_wvel );
              break;

            default:
              std::stringstream msg;
              msg << "Error: Face type not recognized: " << face << std::endl;
              throw InvalidValue(msg.str(), __FILE__, __LINE__);
              break;
            }
          }
        }
      }
    }
  }
}

void
BoundaryCondition::sched_setupNewIntrusionCellType( SchedulerP& sched,
                                                    const LevelP& level,
                                                    const MaterialSet* matls,
                                                    const bool doing_restart )
{
  if ( _using_new_intrusion ) {
    const int i = level->getID();
    _intrusionBC[i]->sched_setCellType( sched, level, matls, doing_restart );
  }
}


void
BoundaryCondition::sched_setupNewIntrusions( SchedulerP& sched,
                                             const LevelP& level,
                                             const MaterialSet* matls )
{

  if ( _using_new_intrusion ) {
    const int i = level->getID();
    _intrusionBC[i]->sched_computeBCArea( sched, level, matls );
    _intrusionBC[i]->sched_computeProperties( sched, level, matls );
    _intrusionBC[i]->sched_setIntrusionVelocities( sched, level, matls );
    _intrusionBC[i]->sched_printIntrusionInformation( sched, level, matls );
    _intrusionBC[i]->prune_per_patch_intrusions( sched, level, matls );
    _intrusionBC[i]->sched_setAlphaG(sched, level, matls, false);
  }

}

void
BoundaryCondition::sched_setIntrusionDensity( SchedulerP& sched,
                                              const LevelP& level,
                                              const MaterialSet* matls )
{
  Task* tsk = scinew Task( "BoundaryCondition::setIntrusionDensity",
                           this, &BoundaryCondition::setIntrusionDensity);
  tsk->requires( Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells, 1 );
  tsk->modifies( d_lab->d_densityCPLabel );
  sched->addTask( tsk, level->eachPatch(), matls );

}

void
BoundaryCondition::setIntrusionDensity( const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {

    if ( _using_new_intrusion ) {

      const Patch* patch = patches->get(p);
      const Level* level = patch->getLevel();
      const int ilvl = level->getID();
      int archIndex = 0; // only one arches material
      int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

      CCVariable<double> density;
      constCCVariable<double> old_density;
      new_dw->getModifiable( density, d_lab->d_densityCPLabel, indx, patch );
      new_dw->get( old_density, d_lab->d_densityCPLabel, indx, patch, Ghost::AroundCells, 1 );

      _intrusionBC[ilvl]->setDensity( patch, density, old_density );

    }
  }
}

Vector BoundaryCondition::getMaxIntrusionVelocity( const Level* level ){

  const int ilvl = level->getID();
  if ( _using_new_intrusion ){
    bool has_intrusion_inlets = _intrusionBC[ilvl]->has_intrusion_inlets();
    if ( has_intrusion_inlets ){
      return _intrusionBC[ilvl]->getMaxVelocity();
    }
  }

  return Vector(0,0,0);

}

void
BoundaryCondition::sched_wallStressConstSmag( Task::WhichDW dw, Task* tsk ){

  const VarLabel* IsImag_label = VarLabel::find("strainMagnitudeLabel");
  if ( IsImag_label != nullptr ){
    tsk->requires( dw, IsImag_label, Ghost::AroundCells, 1 );
  } else {
    throw InvalidValue("Error: Strain rate mag. not found for const Smag. wall model.", __FILE__, __LINE__ );
  }

}

void
BoundaryCondition::wallStressConstSmag( const Patch* p,
                                        DataWarehouse* dw,
                                        const double csmag_wall,
                                        const int standoff,
                                        constSFCXVariable<double>& uvel,
                                        constSFCYVariable<double>& vvel,
                                        constSFCZVariable<double>& wvel,
                                        SFCXVariable<double>& Su,
                                        SFCYVariable<double>& Sv,
                                        SFCZVariable<double>& Sw,
                                        constCCVariable<double>& rho,
                                        constCCVariable<double>& eps )
{

  Vector Dx = p->dCell();

  double viscos; // molecular viscosity
  viscos = d_physicalConsts->getMolecularViscosity();
  const double delta = std::pow( Dx.x()*Dx.y()*Dx.z(),1./3.);
  const double area_ew = Dx.y() * Dx.z();
  const double area_ns = Dx.z() * Dx.x();
  const double area_tb = Dx.x() * Dx.y();
  const double dx = Dx.x();
  const double dy = Dx.y();
  const double dz = Dx.z();

  constCCVariable<double> IsImag;
  const int indx = 0;
  const Level* level = p->getLevel();
  const int ilvl = level->getID();

  const VarLabel* IsImag_label = VarLabel::find("strainMagnitudeLabel");

  if ( dw->exists( IsImag_label, indx, p ) ){

    dw->get( IsImag, IsImag_label, indx, p, Ghost::AroundCells, 1 );

    bool has_intrusion_inlets = false;
    if ( _using_new_intrusion ){
      has_intrusion_inlets = _intrusionBC[ilvl]->has_intrusion_inlets();
    }

    for (CellIterator iter=p->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      IntVector xm = *iter - IntVector(1,0,0);
      IntVector xp = *iter + IntVector(1,0,0);
      IntVector ym = *iter - IntVector(0,1,0);
      IntVector yp = *iter + IntVector(0,1,0);
      IntVector zm = *iter - IntVector(0,0,1);
      IntVector zp = *iter + IntVector(0,0,1);
      IntVector xmym = *iter + IntVector(-1,-1,0);
      IntVector xmyp = *iter + IntVector(-1,1,0);
      IntVector xmzm = *iter + IntVector(-1,0,-1);
      IntVector xmzp = *iter + IntVector(-1,0,1);
      IntVector xpym = *iter + IntVector(1,-1,0);
      IntVector xpzm = *iter + IntVector(1,0,-1);
      IntVector ymzm = *iter + IntVector(0,-1,-1);
      IntVector ymzp = *iter + IntVector(0,-1,1);
      IntVector ypzm = *iter + IntVector(0,1,-1);
      IntVector x_so = IntVector(standoff,0,0);
      IntVector y_so = IntVector(0,standoff,0);
      IntVector z_so = IntVector(0,0,standoff);
      //apply u-mom bc -
      if ( eps[xm] * eps[c] > .5 ){
        // Y-
        if ( eps[ym] * eps[xmym] < .5 ){
          const double i_so = ( eps[c+y_so] * eps[xm+y_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(0,i_so,0)] + IsImag[c+IntVector(-1,i_so,0)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;
          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, c, has_condition,
                                                   velocity_cond );
          }

          Su[c] += 2.0 * area_ns * ( mu_t + viscos )
                   * (velocity_cond[0] - uvel[c]) / dy;

        }
        // Y+
        if ( eps[yp] * eps[xmyp] < .5 ){
          const double i_so = ( eps[c-y_so] * eps[xm-y_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5*(IsImag[c+IntVector(0,-i_so,0)] + IsImag[c+IntVector(-1,-i_so,0)]);
          const double mu_t = pow( csmag_wall*delta, 2.0 ) * rho[c] * ISI;
          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, yp, has_condition,
                                                   velocity_cond );
          }

          Su[c] += 2.0 * area_ns * ( mu_t + viscos )
                    * (velocity_cond[0] - uvel[c]) / dy;

        }

        // Z-
        if ( eps[zm] * eps[xmzm] < .5 ){
          const double i_so = ( eps[c+z_so] * eps[xm+z_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(0,0,i_so)] + IsImag[c+IntVector(-1,0,i_so)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;

          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, c, has_condition,
                                                   velocity_cond );
          }

          Su[c] += 2.0 * area_tb * ( mu_t + viscos )
                   * (velocity_cond[0] - uvel[c]) / dz;

        }

        // Z+
        if ( eps[zp] * eps[xmzp] < .5 ){
          const double i_so = ( eps[c-z_so] * eps[xm-z_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5*(IsImag[c+IntVector(0,0,-i_so)] + IsImag[c+IntVector(-1,0,-i_so)]);
          const double mu_t = pow( csmag_wall*delta, 2.0 ) * rho[c] * ISI;

          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, zp, has_condition,
                                                   velocity_cond );
          }

          Su[c] += 2.0 * area_tb * ( mu_t + viscos )
                    * (velocity_cond[0] - uvel[c]) / dz;

        }
      }
      //apply v-mom bc -
      if ( eps[ym] * eps[c] > 0.5 ) {
        // X-
        if ( eps[xm] * eps[xmym] < .5 ){
          const double i_so = ( eps[c+x_so] * eps[ym+x_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(i_so,0,0)] + IsImag[c+IntVector(i_so,-1,0)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;
          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, c, has_condition,
                                                   velocity_cond );
          }

          Sv[c] += 2.0 * area_ew * ( mu_t + viscos )
                    * (velocity_cond[1] - vvel[c] )/ dx;

        }
        // X+
        if ( eps[xp] * eps[xpym] < .5 ){
          const double i_so = ( eps[c-x_so] * eps[ym-x_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(-i_so,0,0)] + IsImag[c+IntVector(-i_so,-1,0)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;
          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, xp, has_condition,
                                                   velocity_cond );
          }

          Sv[c] += 2.0 * area_ew * ( mu_t + viscos )
                    * (velocity_cond[1] - vvel[c] )/ dx;

        }
        // Z-
        if ( eps[zm] * eps[ymzm] < .5 ){
          const double i_so = ( eps[c+z_so] * eps[ym+z_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(0,0,i_so)] + IsImag[c+IntVector(0,-1,i_so)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;
          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, c, has_condition,
                                                   velocity_cond );
          }

          Sv[c] += 2.0 * area_tb * ( mu_t + viscos )
            * ( velocity_cond[1] - vvel[c] ) / dz;

        }
        // Z+
        if ( eps[zp] * eps[ymzp] < .5 ){
          const double i_so = ( eps[c-z_so] * eps[ym-z_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(0,0,-i_so)] + IsImag[c+IntVector(0,-1,-i_so)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;

          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, zp, has_condition,
                                                   velocity_cond );
          }

          Sv[c] += 2.0 * area_tb * ( mu_t + viscos )
            * ( velocity_cond[1] - vvel[c] ) / dz;

        }
      }
      //apply w-mom bc -
      if ( eps[zm] * eps[c] > 0.5 ) {
        // X-
        if ( eps[xm] * eps[xmzm] < .5 ){
          const double i_so = ( eps[c+x_so] * eps[zm+x_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(i_so,0,0)] + IsImag[c+IntVector(i_so,0,-1)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;

          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, c, has_condition,
                                                   velocity_cond );
          }

          Sw[c] += 2.0 * area_ew * ( mu_t + viscos )
            * ( velocity_cond[2] - wvel[c] ) / dx;

        }
        // X+
        if ( eps[xp] * eps[xpzm] < .5 ){
          const double i_so = ( eps[c-x_so] * eps[zm-x_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(-i_so,0,0)] + IsImag[c+IntVector(-i_so,0,-1)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;

          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, xp, has_condition,
                                                   velocity_cond );
          }

          Sw[c] += 2.0 * area_ew * ( mu_t + viscos )
            * ( velocity_cond[2] - wvel[c] ) / dx;

        }
        // Y-
        if ( eps[ym] * eps[ymzm] < .5 ){
          const double i_so = ( eps[c+y_so] * eps[zm+y_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5 * ( IsImag[c+IntVector(0,i_so,0)] + IsImag[c+IntVector(0,i_so,-1)] );
          const double mu_t = pow( csmag_wall * delta, 2.0 ) * rho[c] * ISI;

          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, c, has_condition,
                                                   velocity_cond );
          }

          Sw[c] += 2.0 * area_ns * ( mu_t + viscos )
            * ( velocity_cond[2] - wvel[c] ) / dy;

        }
        // Y+
        if ( eps[yp] * eps[ypzm] < .5 ){
          const double i_so = ( eps[c-y_so] * eps[zm-y_so] > .5 ) ?
                               standoff :
                               0;
          const double ISI = 0.5*(IsImag[c+IntVector(0,-i_so,0)] + IsImag[c+IntVector(0,-i_so,-1)]);
          const double mu_t = pow( csmag_wall*delta, 2.0 ) * rho[c] * ISI;

          Vector velocity_cond(0,0,0);
          if ( has_intrusion_inlets ){
            bool has_condition = false;
            _intrusionBC[ilvl]->getVelocityCondition( p, yp, has_condition,
                                                   velocity_cond );
          }

          Sw[c] += 2.0 * area_ns * ( mu_t + viscos )
            * ( velocity_cond[2] - wvel[c] ) / dy;

        }
      }
    }// end cell loop
  } // If strainRate exists
}// end function

void
BoundaryCondition::wallStressLog( const Patch* patch,
                               ArchesVariables* vars,
                               ArchesConstVariables* constvars,
                               constCCVariable<double>& volFraction )
{

  Vector Dx = patch->dCell();
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
     int flow=-1;
     double NonLinearX =0.0;
     double NonLinearY =0.0;
     double NonLinearZ =0.0;
     double yplusCrit=11.63;
     double molvis=0.0;
     double densityup=0.0;

     constCCVariable<double > uctr=constvars->CCUVelocity;
     constCCVariable<double > vctr=constvars->CCVVelocity;
     constCCVariable<double > wctr=constvars->CCWVelocity;
     constSFCXVariable<double > ufac=constvars->uVelocity;
     constSFCYVariable<double > vfac=constvars->vVelocity;
     constSFCZVariable<double > wfac=constvars->wVelocity;

     constCCVariable<double> Gasdensity=constvars->density;
     molvis=d_physicalConsts->getMolecularViscosity();
     //constCCVariable<double> molvis=constvars->viscosity;
                              // need to see it

     double UMtotal=std::sqrt(uctr(i,j,k)*uctr(i,j,k)+vctr(i,j,k)*vctr(i,j,k) +wctr(i,j,k)*wctr(i,j,k));

     //-------------------------X direction ----------------------------------
     // for the log-log velocity distribution to solve the utau => wall stress

    // x- direction

      bool If_wallxmVd=  ( volFraction(i-1,j,k) *volFraction(i-1,j-1,k)<0.5);
      // vcell at x- direction wall
     if( If_wallxmVd && volFraction(i,j,k)*volFraction(i,j-1,k) > 0.5 ){

       const double XmVtot=0.5*std::sqrt((uctr(i+1,j,k)+uctr(i+1,j-1,k))* (uctr(i+1,j,k)+uctr(i+1,j-1,k))
                                       +(vctr(i+1,j,k)+vctr(i+1,j-1,k))* (vctr(i+1,j,k)+vctr(i+1,j-1,k))
                                       +(wctr(i+1,j,k)+wctr(i+1,j-1,k))* (wctr(i+1,j,k)+wctr(i+1,j-1,k)));

      densityup=0.5*Gasdensity(i+1,j,k)+0.5*Gasdensity(i+1,j-1,k);

      double XmVtauGuess=yplusCrit*molvis/densityup/Dx.x();

      newton_solver(XmVtot,Dx.x()*1.5,densityup, molvis, XmVtauGuess);

      double tauXmVmom= ( XmVtauGuess*densityup*Dx.x()*1.5/ molvis < yplusCrit) ?     molvis*vfac(i+1,j,k)/1.5/Dx.x()      : XmVtauGuess*XmVtauGuess*(0.5*Gasdensity(i+1,j,k)+0.5*Gasdensity(i+1,j-1,k));

      NonLinearY = (If_wallxmVd && constvars->cellType(i,j-1,k)== flow && XmVtot!=0) ?  NonLinearY- Dx.z() * Dx.y() * tauXmVmom* vfac(i+1,j,k)/XmVtot  : NonLinearY;
      }


      // wcell at x- direction wall
      bool If_wallxmWd=  ( volFraction(i-1,j,k) *volFraction(i-1,j,k-1)<0.5);

      if (If_wallxmWd && volFraction(i,j,k)*volFraction(i,j,k-1) > 0.5) {

        const double XmWtot=0.5*std::sqrt((uctr(i+1,j,k)+uctr(i+1,j,k-1))* (uctr(i+1,j,k)+uctr(i+1,j,k-1))
                                       +(vctr(i+1,j,k)+vctr(i+1,j,k-1))* (vctr(i+1,j,k)+vctr(i+1,j,k-1))
                                       +(wctr(i+1,j,k)+wctr(i+1,j,k-1))* (wctr(i+1,j,k)+wctr(i+1,j,k-1)));

      densityup=0.5*Gasdensity(i+1,j,k)+0.5*Gasdensity(i+1,j,k-1);

       double    XmWtauGuess=yplusCrit*molvis/densityup/Dx.x();

       newton_solver(XmWtot,Dx.x()*1.5,densityup, molvis, XmWtauGuess);

      double tauXmWmom= ( XmWtauGuess*densityup*Dx.x()*1.5/ molvis < yplusCrit)   ?     molvis*wfac(i+1,j,k)/1.5/Dx.x()   : XmWtauGuess* XmWtauGuess*(0.5*Gasdensity(i+1,j,k)+0.5*Gasdensity(i+1,j,k-1));

      NonLinearZ = (If_wallxmWd && constvars->cellType(i,j,k-1)== flow&& XmWtot!=0) ?  NonLinearZ- Dx.z() * Dx.y() * tauXmWmom*wfac(i+1,j,k)/XmWtot  : NonLinearZ;
        }

      // x+ -----------------------------------------------------------------------------direction

      bool If_wallxpVd=  ( volFraction(i+1,j,k) *volFraction(i+1,j-1,k)<0.5);
      // vcell at x+ direction wall

      if(If_wallxpVd && volFraction(i,j,k)*volFraction(i,j-1,k) > 0.5 ){

      const double XpVtot=0.5*std::sqrt((uctr(i-1,j,k)+uctr(i-1,j-1,k))* (uctr(i-1,j,k)+uctr(i-1,j-1,k))
                                       +(vctr(i-1,j,k)+vctr(i-1,j-1,k))* (vctr(i-1,j,k)+vctr(i-1,j-1,k))
                                       +(wctr(i-1,j,k)+wctr(i-1,j-1,k))* (wctr(i+1,j,k)+wctr(i+1,j-1,k)));
      densityup=0.5*Gasdensity(i-1,j,k)+0.5*Gasdensity(i-1,j-1,k);

      double XpVtauGuess=yplusCrit*molvis/densityup/Dx.x();

      newton_solver(XpVtot,Dx.x()*1.5,densityup, molvis,XpVtauGuess);

      double tauXpVmom=  ( XpVtauGuess*densityup*Dx.x()*1.5/ molvis < yplusCrit) ?     molvis*vfac(i-1,j,k)/1.5/Dx.x()        : XpVtauGuess*XpVtauGuess*(0.5*Gasdensity(i-1,j,k)+0.5*Gasdensity(i-1,j-1,k));

      NonLinearY = (If_wallxpVd && constvars->cellType(i,j-1,k)== flow && XpVtot!=0) ?  NonLinearY- Dx.z() * Dx.y() * tauXpVmom* vfac(i-1,j,k)/XpVtot  : NonLinearY;
      }

      // Wcell at x+ direction wall
      bool If_wallxpWd=  ( volFraction(i+1,j,k) *volFraction(i+1,j,k-1)<0.5);

      if(If_wallxpWd && volFraction(i,j,k)*volFraction(i,j,k-1) > 0.5){

      const double XpWtot=0.5*std::sqrt((uctr(i-1,j,k)+uctr(i-1,j,k-1))* (uctr(i-1,j,k)+uctr(i-1,j,k-1))
                                       +(vctr(i-1,j,k)+vctr(i-1,j,k-1))* (vctr(i-1,j,k)+vctr(i-1,j,k-1))
                                       +(wctr(i-1,j,k)+wctr(i-1,j,k-1))* (wctr(i-1,j,k)+wctr(i-1,j,k-1)));
      densityup=0.5*Gasdensity(i-1,j,k)+0.5*Gasdensity(i-1,j,k-1);

      double      XpWtauGuess=yplusCrit*molvis/densityup/Dx.x();

      newton_solver(XpWtot,Dx.x()*1.5,densityup, molvis,XpWtauGuess);

      double tauXpWmom= ( XpWtauGuess*densityup*Dx.x()*1.5/ molvis < yplusCrit) ?     molvis*wfac(i-1,j,k)/1.5/Dx.x()    : XpWtauGuess*XpWtauGuess*0.5*(Gasdensity(i-1,j,k)+0.5*Gasdensity(i-1,j,k-1));

      NonLinearZ = (If_wallxpWd && constvars->cellType(i,j,k-1)== flow && XpWtot!=0) ?  NonLinearZ- Dx.z() * Dx.y() *  tauXpWmom*wfac(i-1,j,k)/XpWtot: NonLinearZ;
      }

   // y- direction
 // ucell at y-direction
    bool If_wallymUd= ( volFraction(i,j-1,k) *volFraction(i-1,j-1,k)<0.5);

    if(If_wallymUd && volFraction(i,j,k)*volFraction(i-1,j,k) >0.5){

    const double YmUtot=0.5*std::sqrt((uctr(i,j+1,k)+uctr(i-1,j+1,k))* (uctr(i,j+1,k)+uctr(i-1,j+1,k))
                                     +(vctr(i,j+1,k)+vctr(i-1,j+1,k))* (vctr(i,j+1,k)+vctr(i-1,j+1,k))
                                     +(wctr(i,j+1,k)+wctr(i-1,j+1,k))* (wctr(i,j+1,k)+wctr(i-1,j+1,k)));

    densityup=0.5*Gasdensity(i,j+1,k)+0.5*Gasdensity(i-1,j+1,k);

    double YmUtauGuess=yplusCrit*molvis/densityup/Dx.y();

    newton_solver(YmUtot,Dx.y()*1.5, densityup, molvis,YmUtauGuess);

    double tauYmUmom=   ( YmUtauGuess*densityup*Dx.y()*1.5/ molvis < yplusCrit) ?     molvis*ufac(i,j+1,k)/1.5/Dx.y()  :  YmUtauGuess*YmUtauGuess*(0.5*Gasdensity(i,j+1,k)+0.5*Gasdensity(i-1,j+1,k));

    NonLinearX = (If_wallymUd && constvars->cellType(i-1,j,k)== flow&& YmUtot!=0) ?  NonLinearX- Dx.x() * Dx.z() *  tauYmUmom*ufac(i,j+1,k)/YmUtot : NonLinearX;


    }


 // wcell at y-direction
    bool If_wallymWd= ( volFraction(i,j-1,k) *volFraction(i,j-1,k-1)<0.5);

    if(If_wallymWd && volFraction(i,j,k)*volFraction(i,j,k-1) >0.5 ){

    const double YmWtot=0.5*std::sqrt((uctr(i,j+1,k)+uctr(i,j+1,k-1))* (uctr(i,j+1,k)+uctr(i,j+1,k-1))
                                     +(vctr(i,j+1,k)+vctr(i,j+1,k-1))* (vctr(i,j+1,k)+vctr(i,j+1,k-1))
                                     +(wctr(i,j+1,k)+wctr(i,j+1,k-1))* (wctr(i,j+1,k)+wctr(i,j+1,k-1)) );

    densityup=0.5*Gasdensity(i,j+1,k)+0.5*Gasdensity(i,j+1,k-1);

    double     YmWtauGuess=yplusCrit*molvis/densityup/Dx.y();

      newton_solver(YmWtot,Dx.y()*1.5, densityup, molvis,YmWtauGuess);

    double tauYmWmom=  ( YmWtauGuess*densityup*Dx.y()*1.5/ molvis < yplusCrit) ?     molvis*wfac(i,j+1,k)/1.5/Dx.y()  :   YmWtauGuess*YmWtauGuess*(0.5*Gasdensity(i,j+1,k)+0.5*Gasdensity(i,j+1,k-1));

     NonLinearZ = (If_wallymWd && constvars->cellType(i,j,k-1)== flow && YmWtot !=0 ) ?  NonLinearZ- Dx.x() * Dx.z() *  tauYmWmom*wfac(i,j+1,k)/YmWtot : NonLinearZ;
    }

    // y+ direction
 // ucell at y+direction
    bool If_wallypUd= ( volFraction(i,j+1,k) *volFraction(i-1,j+1,k)<0.5);

    if(If_wallypUd && volFraction(i,j,k)*volFraction(i-1,j,k) >0.5 ){

      const double YpUtot=0.5*std::sqrt((uctr(i,j-1,k)+uctr(i-1,j-1,k))* (uctr(i,j-1,k)+uctr(i-1,j-1,k))
                                     +(vctr(i,j-1,k)+vctr(i-1,j-1,k))* (vctr(i,j-1,k)+vctr(i-1,j-1,k))
                                     +(wctr(i,j-1,k)+wctr(i-1,j-1,k))* (wctr(i,j-1,k)+wctr(i-1,j-1,k)));

      densityup=0.5*Gasdensity(i,j-1,k)+0.5*Gasdensity(i-1,j-1,k);

    double  YpVtauGuess=yplusCrit*molvis/densityup/Dx.y();

       newton_solver(YpUtot,Dx.y()*1.5, densityup,molvis,YpVtauGuess) ;

     double tauYpUmom=  ( YpVtauGuess*densityup*Dx.y()*1.5/ molvis < yplusCrit) ?     molvis*ufac(i,j-1,k)/1.5/Dx.y()  :   YpVtauGuess*YpVtauGuess*(0.5*Gasdensity(i,j-1,k)+0.5*Gasdensity(i-1,j-1,k));

     NonLinearX = (If_wallypUd && constvars->cellType(i-1,j,k)== flow && YpUtot !=0) ?  NonLinearX- Dx.x() * Dx.z() *  tauYpUmom*ufac(i,j-1,k)/YpUtot : NonLinearX;



    }


 // wcell at y+direction
    bool If_wallypWd= ( volFraction(i,j+1,k) *volFraction(i,j+1,k-1)<0.5);

    if(If_wallypWd && volFraction(i,j,k)*volFraction(i,j,k-1) > 0.5 ){

      const double YpWtot=0.5*std::sqrt((uctr(i,j-1,k)+uctr(i,j-1,k-1))* (uctr(i,j-1,k)+uctr(i,j-1,k-1))
                                     +(vctr(i,j-1,k)+vctr(i,j-1,k-1))* (vctr(i,j-1,k)+vctr(i,j-1,k-1))
                                     +(wctr(i,j-1,k)+wctr(i,j-1,k-1))* (wctr(i,j-1,k)+wctr(i,j-1,k-1)) );

    densityup=0.5*Gasdensity(i,j-1,k)+0.5*Gasdensity(i,j-1,k-1);

    double YpWtauGuess=yplusCrit*molvis/densityup/Dx.y();

    newton_solver(YpWtot,Dx.y()*1.5,densityup, molvis,YpWtauGuess) ;

    double tauYpWmom=   ( YpWtauGuess*densityup*Dx.y()*1.5/ molvis < yplusCrit) ?     molvis*wfac(i,j-1,k)/1.5/Dx.y() :   YpWtauGuess*YpWtauGuess*(0.5*Gasdensity(i,j-1,k)+0.5*Gasdensity(i,j-1,k-1));

     NonLinearZ = (If_wallypWd && constvars->cellType(i,j,k-1)== flow && YpWtot !=0) ?  NonLinearZ- Dx.x() * Dx.z() *  tauYpWmom*wfac(i,j-1,k)/YpWtot : NonLinearZ;


     }


   //-------------------------Z direction ---------------------------------

   // z- direction
   // ucell  z-direction
   bool  If_wallzmUd=  ( volFraction(i,j,k-1) *volFraction(i-1,j,k-1)<0.5);

   if(If_wallzmUd && volFraction(i,j,k)*volFraction(i-1,j,k) > 0.5){

       const double ZmUtot=0.5*std::sqrt((uctr(i,j,k+1)+uctr(i-1,j,k+1))* (uctr(i,j,k+1)+uctr(i-1,j,k+1))
                                     +(vctr(i,j,k+1)+vctr(i-1,j,k+1))* (vctr(i,j,k+1)+vctr(i-1,j,k+1))
                                     +(wctr(i,j,k+1)+wctr(i-1,j,k+1))* (wctr(i,j,k+1)+wctr(i-1,j,k+1)));
   densityup=0.5*Gasdensity(i,j,k+1)+0.5*Gasdensity(i-1,j,k+1);

   double ZmUtauGuess=yplusCrit*molvis/densityup/Dx.z();

   newton_solver(ZmUtot,Dx.z()*1.5,densityup,molvis,ZmUtauGuess) ;
    double tauZmUmom=   ( ZmUtauGuess*densityup*Dx.z()*1.5/ molvis < yplusCrit) ?     molvis*ufac(i,j,k+1)/1.5/Dx.z() :   ZmUtauGuess*ZmUtauGuess*(0.5*Gasdensity(i,j,k+1)+0.5*Gasdensity(i-1,j,k+1));

    NonLinearX = (If_wallzmUd && constvars->cellType(i-1,j,k)== flow && ZmUtot !=0) ?  NonLinearX- Dx.x() * Dx.y() *  tauZmUmom*ufac(i,j,k+1)/ZmUtot : NonLinearX;
    }

   // vcell  z-direction
   bool  If_wallzmVd=  ( volFraction(i,j,k-1) *volFraction(i,j-1,k-1)<0.5);

    if(If_wallzmVd && volFraction(i,j,k)*volFraction(i,j-1,k) > 0.5){

      const double ZmVtot=0.5*std::sqrt((uctr(i,j,k+1)+uctr(i,j-1,k+1))* (uctr(i,j,k+1)+uctr(i,j-1,k+1))
                                    +(vctr(i,j,k+1)+vctr(i,j-1,k+1))* (vctr(i,j,k+1)+vctr(i,j-1,k+1))
                                    +(wctr(i,j,k+1)+wctr(i,j-1,k+1))* (wctr(i,j,k+1)+wctr(i,j-1,k+1)));

    densityup=0.5*Gasdensity(i,j-1,k+1)+0.5*Gasdensity(i,j,k+1);

    double   ZmVtauGuess=yplusCrit*molvis/densityup/Dx.z();

    newton_solver(ZmVtot,Dx.z()*1.5,densityup,molvis,ZmVtauGuess) ;
    double tauZmVmom=   ( ZmVtauGuess*densityup*Dx.z()*1.5/ molvis < yplusCrit) ?     molvis*vfac(i,j,k+1)/1.5/Dx.z() :    ZmVtauGuess*ZmVtauGuess*(0.5*Gasdensity(i,j,k+1)+0.5*Gasdensity(i,j-1,k+1));

     NonLinearY = (If_wallzmVd && constvars->cellType(i,j-1,k)== flow && ZmVtot !=0) ?  NonLinearY- Dx.x() * Dx.y() *  tauZmVmom*vfac(i,j,k+1)/ZmVtot : NonLinearY;
    }

    // z+ direction
   // ucell  z+direction
   bool  If_wallzpUd=  ( volFraction(i,j,k+1) *volFraction(i-1,j,k+1)<0.5);

   if(If_wallzpUd && volFraction(i,j,k)*volFraction(i-1,j,k) > 0.5 ){

     const double ZpUtot=0.5*std::sqrt((uctr(i,j,k-1)+uctr(i-1,j,k-1))* (uctr(i,j,k-1)+uctr(i-1,j,k-1))
                                     +(vctr(i,j,k-1)+vctr(i-1,j,k-1))* (vctr(i,j,k-1)+vctr(i-1,j,k-1))
                                     +(wctr(i,j,k-1)+wctr(i-1,j,k-1))* (wctr(i,j,k-1)+wctr(i-1,j,k-1)));
    densityup=0.5*Gasdensity(i,j,k-1)+0.5*Gasdensity(i-1,j,k-1);

      double      ZpUtauGuess=yplusCrit*molvis/densityup/Dx.z();

   newton_solver(ZpUtot,Dx.z()*1.5,densityup,molvis,ZpUtauGuess) ;

    double tauZpUmom=   ( ZpUtauGuess*densityup*Dx.z()*1.5/ molvis < yplusCrit) ?     molvis*ufac(i,j,k-1)/1.5/Dx.z() :    ZpUtauGuess*ZpUtauGuess*(0.5*Gasdensity(i,j,k-1)+0.5*Gasdensity(i-1,j,k-1));

    NonLinearX = (If_wallzpUd && constvars->cellType(i-1,j,k)== flow && ZpUtot !=0) ?  NonLinearX- Dx.x() * Dx.y() *  tauZpUmom*ufac(i,j,k-1)/ZpUtot : NonLinearX;
   }

   // vcell  z+direction
   bool  If_wallzpVd=  ( volFraction(i,j,k+1) *volFraction(i,j-1,k+1)<0.5);

  if(If_wallzpVd && volFraction(i,j,k)*volFraction(i,j-1,k)  > 0.5){

    const double ZpVtot=0.5*std::sqrt((uctr(i,j,k-1)+uctr(i,j-1,k-1))* (uctr(i,j,k-1)+uctr(i,j-1,k-1))
                                    +(vctr(i,j,k-1)+vctr(i,j-1,k-1))* (vctr(i,j,k-1)+vctr(i,j-1,k-1))
                                    +(wctr(i,j,k-1)+wctr(i,j-1,k-1))* (wctr(i,j,k-1)+wctr(i,j-1,k-1)));

    densityup=0.5*Gasdensity(i,j-1,k-1)+0.5*Gasdensity(i,j,k-1);

    double   ZpVtauGuess=yplusCrit*molvis/densityup/Dx.z();

       newton_solver(ZpVtot,Dx.z()*1.5, densityup, molvis,ZpVtauGuess) ;

       double tauZpVmom=   ( ZpVtauGuess*densityup*Dx.z()*1.5/ molvis < yplusCrit) ?   molvis*vfac(i,j,k-1)/1.5/Dx.z() :    ZpVtauGuess*ZpVtauGuess*(0.5*Gasdensity(i,j,k-1)+0.5*Gasdensity(i,j-1,k-1));

     NonLinearY = (If_wallzpVd && constvars->cellType(i,j-1,k)== flow && ZpVtot) ?  NonLinearY- Dx.x() * Dx.y() *  tauZpVmom*vfac(i,j,k-1)/ZpVtot : NonLinearY;


  }

    vars->uVelNonlinearSrc(i,j,k)=(!d_slip && constvars->cellType(i,j,k)==flow && UMtotal != 0.0) ? vars->uVelNonlinearSrc(i,j,k)+ NonLinearX :  vars->uVelNonlinearSrc(i,j,k);
    vars->vVelNonlinearSrc(i,j,k)=(!d_slip && constvars->cellType(i,j,k)==flow && UMtotal != 0.0) ? vars->vVelNonlinearSrc(i,j,k)+ NonLinearY :  vars->vVelNonlinearSrc(i,j,k);
    vars->wVelNonlinearSrc(i,j,k)=(!d_slip && constvars->cellType(i,j,k)==flow && UMtotal != 0.0) ? vars->wVelNonlinearSrc(i,j,k)+ NonLinearZ :  vars->wVelNonlinearSrc(i,j,k);

  });

}
//----------------------------------
void
BoundaryCondition::newton_solver( const double& up, const double& yp, const double& density, const double& viscosity,double& utau)
{
  // solver constants
  const double d_tol    = 1e-10;
  const double E=9.8;
  const double kappa=0.42;
  const double A=E*density*yp/viscosity;
  bool        converged=false;
  double      fprime=0.0;
  double      f=0.0;
  double      wrk=0.0;
  double      df=0.0;
  double  uinit= utau;
  // newton solve
  for ( int iterT=0; iterT < 50; iterT++) {
    wrk=log(A*utau);
    fprime=-(1.0+wrk);
    f=kappa*up-utau*wrk;
    df=f/fprime;
    utau=utau-df;
     ///std::cout<< "iterT="<<iterT<<"utau="<<utau<<"A="<<A<<"\n";
    if (std::abs(df) < d_tol){
      converged =true;
      break;
      }
    }// end for solver looping process

  if (!converged && std::abs(up)> 1e-5 && density != 0.0 && viscosity !=0)
  { std::cout<<"diverge for solving log wall stress, not converged"<<std::endl;
    std::cout<<"up="<<up<<"yp="<<yp<<"density="<<density<<"visocisty="<<viscosity<<"utau="<<utau<< " uinit ="<< uinit<<std::endl;
  }

}

void
BoundaryCondition::wallStressMolecular( const Patch* p,
                                        constSFCXVariable<double>& uvel,
                                        constSFCYVariable<double>& vvel,
                                        constSFCZVariable<double>& wvel,
                                        SFCXVariable<double>& Su,
                                        SFCYVariable<double>& Sv,
                                        SFCZVariable<double>& Sw,
                                        constCCVariable<double>& eps )
{

  Vector Dx = p->dCell();

  double viscos; // molecular viscosity
  viscos = d_physicalConsts->getMolecularViscosity();
  const double area_ew = Dx.y() * Dx.z();
  const double area_ns = Dx.z() * Dx.x();
  const double area_tb = Dx.x() * Dx.y();
  const double dx = Dx.x();
  const double dy = Dx.y();
  const double dz = Dx.z();

  for (CellIterator iter=p->getCellIterator(); !iter.done(); iter++) {
    IntVector c = *iter;
    IntVector xm = *iter - IntVector(1,0,0);
    IntVector xp = *iter + IntVector(1,0,0);
    IntVector ym = *iter - IntVector(0,1,0);
    IntVector yp = *iter + IntVector(0,1,0);
    IntVector zm = *iter - IntVector(0,0,1);
    IntVector zp = *iter + IntVector(0,0,1);
    IntVector xmym = *iter + IntVector(-1,-1,0);
    IntVector xmyp = *iter + IntVector(-1,1,0);
    IntVector xmzm = *iter + IntVector(-1,0,-1);
    IntVector xmzp = *iter + IntVector(-1,0,1);
    IntVector xpym = *iter + IntVector(1,-1,0);
    IntVector xpzm = *iter + IntVector(1,0,-1);
    IntVector ymzm = *iter + IntVector(0,-1,-1);
    IntVector ymzp = *iter + IntVector(0,-1,1);
    IntVector ypzm = *iter + IntVector(0,1,-1);

    //apply u-mom bc -
    if ( eps[xm] * eps[c] > .5 ){
      // Y-
      if ( eps[ym] * eps[xmym] < .5 ){
        Su[c] -= 2.0 * area_ns * ( viscos ) * uvel[c] / dy;
      }
      // Y+
      if ( eps[yp] * eps[xmyp] < .5 ){
        Su[c] -= 2.0 * area_ns * ( viscos ) * uvel[c] / dy;
      }
      // Z-
      if ( eps[zm] * eps[xmzm] < .5 ){
        Su[c] -= 2.0 * area_tb * ( viscos ) * uvel[c] / dz;
      }
      // Z+
      if ( eps[zp] * eps[xmzp] < .5 ){
        Su[c] -= 2.0 * area_tb * ( viscos ) * uvel[c] / dy;
      }
    }
    //apply v-mom bc -
    if ( eps[ym] * eps[c] > 0.5 ) {
      // X-
      if ( eps[xm] * eps[xmym] < .5 ){
        Sv[c] -= 2.0 * area_ew * ( viscos ) * vvel[c] / dx;
      }
      // X+
      if ( eps[xp] * eps[xpym] < .5 ){
        Sv[c] -= 2.0 * area_ew * ( viscos ) * vvel[c] / dx;
      }
      // Z-
      if ( eps[zm] * eps[ymzm] < .5 ){
        Sv[c] -= 2.0 * area_tb * ( viscos ) * vvel[c] / dz;
      }
      // Z+
      if ( eps[zp] * eps[ymzp] < .5 ){
        Sv[c] -= 2.0 * area_tb * ( viscos ) * vvel[c] / dz;
      }
    }
    //apply w-mom bc -
    if ( eps[zm] * eps[c] > 0.5 ) {
      // X-
      if ( eps[xm] * eps[xmzm] < .5 ){
        Sw[c] -= 2.0 * area_ew * ( viscos ) * wvel[c] / dx;
      }
      // X+
      if ( eps[xp] * eps[xpzm] < .5 ){
        Sw[c] -= 2.0 * area_ew * ( viscos ) * wvel[c] / dx;
      }
      // Y-
      if ( eps[ym] * eps[ymzm] < .5 ){
        Sw[c] -= 2.0 * area_ns * ( viscos ) * wvel[c] / dy;
      }
      // Y+
      if ( eps[yp] * eps[ypzm] < .5 ){
        Sw[c] -= 2.0 * area_ns * ( viscos ) * wvel[c] / dy;
      }
    }
  }// end cell loop
}// end function

void
BoundaryCondition::wallStressDynSmag( const Patch* p,
                                      const int standoff,
                                      constCCVariable<double>& mu_t,
                                      constSFCXVariable<double>& uvel,
                                      constSFCYVariable<double>& vvel,
                                      constSFCZVariable<double>& wvel,
                                      SFCXVariable<double>& Su,
                                      SFCYVariable<double>& Sv,
                                      SFCZVariable<double>& Sw,
                                      constCCVariable<double>& rho,
                                      constCCVariable<double>& eps )
{

  // mu_t coming in here has the mol visc added.
  //
  Vector Dx = p->dCell();

  const double area_ew = Dx.y() * Dx.z();
  const double area_ns = Dx.z() * Dx.x();
  const double area_tb = Dx.x() * Dx.y();
  const double dx = Dx.x();
  const double dy = Dx.y();
  const double dz = Dx.z();

  for (CellIterator iter=p->getCellIterator(); !iter.done(); iter++) {
    IntVector c = *iter;
    IntVector xm = *iter - IntVector(1,0,0);
    IntVector xp = *iter + IntVector(1,0,0);
    IntVector ym = *iter - IntVector(0,1,0);
    IntVector yp = *iter + IntVector(0,1,0);
    IntVector zm = *iter - IntVector(0,0,1);
    IntVector zp = *iter + IntVector(0,0,1);
    IntVector xmym = *iter + IntVector(-1,-1,0);
    IntVector xmyp = *iter + IntVector(-1,1,0);
    IntVector xmzm = *iter + IntVector(-1,0,-1);
    IntVector xmzp = *iter + IntVector(-1,0,1);
    IntVector xpym = *iter + IntVector(1,-1,0);
    IntVector xpzm = *iter + IntVector(1,0,-1);
    IntVector ymzm = *iter + IntVector(0,-1,-1);
    IntVector ymzp = *iter + IntVector(0,-1,1);
    IntVector ypzm = *iter + IntVector(0,1,-1);
    IntVector x_so = IntVector(standoff,0,0);
    IntVector y_so = IntVector(0,standoff,0);
    IntVector z_so = IntVector(0,0,standoff);
    //apply u-mom bc -
    if ( eps[xm] * eps[c] > .5 ){
      // Y-
      if ( eps[ym] * eps[xmym] < .5 ){
        const double i_so = ( eps[c+y_so] * eps[xm+y_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(0,i_so,0)] + mu_t[c+IntVector(-1,i_so,0)] );
        Su[c] -= 2.0 * area_ns * ( mu ) * uvel[c] / dy;
      }
      // Y+
      if ( eps[yp] * eps[xmyp] < .5 ){
        const double i_so = ( eps[c-y_so] * eps[xm-y_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5*(mu_t[c+IntVector(0,-i_so,0)] + mu_t[c+IntVector(-1,-i_so,0)]);
        Su[c] -= 2.0 * area_ns * ( mu ) * uvel[c] / dy;
      }
      // Z-
      if ( eps[zm] * eps[xmzm] < .5 ){
        const double i_so = ( eps[c+z_so] * eps[xm+z_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(0,0,i_so)] + mu_t[c+IntVector(-1,0,i_so)] );
        Su[c] -= 2.0 * area_tb * ( mu ) * uvel[c] / dz;
      }
      // Z+
      if ( eps[zp] * eps[xmzp] < .5 ){
        const double i_so = ( eps[c-z_so] * eps[xm-z_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5*(mu_t[c+IntVector(0,0,-i_so)] + mu_t[c+IntVector(-1,0,-i_so)]);
        Su[c] -= 2.0 * area_tb * ( mu ) * uvel[c] / dy;
      }
    }
    //apply v-mom bc -
    if ( eps[ym] * eps[c] > 0.5 ) {
      // X-
      if ( eps[xm] * eps[xmym] < .5 ){
        const double i_so = ( eps[c+x_so] * eps[ym+x_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(i_so,0,0)] + mu_t[c+IntVector(i_so,-1,0)] );
        Sv[c] -= 2.0 * area_ew * ( mu ) * vvel[c] / dx;
      }
      // X+
      if ( eps[xp] * eps[xpym] < .5 ){
        const double i_so = ( eps[c-x_so] * eps[ym-x_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(-i_so,0,0)] + mu_t[c+IntVector(-i_so,-1,0)] );
        Sv[c] -= 2.0 * area_ew * ( mu ) * vvel[c] / dx;
      }
      // Z-
      if ( eps[zm] * eps[ymzm] < .5 ){
        const double i_so = ( eps[c+z_so] * eps[ym+z_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(0,0,i_so)] + mu_t[c+IntVector(0,-1,i_so)] );
        Sv[c] -= 2.0 * area_tb * ( mu ) * vvel[c] / dz;
      }
      // Z+
      if ( eps[zp] * eps[ymzp] < .5 ){
        const double i_so = ( eps[c-z_so] * eps[ym-z_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(0,0,-i_so)] + mu_t[c+IntVector(0,-1,-i_so)] );
        Sv[c] -= 2.0 * area_tb * ( mu ) * vvel[c] / dz;
      }
    }
    //apply w-mom bc -
    if ( eps[zm] * eps[c] > 0.5 ) {
      // X-
      if ( eps[xm] * eps[xmzm] < .5 ){
        const double i_so = ( eps[c+x_so] * eps[zm+x_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(i_so,0,0)] + mu_t[c+IntVector(i_so,0,-1)] );
        Sw[c] -= 2.0 * area_ew * ( mu ) * wvel[c] / dx;
      }
      // X+
      if ( eps[xp] * eps[xpzm] < .5 ){
        const double i_so = ( eps[c-x_so] * eps[zm-x_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(-i_so,0,0)] + mu_t[c+IntVector(-i_so,0,-1)] );
        Sw[c] -= 2.0 * area_ew * ( mu ) * wvel[c] / dx;
      }
      // Y-
      if ( eps[ym] * eps[ymzm] < .5 ){
        const double i_so = ( eps[c+y_so] * eps[zm+y_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5 * ( mu_t[c+IntVector(0,i_so,0)] + mu_t[c+IntVector(0,i_so,-1)] );
        Sw[c] -= 2.0 * area_ns * ( mu ) * wvel[c] / dy;
      }
      // Y+
      if ( eps[yp] * eps[ypzm] < .5 ){
        const double i_so = ( eps[c-y_so] * eps[zm-y_so] > .5 ) ?
                             standoff :
                             0;
        const double mu = 0.5*(mu_t[c+IntVector(0,-i_so,0)] + mu_t[c+IntVector(0,-i_so,-1)]);
        Sw[c] -= 2.0 * area_ns * ( mu ) * wvel[c] / dy;
      }
    }
  }// end cell loop
}// end function

void
BoundaryCondition::sched_checkMomBCs( SchedulerP& sched, const LevelP& level, const MaterialSet* matls )
{
  string taskname = "BoundaryCondition::checkMomBCs";
  Task* tsk = scinew Task(taskname, this, &BoundaryCondition::checkMomBCs );

  sched->addTask( tsk, level->eachPatch(), matls );
}

void
BoundaryCondition::checkMomBCs( const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    const Vector Dx = patch->dCell();
    double dx=0, dy=0;

    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);
    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++) {
      Patch::FaceType face = *bf_iter;

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++) {

        for ( std::vector<std::string>::iterator iname = d_all_v_inlet_names.begin();
              iname != d_all_v_inlet_names.end(); iname++ ) {

          std::string bc_s_value = "NA";

          Iterator bound_ptr;
          string bc_kind = "NotSet";
          string face_name;

          getBCKind( patch, face, child, *iname, matlIndex, bc_kind, face_name );

          std::ofstream outputfile;
          std::stringstream fname;
          fname << "handoff_velocity_" << face_name <<  "." << patch->getID();
          bool file_is_open = false;


          string whichface;
          int index=0;

          if (face == 0) {
            whichface = "x-";
            index = 0;
            dx = Dx[1];
            dy = Dx[2];
          } else if (face == 1) {
            whichface = "x+";
            index = 0;
            dx = Dx[1];
            dy = Dx[2];
          } else if (face == 2) {
            whichface = "y-";
            index = 1;
            dx = Dx[2];
            dy = Dx[0];
          } else if (face == 3) {
            whichface = "y+";
            index = 1;
            dx = Dx[2];
            dy = Dx[0];
          } else if (face == 4) {
            whichface = "z-";
            index = 2;
            dx = Dx[0];
            dy = Dx[1];
          } else if (face == 5) {
            whichface = "z+";
            index = 2;
            dx = Dx[0];
            dy = Dx[1];
          }

          // need to map x,y,z -> i,j,k for the FromFile option
          bool foundIterator = false;
          if ( bc_kind == "VelocityFileInput" ) {
            foundIterator =
                    getIteratorBCValue<std::string>( patch, face, child, *iname, matlIndex, bc_s_value, bound_ptr);
          }

          BoundaryCondition::FaceToInput::iterator i_uvel_bc_storage = _u_input.find( face_name );
          BoundaryCondition::FaceToInput::iterator i_vvel_bc_storage = _v_input.find( face_name );
          BoundaryCondition::FaceToInput::iterator i_wvel_bc_storage = _w_input.find( face_name );

          //check the grid spacing:
          if ( i_uvel_bc_storage != _u_input.end() ) {
            proc0cout <<  endl << "For momentum handoff file named: " << i_uvel_bc_storage->second.name << endl;
            proc0cout <<          "  Grid and handoff spacing relative differences are: ["
                      << std::abs(i_uvel_bc_storage->second.dx - dx)/dx << ", "
                      << std::abs(i_uvel_bc_storage->second.dy - dy)/dy << "]" << endl << endl;
          }

          if (foundIterator) {

            //if we are here, then we are of type "FromFile"
            bound_ptr.reset();

            //this should assign the correct normal direction xyz value without forcing the user to have
            //to know what it is.
            Vector ref_point;
            if ( index == 0 ) {
              i_uvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              i_vvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              i_wvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              ref_point = i_uvel_bc_storage->second.relative_xyz;
            } else if ( index == 1 ) {
              i_uvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              i_vvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              i_wvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              ref_point = i_vvel_bc_storage->second.relative_xyz;
            } else if ( index == 2 ) {
              i_uvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              i_vvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              i_wvel_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
              ref_point = i_wvel_bc_storage->second.relative_xyz;
            }

            Point xyz(ref_point[0],ref_point[1],ref_point[2]);

            IntVector ijk = patch->getLevel()->getCellIndex( xyz );

            i_uvel_bc_storage->second.relative_ijk = ijk;
            i_vvel_bc_storage->second.relative_ijk = ijk;
            i_wvel_bc_storage->second.relative_ijk = ijk;
            i_uvel_bc_storage->second.relative_ijk[index] = 0;
            i_vvel_bc_storage->second.relative_ijk[index] = 0;
            i_wvel_bc_storage->second.relative_ijk[index] = 0;

            int face_index_value=10;

            //now check to make sure that there is a bc set for each iterator:
            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              if ( index == 0 ) {
                //is this cell contained in list?
                //The next three lines are needed because we are ignoring the user input
                //for the normal index but still loading it into memory
                IntVector mod_bound_ptr = (*bound_ptr);
                face_index_value = mod_bound_ptr[index];
                mod_bound_ptr[index] = (i_uvel_bc_storage->second.values.begin()->first)[index];
                CellToValue::iterator check_iter = i_uvel_bc_storage->second.values.find( mod_bound_ptr - i_uvel_bc_storage->second.relative_ijk );
                if ( check_iter == i_uvel_bc_storage->second.values.end() ) {
                  std::stringstream out;
                  out << "Vel BC: " << *iname << " - No UINTAH boundary cell " << mod_bound_ptr - i_uvel_bc_storage->second.relative_ijk << " (relative) " << mod_bound_ptr << " (absolute) in the handoff file." << endl;
                  if ( !file_is_open ) {
                    file_is_open = true;
                    outputfile.open(fname.str().c_str());
                    outputfile << "Patch Dimentions (exclusive): \n";
                    outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                    outputfile << " high = " << patch->getCellHighIndex() << "\n";
                    outputfile << out.str();
                  } else {
                    outputfile << out.str();
                  }
                }
              } else if ( index == 1 ) {
                //is this cell contained in list?
                //The next three lines are needed because we are ignoring the user input
                //for the normal index but still loading it into memory
                IntVector mod_bound_ptr = (*bound_ptr);
                face_index_value = mod_bound_ptr[index];
                mod_bound_ptr[index] = (i_vvel_bc_storage->second.values.begin()->first)[index];
                CellToValue::iterator check_iter = i_vvel_bc_storage->second.values.find( mod_bound_ptr - i_vvel_bc_storage->second.relative_ijk );
                if ( check_iter == i_vvel_bc_storage->second.values.end() ) {
                  std::stringstream out;
                  out << "Vel BC: " << *iname << " - No UINTAH boundary cell " << mod_bound_ptr - i_vvel_bc_storage->second.relative_ijk << " (relative) " << mod_bound_ptr << " (absolute) in the handoff file." << endl;
                  if ( !file_is_open ) {
                    file_is_open = true;
                    outputfile.open(fname.str().c_str());
                    outputfile << "Patch Dimensions (exclusive): \n";
                    outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                    outputfile << " high = " << patch->getCellHighIndex() << "\n";
                    outputfile << out.str();
                  } else {
                    outputfile << out.str();
                  }
                }
              } else if ( index == 2 ) {
                //is this cell contained in list?
                //The next three lines are needed because we are ignoring the user input
                //for the normal index but still loading it into memory
                IntVector mod_bound_ptr = (*bound_ptr);
                face_index_value = mod_bound_ptr[index];
                mod_bound_ptr[index] = (i_wvel_bc_storage->second.values.begin()->first)[index];
                CellToValue::iterator check_iter = i_wvel_bc_storage->second.values.find( mod_bound_ptr - i_wvel_bc_storage->second.relative_ijk );
                if ( check_iter == i_wvel_bc_storage->second.values.end() ) {
                  std::stringstream out;
                  out << "Vel BC: " << *iname << " - No UINTAH boundary cell " << mod_bound_ptr - i_wvel_bc_storage->second.relative_ijk << " (relative) " << mod_bound_ptr << " (absolute) in the handoff file." << endl;
                  if ( !file_is_open ) {
                    file_is_open = true;
                    outputfile.open(fname.str().c_str());
                    outputfile << "Patch Dimentions (exclusive): \n";
                    outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                    outputfile << " high = " << patch->getCellHighIndex() << "\n";
                    outputfile << out.str();
                  } else {
                    outputfile << out.str();
                  }
                }
              }
            }

            //now check the reverse -- does the handoff file have an associated boundary ptr
            if ( index == 0 ) {

              CellToValue temp_map;
              for ( CellToValue::iterator check_iter = i_uvel_bc_storage->second.values.begin(); check_iter !=
                    i_uvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign the values now with the correct index for the face direction
              i_uvel_bc_storage->second.values = temp_map;

              temp_map.clear();
              for ( CellToValue::iterator check_iter = i_vvel_bc_storage->second.values.begin(); check_iter !=
                    i_vvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign the values now with the correct index for the face direction
              i_vvel_bc_storage->second.values = temp_map;

              temp_map.clear();
              for ( CellToValue::iterator check_iter = i_wvel_bc_storage->second.values.begin(); check_iter !=
                    i_wvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign the values now with the correct index for the face direction
              i_wvel_bc_storage->second.values = temp_map;

              for ( CellToValue::iterator check_iter = i_uvel_bc_storage->second.values.begin(); check_iter !=
                    i_uvel_bc_storage->second.values.end(); check_iter++ ) {

                bool found_it = false;
                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  if ( *bound_ptr == (check_iter->first + i_uvel_bc_storage->second.relative_ijk) )
                    found_it = true;
                }
                if ( !found_it && patch->containsCell(check_iter->first + i_uvel_bc_storage->second.relative_ijk) ) {
                  std::stringstream out;
                  out << "Vel BC: " << *iname << " - No HANDOFF cell " << check_iter->first << " (relative), " << check_iter->first + i_uvel_bc_storage->second.relative_ijk << " (absolute) in the Uintah geometry object." << endl;
                  if ( !file_is_open ) {
                    file_is_open = true;
                    outputfile.open(fname.str().c_str());
                    outputfile << "Patch Dimentions (exclusive): \n";
                    outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                    outputfile << " high = " << patch->getCellHighIndex() << "\n";
                    outputfile << out.str();
                  } else {
                    outputfile << out.str();
                  }
                }

              }
            } else if ( index == 1 ) {

              CellToValue temp_map;
              for ( CellToValue::iterator check_iter = i_vvel_bc_storage->second.values.begin(); check_iter !=
                    i_vvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign the values now with the correct index for the face direction
              i_vvel_bc_storage->second.values = temp_map;

              temp_map.clear();
              for ( CellToValue::iterator check_iter = i_uvel_bc_storage->second.values.begin(); check_iter !=
                    i_uvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign
              i_uvel_bc_storage->second.values = temp_map;

              temp_map.clear();
              for ( CellToValue::iterator check_iter = i_wvel_bc_storage->second.values.begin(); check_iter !=
                    i_wvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign
              i_wvel_bc_storage->second.values = temp_map;

              for ( CellToValue::iterator check_iter = i_vvel_bc_storage->second.values.begin(); check_iter !=
                    i_vvel_bc_storage->second.values.end(); check_iter++ ) {

                bool found_it = false;
                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  if ( *bound_ptr == (check_iter->first + i_vvel_bc_storage->second.relative_ijk) )
                    found_it = true;
                }
                if ( !found_it && patch->containsCell(check_iter->first + i_vvel_bc_storage->second.relative_ijk) ) {
                  std::stringstream out;
                  out << "Vel BC: " << *iname << " - No HANDOFF cell " << check_iter->first << " (relative), " << check_iter->first + i_vvel_bc_storage->second.relative_ijk << " (absolute) in the Uintah geometry object." << endl;
                  if ( !file_is_open ) {
                    file_is_open = true;
                    outputfile.open(fname.str().c_str());
                    outputfile << "Patch Dimentions (exclusive): \n";
                    outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                    outputfile << " high = " << patch->getCellHighIndex() << "\n";
                    outputfile << out.str();
                  } else {
                    outputfile << out.str();
                  }
                }

              }
            } else if ( index == 2 ) {

              CellToValue temp_map;
              for ( CellToValue::iterator check_iter = i_wvel_bc_storage->second.values.begin(); check_iter !=
                    i_wvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign the values now with the correct index for the face direction
              i_wvel_bc_storage->second.values = temp_map;

              temp_map.clear();
              for ( CellToValue::iterator check_iter = i_uvel_bc_storage->second.values.begin(); check_iter !=
                    i_uvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign the values now with the correct index for the face direction
              i_uvel_bc_storage->second.values = temp_map;

              temp_map.clear();
              for ( CellToValue::iterator check_iter = i_vvel_bc_storage->second.values.begin(); check_iter !=
                    i_vvel_bc_storage->second.values.end(); check_iter++ ) {

                //need to reset the values to get the right [index] int value for the face
                double value = check_iter->second;
                IntVector location = check_iter->first;
                location[index] = face_index_value;

                temp_map.insert(make_pair(location, value));

              }

              //reassign the values now with the correct index for the face direction
              i_vvel_bc_storage->second.values = temp_map;

              for ( CellToValue::iterator check_iter = i_wvel_bc_storage->second.values.begin(); check_iter !=
                    i_wvel_bc_storage->second.values.end(); check_iter++ ) {

                bool found_it = false;
                for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  if ( *bound_ptr == (check_iter->first + i_wvel_bc_storage->second.relative_ijk) )
                    found_it = true;
                }
                if ( !found_it && patch->containsCell(check_iter->first + i_wvel_bc_storage->second.relative_ijk) ) {
                  std::stringstream out;
                  out << "Vel BC: " << *iname << " - No HANDOFF cell " << check_iter->first << " (relative), " << check_iter->first + i_wvel_bc_storage->second.relative_ijk << " (absolute) in the Uintah geometry object." << endl;
                  if ( !file_is_open ) {
                    file_is_open = true;
                    outputfile.open(fname.str().c_str());
                    outputfile << "Patch Dimentions (exclusive): \n";
                    outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                    outputfile << " high = " << patch->getCellHighIndex() << "\n";
                    outputfile << out.str();
                  } else {
                    outputfile << out.str();
                  }
                }

              }
            }
          }
          if ( file_is_open ) {
            cout << "\n  Notice: Handoff velocity warning information has been printed to file for patch #: " << patch->getID() << "\n";
            outputfile.close();
          }
        }
      }
    }
  }
}
//______________________________________________________________________
//
void
BoundaryCondition::sched_create_radiation_temperature( SchedulerP       & sched,
                                                       const LevelP     & level,
                                                       const MaterialSet* matls,
                                                       bool doing_restart,
                                                       const bool use_old_dw )
{
  SourceTermFactory& srcs = SourceTermFactory::self();

  bool doing_DO_rad    = srcs.source_type_exists("do_radiation");
  bool doing_RMCRT_rad = srcs.source_type_exists( "rmcrt_radiation");

  if ( !doing_DO_rad && !doing_RMCRT_rad ){
    return;
  }

  //__________________________________
  //  Return if restarting and rad temp exists
  DataWarehouse* new_dw = sched->get_dw(1);
  int archIndex = 0;
  int matlIndex = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

  bool radTemp_exists = false;


  const int rank = Uintah::Parallel::getMPIRank();

  // Find the patches on the arches level that this mpi rank owns.
  const Uintah::PatchSet* const ps = sched->getLoadBalancer()->getPerProcessorPatchSet( level );
  const PatchSubset* myPatches     = ps->getSubset( rank );

  // hackish way to determine if radTemp exists.
  for( auto i=0; i<myPatches->size(); i++) {
    const Patch* patch = myPatches->get(i);
    radTemp_exists = new_dw->exists( d_radiation_temperature_label,   matlIndex, patch );
  }

  if( doing_restart && radTemp_exists ){
    return;
  }

  //__________________________________
  //
  // Before you can require varLabel from the new_dw
  // there must be a compute() for that variable.  This is an empty task.
  if ( doing_restart ) {

    Task* t = scinew Task("BoundaryCondition::create_radiation_temperatureHack", this,
                          &BoundaryCondition::create_radiation_temperatureHack );
    t->computes( d_temperature_label );
    sched->addTask(t, level->eachPatch(), matls);
  }

  //__________________________________
  //
  string taskname = "BoundaryCondition::create_radiation_temperature";
  Task* tsk = scinew Task(taskname, this, &BoundaryCondition::create_radiation_temperature, use_old_dw );


  //WARNING! THIS ASSUMES WE ARE DOING RADIATION ONCE PER TIMESTEP ON RK STEP = 0
  if ( use_old_dw ) {
    tsk->requires(Task::OldDW, d_temperature_label, Ghost::None, 0);
  } else {
    tsk->requires(Task::NewDW, d_temperature_label, Ghost::None, 0);
  }

  tsk->computes(d_radiation_temperature_label);

  sched->addTask( tsk, level->eachPatch(), matls );
}
//______________________________________________________________________
//
void
BoundaryCondition::create_radiation_temperature( const ProcessorGroup* pc,
                                                 const PatchSubset* patches,
                                                 const MaterialSubset* matls,
                                                 DataWarehouse* old_dw,
                                                 DataWarehouse* new_dw,
                                                 const bool use_old_dw )
{
  for (int p=0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> radiation_temperature;
    constCCVariable<double> old_temperature;

    new_dw->allocateAndPut( radiation_temperature, d_radiation_temperature_label, matlIndex, patch );

    if ( use_old_dw ) {
      old_dw->get( old_temperature, d_temperature_label, matlIndex, patch, Ghost::None, 0 );
    }
    else {
      new_dw->get( old_temperature, d_temperature_label, matlIndex, patch, Ghost::None, 0 );
      d_newBC->checkForBC( pc, patch, "radiation_temperature");
    }

    radiation_temperature.copyData(old_temperature);

    d_newBC->setExtraCellScalarValueBC<double>( pc, patch, radiation_temperature, "radiation_temperature" );
  }
}
//______________________________________________________________________
//
void
BoundaryCondition::addIntrusionMomRHS( const Patch* patch,
                                       constSFCXVariable<double>& u,
                                       constSFCYVariable<double>& v,
                                       constSFCZVariable<double>& w,
                                       SFCXVariable<double>& usrc,
                                       SFCYVariable<double>& vsrc,
                                       SFCZVariable<double>& wsrc,
                                       constCCVariable<double>& density )
{

  const Level* level = patch->getLevel();
  const int ilvl = level->getID();
  if ( _using_new_intrusion ){
    bool has_intrusion_inlets = _intrusionBC[ilvl]->has_intrusion_inlets();
    if ( has_intrusion_inlets ){
      _intrusionBC[ilvl]->addMomRHS( patch, u, v, w, usrc, vsrc, wsrc, density );
    }
  }

}

void
BoundaryCondition::addIntrusionMassRHS( const Patch* patch,
                                        CCVariable<double>& mass_src )
{

  const Level* level = patch->getLevel();
  const int ilvl = level->getID();
  if ( _using_new_intrusion ){
    bool has_intrusion_inlets = _intrusionBC[ilvl]->has_intrusion_inlets();
    if ( has_intrusion_inlets ){
      _intrusionBC[ilvl]->addMassRHS( patch, mass_src );
    }
  }

}
