/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
  is hereby granted, free of charge, to any person obtaining a copy
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

#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/Utility/MassFlowRate.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/UPSHelper.h>

using namespace Uintah;
using namespace ArchesCore;

// Constructor -----------------------------------------------------------------
MassFlowRate::MassFlowRate( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
}

// Destructor ------------------------------------------------------------------
MassFlowRate::~MassFlowRate(){
}

// Define ----------------------------------------------------------------------
void MassFlowRate::problemSetup( ProblemSpecP& db ){

  // Parse from UPS file
  m_g_uVel_name = parse_ups_for_role( UVELOCITY, db, "uVelocitySPBC" );
  m_g_vVel_name = parse_ups_for_role( VVELOCITY, db, "vVelocitySPBC" );
  m_g_wVel_name = parse_ups_for_role( WVELOCITY, db, "wVelocitySPBC" );

  particleMethod_bool = check_for_particle_method( db, DQMOM_METHOD );

  const ProblemSpecP db_root = db->getRootNode();

  if(particleMethod_bool){

    m_Nenv    = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );

    if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables") ){

      m_w_base_name = "w";                                                                        // weights
      m_RC_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);   // RawCoal
      m_CH_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);      // Char
      m_p_uVel_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL);  // uVel
      m_p_vVel_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL);  // vVel
      m_p_wVel_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL);  // wVel

      for ( int qn = 0; qn < m_Nenv; qn++ ){

        m_w_names.push_back(ArchesCore::append_qn_env( m_w_base_name, qn));
        m_RC_names.push_back(ArchesCore::append_qn_env( m_RC_base_name, qn));
        m_CH_names.push_back(ArchesCore::append_qn_env( m_CH_base_name, qn));

        m_p_uVel_names.push_back(ArchesCore::append_qn_env( m_p_uVel_base_name, qn));
        m_p_vVel_names.push_back(ArchesCore::append_qn_env( m_p_vVel_base_name, qn));
        m_p_wVel_names.push_back(ArchesCore::append_qn_env( m_p_wVel_base_name, qn));

        m_w_scaling_constant.push_back(ArchesCore::get_scaling_constant(db, m_w_base_name, qn));
        m_RC_scaling_constant.push_back(ArchesCore::get_scaling_constant(db, m_RC_base_name, qn));
        m_CH_scaling_constant.push_back(ArchesCore::get_scaling_constant(db, m_CH_base_name, qn));

        m_p_uVel_scaling_constant.push_back(ArchesCore::get_scaling_constant(db, m_p_uVel_base_name, qn));
        m_p_vVel_scaling_constant.push_back(ArchesCore::get_scaling_constant(db, m_p_vVel_base_name, qn));
        m_p_wVel_scaling_constant.push_back(ArchesCore::get_scaling_constant(db, m_p_wVel_base_name, qn));
      }
    }
  }
}

// Declaration -----------------------------------------------------------------
void
MassFlowRate::create_local_labels(){

  register_new_variable<sum_vartype> ("m_dot_g");  // Gas phase mass flow rate
  register_new_variable<sum_vartype> ("m_dot_p");  // Particle phase mass flow rate

}

// Initialization --------------------------------------------------------------
void MassFlowRate::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks ){

  register_massFlowRate( variable_registry, packed_tasks );
}

void MassFlowRate::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  eval_massFlowRate( patch, tsk_info );
}

// Timestep Eval ---------------------------------------------------------------
void MassFlowRate::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

  register_massFlowRate( variable_registry, packed_tasks );

}

void MassFlowRate::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  eval_massFlowRate( patch, tsk_info );
}

// Definitions -----------------------------------------------------------------
void MassFlowRate::register_massFlowRate( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks ){

  typedef ArchesFieldContainer AFC;

  register_variable( "m_dot_g"    , AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( "m_dot_p"    , AFC::COMPUTES, variable_registry, m_task_name );

  register_variable( "density"       , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

  register_variable( m_g_uVel_name   , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( m_g_vVel_name   , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
  register_variable( m_g_wVel_name   , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

  if(particleMethod_bool){

    for ( int qn = 0; qn < m_Nenv; qn++ ){

      register_variable( m_w_names[qn] , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
      register_variable( m_RC_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
      register_variable( m_CH_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

      register_variable( m_p_uVel_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
      register_variable( m_p_vVel_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );
      register_variable( m_p_wVel_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry );

    }
  }
}

// -----------------------------------------------------------------------------
void MassFlowRate::eval_massFlowRate( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  constCCVariable<double>& density = tsk_info->get_const_uintah_field_add<constCCVariable<double> >("density");

  DataWarehouse* new_dw = tsk_info->getNewDW();

  Vector Dx = patch->dCell();
  const double area_x = Dx.y() * Dx.z();
  const double area_y = Dx.z() * Dx.x();
  const double area_z = Dx.x() * Dx.y();

  // Get the  helper information about boundaries (global) per patch :
  const BndMapT& bc_info = m_bcHelper->get_boundary_information();

  double m_dot_gas  = 0.0;
  double m_dot_coal = 0.0;

  if(particleMethod_bool){

    for ( int i = 0; i < m_Nenv; i++ ){

      // Gas phase
      constSFCXVariable<double>& uvel_g  = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >(m_g_uVel_name);
      constSFCYVariable<double>& vvel_g  = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >(m_g_vVel_name);
      constSFCZVariable<double>& wvel_g  = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >(m_g_wVel_name);

      // Coal phase
      constCCVariable<double>& wqn  = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_w_names[i]  ));
      constCCVariable<double>& RCqn = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_RC_names[i] ));
      constCCVariable<double>& CHqn = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_CH_names[i] ));

      constCCVariable<double>& uvel_p = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_p_uVel_names[i] ));
      constCCVariable<double>& vvel_p = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_p_vVel_names[i] ));
      constCCVariable<double>& wvel_p = *(tsk_info->get_const_uintah_field<constCCVariable<double> >( m_p_wVel_names[i] ));

      for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

        // Sweep through, for type == Inlet, then sweep through the specified cells
        if ( i_bc->second.type == INLET ){

          // Get the cell iterator - range of cellID:
          Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

          // Get the face direction (this is the outward facing normal):
          IntVector normal     = patch->faceDirection(i_bc->second.face);
          IntVector normalFace = patch->faceDirection(i_bc->second.face);
          if((i_bc->second.name == "x+") || (i_bc->second.name == "y+") || (i_bc->second.name == "z+")){
            normalFace = IntVector(0,0,0);
          }

          //Now loop through the cells:
            parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {

            const int ic = i - normal[0]; // interior cell index
            const int jc = j - normal[1];
            const int kc = k - normal[2];

            const int iF = i - normalFace[0]; // interior FACE index
            const int jF = j - normalFace[1];
            const int kF = k - normalFace[2];


            double rho_interp  = 0.5 * ( density(i,j,k) + density(ic,jc,kc) );

            double RCqn_interp = 0.5 * ( RCqn(i,j,k)    + RCqn(ic,jc,kc) );
            double CHqn_interp = 0.5 * ( CHqn(i,j,k)    + CHqn(ic,jc,kc) );
            double wqn_interp  = 0.5 * ( wqn (i,j,k)    + wqn (ic,jc,kc) );

            double uvel_p_interp  = 0.5 * ( uvel_p(i,j,k) + uvel_p(ic,jc,kc) );
            double vvel_p_interp  = 0.5 * ( vvel_p(i,j,k) + vvel_p(ic,jc,kc) );
            double wvel_p_interp  = 0.5 * ( wvel_p(i,j,k) + wvel_p(ic,jc,kc) );

            double uvel_p_i = uvel_p_interp * m_p_uVel_scaling_constant[i] / wqn_interp ;
            double vvel_p_i = vvel_p_interp * m_p_vVel_scaling_constant[i] / wqn_interp ;
            double wvel_p_i = wvel_p_interp * m_p_wVel_scaling_constant[i] / wqn_interp ;

            double RCi = RCqn_interp * m_RC_scaling_constant[i] * m_w_scaling_constant[i]; // kg of i / m3
            double CHi = CHqn_interp * m_CH_scaling_constant[i] * m_w_scaling_constant[i];

            if(i_bc->second.face == Patch::xminus || i_bc->second.face == Patch::xplus ){

              m_dot_gas += (i==0) ? rho_interp * std::abs(uvel_g(iF,jF,kF)) * area_x : 0.0; // gas is only computed once
              m_dot_coal += ( RCi + CHi ) * std::abs(uvel_p_i) * area_x;
            }
            else if(i_bc->second.face == Patch::yminus || i_bc->second.face == Patch::yplus ){
              m_dot_gas += (i==0) ? rho_interp * std::abs(vvel_g(iF,jF,kF)) * area_y : 0.0;
              m_dot_coal += ( RCi + CHi ) * std::abs(vvel_p_i) * area_y;
            }
            // i_bc->second.name == "z-" || i_bc->second.name == "z+"
            else{
              m_dot_gas += (i==0) ? rho_interp * std::abs(wvel_g(iF,jF,kF)) * area_z : 0.0;
              m_dot_coal += ( RCi + CHi ) * std::abs(wvel_p_i) * area_z;
            }
          });
        }
      }
    }

    new_dw->put(sum_vartype(m_dot_gas ), VarLabel::find("m_dot_g"));
    new_dw->put(sum_vartype(m_dot_coal), VarLabel::find("m_dot_p"));
    // ----------------------------------------------------------------------

    proc0cout << "\n Mass flow Gas  : Inlet face = " << m_dot_gas << " [kg/s]" << std::endl;
    proc0cout << " Mass flow Coal : Inlet face = " << m_dot_coal << " [kg/s]\n" << std::endl;
  }
  else{

    constSFCXVariable<double>& uvel_g  = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >( m_g_uVel_name );
    constSFCYVariable<double>& vvel_g  = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >( m_g_vVel_name );
    constSFCZVariable<double>& wvel_g  = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >( m_g_wVel_name );

    for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){
      // Sweep through, for type == Inlet, then sweep through the specified cells

      if ( i_bc->second.type == INLET ){

        // Get the cell iterator - range of cellID:
        Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

        // Get the face direction (this is the outward facing normal):
        IntVector normal     = patch->faceDirection(i_bc->second.face);
        IntVector normalFace = patch->faceDirection(i_bc->second.face);
        if((i_bc->second.name == "x+") || (i_bc->second.name == "y+") || (i_bc->second.name == "z+")){
          normalFace = IntVector(0,0,0);
        }

        //Now loop through the cells:
        parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {

          const int ic = i - normal[0]; // interior cell index
          const int jc = j - normal[1];
          const int kc = k - normal[2];

          const int iF = i - normalFace[0]; // interior FACE index
          const int jF = j - normalFace[1];
          const int kF = k - normalFace[2];

          double rho_interp  = 0.5 * ( density(i,j,k) + density(ic,jc,kc) );

          if(i_bc->second.face == Patch::xminus || i_bc->second.face == Patch::xplus ){
            m_dot_gas += rho_interp * std::abs(uvel_g(iF,jF,kF)) * area_x;
          }
          else if(i_bc->second.face == Patch::yminus || i_bc->second.face == Patch::yplus ){
            m_dot_gas += rho_interp * std::abs(vvel_g(iF,jF,kF)) * area_y;
          }
          // i_bc->second.name == "z-" || i_bc->second.name == "z+"
          else{
            m_dot_gas += rho_interp * std::abs(wvel_g(iF,jF,kF)) * area_z;
          }
        });
      }
    }
    new_dw->put(sum_vartype(m_dot_gas), VarLabel::find("m_dot_g"));
    // ----------------------------------------------------------------------

    proc0cout << "\n Mass flow Gas  : Inlet face = " << m_dot_gas << " [kg/s]\n" << std::endl;
  }
}
