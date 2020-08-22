/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

// NOTE: This code is computing the mass flow of the gas and
//       mass flow of the organics in the particle phase.
//       To get the total mass flow rate of the particles you
//       need to compute it from the fixed ash and moisture content:
//       mDot_part = m_dot_org / ( mass fraction of organics )
//       where m_dot_org is what is computed here.

// Constructor -----------------------------------------------------------------
MassFlowRate::MassFlowRate( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
}

// Destructor ------------------------------------------------------------------
MassFlowRate::~MassFlowRate(){
}

//---------------------------------------------------------------------------
//Method: Load task function pointers for portability
//---------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MassFlowRate::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MassFlowRate::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &MassFlowRate::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &MassFlowRate::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &MassFlowRate::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MassFlowRate::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &MassFlowRate::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &MassFlowRate::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &MassFlowRate::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MassFlowRate::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace MassFlowRate::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//-------------------------------------------------------------------------------------------------------
void MassFlowRate::problemSetup( ProblemSpecP& db ){

  // Parse from UPS file
  m_g_uVel_name = parse_ups_for_role( UVELOCITY_ROLE, db, "uVelocitySPBC" );
  m_g_vVel_name = parse_ups_for_role( VVELOCITY_ROLE, db, "vVelocitySPBC" );
  m_g_wVel_name = parse_ups_for_role( WVELOCITY_ROLE, db, "wVelocitySPBC" );

  m_volFraction_name = "volFraction";
  m_particleMethod_bool = check_for_particle_method( db, DQMOM_METHOD );

  const ProblemSpecP db_root = db->getRootNode();

  if(m_particleMethod_bool){

    m_Nenv    = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );

    if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables") ){

      m_w_base_name = "w";                                                                        // weights
      m_RC_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);   // RawCoal
      m_CH_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);      // Char
      m_p_uVel_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_XVEL);  // uVel
      m_p_vVel_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_YVEL);  // vVel
      m_p_wVel_base_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_ZVEL);  // wVel

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

  // Get name of inlets
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");
  for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr;
        db_face = db_face->findNextBlock("Face") ) {

    std::string Type;
    db_face->getAttribute("type", Type );
    if (Type =="Inlet" ) {

      std::string faceName;
      db_face->getAttribute("name",faceName);
      MFInfo info_g ;
      info_g.name = m_task_name+"_mdot_g_in_" + faceName;
      info_g.face_name = faceName;
      info_g.value = 0;
      m_m_gas_info.push_back(info_g);

      if ( m_particleMethod_bool ){
        MFInfo info_p ;
        info_p.name = m_task_name+"_mdot_p_organics_in_" + faceName;
        info_p.face_name = faceName;
        info_p.value = 0;
        m_m_p_info.push_back(info_p);
      }

    }
  }

  // Get name of outlets
  //ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");
  for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr;
        db_face = db_face->findNextBlock("Face") ) {

    std::string Type;
    db_face->getAttribute("type", Type );
    if (Type =="Outflow" ) {

      std::string faceName;
      db_face->getAttribute("name",faceName);
      MFInfo info_g ;
      info_g.name = m_task_name+"_mdot_g_out_" + faceName;
      info_g.face_name = faceName;
      info_g.value = 0;
      m_m_gas_info.push_back(info_g);

      if ( m_particleMethod_bool ){
        MFInfo info_p ;
        info_p.name = m_task_name+"_mdot_p_organics_out_" + faceName;
        info_p.face_name = faceName;
        info_p.value = 0;
        m_m_p_info.push_back(info_p);
      }

    }
  }

  // Check to see if there is a scalar associated with this balance:
  if ( db->findBlock("scalar") ){
    db->require("scalar", m_scalar_name);
  }

}

// Declaration -----------------------------------------------------------------
void
MassFlowRate::create_local_labels(){

  register_new_variable<sum_vartype> (m_task_name+"_mdot_g_in");  // Gas phase mass flow rate
  register_new_variable<sum_vartype> (m_task_name+"_mdot_g_out");  // Gas phase mass flow rate

  for (auto iface = m_m_gas_info.begin(); iface != m_m_gas_info.end(); iface++){
    MFInfo info = *iface;
    register_new_variable<sum_vartype> (info.name);
  }

  if ( m_particleMethod_bool ){
    for (auto iface = m_m_p_info.begin(); iface != m_m_p_info.end(); iface++){
      MFInfo info = *iface;
      register_new_variable<sum_vartype> (info.name);
    }
    register_new_variable<sum_vartype> (m_task_name+"_mdot_p_organics_out");  // Particle phase mass flow rate
    register_new_variable<sum_vartype> (m_task_name+"_mdot_p_organics_in");  // Particle phase mass flow rate
  }

}

// Timestep Eval ---------------------------------------------------------------
void MassFlowRate::register_timestep_eval(

  std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
  const int time_substep , const bool packed_tasks){

  typedef ArchesFieldContainer AFC;

  register_variable( m_task_name+"_mdot_g_in"    , AFC::COMPUTES, variable_registry, m_task_name );
  register_variable( m_task_name+"_mdot_g_out"    , AFC::COMPUTES, variable_registry, m_task_name );

  for (auto iface = m_m_gas_info.begin(); iface != m_m_gas_info.end(); iface++){
    MFInfo info = *iface;
    register_variable( info.name    , AFC::COMPUTES, variable_registry, m_task_name );
  }

  if ( m_particleMethod_bool ){
    for (auto iface = m_m_p_info.begin(); iface != m_m_p_info.end(); iface++){
      MFInfo info = *iface;
      register_variable( info.name    , AFC::COMPUTES, variable_registry, m_task_name );
    }
    register_variable( m_task_name+"_mdot_p_organics_out"    , AFC::COMPUTES, variable_registry, m_task_name );
    register_variable( m_task_name+"_mdot_p_organics_in"    , AFC::COMPUTES, variable_registry, m_task_name );
  }

  register_variable( "density"       , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_g_uVel_name   , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_g_vVel_name   , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_g_wVel_name   , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_volFraction_name, AFC::REQUIRES, 1, AFC::OLDDW, variable_registry, time_substep, m_task_name );

  if(m_particleMethod_bool){

    for ( int qn = 0; qn < m_Nenv; qn++ ){

      register_variable( m_w_names[qn] , AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
      register_variable( m_RC_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
      register_variable( m_CH_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );

      register_variable( m_p_uVel_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
      register_variable( m_p_vVel_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );
      register_variable( m_p_wVel_names[qn], AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name );

    }
  }

  if (m_scalar_name != "NULL"){
    register_variable(m_scalar_name, AFC::REQUIRES, 1, AFC::NEWDW, variable_registry, time_substep, m_task_name);
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void MassFlowRate::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  eval_massFlowRate( patch, tsk_info , execObj );
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void MassFlowRate::eval_massFlowRate( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace> execObj ){

  DataWarehouse* new_dw = tsk_info->getNewDW();

  Vector Dx = patch->dCell();
  const double area_x = Dx.y() * Dx.z();
  const double area_y = Dx.z() * Dx.x();
  const double area_z = Dx.x() * Dx.y();

  // Get the  helper information about boundaries (global) per patch :
  const BndMapT& bc_info = m_bcHelper->get_boundary_information();

  // Gas phase
  constSFCXVariable<double>& uvel_g = tsk_info->get_field<constSFCXVariable<double> >(m_g_uVel_name);
  constSFCYVariable<double>& vvel_g = tsk_info->get_field<constSFCYVariable<double> >(m_g_vVel_name);
  constSFCZVariable<double>& wvel_g = tsk_info->get_field<constSFCZVariable<double> >(m_g_wVel_name);
  constCCVariable<double>& density = tsk_info->get_field<constCCVariable<double> >("density");
  constCCVariable<double>& eps = tsk_info->get_field<constCCVariable<double> >(m_volFraction_name);

  if(m_particleMethod_bool){

    double mDot_coal = 0.0;

    for ( int qn = 0; qn < m_Nenv; qn++ ){

      // Coal phase
      constCCVariable<double>& wqn = tsk_info->get_field<constCCVariable<double> >( m_w_names[qn]  );
      constCCVariable<double>& RCqn = tsk_info->get_field<constCCVariable<double> >( m_RC_names[qn] );
      constCCVariable<double>& CHqn = tsk_info->get_field<constCCVariable<double> >( m_CH_names[qn] );

      constCCVariable<double>& uvel_p = tsk_info->get_field<constCCVariable<double> >( m_p_uVel_names[qn] );
      constCCVariable<double>& vvel_p = tsk_info->get_field<constCCVariable<double> >( m_p_vVel_names[qn] );
      constCCVariable<double>& wvel_p = tsk_info->get_field<constCCVariable<double> >( m_p_wVel_names[qn] );

      for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

        const bool on_this_patch = i_bc->second.has_patch(patch->getID());

        if ( on_this_patch ){

          // Sweep through, for type == Inlet, then sweep through the specified cells
          if ( i_bc->second.type == INLET_BC ){

            // Get the cell iterator - range of cellID:
            Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

            // Get the face direction (this is the outward facing normal):
            const IntVector iDir     = patch->faceDirection(i_bc->second.face);

            //Now loop through the cells:
            double value_m_dot = 0;
            parallel_for_unstructured(execObj,cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), [&] (const int i,const int j,const int k) {

            const int ic = i - iDir[0]; // interior cell index
            const int jc = j - iDir[1];
            const int kc = k - iDir[2];

            const double eps_interp  =  eps(i,j,k)*eps(ic,jc,kc);

            const double RCqn_interp = 0.5 * ( RCqn(i,j,k) + RCqn(ic,jc,kc) );
            const double CHqn_interp = 0.5 * ( CHqn(i,j,k) + CHqn(ic,jc,kc) );
            const double wqn_interp  = 0.5 * ( wqn (i,j,k) + wqn (ic,jc,kc) );

            const double uvel_p_interp  = 0.5 * ( uvel_p(i,j,k) + uvel_p(ic,jc,kc) );
            const double vvel_p_interp  = 0.5 * ( vvel_p(i,j,k) + vvel_p(ic,jc,kc) );
            const double wvel_p_interp  = 0.5 * ( wvel_p(i,j,k) + wvel_p(ic,jc,kc) );

            const double uvel_p_i = uvel_p_interp * m_p_uVel_scaling_constant[qn] / wqn_interp ;
            const double vvel_p_i = vvel_p_interp * m_p_vVel_scaling_constant[qn] / wqn_interp ;
            const double wvel_p_i = wvel_p_interp * m_p_wVel_scaling_constant[qn] / wqn_interp ;

            const double RCi = RCqn_interp * m_RC_scaling_constant[qn] / wqn_interp; // kg of i / m3
            const double CHi = CHqn_interp * m_CH_scaling_constant[qn] / wqn_interp;

            const double wi =  wqn_interp * m_w_scaling_constant[qn];

            value_m_dot += -( RCi + CHi ) * uvel_p_i * area_x * wi * eps_interp * iDir[0]
                           -( RCi + CHi ) * vvel_p_i * area_y * wi * eps_interp * iDir[1]
                           -( RCi + CHi ) * wvel_p_i * area_z * wi * eps_interp * iDir[2];

            });

            mDot_coal += value_m_dot;

            new_dw->put(sum_vartype(value_m_dot), VarLabel::find(m_task_name+"_mdot_p_organics_in_"+i_bc->second.name));

          }
        }
      }
    }

    new_dw->put(sum_vartype(mDot_coal), VarLabel::find(m_task_name+"_mdot_p_organics_in"));

  }

  double mDot_gas  = 0.0;
  double mDot_gas_out = 0.0;

  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());

    if ( on_this_patch ){

      // Sweep through, for type == Inlet, then sweep through the specified cells
      if ( i_bc->second.type == INLET_BC || i_bc->second.type == OUTLET_BC ){

        // Get the cell iterator - range of cellID:
        Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

        // Get the face direction (this is the outward facing normal):
        const IntVector iDir     = patch->faceDirection(i_bc->second.face);

        //Now loop through the cells:
        double value_m_dot = 0;
        if ( m_scalar_name == "NULL" ){

          parallel_for_unstructured(execObj,cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), [&] (const int i,const int j,const int k) {

            const int im = i - iDir[0]; // interior cell index
            const int jm = j - iDir[1];
            const int km = k - iDir[2];

            const double rho_interp  = 0.5 * ( density(i,j,k) + density(im,jm,km) );
            const double eps_interp  =  eps(i,j,k)*eps(im,jm,km);

            // x- => (-1,0,0)  x+ => (1,0,0)
            // y- => (0,-1,0)  y+ => (0,1,0)
            // z- => (0,0,-1)  z+ => (0,0,1)
            value_m_dot += -rho_interp * uvel_g(i,j,k) * area_x * eps_interp * iDir[0]
                           -rho_interp * vvel_g(i,j,k) * area_y * eps_interp * iDir[1]
                           -rho_interp * wvel_g(i,j,k) * area_z * eps_interp * iDir[2];


          });

        } else {

          auto scalar = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>( m_scalar_name );

          parallel_for_unstructured(execObj,cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), [&] (const int i,const int j,const int k) {

            const int im = i - iDir[0]; // interior cell index
            const int jm = j - iDir[1];
            const int km = k - iDir[2];

            const double rho_interp  = 0.5 * ( density(i,j,k) + density(im,jm,km) );
            const double eps_interp  =  eps(i,j,k)*eps(im,jm,km);
            const double scalar_interp = 0.5 * ( scalar(i,j,k) + scalar(im,jm,km) );

              // x- => (-1,0,0)  x+ => (1,0,0)
              // y- => (0,-1,0)  y+ => (0,1,0)
              // z- => (0,0,-1)  z+ => (0,0,1)
              value_m_dot += -rho_interp * uvel_g(i,j,k) * area_x * eps_interp * scalar_interp * iDir[0]
                             -rho_interp * vvel_g(i,j,k) * area_y * eps_interp * scalar_interp * iDir[1]
                             -rho_interp * wvel_g(i,j,k) * area_z * eps_interp * scalar_interp * iDir[2];


          });
        }

        if ( i_bc->second.type == OUTLET_BC ){
          new_dw->put(sum_vartype(value_m_dot), VarLabel::find(m_task_name+"_mdot_g_out_"+i_bc->second.name));
          mDot_gas_out += value_m_dot;
        } else {
          new_dw->put(sum_vartype(value_m_dot), VarLabel::find(m_task_name+"_mdot_g_in_"+i_bc->second.name));
          mDot_gas += value_m_dot;
        }

      }
    }
  }

  new_dw->put(sum_vartype(mDot_gas), VarLabel::find(m_task_name+"_mdot_g_in"));
  new_dw->put(sum_vartype(mDot_gas_out), VarLabel::find(m_task_name+"_mdot_g_out"));

}
