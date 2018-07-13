#include <CCA/Components/Arches/Utility/SurfaceVolumeFractionCalc.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/Grid/Box.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceVolumeFractionCalc::loadTaskEvalFunctionPointers(){

  TaskAssignedExecutionSpace assignedTag{};
  LOAD_ARCHES_EVAL_TASK_2TAGS(UINTAH_CPU_TAG, KOKKOS_OPENMP_TAG, assignedTag, SurfaceVolumeFractionCalc::eval);
  return assignedTag;

}

//--------------------------------------------------------------------------------------------------
void SurfaceVolumeFractionCalc::problemSetup( ProblemSpecP& db ){

  //Collect all intrusions:
  ProblemSpecP db_intrusions =
    db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("BoundaryConditions");
  if ( db_intrusions ){
    db_intrusions = db_intrusions->findBlock("intrusions");
    if ( db_intrusions ){
      for ( ProblemSpecP db_intrusion = db_intrusions->findBlock("intrusion");
        db_intrusion != nullptr; db_intrusion = db_intrusion->findNextBlock("intrusion") ){

          IntrusionBoundary I;

          //get the geom_object:
          ProblemSpecP geometry_db = db_intrusion->findBlock("geom_object");
          if ( geometry_db == nullptr ){
            throw ProblemSetupException("Error: Make sure all intrusions have a valid geom_object.", __FILE__, __LINE__ );
          }
          GeometryPieceFactory::create( geometry_db, I.geometry );

          m_intrusions.push_back( I );

      }
    }
  }

}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::create_local_labels(){

  // CC fields
  register_new_variable<CCVariable<double> >( "volFraction" );
  register_new_variable<SFCXVariable<double> >( "fx_volume_fraction" );
  register_new_variable<SFCYVariable<double> >( "fy_volume_fraction" );
  register_new_variable<SFCZVariable<double> >( "fz_volume_fraction" );

  m_var_names.push_back( "volFraction" );
  m_var_names.push_back( "fx_volume_fraction" );
  m_var_names.push_back( "fy_volume_fraction" );
  m_var_names.push_back( "fz_volume_fraction" );

}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::register_initialize( ArchesVIVector& variable_registry , const bool packed_tasks){

  for ( auto i = m_var_names.begin(); i != m_var_names.end(); i++ ){
    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
  }
}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject& executionObject ){

  typedef CCVariable<double> T;

  T& cc_vf = tsk_info->get_uintah_field_add<T>("volFraction");
  SFCXVariable<double>& fx_vf = tsk_info->get_uintah_field_add<SFCXVariable<double> >("fx_volume_fraction");
  SFCYVariable<double>& fy_vf = tsk_info->get_uintah_field_add<SFCYVariable<double> >("fy_volume_fraction");
  SFCZVariable<double>& fz_vf = tsk_info->get_uintah_field_add<SFCZVariable<double> >("fz_volume_fraction");

  cc_vf.initialize(1.0);
  fx_vf.initialize(1.0);
  fy_vf.initialize(1.0);
  fz_vf.initialize(1.0);

  //Clean out all intrusions that don't intersect with this patch:
  for ( auto i = m_intrusions.begin(); i != m_intrusions.end(); i++ ){

    std::vector<GeometryPieceP> intersecting_geometry;

    for ( auto i_geom  = i->geometry.begin(); i_geom != i->geometry.end(); i_geom++ ){

      //Adding a buffer to ensure that patch-to-patch boundaries arent ignored
      IntVector low(patch->getCellLowIndex() - IntVector(2,2,2));
      IntVector high(patch->getCellHighIndex() + IntVector(2,2,2));
      Point low_pt = patch->getCellPosition(low);
      Point high_pt = patch->getCellPosition(high);
      Box patch_box(low_pt, high_pt);
      GeometryPieceP geom = *i_geom;
      Box intersecting_box = patch_box.intersect( geom->getBoundingBox() );

      if ( !intersecting_box.degenerate() ){

        intersecting_geometry.push_back(geom);

        for ( CellIterator icell = patch->getExtraCellIterator(); !icell.done(); icell++ ){

          IntVector c = *icell;

          Point p = patch->cellPosition( *icell );
          if ( geom->inside(p) ){

            //PCELL
            cc_vf[c] = 0.0;

          }
        }
      }
    }

    i->geometry = intersecting_geometry;

  }

  //Now get the boundary conditions:
  const BndMapT& bc_info = m_bcHelper->get_boundary_information();

  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());

    if ( on_this_patch ){

      if ( i_bc->second.type == WALL ){
        //Get the iterator
        Uintah::Iterator cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

        for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){
          cc_vf[*cell_iter] = 0.0;
        }

      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks){

  for ( auto i = m_var_names.begin(); i != m_var_names.end(); i++ ){
    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry, _task_name );
    register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, _task_name );
  }
}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& cc_vol_frac = tsk_info->get_uintah_field_add<CCVariable<double> >("volFraction");
  constCCVariable<double>& cc_vol_frac_old = tsk_info->get_const_uintah_field_add<constCCVariable<double> >("volFraction");

  cc_vol_frac.copyData(cc_vol_frac_old);

  SFCXVariable<double>& fx_vol_frac = tsk_info->get_uintah_field_add<SFCXVariable<double> >("fx_volume_fraction");
  constSFCXVariable<double>& fx_vol_frac_old = tsk_info->get_const_uintah_field_add<constSFCXVariable<double> >("fx_volume_fraction");
  fx_vol_frac.copyData(fx_vol_frac_old);

  SFCYVariable<double>& fy_vol_frac = tsk_info->get_uintah_field_add<SFCYVariable<double> >("fy_volume_fraction");
  constSFCYVariable<double>& fy_vol_frac_old = tsk_info->get_const_uintah_field_add<constSFCYVariable<double> >("fy_volume_fraction");
  fy_vol_frac.copyData(fy_vol_frac_old);

  SFCZVariable<double>& fz_vol_frac = tsk_info->get_uintah_field_add<SFCZVariable<double> >("fz_volume_fraction");
  constSFCZVariable<double>& fz_vol_frac_old = tsk_info->get_const_uintah_field_add<constSFCZVariable<double> >("fz_volume_fraction");
  fz_vol_frac.copyData(fz_vol_frac_old);

}
