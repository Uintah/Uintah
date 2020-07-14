#include <CCA/Components/Arches/Utility/SurfaceVolumeFractionCalc.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/Grid/Box.h>

using namespace Uintah;

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceVolumeFractionCalc::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceVolumeFractionCalc::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &SurfaceVolumeFractionCalc::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &SurfaceVolumeFractionCalc::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &SurfaceVolumeFractionCalc::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceVolumeFractionCalc::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceVolumeFractionCalc::loadTaskTimestepInitFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
                                     , &SurfaceVolumeFractionCalc::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &SurfaceVolumeFractionCalc::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &SurfaceVolumeFractionCalc::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace SurfaceVolumeFractionCalc::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
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
  register_new_variable<SFCXVariable<double> >( "volFractionX" );
  register_new_variable<SFCYVariable<double> >( "volFractionY" );
  register_new_variable<SFCZVariable<double> >( "volFractionZ" );
  register_new_variable<CCVariable<int> >("cellType");

  m_var_names.push_back( "volFraction" );
  m_var_names.push_back( "volFractionX" );
  m_var_names.push_back( "volFractionY" );
  m_var_names.push_back( "volFractionZ" );
  m_var_names.push_back( "cellType" );

}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::register_initialize( ArchesVIVector& variable_registry , const bool packed_tasks){

  for ( auto i = m_var_names.begin(); i != m_var_names.end(); i++ ){
    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
  }

}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void SurfaceVolumeFractionCalc::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  typedef CCVariable<double> T;

  auto cc_vf = tsk_info->get_field<T, double, MemSpace>("volFraction");
  auto fx_vf = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("volFractionX");
  auto fy_vf = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("volFractionY");
  auto fz_vf = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("volFractionZ");
  auto cell_type = tsk_info->get_field<CCVariable<int>, int, MemSpace>("cellType");

  parallel_initialize(execObj,1.0, cc_vf,fx_vf,fy_vf,fz_vf);
  parallel_initialize(execObj,-1, cell_type);

  //Get the boundary conditions:
  const BndMapT& bc_info = m_bcHelper->get_boundary_information();

  const int pID = patch->getID();

  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    const bool on_this_patch = i_bc->second.has_patch(patch->getID());

    if ( on_this_patch ){

      //Handle cell type first
      Uintah::ListOfCellsIterator& cell_iter_ct  = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
      parallel_for_unstructured(execObj,cell_iter_ct.get_ref_to_iterator(execObj),cell_iter_ct.size(), KOKKOS_LAMBDA (int i,int j,int k) {
        cell_type(i,j,k) = i_bc->second.type;
      });

      if ( i_bc->second.type == WALL_BC ){

        //Get the iterator
        Uintah::ListOfCellsIterator& cell_iter  = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

      parallel_for_unstructured(execObj,cell_iter.get_ref_to_iterator(execObj),cell_iter.size(), KOKKOS_LAMBDA (int i,int j,int k) {

          cc_vf(i,j,k)= 0.0;
          fx_vf(i,j,k)= 0.0;
          fy_vf(i,j,k)= 0.0;
          fz_vf(i,j,k)= 0.0;

          if ( i_bc->second.face == Patch::xminus || i_bc->second.face == Patch::xplus ){
            fx_vf(i+1,j,k) = 0.0;
          }

          if ( i_bc->second.face == Patch::yminus || i_bc->second.face == Patch::yplus ){
            fy_vf(i,j+1,k) = 0.0;
          }

          if ( i_bc->second.face == Patch::zminus || i_bc->second.face == Patch::zplus ){
            fz_vf(i,j,k+1) = 0.0;
          }

        });

      }
    }
  }

  //Clean out all intrusions that don't intersect with this patch:
  // This needs to be done to limit the amount of information/patch
  // in cases where there are many intrusions.
  // NEEDS TO BE MADE THREAD SAFE - lock m_intrusion_map
  std::vector<IntrusionBoundary> intrusions;

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());

  for ( auto i = m_intrusions.begin(); i != m_intrusions.end(); i++ ){

    std::vector<GeometryPieceP> intersecting_geometry;

    for ( auto i_geom  = i->geometry.begin(); i_geom != i->geometry.end(); i_geom++ ){

      //Adding a buffer to ensure that patch-to-patch boundaries arent ignored
      const int Nbuff = 2;
      IntVector low(patch->getCellLowIndex() - IntVector(Nbuff, Nbuff, Nbuff));
      IntVector high(patch->getCellHighIndex() + IntVector(Nbuff, Nbuff, Nbuff));
      Point low_pt = patch->getCellPosition(low);
      Point high_pt = patch->getCellPosition(high);
      Box patch_box(low_pt, high_pt);
      GeometryPieceP geom = *i_geom;
      Box intersecting_box = patch_box.intersect( geom->getBoundingBox() );

      if ( !intersecting_box.degenerate() ){

        intersecting_geometry.push_back(geom);

        parallel_for(execObj,range, KOKKOS_LAMBDA (int i,int j,int k){

          Point p = patch->cellPosition(IntVector(i,j,k) );
          if ( geom->inside(p) ){

            //PCELL
            cc_vf(i,j,k) = 0.0;
            cell_type(i,j,k) = INTRUSION_BC;

          }

          // X-dir
          IntVector ix = IntVector(i,j,k) - IntVector(1,0,0);
          Point px = patch->cellPosition( ix );
          if ( patch->containsCell( ix ) ){
            if ( geom->inside(px) || geom->inside(p) ){
              fx_vf(i,j,k) = 0.0;
            }
          }

          // y-dir
          IntVector iy = IntVector(i,j,k) - IntVector(0,1,0);
          Point py = patch->cellPosition( iy );
          if ( patch->containsCell( iy ) ){
            if ( geom->inside(py) || geom->inside(p) ){
              fy_vf(i,j,k) = 0.0;
            }
          }

          // z-dir
          IntVector iz = IntVector(i,j,k) - IntVector(0,0,1);
          Point pz = patch->cellPosition( iz );
          if ( patch->containsCell( iy ) ){
            if ( geom->inside(pz) || geom->inside(p) ){
              fz_vf(i,j,k) = 0.0;
            }
          }

        });
      }
    }

    IntrusionBoundary IB;
    IB.geometry = intersecting_geometry;

    intrusions.push_back(IB);

  }

  m_intrusion_lock.lock();

  m_intrusion_map.insert(std::make_pair(pID, intrusions));

  m_intrusion_lock.unlock();

}

//--------------------------------------------------------------------------------------------------
void
SurfaceVolumeFractionCalc::register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks){

  for ( auto i = m_var_names.begin(); i != m_var_names.end(); i++ ){
    register_variable( *i, ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
    register_variable( *i, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, m_task_name );
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace> void
SurfaceVolumeFractionCalc::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto cc_vol_frac = tsk_info->get_field<CCVariable<double>, double, MemSpace>("volFraction");
  auto cc_vol_frac_old = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>("volFraction");

  auto cellType = tsk_info->get_field<CCVariable<int>, int, MemSpace>("cellType");
  auto cellType_old = tsk_info->get_field<constCCVariable<int>, const int, MemSpace>("cellType");

  auto fx_vol_frac = tsk_info->get_field<SFCXVariable<double>, double, MemSpace>("volFractionX");
  auto fx_vol_frac_old = tsk_info->get_field<constSFCXVariable<double>, const double, MemSpace>("volFractionX");

  auto fy_vol_frac = tsk_info->get_field<SFCYVariable<double>, double, MemSpace>("volFractionY");
  auto fy_vol_frac_old = tsk_info->get_field<constSFCYVariable<double>, const double, MemSpace>("volFractionY");

  auto fz_vol_frac = tsk_info->get_field<SFCZVariable<double>, double, MemSpace>("volFractionZ");
  auto fz_vol_frac_old = tsk_info->get_field<constSFCZVariable<double>, const double, MemSpace>("volFractionZ");

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());

  parallel_for( execObj, range, KOKKOS_LAMBDA (int i,int j,int k){
    cc_vol_frac(i,j,k) = cc_vol_frac_old(i,j,k);
    fx_vol_frac(i,j,k) = fx_vol_frac_old(i,j,k);
    fy_vol_frac(i,j,k) = fy_vol_frac_old(i,j,k);
    fz_vol_frac(i,j,k) = fz_vol_frac_old(i,j,k);
    cellType(i,j,k)    = cellType_old(i,j,k);
    //std::cout << i << "  " << j << "  " << k << "  " <<  (int) i_bc->second.type << " \n";
    //std::cout << i << "  " << j << "  " << k << "  " <<  cellType_old(i,j,k) << " \n";
  });

}
