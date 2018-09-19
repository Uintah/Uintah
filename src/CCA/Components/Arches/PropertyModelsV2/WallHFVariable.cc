#include <CCA/Components/Arches/PropertyModelsV2/WallHFVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>

#define SMALLNUM 1e-100

namespace Uintah{

WallHFVariable::WallHFVariable( std::string task_name, int matl_index, MaterialManagerP materialManager ) :
  TaskInterface( task_name, matl_index ), _materialManager(materialManager) {

  _flux_x = task_name + "_x";
  _flux_y = task_name + "_y";
  _flux_z = task_name + "_z";
  _net_power = task_name + "_power";

}

WallHFVariable::~WallHFVariable(){
}

void
WallHFVariable::problemSetup( ProblemSpecP& db ){

  db->getWithDefault("frequency",_f,1);

  _new_variables = false;
  if ( db->findBlock("new_model"))
    _new_variables = true;

  _area = m_task_name + "_area";

}

void
WallHFVariable::create_local_labels(){

  register_new_variable<CCVariable<double> >( _flux_x );
  register_new_variable<CCVariable<double> >( _flux_y );
  register_new_variable<CCVariable<double> >( _flux_z );
  register_new_variable<CCVariable<double> >( _net_power );
  register_new_variable<CCVariable<double> >( m_task_name );
  register_new_variable<CCVariable<double> >( _area );

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void
WallHFVariable::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks){

  register_variable( _flux_x, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _flux_y, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _flux_z, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _net_power, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _area, ArchesFieldContainer::COMPUTES, variable_registry );

}

void
WallHFVariable::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& flux_x = *(tsk_info->get_uintah_field<CCVariable<double> >(_flux_x));
  CCVariable<double>& flux_y = *(tsk_info->get_uintah_field<CCVariable<double> >(_flux_y));
  CCVariable<double>& flux_z = *(tsk_info->get_uintah_field<CCVariable<double> >(_flux_z));
  CCVariable<double>& power  = *(tsk_info->get_uintah_field<CCVariable<double> >(_net_power));
  CCVariable<double>& total  = *(tsk_info->get_uintah_field<CCVariable<double> >(m_task_name));
  CCVariable<double>& area   = *(tsk_info->get_uintah_field<CCVariable<double> >(_area));

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    flux_x(i,j,k) = 0.0;
    flux_y(i,j,k) = 0.0;
    flux_z(i,j,k) = 0.0;
    power(i,j,k) = 0.0;
    total(i,j,k) = 0.0;
    area(i,j,k)= 0.0;
  });

}

void
WallHFVariable::register_restart_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

  if ( _new_variables ) {

    register_variable( _flux_x, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( _flux_y, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( _flux_z, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( _net_power, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( _area, ArchesFieldContainer::COMPUTES, variable_registry );

  }

}

void
WallHFVariable::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& flux_x = *(tsk_info->get_uintah_field<CCVariable<double> >(_flux_x));
  CCVariable<double>& flux_y = *(tsk_info->get_uintah_field<CCVariable<double> >(_flux_y));
  CCVariable<double>& flux_z = *(tsk_info->get_uintah_field<CCVariable<double> >(_flux_z));
  CCVariable<double>& power  = *(tsk_info->get_uintah_field<CCVariable<double> >(_net_power));
  CCVariable<double>& total  = *(tsk_info->get_uintah_field<CCVariable<double> >(m_task_name));
  CCVariable<double>& area   = *(tsk_info->get_uintah_field<CCVariable<double> >(_area));

  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range, [&](int i, int j, int k){
    flux_x(i,j,k) = 0.0;
    flux_y(i,j,k) = 0.0;
    flux_z(i,j,k) = 0.0;
    power(i,j,k) = 0.0;
    total(i,j,k) = 0.0;
    area(i,j,k)= 0.0;
  });

}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void
WallHFVariable::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

  register_variable( _flux_x, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _flux_y, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _flux_z, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _net_power, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( _area, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( "radiationFluxE", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "radiationFluxW", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "radiationFluxN", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "radiationFluxS", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "radiationFluxT", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "radiationFluxB", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "volFraction", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( "radiation_temperature", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::NEWDW, variable_registry );
  register_variable( _flux_x, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( _flux_y, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( _flux_z, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( _net_power, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( m_task_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );
  register_variable( _area, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry );

}

void
WallHFVariable::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  double sigma=5.67e-8;  //  w / m^2 k^4

  constCCVariable<double>* Fe = tsk_info->get_const_uintah_field<constCCVariable<double> >("radiationFluxE");
  constCCVariable<double>* Fw = tsk_info->get_const_uintah_field<constCCVariable<double> >("radiationFluxW");
  constCCVariable<double>* Fn = tsk_info->get_const_uintah_field<constCCVariable<double> >("radiationFluxN");
  constCCVariable<double>* Fs = tsk_info->get_const_uintah_field<constCCVariable<double> >("radiationFluxS");
  constCCVariable<double>* Ft = tsk_info->get_const_uintah_field<constCCVariable<double> >("radiationFluxT");
  constCCVariable<double>* Fb = tsk_info->get_const_uintah_field<constCCVariable<double> >("radiationFluxB");
  constCCVariable<double>* T = tsk_info->get_const_uintah_field<constCCVariable<double> >("radiation_temperature");
  constCCVariable<double>* volFraction = tsk_info->get_const_uintah_field<constCCVariable<double> >("volFraction");

  CCVariable<double>* flux_x = tsk_info->get_uintah_field<CCVariable<double> >(_flux_x);
  CCVariable<double>* flux_y = tsk_info->get_uintah_field<CCVariable<double> >(_flux_y);
  CCVariable<double>* flux_z = tsk_info->get_uintah_field<CCVariable<double> >(_flux_z);
  CCVariable<double>* power  = tsk_info->get_uintah_field<CCVariable<double> >(_net_power);
  CCVariable<double>* total  = tsk_info->get_uintah_field<CCVariable<double> >(m_task_name);
  CCVariable<double>* area   = tsk_info->get_uintah_field<CCVariable<double> >(_area);

  (*flux_x).initialize(0.0);
  (*flux_y).initialize(0.0);
  (*flux_z).initialize(0.0);
  (*power).initialize(0.0);
  (*total).initialize(0.0);
  (*area).initialize(0.0);

  Vector DX = patch->dCell();

//   int timeStep = _materialManager->getCurrentTopLevelTimeStep();
  int timeStep = tsk_info->get_timeStep();
 
  //if ( ( timeStep )%_f + 1 == 1 ){
  if ( ( timeStep )%_f  == 0 ) {

    for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ) {

      IntVector c = *iter;
      IntVector cxp = c + IntVector(1,0,0);
      IntVector cxm = c - IntVector(1,0,0);
      IntVector cyp = c + IntVector(0,1,0);
      IntVector cym = c - IntVector(0,1,0);
      IntVector czp = c + IntVector(0,0,1);
      IntVector czm = c - IntVector(0,0,1);

      if ( (*volFraction)[c] < 1.0 ) {

        double Q_in = 0.0;
        double Q_emit = 0.0;
        double darea = 0.0;

        ////check neighbors to see if we populate a flux here:
        double a = DX.y()*DX.z();
        if ( (*volFraction)[cxm] > 0.0 ) {
          (*flux_x)[c] = (*flux_x)[c] + (*Fe)[cxm];
          Q_in += (*Fe)[cxm]*a;
          Q_emit += sigma*(*T)[c]*(*T)[c]*(*T)[c]*(*T)[c]*a;
          darea += a;
        }
        if ( (*volFraction)[cxp] > 0.0 ) {
          (*flux_x)[c] = (*flux_x)[c] + (*Fw)[cxp];
          Q_in += (*Fw)[cxp]*a;
          Q_emit += sigma*(*T)[c]*(*T)[c]*(*T)[c]*(*T)[c]*a;
          darea += a;
        }
        a = DX.x()*DX.z();
        if ( (*volFraction)[cym] > 0.0 ) {
          (*flux_y)[c] = (*flux_y)[c] + (*Fn)[cym];
          Q_in += (*Fn)[cym]*a;
          Q_emit += sigma*(*T)[c]*(*T)[c]*(*T)[c]*(*T)[c]*a;
          darea += a;
        }
        if ( (*volFraction)[cyp] > 0.0 ) {
          (*flux_y)[c] = (*flux_y)[c] + (*Fs)[cyp];
          Q_in += (*Fs)[cyp]*a;
          Q_emit += sigma*(*T)[c]*(*T)[c]*(*T)[c]*(*T)[c]*a;
          darea += a;
        }
        a = DX.x()*DX.y();
        if ( (*volFraction)[czm] > 0.0 ) {
          (*flux_z)[c] = (*flux_z)[c] + (*Ft)[czm];
          Q_in += (*Ft)[czm]*a;
          Q_emit += sigma*(*T)[c]*(*T)[c]*(*T)[c]*(*T)[c]*a;
          darea += a;
        }
        if ( (*volFraction)[czp] > 0.0 ) {
          (*flux_z)[c] = (*flux_z)[c] + (*Fb)[czp];
          Q_in += (*Fb)[czp]*a;
          Q_emit += sigma*(*T)[c]*(*T)[c]*(*T)[c]*(*T)[c]*a;
          darea += a;
        }

        (*total)[c] = Q_in/(darea+SMALLNUM);
        (*area)[c] = darea;
        (*power)[c] = Q_in - Q_emit;

      }
    }

    std::vector<Patch::FaceType>::const_iterator bf_iter;
    std::vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    Vector A = Vector(DX.y()*DX.z(), DX.x()*DX.z(), DX.x()*DX.y());

    for (bf_iter = bf.begin(); bf_iter !=bf.end(); bf_iter++) {

      Patch::FaceType face = *bf_iter;
      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
      IntVector insideCellDir = patch->faceDirection(face);
      int P_dir = patch->getFaceAxes(face)[0];
      double a = A[P_dir];

      for (CellIterator citer =  patch->getFaceIterator(face, PEC); !citer.done(); citer++) {

        IntVector c = *citer;
        IntVector cxp = c - insideCellDir;

        double Q_in = 0.;
        double Q_emit = 0.;
        double darea = 0.;

        constCCVariable<double>* F;
        CCVariable<double>* flux;
        if ( P_dir == 0 ) {
          flux = flux_x;
          if ( insideCellDir[0] == -1 ) {
            F = Fw;
          } else {
            F = Fe;
          }
        } else if ( P_dir == 1 ) {
          flux = flux_y;
          if ( insideCellDir[1] == -1 ) {
            F = Fs;
          } else {
            F = Fn;
          }
        } else {
          flux = flux_z;
          if ( insideCellDir[2] == -1 ) {
            F = Fb;
          } else {
            F = Ft;
          }
        }

        if ( (*volFraction)[c] < SMALLNUM ) {
          if ( (*volFraction)[cxp] > 0.0 ) {
            (*flux)[c] = (*flux)[c] + (*F)[cxp];
            Q_in   += (*F)[cxp]*a;
            Q_emit += sigma*(*T)[c]*(*T)[c]*(*T)[c]*(*T)[c]*a;
            darea  += a;
          }
        }

        (*total)[c] = Q_in / (darea+SMALLNUM);
        (*area)[c]  = darea;
        (*power)[c] = Q_in - Q_emit;

      }
    }
  } else {

    constCCVariable<double>* old_flux_x = tsk_info->get_const_uintah_field<constCCVariable<double> >(_flux_x);
    constCCVariable<double>* old_flux_y = tsk_info->get_const_uintah_field<constCCVariable<double> >(_flux_y);
    constCCVariable<double>* old_flux_z = tsk_info->get_const_uintah_field<constCCVariable<double> >(_flux_z);
    constCCVariable<double>* old_power  = tsk_info->get_const_uintah_field<constCCVariable<double> >(_net_power);
    constCCVariable<double>* old_total  = tsk_info->get_const_uintah_field<constCCVariable<double> >(m_task_name);
    constCCVariable<double>* old_area   = tsk_info->get_const_uintah_field<constCCVariable<double> >(_area);

    (*flux_x).copyData(*old_flux_x);
    (*flux_y).copyData(*old_flux_y);
    (*flux_z).copyData(*old_flux_z);
    (*power).copyData(*old_power);
    (*total).copyData((*old_total));
    (*area).copyData(*old_area);

  }
}
} //namspace Uintah
