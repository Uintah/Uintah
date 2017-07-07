#ifndef Uintah_Component_Arches_VariableStats_h
#define Uintah_Component_Arches_VariableStats_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{

  class VariableStats : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    VariableStats( std::string task_name, int matl_index );
    ~VariableStats();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ); 

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();


    //Build instructions for this (VariableStats) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      VariableStats* build()
      { return scinew VariableStats( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int _matl_index;

    };

private:

    std::vector<const VarLabel*> _ave_sum_labels;
    std::vector<const VarLabel*> _ave_flux_sum_labels;
    std::vector<const VarLabel*> _sqr_sum_labels;

    //single variables
    std::vector<std::string> _ave_sum_names;
    std::vector<std::string> _base_var_names;
    std::vector<std::string> _new_variables;
    std::vector<std::string> _sqr_variable_names;

    std::string _rho_name;

    //fluxes
    bool _no_flux;
    struct FluxInfo{
      std::string phi;
      bool do_phi;
    };

    std::vector<std::string> _ave_x_flux_sum_names;
    std::vector<std::string> _ave_y_flux_sum_names;
    std::vector<std::string> _ave_z_flux_sum_names;

    std::vector<std::string> _x_flux_sqr_sum_names;
    std::vector<std::string> _y_flux_sqr_sum_names;
    std::vector<std::string> _z_flux_sqr_sum_names;
    std::vector<FluxInfo>    _flux_sum_info;


  }; //end class header
}
#endif

    ////Uintah implementation
    ////Single Variables
    //for ( int i = 0; i < N; i++ ){

      //CCVariable<double>* sump          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_sum_names[i]);
      //constCCVariable<double>* varp     = tsk_info->get_const_uintah_field<constCCVariable<double> >(_base_var_names[i]);
      //constCCVariable<double>* old_sump = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_sum_names[i]);

      //CCVariable<double>& sum          = *sump;
      //constCCVariable<double>& var     = *varp;
      //constCCVariable<double>& old_sum = *old_sump;

      //sum.initialize(0.0);

      //for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {

        //IntVector c = *iter;

        //sum[c] = old_sum[c] + dt * var[c];

      //}
    //}

    ////Fluxes
    //if ( !_no_flux ){
      //constCCVariable<double>* rhop = tsk_info->get_const_uintah_field<constCCVariable<double> >(_rho_name);
      //constSFCXVariable<double>* up = tsk_info->get_const_uintah_field<constSFCXVariable<double> >("uVelocitySPBC");
      //constSFCYVariable<double>* vp = tsk_info->get_const_uintah_field<constSFCYVariable<double> >("vVelocitySPBC");
      //constSFCZVariable<double>* wp = tsk_info->get_const_uintah_field<constSFCZVariable<double> >("wVelocitySPBC");

      //constCCVariable<double>& rho = *rhop;
      //constSFCXVariable<double>& u = *up;
      //constSFCYVariable<double>& v = *vp;
      //constSFCZVariable<double>& w = *wp;

      //N = _ave_x_flux_sum_names.size();

      //for ( int i = 0; i < N; i++ ){

        //CCVariable<double>* sump_x          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_x_flux_sum_names[i]);
        //constCCVariable<double>* old_sump_x = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_x_flux_sum_names[i]);
        //CCVariable<double>* sump_y          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_y_flux_sum_names[i]);
        //constCCVariable<double>* old_sump_y = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_y_flux_sum_names[i]);
        //CCVariable<double>* sump_z          = tsk_info->get_uintah_field<CCVariable<double> >(_ave_z_flux_sum_names[i]);
        //constCCVariable<double>* old_sump_z = tsk_info->get_const_uintah_field<constCCVariable<double> >(_ave_z_flux_sum_names[i]);
        //constCCVariable<double>* phip;

        //if ( _flux_sum_info[i].do_phi)
          //phip = tsk_info->get_const_uintah_field<constCCVariable<double> >(_flux_sum_info[i].phi);

        //CCVariable<double>& sum_x          = *sump_x;
        //constCCVariable<double>& old_sum_x = *old_sump_x;
        //CCVariable<double>& sum_y          = *sump_y;
        //constCCVariable<double>& old_sum_y = *old_sump_y;
        //CCVariable<double>& sum_z          = *sump_z;
        //constCCVariable<double>& old_sum_z = *old_sump_z;

        //sum_x.initialize(12.0);
        //sum_y.initialize(0.0);
        //sum_z.initialize(0.0);

        //for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {

          //IntVector c = *iter;

          //if ( _flux_sum_info[i].do_phi ){

            //sum_x[c] = old_sum_x[c] + dt * rho[c] * ( u[c] + u[c+IntVector(1,0,0)] )/2.0 * (*phip)[c];
            //sum_y[c] = old_sum_y[c] + dt * rho[c] * ( v[c] + v[c+IntVector(0,1,0)] )/2.0 * (*phip)[c];
            //sum_z[c] = old_sum_z[c] + dt * rho[c] * ( w[c] + w[c+IntVector(0,0,1)] )/2.0 * (*phip)[c];

          //} else {

            //sum_x[c] = old_sum_x[c] + dt * rho[c] * ( u[c] + u[c+IntVector(1,0,0)] )/2.0 ;
            //sum_y[c] = old_sum_y[c] + dt * rho[c] * ( v[c] + v[c+IntVector(0,1,0)] )/2.0 ;
            //sum_z[c] = old_sum_z[c] + dt * rho[c] * ( w[c] + w[c+IntVector(0,0,1)] )/2.0 ;

          //}

        //}
      //}
    //}
