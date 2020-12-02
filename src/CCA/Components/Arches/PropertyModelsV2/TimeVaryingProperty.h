#ifndef Uintah_Component_Arches_TimeVaryingProperty_h
#define Uintah_Component_Arches_TimeVaryingProperty_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

namespace Uintah{

  template <typename T>
  class TimeVaryingProperty : public TaskInterface {

public:

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    TimeVaryingProperty<T>( std::string task_name, int matl_index );
    ~TimeVaryingProperty<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels(){
      register_new_variable<T>( m_task_name );
    }

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){};

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    //Build instructions for this (TimeVaryingProperty) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
        : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      TimeVaryingProperty* build()
      { return scinew TimeVaryingProperty( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    typedef std::vector<ArchesFieldContainer::VariableInformation> AVarInfo;

    double m_constant;
    bool m_has_regions;
    std::string m_func;

    //sine function:
    double m_c; ///< Center
    double m_P; ///< Period
    double m_A; ///< Amplitute

    //linear function
    double m_m; ///< slope
    double m_b; ///< intercept

    struct ConstantGeomContainer{
      double constant;
      bool inverted;
      std::vector<GeometryPieceP> geometry;
    };

    std::vector<ConstantGeomContainer> m_region_info;

    double sineFun(const double c, const double A, const double p, const double time){
      const double pi = acos(-1);
      return c + A*sin(2.*pi/p*time);
    }

    double lineFun(const double m, const double b, const double time){
      return m*time + b;
    }

  };

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TimeVaryingProperty<T>::TimeVaryingProperty( std::string task_name, int matl_index ) :
  TaskInterface( task_name, matl_index ){
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TimeVaryingProperty<T>::~TimeVaryingProperty(){}

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TimeVaryingProperty<T>::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TimeVaryingProperty<T>::loadTaskInitializeFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TimeVaryingProperty<T>::loadTaskEvalFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TimeVaryingProperty<T>::loadTaskTimestepInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace TimeVaryingProperty<T>::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeVaryingProperty<T>::problemSetup( ProblemSpecP& db ){

    if ( db->findBlock("regions") ){

      ProblemSpecP db_regions = db->findBlock("regions");

      //geometrically bound constants
      m_has_regions = true;

      for ( ProblemSpecP db_r = db_regions->findBlock("region"); db_r != nullptr;
            db_r = db_r->findNextBlock("region") ){

        ConstantGeomContainer region_info;

        if ( db_r->findBlock("inverted")){
          region_info.inverted = true;
        } else {
          region_info.inverted = false;
        }

        ProblemSpecP db_geom = db_r->findBlock("geom_object");
        GeometryPieceFactory::create( db_geom, region_info.geometry );
        db_r->require("value", region_info.constant);
        m_region_info.push_back( region_info );

      }

    } else {

      //just one global constant
      m_has_regions = false;

    }

    db->require("function", m_func);

    if ( m_func == "sine" ){
      // y = c + Amp * sin( 2*pi/P * x)
      db->require("offset", m_c);
      db->require("P", m_P);
      db->require("Amp", m_A);
    } else if ( m_func == "linear" ){
      // y = m*x + intercept
      db->require("m", m_m);
      db->require("intercept", m_b);
    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeVaryingProperty<T>::register_initialize( AVarInfo& variable_registry , const bool pack_tasks){
    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeVaryingProperty<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& property = tsk_info->get_field<T>( m_task_name );

    const double time = tsk_info->get_time();
    double value = 0;

    if ( m_func == "linear"){
      value = lineFun( m_m, m_b, time );
    } else if ( m_func == "sine" ){
      value = sineFun( m_c, m_A, m_P, time );
    }

    if ( m_has_regions ){

      for ( auto region_iter = m_region_info.begin(); region_iter != m_region_info.end();
            region_iter++){

        Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
        Uintah::parallel_for( range, [&](int i, int j, int k){

          //not worried about efficiency since this is only at startup:
          Point p = patch->cellPosition(IntVector(i,j,k));


          for ( auto geom_iter = region_iter->geometry.begin();
                geom_iter != region_iter->geometry.end(); geom_iter++ ){

            GeometryPieceP geom = *geom_iter;

            if ( !region_iter->inverted ){
              if ( geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            } else {
              if ( !geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            }
          }

        });

      }

    } else {

      property.initialize(value);

    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeVaryingProperty<T>::register_restart_initialize( AVarInfo& variable_registry , const bool packed_tasks){
    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeVaryingProperty<T>::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& property = tsk_info->get_field<T>( m_task_name );
    const double time = tsk_info->get_time();
    double value = 0;

    if ( m_func == "linear"){
      value = lineFun( m_m, m_b, time );
    } else if ( m_func == "sine" ){
      value = sineFun( m_c, m_A, m_P, time );
    }

    if ( m_has_regions ){

      for ( auto region_iter = m_region_info.begin(); region_iter != m_region_info.end();
            region_iter++){

        Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
        Uintah::parallel_for( range, [&](int i, int j, int k){

          //not worried about efficiency since this is only at startup:
          Point p = patch->cellPosition(IntVector(i,j,k));


          for ( auto geom_iter = region_iter->geometry.begin();
                geom_iter != region_iter->geometry.end(); geom_iter++ ){

            GeometryPieceP geom = *geom_iter;

            if ( !region_iter->inverted ){
              if ( geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            } else {
              if ( !geom->inside(p) ){
                property(i,j,k) = region_iter->constant;
              }
            }
          }

        });

      }

    } else {

      property.initialize(value);

    }

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeVaryingProperty<T>::register_timestep_init( AVarInfo& variable_registry , const bool packed_tasks){
    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void TimeVaryingProperty<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& property = tsk_info->get_field<T>( m_task_name );

    const double time = tsk_info->get_time();
    double value = 0;

    if ( m_func == "linear"){
      value = lineFun( m_m, m_b, time );
    } else if ( m_func == "sine" ){
      value = sineFun( m_c, m_A, m_P, time );
    }
    property.initialize(value);

  }
}
#endif
