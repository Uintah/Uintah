#ifndef Uintah_Component_Arches_SurfaceVolumeFractionCalc_h
#define Uintah_Component_Arches_SurfaceVolumeFractionCalc_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

namespace Uintah{

  class SurfaceVolumeFractionCalc : TaskInterface {

  public:

    SurfaceVolumeFractionCalc( std::string task_name, int matl_index ) : TaskInterface( task_name,
      matl_index){};
    ~SurfaceVolumeFractionCalc(){};

    typedef std::vector<ArchesFieldContainer::VariableInformation> ArchesVIVector;

    void problemSetup( ProblemSpecP& db );

    void register_initialize( ArchesVIVector& variable_registry , const bool packed_tasks);

    void register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks);

    void register_timestep_eval( ArchesVIVector& variable_registry,
                                 const int time_substep, const bool packed_tasks ){};

    void register_compute_bcs( ArchesVIVector& variable_registry,
                               const int time_substep, const bool packed_tasks ){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){};

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){};

    void create_local_labels();

    //Build instructions for this (KScalarRHS) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index )
      : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      SurfaceVolumeFractionCalc* build()
      { return scinew SurfaceVolumeFractionCalc( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

  private:

    std::vector<std::string> m_var_names;

    struct IntrusionBoundary{
      std::vector<GeometryPieceP> geometry;
    };

    std::vector<IntrusionBoundary> m_intrusions;

  }; // class SurfaceVolumeFractionCalc

} //end namespace Uintah

#endif
