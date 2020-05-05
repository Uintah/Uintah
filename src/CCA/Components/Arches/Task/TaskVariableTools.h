#ifndef Uintah_Component_Arches_TaskVariableTools_h
#define Uintah_Component_Arches_TaskVariableTools_h

#include <CCA/Components/Arches/Task/FieldContainer.h>

//==================================================================================================

/**
* @class  Task Variable Tools
* @author Jeremy Thornock
* @date   2016
*
* @brief Tools associated with variable management within a Task.
*
**/

//==================================================================================================

namespace Uintah {

  /** @brief A container to hold a small amount of other information to pass into the task exe. **/
  struct SchedToTaskInfo {
    int    time_substep{99};
    int    timeStep{0};
    double time{0.0};
    double dt{0.};
    bool   packed_tasks{false};
  };

  /** @brief A class for managing the retrieval of uintah/so fields during task exe **/
  class ArchesTaskInfoManager {

    public:

      enum MAPCHECK { CHECK_FIELD, CONST_FIELD, NONCONST_FIELD };

      ArchesTaskInfoManager(       std::vector<ArchesFieldContainer::VariableInformation> & var_reg
                           , const Patch                                                  * patch
                           ,       SchedToTaskInfo                                        & info
                           )
        : _var_reg( var_reg )
        , _patch( patch )
        , _tsk_info( info )
      {};

      ~ArchesTaskInfoManager(){};

      /** @brief return the time substep **/
      inline int get_time_substep(){ return _tsk_info.time_substep; };

      /** @brief return the dt **/
      inline double get_dt(){ return _tsk_info.dt; };

      /** @brief return the time step **/
      inline int get_timeStep(){ return _tsk_info.timeStep; };

      /** @brief return the time **/
      inline double get_time(){ return _tsk_info.time; };

      /** @brief set the time **/
      inline void set_time(const double time){ _tsk_info.time = time; };

      /** @brief Return a bool to indicate if this Arches Task is a subset of a larger, single Uintah task. **/
      inline bool packed_tasks(){ return _tsk_info.packed_tasks; }

      /** @brief return the variable registry **/
      inline std::vector<ArchesFieldContainer::VariableInformation>& get_variable_reg(){ return _var_reg; }

      /** @brief Set the references to the variable maps in the Field Collector for easier
       *         management of the fields when trying to retrieve from the DW **/
      void set_field_container(ArchesFieldContainer* field_container){ _field_container = field_container; }

      /** @brief Get a reference to the field container **/
      ArchesFieldContainer* getFieldContainer(){ return _field_container; };

      /** @brief Return the spp time factor: t_ssp = t + factor * dt **/
      inline double get_ssp_time_factor( const int rk_step )
      {
        if ( rk_step == 0 ) {
          return 0.0;
        } else if ( rk_step == 1 ) {
          return 1.0;
        } else {
          return 0.5;
        }
      }

      //====================================================================================
      // GRID VARIABLE ACCESS
      //====================================================================================

      /** @brief Return a CONST grid variable **/
      template <typename T, typename... Args>
      typename std::enable_if<std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value, T&>::type
      get_field( const std::string name, Args... args )
      {
        return *( _field_container->get_const_field<T>(name, args...) );
      }

      /** @brief Return a CONST grid variable **/
      template <typename T, typename PODType, typename MemSpace, typename... Args>
      typename std::enable_if<std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value && std::is_same<MemSpace, UintahSpaces::HostSpace>::value, T>::type
      get_field( const std::string name, Args... args )
      {
        return *( _field_container->get_const_field<T>(name, args...) );
      }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
      /** @brief Return a CONST grid variable **/
      template <typename T, typename PODType, typename MemSpace, typename... Args>
      typename std::enable_if<std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value && std::is_same<MemSpace, Kokkos::HostSpace>::value, KokkosView3<PODType, Kokkos::HostSpace>>::type
      get_field( const std::string name, Args... args )
      {
        return (*( _field_container->get_const_field<T>(name, args...) )).getKokkosView();
      }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
      /** @brief Return a CONST grid variable **/
      template <typename T, typename PODType, typename MemSpace>
      typename std::enable_if<std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value && std::is_same<MemSpace, Kokkos::CudaSpace>::value, KokkosView3<PODType, Kokkos::CudaSpace>>::type
      get_field( const std::string name )
      {
        int which_dw = _field_container->get_variable_information( name, true ).uintah_task_dw;

        if ( which_dw == ArchesFieldContainer::OLDDW || ( which_dw == ArchesFieldContainer::LATEST && _tsk_info.time_substep == 0 ) ) {
          return getOldDW()->getGPUDW()->getKokkosView<PODType>( name.c_str(), _field_container->getPatch()->getID(), _field_container->getMaterialIndex(), _field_container->getPatch()->getLevel()->getIndex() );
        } else {
          return getNewDW()->getGPUDW()->getKokkosView<PODType>( name.c_str(), _field_container->getPatch()->getID(), _field_container->getMaterialIndex(), _field_container->getPatch()->getLevel()->getIndex() );
        }
      }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
      /** @brief Return a CONST grid variable **/
      template <typename T, typename PODType, typename MemSpace>
      typename std::enable_if<std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value && std::is_same<MemSpace, Kokkos::CudaSpace>::value, KokkosView3<PODType, Kokkos::CudaSpace>>::type
      get_field( const std::string                    name
               ,       ArchesFieldContainer::WHICH_DW which_dw
               )
      {
        if ( which_dw == ArchesFieldContainer::OLDDW || ( which_dw == ArchesFieldContainer::LATEST && _tsk_info.time_substep == 0 ) ) {
          return getOldDW()->getGPUDW()->getKokkosView<PODType>( name.c_str(), _field_container->getPatch()->getID(), _field_container->getMaterialIndex(), _field_container->getPatch()->getLevel()->getIndex() );
        } else {
          return getNewDW()->getGPUDW()->getKokkosView<PODType>( name.c_str(), _field_container->getPatch()->getID(), _field_container->getMaterialIndex(), _field_container->getPatch()->getLevel()->getIndex() );
        }
      }
#endif

      /** @brief Return a NON-CONST grid variable **/
      template <typename T, typename... Args>
      typename std::enable_if<!std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value, T&>::type
      get_field( const std::string name, Args... args )
      {
        return  *(_field_container->get_field<T>(name, args...));
      }

      /** @brief Return a NON-CONST grid variable **/
      template <typename T, typename PODType, typename MemSpace, typename... Args>
      typename std::enable_if<!std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value && std::is_same<MemSpace, UintahSpaces::HostSpace>::value, T>::type
      get_field( const std::string name, Args... args )
      {
        return  *(_field_container->get_field<T>(name, args...));
      }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
      /** @brief Return a NON-CONST grid variable **/
      template <typename T, typename PODType, typename MemSpace, typename... Args>
      typename std::enable_if<!std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value && std::is_same<MemSpace, Kokkos::HostSpace>::value, KokkosView3<PODType, Kokkos::HostSpace>>::type
      get_field( const std::string name, Args... args )
      {
        return (*( _field_container->get_field<T>(name, args...))).getKokkosView();
      }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
      /** @brief Return a NON-CONST grid variable **/
      template <typename T, typename PODType, typename MemSpace>
      typename std::enable_if<!std::is_base_of<Uintah::constVariableBase<Uintah::GridVariableBase>, T>::value && std::is_same<MemSpace, Kokkos::CudaSpace>::value, KokkosView3<PODType, Kokkos::CudaSpace>>::type
      get_field( const std::string name )
      {
        return getNewDW()->getGPUDW()->getKokkosView<PODType>( name.c_str(), _field_container->getPatch()->getID(), _field_container->getMaterialIndex(), _field_container->getPatch()->getLevel()->getIndex() );
      }
#endif

      template <typename T, typename PODType, typename MemSpace>
      inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, T >::type
      get_empty_field()
      {
        return T();
      }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
      template <typename T, typename PODType, typename MemSpace>
      inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, KokkosView3<PODType, Kokkos::HostSpace> >::type
      get_empty_field()
      {
        KokkosView3<PODType, Kokkos::HostSpace> temp;
        return temp;
      }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
      template <typename T, typename PODType, typename MemSpace>
      inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, KokkosView3<PODType, Kokkos::CudaSpace> >::type
      get_empty_field()
      {
        KokkosView3<PODType, Kokkos::CudaSpace> temp;
        return temp;
      }
#endif

      /** @brief Return a UINTAH field allowing the user to manage the memory. **/
      template <typename FIELD_TYPE>
      inline void
      get_unmanaged_uintah_field( const std::string   name
                                ,       FIELD_TYPE  & field
                                )
      {
        _field_container->get_unmanaged_field( name, field );
      }

      template <typename T, typename ElemType, typename MemSpace, typename FIELD_TYPE>
      inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, void >::type
      get_unmanaged_uintah_field(       FIELD_TYPE  & field
                                , const std::string   name
                                , const int           patch
                                , const int           matl_indx
                                , const int           DW_index
                                )
      {
        _field_container->get_unmanaged_field<T>( name, field );
      }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
      template <typename T, typename ElemType, typename MemSpace, typename FIELD_TYPE>
      inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, void >::type
      get_unmanaged_uintah_field(       FIELD_TYPE  & field
                                , const std::string   name
                                , const int           patch
                                , const int           matl_indx
                                , const int           DW_index
                                )
      {
        field = (*(_field_container->get_field<T>( name ))).getKokkosView();
      }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
      template <typename T, typename ElemType, typename MemSpace, typename FIELD_TYPE>
      inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, void >::type
      get_unmanaged_uintah_field(       FIELD_TYPE  & field
                                , const std::string   name
                                , const int           patch
                                , const int           matl_indx
                                , const int           DW_index
                                )
      {
        field = getNewDW()->getGPUDW()->getKokkosView<ElemType>( name.c_str(), patch, matl_indx, _field_container->getPatch()->getLevel()->getIndex() );
      }
#endif

      template <typename T, typename ElemType, typename MemSpace>
      inline typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, void >::type
      get_unmanaged_uintah_field( const std::string   name
                                ,       T           & field
                                )
      {
        _field_container->get_unmanaged_field( name, field );
      }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
      template <typename T, typename ElemType, typename MemSpace>
      inline typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, void >::type
      get_unmanaged_uintah_field( const std::string                     name
                                ,       KokkosView3<double, MemSpace> & field
                                )
      {
        field = (*(_field_container->get_field<T>( name ))).getKokkosView();
      }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
      template <typename T, typename ElemType, typename MemSpace>
      inline typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, void >::type
      get_unmanaged_uintah_field( const std::string                     name
                                ,       KokkosView3<double, MemSpace> & field
                                )
      {
        field = getNewDW()->getGPUDW()->getKokkosView<ElemType>( name.c_str(), _field_container->getPatch()->getID(), _field_container->getMaterialIndex(), _field_container->getPatch()->getLevel()->getIndex() );
      }
#endif

      /** @brief Return a CONST UINTAH field allowing the user to manage the memory. **/
      template <typename T>
      inline void
      get_const_unmanaged_uintah_field( const std::string   name
                                      ,       T           & field
                                      )
      {
        _field_container->get_const_unmanaged_field<T>( name, field );
      }

      /** @brief Return a UINTAH particle field **/
      std::tuple<ParticleVariable<double>*, ParticleSubset*>
      get_uintah_particle_field( const std::string name )
      {
        return _field_container->get_uintah_particle_field( name );
      }

      /** @brief Return a const UINTAH particle field **/
      std::tuple<constParticleVariable<double>*, ParticleSubset*>
      get_const_uintah_particle_field( const std::string name )
      {
        return _field_container->get_const_uintah_particle_field( name );
      }

      /** @brief Get the current patch ID **/
      inline int get_patch_id(){ return _patch->getID(); }

      /** @brief get NEW DW reference **/
      DataWarehouse* getNewDW(){ return _field_container->getNewDW(); }

      /** @brief get an OLD DW reference **/
      DataWarehouse* getOldDW(){ return _field_container->getOldDW(); }

    private:

      ArchesFieldContainer* _field_container{nullptr};

            std::vector<ArchesFieldContainer::VariableInformation>   _var_reg;
      const Patch                                                  * _patch{nullptr};
            SchedToTaskInfo                                        & _tsk_info;

  }; // ArchesTaskInfoManager

  // The createContainer() calls.
  // Legacy tasks need to create a container of things like CCVariable<double>.  Vector are ok here.
  // Kokkos tasks can't use vectors (not allowed for CUDA).  Further, the containers must be of Kokkos views of
  // an internal data type.  For portability, both kinds of types are passed in, and only one of the two is used
  // depending on the portability mode.

  template <typename T, typename elemType, unsigned int MaxCapacity, typename MemSpace>
  typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, struct1DArray<T, MaxCapacity> >::type
  createContainer( int num = MaxCapacity )
  {
    return struct1DArray<T, MaxCapacity>( num ); // perform deep copy (should be ok since it is an empty CCVariable?)
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename T, typename elemType, unsigned int MaxCapacity, typename MemSpace>
  typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, struct1DArray<KokkosView3<elemType, MemSpace>, MaxCapacity> >::type
  createContainer( int num = MaxCapacity )
  {
    return struct1DArray<KokkosView3<elemType, MemSpace>, MaxCapacity>( num );
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename T, typename elemType, unsigned int MaxCapacity, typename MemSpace>
  typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, struct1DArray<KokkosView3<elemType, MemSpace>, MaxCapacity> >::type
  createContainer( int num = MaxCapacity )
  {
    return struct1DArray<KokkosView3<elemType, MemSpace>, MaxCapacity>( num );
  }
#endif

  template <typename T, typename elemType, unsigned int MaxCapacity, typename MemSpace>
  typename std::enable_if< std::is_same< MemSpace, UintahSpaces::HostSpace >::value, struct1DArray<T, MaxCapacity> >::type
  createConstContainer( int num = MaxCapacity )
  {
    return struct1DArray<T, MaxCapacity>( num ); // perform deep copy (should be ok since it is an empty CCVariable?)
  }

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
  template <typename T, typename elemType, unsigned int MaxCapacity, typename MemSpace>
  typename std::enable_if< std::is_same< MemSpace, Kokkos::HostSpace >::value, struct1DArray<KokkosView3<elemType, MemSpace>, MaxCapacity> >::type
  createConstContainer( int num = MaxCapacity )
  {
    return struct1DArray<KokkosView3<const elemType, MemSpace>, MaxCapacity>( num );
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template <typename T, typename elemType, unsigned int MaxCapacity, typename MemSpace>
  typename std::enable_if< std::is_same< MemSpace, Kokkos::CudaSpace >::value, struct1DArray<KokkosView3<elemType, MemSpace>, MaxCapacity> >::type
  createConstContainer( int num = MaxCapacity )
  {
    return struct1DArray<KokkosView3<const elemType, MemSpace>, MaxCapacity>( num );
  }
#endif

  /** @brief Builds a struct for each variable containing all pertinent uintah DW information **/
  void register_variable_work(       std::string                              name
                             ,       ArchesFieldContainer::VAR_DEPEND         dep
                             ,       int                                      nGhost
                             ,       ArchesFieldContainer::WHICH_DW           dw
                             ,       ArchesFieldContainer::VariableRegistry & variable_registry
                             , const int                                      time_substep
                             , const std::string                              task_name
                             );

  /** @brief Inteface to register_variable_work -- this function is overloaded. **/
  void register_variable(       std::string                                              name
                        ,       ArchesFieldContainer::VAR_DEPEND                         dep
                        ,       int                                                      nGhost
                        ,       ArchesFieldContainer::WHICH_DW                           dw
                        ,       std::vector<ArchesFieldContainer::VariableInformation> & var_reg
                        , const int                                                      time_substep
                        ,       std::string task_name          = "(Arches task name not not communicated to this variable registration)"
                        , const bool        temporary_variable = false
                        );

  /** @brief Inteface to register_variable_work -- this function is overloaded. **/
  void register_variable(       std::string                                              name
                        ,       ArchesFieldContainer::VAR_DEPEND                         dep
                        ,       int                                                      nGhost
                        ,       ArchesFieldContainer::WHICH_DW                           dw
                        ,       std::vector<ArchesFieldContainer::VariableInformation> & var_reg
                        ,       std::string task_name          = "(Arches task name not not communicated to this variable registration)"
                        , const bool        temporary_variable = false
                        );

  /** @brief Inteface to register_variable_work -- this function is overloaded.
   *         This version assumes NewDW and zero ghosts. **/
  void register_variable(       std::string                                              name
                        ,       ArchesFieldContainer::VAR_DEPEND                         dep
                        ,       std::vector<ArchesFieldContainer::VariableInformation> & var_reg
                        ,       std::string task_name          = "(Arches task name not not communicated to this variable registration)"
                        , const bool        temporary_variable = false
                        );

  /** @brief Inteface to register_variable_work -- this function is overloaded.
   *         This version assumes NewDW and zero ghosts and passes the timesubstep. **/
  void register_variable(       std::string                                              name
                        ,       ArchesFieldContainer::VAR_DEPEND                         dep
                        ,       std::vector<ArchesFieldContainer::VariableInformation> & var_reg
                        , const int                                                      timesubstep
                        ,       std::string task_name          = "(Arches task name not not communicated to this variable registration)"
                        , const bool        temporary_variable = false
                        );

  /** @brief Helper struct when resolving get_field becomes tricky **/
  template <typename T>
  struct FieldTool
  {
    FieldTool(ArchesTaskInfoManager* tsk_info){
      throw InvalidValue("Error: I should not be in this constructor.",__FILE__,__LINE__);
    }

    T* get( const std::string name ) {
      throw InvalidValue("Error: I should not be called.",__FILE__,__LINE__);
    }
  };

  template <>
  struct FieldTool<CCVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    CCVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_field<CCVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

  template <>
  struct FieldTool<constCCVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    constCCVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_const_field<constCCVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

  template <>
  struct FieldTool<SFCXVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    SFCXVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_field<SFCXVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

  template <>
  struct FieldTool<constSFCXVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    constSFCXVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_const_field<constSFCXVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

  template <>
  struct FieldTool<SFCYVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    SFCYVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_field<SFCYVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

  template <>
  struct FieldTool<constSFCYVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    constSFCYVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_const_field<constSFCYVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

  template <>
  struct FieldTool<SFCZVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    SFCZVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_field<SFCZVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

  template <>
  struct FieldTool<constSFCZVariable<double> >
  {
    FieldTool(ArchesTaskInfoManager* tsk_info):m_tsk_info(tsk_info){}

    constSFCZVariable<double>& get( const std::string name ){
      ArchesFieldContainer* field_container = m_tsk_info->getFieldContainer();
      return *(field_container->get_const_field<constSFCZVariable<double> >( name ));
    }

    private:
      ArchesTaskInfoManager* m_tsk_info;
  };

} // namespace Uintah

#endif
