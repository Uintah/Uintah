#ifndef Uintah_Component_Arches_ArchesFieldContainer_h
#define Uintah_Component_Arches_ArchesFieldContainer_h

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>

//==================================================================================================

/**
* @class  Field Interface for Uintah variables
* @author Jeremy Thornock
* @date   2014
*
* @brief Holds fields for use during task execution
*        Also deletes them when the task is finished.
*        This class should remain as lightweight as possible.
*
* @TODO This class is somewhat large. Perhaps the struct definitions should be moved outside this
*       class?
*
**/

//==================================================================================================
namespace Uintah{

  class ArchesFieldContainer{

    public:

      enum VAR_DEPEND { COMPUTES, MODIFIES, REQUIRES };
      enum WHICH_DW { OLDDW, NEWDW, LATEST };

      /** @brief The variable registry information. Each task variable has one of these.
                 It is constructed at schedule time and used to retrieve the Uintah
                 grid varaible during the task callback. **/
      struct VariableInformation {

        std::string name;
        VAR_DEPEND  depend;
        WHICH_DW    dw;
        int         nGhost;
        const VarLabel* label;
        Task::WhichDW uintah_task_dw;
        Ghost::GhostType ghost_type;
        bool        local;
        bool        is_constant;

      };

      typedef std::vector<VariableInformation > VariableRegistry;

      ArchesFieldContainer( const WasatchCore::AllocInfo& alloc_info,
                            const Patch* patch,
                            const int matl_index,
                            const VariableRegistry variable_registry,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw );

      /** @brief A local struct for dealing with the Uintah grid variable. This is
                 used to delete the grid variable. **/
      struct FieldContainer{

        public:
          void set_field( GridVariableBase* field ){ _field = field; }
          void set_label( const VarLabel* label ){ _label = label; }

          template <class T>
          inline
          T* get_field(){ return dynamic_cast<T*>(_field); }

          int get_n_ghost(){ return 0; } //Not allowing for ghosts currently on modifiable fields

          const VarLabel* get_label(){ return _label; }

        private:
          const VarLabel* _label;
          GridVariableBase* _field;
          int _n_ghosts;

      };

      /** @brief A local struct for dealing with the Uintah grid variable. This is
                 used to delete the grid variable. **/
      struct ConstFieldContainer{

        public:
          void set_field( constVariableBase<GridVariableBase>* field ){ _field = field; }
          void set_label( const VarLabel* label ){ _label = label; }

          void set_ghosts( int n_ghosts ){_n_ghosts = n_ghosts; }

          template <class T>
          inline
          T* get_field(){ return dynamic_cast<T*>(_field); }

          //Not allowing for ghosts currently on modifiable fields
          int get_n_ghost(){ return _n_ghosts; }

          const VarLabel* get_label(){ return _label; }

        private:
          constVariableBase<GridVariableBase>* _field;
          const VarLabel* _label;
          int _n_ghosts;

      };

      /** @brief A local struct for dealing with the Uintah grid variable. This is
                 used to delete the grid variable. **/
      struct ParticleFieldContainer{

        public:
          void set_field( ParticleVariable<double>* field ){ _field = field; }
          void set_label( const VarLabel* label ){ _label = label; }

          void set_ghosts( int n_ghosts ){ _n_ghosts = n_ghosts; }

          inline ParticleVariable<double>* get_field(){ return _field; }

          int get_n_ghost(){ return _n_ghosts; }

          const VarLabel* get_label(){ return _label; }

        private:

          //this is specialized now...is there a point to specializing it?
          ParticleVariable<double>* _field;
          const VarLabel* _label;
          int _n_ghosts;

      };

      /** @brief A local struct for dealing with the Uintah grid variable. This is
                 used to delete the grid variable. **/
      struct ConstParticleFieldContainer{

        public:
          void set_field( constParticleVariable<double>* field ){ _field = field; }
          void set_label( const VarLabel* label ){ _label = label; }

          void set_ghosts( int n_ghosts ){ _n_ghosts = n_ghosts; }

          inline constParticleVariable<double>* get_field(){ return _field; }

          const VarLabel* get_label(){ return _label; }

          int get_n_ghost(){ return _n_ghosts; }

        private:

          //this is specialized now...is there a point to specializing it?
          constParticleVariable<double>* _field;
          const VarLabel* _label;
          int _n_ghosts;

      };

      typedef std::map<std::string, FieldContainer> FieldContainerMap;
      typedef std::map<std::string, ConstFieldContainer> ConstFieldContainerMap;
      typedef std::map<std::string, ParticleFieldContainer > UintahParticleMap;
      typedef std::map<std::string, ConstParticleFieldContainer > ConstUintahParticleMap;

      ~ArchesFieldContainer(){
        //delete the fields
        delete_fields();
      }

      /** @brief Add a variable to the non-const variable map **/
      void add_variable( std::string name, FieldContainer container ){
        FieldContainerMap::iterator iter = _nonconst_var_map.find(name);
        if ( iter == _nonconst_var_map.end() ){
          _nonconst_var_map.insert(std::make_pair(name, container));
        } else {
          std::stringstream msg;
          msg << "Error: Trying to add a variable to non_cont field map that is already present: "
            << name << " Check for duplicate request in task." << std::endl;
          throw InvalidValue(msg.str(), __FILE__,__LINE__);
        }
      }

      /** @brief Add a variable to the const variable map **/
      void add_const_variable( std::string name, ConstFieldContainer container ){
        ConstFieldContainerMap::iterator iter = _const_var_map.find(name);
        if ( iter == _const_var_map.end() ){
          _const_var_map.insert(std::make_pair(name, container));
        } else {
          std::stringstream msg;
          msg << "Error: Trying to add a variable to cont field map that is already present: "
            << name << " Check for duplicate request in task." << std::endl;
          throw InvalidValue(msg.str(), __FILE__,__LINE__);
        }
      }

      /** @brief Add a particle variable to the non-const variable map **/
      void add_particle_variable( std::string name, ParticleFieldContainer container ){
        UintahParticleMap::iterator iter = _particle_map.find(name);

        if ( iter == _particle_map.end() ){
          _particle_map.insert(std::make_pair(name, container));
        } else {
          std::stringstream msg;
          msg << "Error: Trying to add particle variable to cont field map that is already present: "
            << name << " Check for duplicate request in task." << std::endl;
          throw InvalidValue(msg.str(), __FILE__,__LINE__);
        }
      }

      /** @brief Add a particle variable to the const variable map **/
      void add_const_particle_variable( std::string name, ConstParticleFieldContainer container ){
        ConstUintahParticleMap::iterator iter = _const_particle_map.find(name);

        if ( iter == _const_particle_map.end() ){
          _const_particle_map.insert(std::make_pair(name, container));
        } else {
          std::stringstream msg;
          msg << "Error: Trying to add const particle variable to cont field map that is " <<
          "already present: "<< name << " Check for duplicate request in task." << std::endl;
          throw InvalidValue(msg.str(), __FILE__,__LINE__);
        }
      }

      //--------------------------------------------------------------------------------------------
      //UINTAH VARIABLE TASK ACCESS:

      /** @brief Get a modifiable uintah variable **/
      template <typename T>
      inline T* get_const_field( const std::string name ){

        ConstFieldContainerMap::iterator icheck = _const_var_map.find( name );
        if ( icheck != _const_var_map.end() ){
          return icheck->second.get_field<T>();
        }

        VariableInformation ivar = get_variable_information( name, true );
        T* field = scinew T;
        if ( ivar.dw == OLDDW ){
          _old_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, ivar.nGhost );
        } else {
          _new_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, ivar.nGhost );
        }

        ConstFieldContainer icontain;
        icontain.set_field(field);
        icontain.set_label(ivar.label);
        this->add_const_variable(name, icontain);

        return field;

      }

      /** @brief Get a modifiable uintah variable with specified DW **/
      template <typename T>
      inline T* get_const_field( const std::string name, WHICH_DW which_dw ){

        std::ostringstream dw_value;
        dw_value << which_dw;

        ConstFieldContainerMap::iterator icheck = _const_var_map.find( name+"_"+dw_value.str() );
        if ( icheck != _const_var_map.end() ){
          return icheck->second.get_field<T>();
        }

        VariableInformation ivar = get_variable_information( name, true, which_dw );
        T* field = scinew T;
        if ( ivar.dw == OLDDW ){
          _old_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, ivar.nGhost );
        } else {
          _new_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, ivar.nGhost );
        }

        ConstFieldContainer icontain;
        icontain.set_field(field);
        icontain.set_label(ivar.label);
        this->add_const_variable(name+"_"+dw_value.str(), icontain);

        return field;

      }

      /** @brief Get a const uintah variable **/
      template <typename T>
      inline T* get_field( const std::string name ){

        FieldContainerMap::iterator icheck = _nonconst_var_map.find( name );
        if ( icheck != _nonconst_var_map.end() ){
          return icheck->second.get_field<T>();
        }

        VariableInformation ivar = get_variable_information( name, false );

        T* field = scinew T;

        if ( ivar.depend == MODIFIES ){
          _new_dw->getModifiable( *field, ivar.label, _matl_index, _patch );
        } else {
          _new_dw->allocateAndPut( *field, ivar.label, _matl_index, _patch );
        }

        FieldContainer icontain;
        icontain.set_field(field);
        icontain.set_label(ivar.label);
        this->add_variable(name, icontain);

        return field;

      }

      //SPATIAL OPS VARIABLE TASK ACCESS:
      // @brief Get a NON-CONSTANT spatialOps representation of the Uintah field.
      template <class ST>
      SpatialOps::SpatFldPtr<ST> get_so_field(const std::string name){

        typedef typename WasatchCore::SelectUintahFieldType<ST>::type MY_TYPE;
        FieldContainerMap::iterator icheck = _nonconst_var_map.find( name );
        VariableInformation ivar = get_variable_information( name, false );

        if ( icheck != _nonconst_var_map.end() ){
           return WasatchCore::wrap_uintah_field_as_spatialops<ST>(
             *(icheck->second.get_field<MY_TYPE>()), this->_wasatch_ainfo, 0 );
        }

        MY_TYPE* field = scinew MY_TYPE;

        if ( ivar.depend == MODIFIES ){
          _new_dw->getModifiable( *field, ivar.label, _matl_index, _patch );
        } else {
          _new_dw->allocateAndPut( *field, ivar.label, _matl_index, _patch );
        }

        FieldContainer icontain;
        icontain.set_field(field);
        icontain.set_label(ivar.label);
        this->add_variable(name, icontain);

        return WasatchCore::wrap_uintah_field_as_spatialops<ST>( *field, this->_wasatch_ainfo, 0 );

      }

      // @brief Get a NON-CONSTANT spatialOps representation of the Uintah field.
      template <class ST>
      SpatialOps::SpatFldPtr<ST> get_const_so_field(const std::string name){

        typedef typename WasatchCore::SelectUintahFieldType<ST>::const_type MY_TYPE;
        VariableInformation ivar = get_variable_information( name, true );
        ConstFieldContainerMap::iterator icheck = _const_var_map.find( name );
        int nGhost = ivar.nGhost;

        if ( icheck != _const_var_map.end() ){
           return WasatchCore::wrap_uintah_field_as_spatialops<ST>(
             *(icheck->second.get_field<MY_TYPE>()), this->_wasatch_ainfo, nGhost );
        }

        MY_TYPE* field = scinew MY_TYPE;

        if ( ivar.dw == OLDDW ){
          _old_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, nGhost );
        } else {
          _new_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, nGhost );
        }

        ConstFieldContainer icontain;
        icontain.set_field(field);
        icontain.set_label(ivar.label);
        this->add_const_variable(name, icontain);

        return WasatchCore::wrap_uintah_field_as_spatialops<ST>( *field, this->_wasatch_ainfo, nGhost );

      }

      // @brief Get a NON-CONSTANT spatialOps representation of the Uintah field with DW specified
      template <class ST>
      SpatialOps::SpatFldPtr<ST> get_const_so_field(const std::string name, WHICH_DW which_dw ){

        std::ostringstream dw_value;
        dw_value << which_dw;

        typedef typename WasatchCore::SelectUintahFieldType<ST>::const_type MY_TYPE;
        VariableInformation ivar = get_variable_information( name, true, which_dw );
        ConstFieldContainerMap::iterator icheck = _const_var_map.find( name+"_"+dw_value.str() );
        int nGhost = ivar.nGhost;

        if ( icheck != _const_var_map.end() ){
           return WasatchCore::wrap_uintah_field_as_spatialops<ST>(
             *(icheck->second.get_field<MY_TYPE>()), this->_wasatch_ainfo, nGhost );
        }

        MY_TYPE* field = scinew MY_TYPE;

        if ( ivar.dw == OLDDW ){
          _old_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, nGhost );
        } else {
          _new_dw->get( *field, ivar.label, _matl_index, _patch, ivar.ghost_type, nGhost );
        }

        ConstFieldContainer icontain;
        icontain.set_field(field);
        icontain.set_label(ivar.label);
        this->add_const_variable(name+"_"+dw_value.str(), icontain);

        return WasatchCore::wrap_uintah_field_as_spatialops<ST>( *field, this->_wasatch_ainfo, nGhost );

      }

      // @brief Get a particle field spatialOps representation of the Uintah field.
      SpatialOps::SpatFldPtr<ParticleField> get_so_particle_field( const std::string name ){

        VariableInformation ivar = get_variable_information( name, false );
        UintahParticleMap::iterator icheck = _particle_map.find(name);

        if ( icheck != _particle_map.end() ){
          return WasatchCore::wrap_uintah_field_as_spatialops<ParticleField>(
            *(icheck->second.get_field()),this->_wasatch_ainfo, ivar.nGhost );
        }

        /// \TODO Resolve the old_dw vs. new_dw for the particle subset. What does Tony say?
        ParticleSubset* subset;
        ParticleVariable<double>* pvar = scinew ParticleVariable<double>;

        if ( _new_dw->haveParticleSubset(_matl_index, _patch) ){
          subset = _new_dw->getParticleSubset( _matl_index, _patch );
        } else {
          subset = _old_dw->getParticleSubset( _matl_index, _patch );
        }

        if ( ivar.depend == MODIFIES ){
          _new_dw->getModifiable( *pvar, ivar.label, subset );
        } else {
          _new_dw->allocateAndPut( *pvar, ivar.label, subset );
        }

        ParticleFieldContainer icontain;
        icontain.set_field(pvar);
        icontain.set_label(ivar.label);
        this->add_particle_variable(name, icontain);

        return WasatchCore::wrap_uintah_field_as_spatialops<ParticleField>(*pvar,
          this->_wasatch_ainfo, ivar.nGhost );

      }

      // @brief Get a CONSTANT particle field spatialOps representation of the Uintah field.
      SpatialOps::SpatFldPtr<ParticleField> get_const_so_particle_field(
        const std::string name ){

        VariableInformation ivar = get_variable_information( name, true );
        ConstUintahParticleMap::iterator icheck = _const_particle_map.find(name);

        if ( icheck != _const_particle_map.end() ){
          return WasatchCore::wrap_uintah_field_as_spatialops<ParticleField>(
            *(icheck->second.get_field()), this->_wasatch_ainfo, ivar.nGhost );
        }

        constParticleVariable<double>* pvar = scinew constParticleVariable<double>;

        if ( ivar.dw == OLDDW ){
          ParticleSubset* subset = _old_dw->getParticleSubset( _matl_index, _patch );
          _old_dw->get( *pvar, ivar.label, subset );
        } else {
          ParticleSubset* subset = _new_dw->getParticleSubset( _matl_index, _patch );
          _new_dw->get( *pvar, ivar.label, subset );
        }

        ConstParticleFieldContainer icontain;
        icontain.set_field(pvar);
        icontain.set_label(ivar.label);
        this->add_const_particle_variable(name, icontain);

        return WasatchCore::wrap_uintah_field_as_spatialops<ParticleField>(*pvar,
          this->_wasatch_ainfo, ivar.nGhost );

      }

      // @brief Get a CONSTANT particle field spatialOps representation of the Uintah field with
      //        specified DW
      SpatialOps::SpatFldPtr<ParticleField> get_const_so_particle_field(
        const std::string name, WHICH_DW which_dw ){

        std::ostringstream dw_value;
        dw_value << which_dw;

        ConstUintahParticleMap::iterator icheck = _const_particle_map.find(name+"_"+dw_value.str());
        VariableInformation ivar = get_variable_information( name, true, which_dw );

        if ( icheck != _const_particle_map.end() ){
          return WasatchCore::wrap_uintah_field_as_spatialops<ParticleField>(
            *(icheck->second.get_field()), this->_wasatch_ainfo, ivar.nGhost );
        }

        constParticleVariable<double>* pvar = scinew constParticleVariable<double>;

        if ( ivar.dw == OLDDW ){
          ParticleSubset* subset = _old_dw->getParticleSubset( _matl_index, _patch );
          _old_dw->get( *pvar, ivar.label, subset );
        } else {
          ParticleSubset* subset = _new_dw->getParticleSubset( _matl_index, _patch );
          _new_dw->get( *pvar, ivar.label, subset );
        }

        ConstParticleFieldContainer icontain;
        icontain.set_field(pvar);
        icontain.set_label(ivar.label);
        this->add_const_particle_variable(name+"_"+dw_value.str(), icontain);

        return WasatchCore::wrap_uintah_field_as_spatialops<ParticleField>(*pvar,
          this->_wasatch_ainfo, ivar.nGhost );

      }

    private:

      FieldContainerMap _nonconst_var_map;
      ConstFieldContainerMap _const_var_map;
      UintahParticleMap _particle_map;
      ConstUintahParticleMap _const_particle_map;
      const WasatchCore::AllocInfo& _wasatch_ainfo;
      const Patch* _patch;

      const int _matl_index;
      DataWarehouse* _old_dw;
      DataWarehouse* _new_dw;
      VariableRegistry _variable_reg;

      /** @brief From the vector of VariableInformation, return a single set of information based
                 variable's name. **/
      VariableInformation get_variable_information( const std::string name, const bool is_constant ){

        VariableRegistry::iterator i = _variable_reg.begin();
        for (; i!=_variable_reg.end(); i++){
          if ( i->name == name ){
            if ( i->is_constant == is_constant ){
              return *i;
            }
          }
        }

        std::stringstream msg;
        msg << "Error: variable with name: " << name << " not found in the registry." <<
        " Did you register it?" << std::endl;
        throw InvalidValue( msg.str(), __FILE__, __LINE__ );

      }

      /** @brief From the vector of VariableInformation, return a single set of information based
                 variable's name with specified DW. **/
      VariableInformation get_variable_information( const std::string name, const bool is_constant,
      WHICH_DW which_dw ){

        VariableRegistry::iterator i = _variable_reg.begin();
        for (; i!=_variable_reg.end(); i++){
          if ( i->name == name ){
            if ( i->is_constant == is_constant ){
              if ( i->dw == which_dw ){
                return *i;
              }
            }
          }
        }

        std::stringstream msg;
        msg << "Error: variable with name" << name << " not found in the registry." <<
        " Did you register it?" << std::endl;
        throw InvalidValue( msg.str(), __FILE__, __LINE__ );

      }

      /** @brief Delete all fields held in the local containers. **/
      void delete_fields(){

        for ( FieldContainerMap::iterator iter = _nonconst_var_map.begin();
              iter != _nonconst_var_map.end(); iter++ ){

          const VarLabel* lab = iter->second.get_label();
          const Uintah::TypeDescription* type = lab->typeDescription();

          if ( type == CCVariable<int>::getTypeDescription() ){
            delete iter->second.get_field<CCVariable<int> >();
          } else if ( type == CCVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<CCVariable<double> >();
          } else if ( type == CCVariable<Vector>::getTypeDescription() ){
            delete iter->second.get_field<CCVariable<Vector> >();
          } else if ( type == SFCXVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<SFCXVariable<double> >();
          } else if ( type == SFCYVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<SFCYVariable<double> >();
          } else if ( type == SFCZVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<SFCZVariable<double> >();
          } else {
            std::stringstream msg;
            msg << "Error: Trying to delete a variable" <<
            " with unsupported type." << std::endl;
            throw InvalidValue(msg.str(),__FILE__,__LINE__);
          }

        }

        for ( ConstFieldContainerMap::iterator iter = _const_var_map.begin();
              iter != _const_var_map.end(); iter++ ){

          const VarLabel* lab = iter->second.get_label();
          const Uintah::TypeDescription* type = lab->typeDescription();

          if ( type == CCVariable<int>::getTypeDescription() ){
            delete iter->second.get_field<constCCVariable<int> >();
          } else if ( type == CCVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<constCCVariable<double> >();
          } else if ( type == CCVariable<Vector>::getTypeDescription() ){
            delete iter->second.get_field<constCCVariable<Vector> >();
          } else if ( type == SFCXVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<constSFCXVariable<double> >();
          } else if ( type == SFCYVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<constSFCYVariable<double> >();
          } else if ( type == SFCZVariable<double>::getTypeDescription() ){
            delete iter->second.get_field<constSFCZVariable<double> >();
          } else {
            std::stringstream msg;
            msg << "Error: Trying to delete a variable" <<
            " with unsupported type." << std::endl;
            throw InvalidValue(msg.str(), __FILE__, __LINE__);
          }

        }

        for ( UintahParticleMap::iterator iter = _particle_map.begin();
        iter != _particle_map.end(); iter++ ){
          delete iter->second.get_field();
        }
        for ( ConstUintahParticleMap::iterator iter = _const_particle_map.begin();
        iter != _const_particle_map.end(); iter++ ){
          delete iter->second.get_field();
        }

      }

  };
}
#endif
