#ifndef Uintah_Component_Arches_ArchesFieldContainer_h
#define Uintah_Component_Arches_ArchesFieldContainer_h

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>

//===============================================================

/**
* @class  Field Interface for Uintah variables
* @author Jeremy Thornock
* @date   2014
*
* @brief Holds fields for use during task execution
*        Also deletes them when the task is finished.
*
**/

//===============================================================
namespace Uintah{
  class ArchesFieldContainer{

    public:

      enum FC_VAR_TYPE { CC_INT, CC_DOUBLE, CC_VEC, FACEX, FACEY, FACEZ, SUM, MAX, MIN, PARTICLE };

      ArchesFieldContainer( const Wasatch::AllocInfo& alloc_info, const Patch* patch );

      struct FieldContainer{

        public:
          void set_field( GridVariableBase* field ){ _field = field; }

          void set_field_type( FC_VAR_TYPE type ){ _my_type = type; }

         // void set_ghosts( int n_ghosts ){_n_ghosts = n_ghosts; }

          template <class T>
          T* get_field(){ return dynamic_cast<T*>(_field); }

          FC_VAR_TYPE get_type(){ return _my_type; }

          int get_n_ghost(){ return 0; } //Not allowing for ghosts currently on modifiable fields

        private:
          GridVariableBase* _field;
          FC_VAR_TYPE _my_type;
          int _n_ghosts;

      };

      struct ConstFieldContainer{

        public:
          void set_field( constVariableBase<GridVariableBase>* field ){ _field = field; }

          void set_field_type( FC_VAR_TYPE type ){ _my_type = type; }

          void set_ghosts( int n_ghosts ){_n_ghosts = n_ghosts; }

          template <class T>
          T* get_field(){ return dynamic_cast<T*>(_field); }

          FC_VAR_TYPE get_type(){ return _my_type; }

          int get_n_ghost(){ return _n_ghosts; } //Not allowing for ghosts currently on modifiable fields

        private:
          constVariableBase<GridVariableBase>* _field;
          FC_VAR_TYPE _my_type;
          int _n_ghosts;

      };

      struct ParticleFieldContainer{

        public:
          void set_field( ParticleVariable<double>* field ){ _field = field; }

          void set_field_type(FC_VAR_TYPE type=PARTICLE){ _my_type = type; }

          void set_ghosts( int n_ghosts ){ _n_ghosts = n_ghosts; }

          ParticleVariable<double>* get_field(){ return _field; }

          FC_VAR_TYPE get_type(){ return _my_type; }

          int get_n_ghost(){ return _n_ghosts; }

        private:

          //this is specialized now...is there a point to specializing it?
          ParticleVariable<double>* _field;
          FC_VAR_TYPE _my_type;
          int _n_ghosts;

      };

      struct ConstParticleFieldContainer{

        public:
          void set_field( constParticleVariable<double>* field ){ _field = field; }

          void set_field_type(FC_VAR_TYPE type=PARTICLE){ _my_type = type; }

          void set_ghosts( int n_ghosts ){ _n_ghosts = n_ghosts; }

          constParticleVariable<double>* get_field(){ return _field; }

          FC_VAR_TYPE get_type(){ return _my_type; }

          int get_n_ghost(){ return _n_ghosts; }

        private:

          //this is specialized now...is there a point to specializing it?
          constParticleVariable<double>* _field;
          FC_VAR_TYPE _my_type;
          int _n_ghosts;

      };

      typedef std::map<std::string, FieldContainer> FieldContainerMap;
      typedef std::map<std::string, ConstFieldContainer> ConstFieldContainerMap;
      typedef std::map<std::string, ParticleFieldContainer > UintahParticleMap;
      typedef std::map<std::string, ConstParticleFieldContainer > ConstUintahParticleMap;

      ~ArchesFieldContainer(){

        //delete the fields
        for ( FieldContainerMap::iterator iter = _nonconst_var_map.begin();
              iter != _nonconst_var_map.end(); iter++ ){
          if ( iter->second.get_type() == ArchesFieldContainer::CC_DOUBLE ){
            CCVariable<double>* var = iter->second.get_field<CCVariable<double> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::CC_INT){
            CCVariable<int>* var = iter->second.get_field<CCVariable<int> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::CC_VEC){
            CCVariable<Vector>* var = iter->second.get_field<CCVariable<Vector> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::FACEX){
            SFCXVariable<double>* var = iter->second.get_field<SFCXVariable<double> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::FACEY){
            SFCYVariable<double>* var = iter->second.get_field<SFCYVariable<double> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::FACEZ){
            SFCZVariable<double>* var = iter->second.get_field<SFCZVariable<double> >();
            delete var;
          }
        }
        for ( ConstFieldContainerMap::iterator iter = _const_var_map.begin();
              iter != _const_var_map.end(); iter++ ){
          if ( iter->second.get_type() == ArchesFieldContainer::CC_DOUBLE ){
            constCCVariable<double>* var = iter->second.get_field<constCCVariable<double> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::CC_INT){
            constCCVariable<int>* var = iter->second.get_field<constCCVariable<int> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::CC_VEC){
            constCCVariable<Vector>* var = iter->second.get_field<constCCVariable<Vector> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::FACEX){
            constSFCXVariable<double>* var = iter->second.get_field<constSFCXVariable<double> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::FACEY){
            constSFCYVariable<double>* var = iter->second.get_field<constSFCYVariable<double> >();
            delete var;
          } else if ( iter->second.get_type() == ArchesFieldContainer::FACEZ){
            constSFCZVariable<double>* var = iter->second.get_field<constSFCZVariable<double> >();
            delete var;
          }
        }

        for ( UintahParticleMap::iterator iter = _particle_map.begin(); iter != _particle_map.end(); iter++ ){
          delete iter->second.get_field();
        }
        for ( ConstUintahParticleMap::iterator iter = _const_particle_map.begin(); iter != _const_particle_map.end(); iter++ ){
          delete iter->second.get_field();
        }
      }

      /** @brief Add a variable to the non-const variable map **/
      void add_variable( std::string name, FieldContainer container ){
        FieldContainerMap::iterator iter = _nonconst_var_map.find(name);
        if ( iter == _nonconst_var_map.end() ){
          _nonconst_var_map.insert(std::make_pair(name, container));
        } else {
          throw InvalidValue("Error: Trying to add a variable to non_const field map which is already present: "+name, __FILE__,__LINE__);
        }
      }

      /** @brief Add a variable to the const variable map **/
      void add_const_variable( std::string name, ConstFieldContainer container ){
        ConstFieldContainerMap::iterator iter = _const_var_map.find(name);
        if ( iter == _const_var_map.end() ){
          _const_var_map.insert(std::make_pair(name, container));
        } else {
          throw InvalidValue("Error: Trying to add a variable to const field map which is already present: "+name, __FILE__,__LINE__);
        }
      }

      /** @brief Add a particle variable to the non-const variable map **/
      void add_particle_variable( std::string name, ParticleFieldContainer container ){
        UintahParticleMap::iterator iter = _particle_map.find(name);

        if ( iter == _particle_map.end() ){
          _particle_map.insert(std::make_pair(name, container));
        } else {
          throw InvalidValue("Error: Trying to add a particle variable to particle map which is already present: "+name, __FILE__,__LINE__);
        }
      }

      /** @brief Add a particle variable to the const variable map **/
      void add_const_particle_variable( std::string name, ConstParticleFieldContainer container ){
        ConstUintahParticleMap::iterator iter = _const_particle_map.find(name);

        if ( iter == _const_particle_map.end() ){
          _const_particle_map.insert(std::make_pair(name, container));
        } else {
          throw InvalidValue("Error: Trying to add a const particle variable to particle map which is already present: "+name, __FILE__,__LINE__);
        }
      }

      //UINTAH VARIABLE TASK ACCESS:

      /** @brief Get a modifiable uintah variable **/
      template <typename T>
      inline T* get_const_field( const std::string name ){
        ConstFieldContainerMap::iterator iter = _const_var_map.find(name);
        if ( iter != _const_var_map.end() )
          return iter->second.get_field<T>();
        throw InvalidValue("Error: Cannot locate const uintah field: "+name, __FILE__, __LINE__);
      }

      /** @brief Get a const uintah variable **/
      template <typename T>
      inline T* get_field( const std::string name ){
        FieldContainerMap::iterator iter = _nonconst_var_map.find(name);
        if ( iter != _nonconst_var_map.end() )
          return iter->second.get_field<T>();
        throw InvalidValue("Error: Cannot locate uintah field: "+name, __FILE__, __LINE__);
      }

      //SPATIAL OPS VARIABLE TASK ACCESS:

      // @brief Get a NON-CONSTANT spatialOps representation of the Uintah field.
      template <class ST>
      SpatialOps::SpatFldPtr<ST> get_so_field(const std::string name){
        //return new_retrieve_so_field<ST,FieldContainerMap>( name, _nonconst_var_map, _patch, this->_wasatch_ainfo );
        FieldContainerMap::iterator iter = _nonconst_var_map.find( name );

        if ( iter != _nonconst_var_map.end() ){
          typedef typename Wasatch::SelectUintahFieldType<ST>::type MY_TYPE;
          MY_TYPE* var = iter->second.template get_field<MY_TYPE>();
          int nGhost = iter->second.get_n_ghost();
          return Wasatch::wrap_uintah_field_as_spatialops<ST>( *var, this->_wasatch_ainfo, nGhost );
        }

        std::ostringstream msg;
        msg << "Arches Task Error: Cannot resolve Uintah variable: "<<name << "\n (being accessed as NON-CONST)" << std::endl;
        throw InvalidValue(msg.str(), __FILE__, __LINE__);

      }

      // @brief Get a CONSTANT spatialOps representation of the Uintah field.
      template <class ST>
      SpatialOps::SpatFldPtr<ST> get_const_so_field(const std::string name){
        //return new_retrieve_const_so_field<ST, ConstFieldContainerMap>( name, _const_var_map, _patch, this->_wasatch_ainfo );
        ConstFieldContainerMap::iterator iter = _const_var_map.find(name);

        if ( iter != _const_var_map.end() ) {
          typedef typename Wasatch::SelectUintahFieldType<ST>::const_type MY_TYPE;
          MY_TYPE* var = iter->second.template get_field<MY_TYPE>();
          int nGhost = iter->second.get_n_ghost();
          return Wasatch::wrap_uintah_field_as_spatialops<ST>( *var, this->_wasatch_ainfo, nGhost );
        }

        std::ostringstream msg;
        msg << "Arches Task Error: Cannot resolve Uintah variable: "<<name << "\n (being accessed as CONST)" << std::endl;
        throw InvalidValue(msg.str(), __FILE__, __LINE__);

      }

      // @brief Get a particle field spatialOps representation of the Uintah field.
      SpatialOps::SpatFldPtr<ParticleField> get_so_particle_field( const std::string name ){

        UintahParticleMap::iterator iter = _particle_map.find(name);

        if ( iter != _particle_map.end() ) {

          ParticleVariable<double>* pvar = iter->second.get_field();
          int nGhost = iter->second.get_n_ghost();
          return Wasatch::wrap_uintah_field_as_spatialops<ParticleField>( *pvar, this->_wasatch_ainfo, nGhost );

        }

        std::ostringstream msg;
        msg << " Arches Task Error: Cannot resolve particle variable: "<< name << "\n" << "(being accessed as non-const)" << std::endl;
        throw InvalidValue(msg.str(), __FILE__, __LINE__);

      }

      // @brief Get a CONSTANT particle field spatialOps representation of the Uintah field.
      SpatialOps::SpatFldPtr<ParticleField> get_const_so_particle_field( const std::string name ){

        ConstUintahParticleMap::iterator iter = _const_particle_map.find(name);

        if ( iter != _const_particle_map.end() ) {

          constParticleVariable<double>* pvar = iter->second.get_field();
          int nGhost = iter->second.get_n_ghost();
          return Wasatch::wrap_uintah_field_as_spatialops<ParticleField>( *pvar, this->_wasatch_ainfo, nGhost );

        }

        std::ostringstream msg;
        msg << " Arches Task Error: Cannot resolve particle variable: "<< name << "\n" << "(being accessed as const)" << std::endl;
        throw InvalidValue(msg.str(), __FILE__, __LINE__);

      }

    private:

      FieldContainerMap _nonconst_var_map;
      ConstFieldContainerMap _const_var_map;
      UintahParticleMap _particle_map;
      ConstUintahParticleMap _const_particle_map;
      const Wasatch::AllocInfo& _wasatch_ainfo;
      const Patch* _patch;

  };
}
#endif
