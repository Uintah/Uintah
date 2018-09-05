#ifndef Uintah_Component_Arches_CLASSNAME_h
#define Uintah_Component_Arches_CLASSNAME_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS

/** 
* @class  ADD
* @author ADD
* @date   ADD
* 
* @brief Computes ADD INFORMATION HERE
*
* ADD INPUT FILE INFORMATION HERE: 
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <PropertyModels>
*     <.......>
*   </PropertyModels>
* \endcode 
*  
*/ 

namespace Uintah{ 

  class CLASSNAME : public PropertyModelBase {

    public: 

      CLASSNAME( std::string prop_name, MaterialManagerP& materialManager );
      ~CLASSNAME(); 

      void problemSetup( const ProblemSpecP& db ); 

      void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ); 
      void computeProp(const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw, 
                       int time_substep );

      void sched_initialize( const LevelP& level, SchedulerP& sched );
      void initialize( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw );

      class Builder
        : public PropertyModelBase::Builder { 

        public: 

          Builder( std::string name, MaterialManagerP& materialManager ) : _name(name), _materialManager(materialManager){};
          ~Builder(){}; 

          CLASSNAME* build()
          { return scinew CLASSNAME( _name, _materialManager ); };

        private: 

          std::string _name; 
          MaterialManagerP& _materialManager; 

      }; // class Builder 

    private: 

  }; // class CLASSNAME
}   // namespace Uintah

#endif
