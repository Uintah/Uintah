#ifndef Uintah_Component_Arches_AlgebraicScalarDiss_h
#define Uintah_Component_Arches_AlgebraicScalarDiss_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>


/** 
 * @class  AlgebraicScalarDiss
 * @author Jeremy Thornock, revised by Alex Abboud
 * @date   Oct, 2012, revision August 2013
 * 
 * @brief computes the scalar dissipation rate
 * this is to be used in the "1 eqn" model for scalar variance as part of a 
 * source term for the second moment of the mixture fraction
 * \f$ \chi_Z = 2*(D+D_t) | \nabla Z |^2 = 2*(D+D_t) \frac{dZ}{dx_i} \frac{dZ}{dx_i} \f$
 * gradient of Z makes this highly dependent on grid resolution
 */ 

namespace Uintah{ 

  class AlgebraicScalarDiss : public PropertyModelBase {

    public: 

      AlgebraicScalarDiss( std::string prop_name, MaterialManagerP& materialManager );
      ~AlgebraicScalarDiss(); 

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

          AlgebraicScalarDiss* build()
          { return scinew AlgebraicScalarDiss( _name, _materialManager ); };

        private: 

          std::string _name; 
          MaterialManagerP& _materialManager; 

      }; // class Builder 

    private: 

      std::string _mf_name; 

      double _Sc_t;                          ///< Turbulent schmidt number
      double _D;                             ///< filtered molecular diffusion coeff.  (assumed constant) 

      const VarLabel* _mf_label; 
      const VarLabel* _mu_t_label;
      const VarLabel* _volfrac_label;

  }; // class AlgebraicScalarDiss
}   // namespace Uintah

#endif
