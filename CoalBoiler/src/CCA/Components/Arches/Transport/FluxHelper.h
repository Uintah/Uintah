#ifndef Uintah_Component_Arches_FluxHelper_h
#define Uintah_Component_Arches_FluxHelper_h
#include <CCA/Components/Arches/Task/FieldContainer.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>

namespace Uintah{ 

  class FluxHelper { 

    public: 

      FluxHelper(){}
      ~FluxHelper(){}

      template <class FieldT, class FluxT>
      inline SpatialOps::SpatFldPtr<FieldT> compute_diffusive_flux( SpatialOps::OperatorDatabase& opr, 
                                                                    SpatialOps::SpatFldPtr<FieldT> gamma, 
                                                                    SpatialOps::SpatFldPtr<FieldT> rho, 
                                                                    SpatialOps::SpatFldPtr<FieldT> phi )
      { 

        using namespace SpatialOps;
        using SpatialOps::operator *; 

        typedef typename OperatorTypeBuilder< SpatialOps::Gradient    , FieldT , FluxT >::type Grad;
        typedef typename OperatorTypeBuilder< SpatialOps::Interpolant , FieldT , FluxT >::type Interp;
        typedef typename OperatorTypeBuilder< SpatialOps::Divergence  , FluxT  , FieldT >::type Div; 

        const Grad*   g = opr.retrieve_operator<Grad>(); 
        const Interp* i = opr.retrieve_operator<Interp>(); 
        const Div*    d = opr.retrieve_operator<Div>(); 

        //      d/dx( Gamma * Rho * d(phi)/dx)
        return (*d)((*i)( *gamma * *rho ) * (*g)(*phi)); 

      }

    private: 

  };

}
#endif 
