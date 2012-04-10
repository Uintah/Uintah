/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/

#ifndef _DEVIATORIC_STRESS_MODEL_H
#define _DEVIATORIC_STRESS_MODEL_H

#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PlasticityState.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DeformationState.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Math/Matrix3.h>


namespace Uintah { 
  using std::cout;
  using std::endl;

  class DeviatoricStressModel{ 

    public: 

      DeviatoricStressModel(){
        d_this = 0; 
      };

      ~DeviatoricStressModel(){
        delete d_this; 
      };

      //______________________________________________________________________
      void create(ProblemSpecP& ps){
      
        ProblemSpecP dsm_ps = ps->findBlock("deviatoric_stress_model");
        
        if(dsm_ps){
          
          string type="NULL";
          
          if(!dsm_ps->getAttribute("type", type)){
            throw ProblemSetupException("No type specified for DeviatoricStress", __FILE__, __LINE__);
          }
          
          if (type == "hypoElastic"){
            d_this = scinew hypoElastic();
            
          } else if (type == "hypoViscoElastic"){
            d_this = scinew hypoViscoElastic();
          
          } else {
            throw ProblemSetupException("Unknown DeviatoricStress type ("+type+")", __FILE__, __LINE__);
          }
        } else{
          d_this = scinew hypoElastic();  // DEFAULT  Deviatoric Stress Model
        }
      };
      
      //__________________________________
      //  This is what's called from the CM
      //  This just passes data through.
      void computeDeviatoricStressInc(const PlasticityState* plaState,
                                      DeformationState* defState,
                                      const double delT){ 
        d_this->computeDevStressInc(plaState, defState, delT);
      };
    //______________________________________________________________________
    //    I N S T A N T I A T I O N S
    //______________________________________________________________________
    private: 

      class DevStressBase { 

        public: 
          DevStressBase() {} 
          virtual ~DevStressBase() {}
          
          virtual void computeDevStressInc(const PlasticityState* plaState,
                                           DeformationState* defState,
                                           const double delT)=0;
      };

      //______________________________________________________________________
      //  HYPOELASTIC
      class hypoElastic : public DevStressBase  {

        public: 
          hypoElastic() {}
          ~hypoElastic() {}
          
          
          //__________________________________
          //
          void computeDevStressInc( const PlasticityState*  plaState,
                                    DeformationState* defState,
                                    const double delT ){
            double mu = plaState->shearModulus;
            defState->devStressInc = defState->tensorEta * (2.0 * mu * delT);
          }; 

        private:
      }; 

      //______________________________________________________________________
      //   HYPOVISCOELASTIC
      class  hypoViscoElastic : public DevStressBase  {

        public: 
          hypoViscoElastic() {}
          ~hypoViscoElastic() {}
          
          //__________________________________
          //
          void computeDevStressInc( const PlasticityState* plaState,
                                    DeformationState* defState,
                                    const double delT ){ 
            cout << " hypoViscoElastic:computeDevStessInc " << endl;
            double mu = plaState->shearModulus;
            defState->devStressInc = defState->tensorEta * (2.0 * mu * delT);
          }; 

        private: 
      }; 

      DeviatoricStressModel::DevStressBase* d_this;
  }; 
} 

#endif
