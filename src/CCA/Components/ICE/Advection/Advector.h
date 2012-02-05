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


#ifndef UINTAH_ADVECTOR_H
#define UINTAH_ADVECTOR_H

#include <CCA/Components/ICE/Advection/FluxDatatypes.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/Schedulers/GPUThreadedMPIScheduler.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Grid.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/ProgressiveWarning.h>

namespace Uintah {

  class DataWarehouse;
  class Patch;

  struct advectVarBasket{
      bool useCompatibleFluxes;
      bool is_Q_massSpecific;
      DataWarehouse* new_dw;
      DataWarehouse* old_dw;
      ICELabel* lb;
      double AMR_subCycleProgressVar;
      int indx;
      string desc;
      const Patch* patch;
      const Level* level;
      bool doRefluxing;
  };
  
  class Advector {

  public:
    Advector();

    virtual ~Advector();
    
    virtual Advector* clone(DataWarehouse* new_dw, 
                            const Patch* patch,
                            const bool isNewGrid) = 0;

    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_FC,
                                     const SFCYVariable<double>& vvel_FC,
                                     const SFCZVariable<double>& wvel_FC,
                                     const double& delT, 
                                     const Patch* patch,
                                     const int& indx,
                                     const bool& bulletProofing_test,
                                     DataWarehouse* new_dw) = 0;

    virtual void inFluxOutFluxVolumeGPU(const VarLabel* uvel_FCMELabel,
                                        const VarLabel* vvel_FCMELabel,
                                        const VarLabel* wvel_FCMELabel,
                                        const double& delT,
                                        const Patch* patch,
                                        const int& indx,
                                        const bool& bulletProofing_test,
                                        DataWarehouse* new_dw,
                                        const int& device,
                                        GPUThreadedMPIScheduler* sched) = 0;

    virtual void  advectQ(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          CCVariable<double>& q_advected,
                          advectVarBasket* vb,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC,
			     DataWarehouse* /*new_dw*/)=0;

    virtual void advectQ(const CCVariable<double>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<double>& q_advected,
                         advectVarBasket* vb)=0;
    
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<Vector>& q_advected,
                         advectVarBasket* vb)=0; 
                         
    virtual void advectMass(const CCVariable<double>& mass,
                           CCVariable<double>& q_advected,
                           advectVarBasket* vb)=0;

                        
    int OF_slab[6];          // outflux slab
    int IF_slab[6];          // influx flab
    IntVector S_ac[6];       // slab adj. cell
    
    SFCXVariable<double> d_notUsedX;
    SFCYVariable<double> d_notUsedY; 
    SFCZVariable<double> d_notUsedZ;
    CCVariable<double> d_notUsed_D;
  }; 
  
  //__________________________________
  void  warning_restartTimestep( vector<IntVector> badCells,
                                 vector<fflux> badOutFlux,
                                 const double vol,
                                 const int indx,
                                 const Patch* patch,
                                 DataWarehouse* new_dw);
				 
  inline double equalZero(double d1, double d2, double d3)
  {
    return d1 == 0.0 ? d2:d3;
  }
    

 /*______________________________________________________________________
 *   C O M M O N L Y   U S E D 
 *______________________________________________________________________*/ 
  
  enum FACE {TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK};
   // These inlined functions are passed into advect() and calculate the face
   // value of q_CC.  Note only one version of advectQ needs to compute q_FC thus
   // we have the ignoreFaceFluxes functions.  This really cuts down on Code
   // bloat by eliminating the need for a specialized version of advect 
  
  class ignore_q_FC_calc_D {     // does nothing
    public:
    inline void operator()( const IntVector&,
			    SFCXVariable<double>&, 
			    SFCYVariable<double>&,  
			    SFCZVariable<double>&,  
			    double[],  
			    double[],
			    const CCVariable<double>&)
    {
    }
  };

  class ignore_q_FC_calc_V {    // does nothing
    public:
    inline void operator()( const IntVector&,
			    SFCXVariable<double>&, 
			    SFCYVariable<double>&,  
			    SFCZVariable<double>&,  
			    double[],  
			    Vector[],
			    const CCVariable<Vector>&)
    {
    }
  };
    
  //__________________________________
  // compute Q at the face center
  class save_q_FC {
    public:
    inline void operator()( const IntVector& c, 
			    SFCXVariable<double>& q_XFC,           
			    SFCYVariable<double>& q_YFC,           
			    SFCZVariable<double>& q_ZFC,           
			    double faceVol[], 
			    double q_face_flux[],
			    const CCVariable<double>& q_CC) 
    {
    
      double tmp_XFC, tmp_YFC, tmp_ZFC, q_tmp;
      q_tmp = q_CC[c];
      tmp_XFC = fabs(q_face_flux[LEFT])  /(faceVol[LEFT]   + 1e-100);
      tmp_YFC = fabs(q_face_flux[BOTTOM])/(faceVol[BOTTOM] + 1e-100);
      tmp_ZFC = fabs(q_face_flux[BACK])  /(faceVol[BACK]   + 1e-100);
    
      // if q_(X,Y,Z)FC = 0.0 then set it equal to q_CC[c]
      tmp_XFC = equalZero(q_face_flux[LEFT],   q_tmp, tmp_XFC);
      tmp_YFC = equalZero(q_face_flux[BOTTOM], q_tmp, tmp_YFC);
      tmp_ZFC = equalZero(q_face_flux[BACK],   q_tmp, tmp_ZFC);
    
      q_XFC[c] = tmp_XFC;
      q_YFC[c] = tmp_YFC;
      q_ZFC[c] = tmp_ZFC;    
    }
  };

} // Uintah namespace
#endif
