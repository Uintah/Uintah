#ifndef UINTAH_ADVECTOR_H
#define UINTAH_ADVECTOR_H
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/ICE/Advection/FluxDatatypes.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
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
      double delT;
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

    virtual void  advectQ(const CCVariable<double>& q_CC,
                          CCVariable<double>& q_advected,
                          advectVarBasket* vb,
                          constSFCXVariable<double>& uvel_FC,
                          constSFCYVariable<double>& vvel_FC,
                          constSFCZVariable<double>& wvel_FC,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC)=0;

    virtual void advectQ(const CCVariable<double>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<double>& q_advected,
                         constSFCXVariable<double>& uvel_FC,
                         constSFCYVariable<double>& vvel_FC,
                         constSFCZVariable<double>& wvel_FC,                         
                         advectVarBasket* vb)=0;
    
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<Vector>& q_advected,
                         constSFCXVariable<double>& uvel_FC,
                         constSFCYVariable<double>& vvel_FC,
                         constSFCZVariable<double>& wvel_FC,
                         advectVarBasket* vb)=0; 
                         
    virtual void advectMass(const CCVariable<double>& mass,
                           CCVariable<double>& q_advected,
                           constSFCXVariable<double>& uvel_FC,
                           constSFCYVariable<double>& vvel_FC,
                           constSFCZVariable<double>& wvel_FC,
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
                                 vector<double> badOutFlux,
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
  
  class ignore_q_FC_D {     // does nothing
    public:
    inline void operator()( double& q_FC, double q_slab[])
    {
    }
  };

  class ignore_q_FC_V {    // does nothing
    public:
    inline void operator()( double& q_FC, double q_slab[])
    {
    }
  };
    
  //__________________________________
  // compute Q at the face center
  class save_q_FC {
    public:
    inline void operator()( double& q_FC, double& q_slab) 
    {
      q_FC = q_slab;   
    }
  };

} // Uintah namespace
#endif
