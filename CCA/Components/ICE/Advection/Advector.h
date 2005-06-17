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
      bool doAMR;
  };
  
  class Advector {

  public:
    Advector();
    virtual ~Advector();
    
    virtual Advector* clone(DataWarehouse* new_dw, const Patch* patch) = 0;


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_FC,
                                     const SFCYVariable<double>& vvel_FC,
                                     const SFCZVariable<double>& wvel_FC,
                                     const double& delT, 
                                     const Patch* patch,
                                     const int& indx,
                                     const bool& bulletProofing_test,
                                     DataWarehouse* new_dw) = 0;

    virtual void  advectQ(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          CCVariable<double>& q_advected,
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
    
    int OF_edge[6][4];      // outflux edge
    int IF_edge[6][4];      // influx edge
    IntVector E_ac[6][4];   // edge adj. cell
    
    int OF_corner[6][4];    // outflux corner
    int IF_corner[6][4];    // influx corner
    IntVector C_ac[6][4];   // corner adj. cell
    
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
  enum EDGE {TOP_R = 0, TOP_FR, TOP_L, TOP_BK, BOT_R, BOT_FR, BOT_L, BOT_BK,
            RIGHT_BK, RIGHT_FR, LEFT_BK, LEFT_FR };
  enum CORNER {TOP_R_BK = 0, TOP_R_FR, TOP_L_BK, TOP_L_FR, BOT_R_BK, 
             BOT_R_FR, BOT_L_BK, BOT_L_FR}; 

  //__________________________________
  // converts patch face into cell Face
  int patchFaceToCellFace(Patch::FaceType face);

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
