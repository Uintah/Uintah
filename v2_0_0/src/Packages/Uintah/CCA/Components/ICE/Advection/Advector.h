#ifndef UINTAH_ADVECTOR_H
#define UINTAH_ADVECTOR_H
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

  class DataWarehouse;
  class Patch;

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
                                 const bool& bulletProofing_test) = 0;

    virtual void  advectQ(const CCVariable<double>& q_CC,
                             const Patch* patch,
                             CCVariable<double>& q_advected,
                             SFCXVariable<double>& q_XFC,
                             SFCYVariable<double>& q_YFC,
                             SFCZVariable<double>& q_ZFC,
				 DataWarehouse* /*new_dw*/) = 0;

    virtual void advectQ(const CCVariable<double>& q_CC,
                      const Patch* patch,
                      CCVariable<double>& q_advected,
			 DataWarehouse* new_dw) = 0;

    virtual void advectQ(const CCVariable<Vector>& q_CC,
                      const Patch* patch,
                      CCVariable<Vector>& q_advected,
			 DataWarehouse* new_dw) = 0;

                        
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
         
  }; 
 /*______________________________________________________________________
 *   C O M M O N L Y   U S E D 
 *______________________________________________________________________*/ 
  
  enum FACE {TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK};    
  enum EDGE {TOP_R = 0, TOP_FR, TOP_L, TOP_BK, BOT_R, BOT_FR, BOT_L, BOT_BK,
            RIGHT_BK, RIGHT_FR, LEFT_BK, LEFT_FR };
  enum CORNER {TOP_R_BK = 0, TOP_R_FR, TOP_L_BK, TOP_L_FR, BOT_R_BK, 
             BOT_R_FR, BOT_L_BK, BOT_L_FR}; 

   // These inlined functions are passed into advect() and calculate the face
   // value of q_CC.  Note only one version of advectQ needs to compute q_FC thus
   // we have the ignoreFaceFluxes functions.  This really cuts down on Code
   // bloat by eliminating the need for a specialized version of advect 
  
  inline void ignoreFaceFluxesD( const IntVector&,
                                 SFCXVariable<double>&, 
                                 SFCYVariable<double>&,  
                                 SFCZVariable<double>&,  
                                 double[],  
                                 double[],
                                 const CCVariable<double>&)
  { 
  }
  inline void ignoreFaceFluxesV( const IntVector&,
                                 SFCXVariable<double>&, 
                                 SFCYVariable<double>&,  
                                 SFCZVariable<double>&,  
                                 double[],  
                                 Vector[],
                                 const CCVariable<Vector>&)
  { 
  }
  
  inline double equalZero(double d1, double d2, double d3)
  {
    return d1 == 0.0 ? d2:d3;
  }
  
  inline void saveFaceFluxes( const IntVector& c, 
                              SFCXVariable<double>& q_XFC,           
                              SFCYVariable<double>& q_YFC,           
                              SFCZVariable<double>& q_ZFC,           
                              double faceVol[], 
                              double q_face_flux[],
                              const CCVariable<double>& q_CC) 
  {
    double d_SMALL_NUM = 1e-100;
    double tmp_XFC, tmp_YFC, tmp_ZFC, q_tmp;
    q_tmp = q_CC[c];
    tmp_XFC = fabs(q_face_flux[LEFT])  /(faceVol[LEFT]   + d_SMALL_NUM);
    tmp_YFC = fabs(q_face_flux[BOTTOM])/(faceVol[BOTTOM] + d_SMALL_NUM);
    tmp_ZFC = fabs(q_face_flux[BACK])  /(faceVol[BACK]   + d_SMALL_NUM);
    
    // if q_(X,Y,Z)FC = 0.0 then set it equal to q_CC[c]
    tmp_XFC = equalZero(q_face_flux[LEFT],   q_tmp, tmp_XFC);
    tmp_YFC = equalZero(q_face_flux[BOTTOM], q_tmp, tmp_YFC);
    tmp_ZFC = equalZero(q_face_flux[BACK],   q_tmp, tmp_ZFC);
    
    q_XFC[c] = tmp_XFC;
    q_YFC[c] = tmp_YFC;
    q_ZFC[c] = tmp_ZFC;    
  }  

  struct fflux { double d_fflux[6]; };          //face flux
  struct eflux { double d_eflux[12]; };         //edge flux
  struct cflux { double d_cflux[8]; };          //corner flux
  
  const TypeDescription* fun_getTypeDescription(fflux*);    
  const TypeDescription* fun_getTypeDescription(eflux*);
  const TypeDescription* fun_getTypeDescription(cflux*); 

}  // Uintah namespace


namespace SCIRun {

  void swapbytes( Uintah::fflux& ); 
  void swapbytes( Uintah::eflux& );
  void swapbytes( Uintah::cflux& );

}


#endif
