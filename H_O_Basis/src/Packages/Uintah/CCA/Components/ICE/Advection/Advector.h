#ifndef UINTAH_ADVECTOR_H
#define UINTAH_ADVECTOR_H
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
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
                                     const bool& bulletProofing_test,
                                     DataWarehouse* new_dw) = 0;

    virtual void  advectQ(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          CCVariable<double>& q_advected,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC,
			     DataWarehouse* /*new_dw*/)=0;

    virtual void advectQ(const bool useCompatibleFluxes,
                         const bool is_Q_massSpecific,
                         const CCVariable<double>& q_CC,
                         const CCVariable<double>& mass,
                         const Patch* patch,
                         CCVariable<double>& q_advected,
                         DataWarehouse* new_dw)=0;
    
    virtual void advectQ(const bool useCompatibleFluxes,
                         const bool is_Q_massSpecific,
                         const CCVariable<Vector>& q_CC,
                         const CCVariable<double>& mass,
                         const Patch* patch,
                         CCVariable<Vector>& q_advected,
                         DataWarehouse* new_dw)=0; 
                         
    virtual void advectMass(const CCVariable<double>& mass,
                           const Patch* patch,
                           CCVariable<double>& q_advected,
			      DataWarehouse* new_dw)=0;

                        
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

   // These inlined functions are passed into advect() and calculate the face
   // value of q_CC.  Note only one version of advectQ needs to compute q_FC thus
   // we have the ignoreFaceFluxes functions.  This really cuts down on Code
   // bloat by eliminating the need for a specialized version of advect 
  
  class ignoreFaceFluxesD {
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

  class ignoreFaceFluxesV {
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
  class saveFaceFluxes {
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
 /*______________________________________________________________________
 *   different data types 
 *______________________________________________________________________*/ 
  struct fflux { double d_fflux[6]; };          //face flux
  struct eflux { double d_eflux[12]; };         //edge flux
  struct cflux { double d_cflux[8]; };          //corner flux

  //__________________________________
  // face data
  template <class T> struct facedata {
    T d_data[6];
  };
  
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                // This turns off SGI compiler warning.
  template<class T>
  MPI_Datatype makeMPI_facedata()
  {
    ASSERTEQ(sizeof(facedata<T>), sizeof(T)*6);
    const TypeDescription* td = fun_getTypeDescription((T*)0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 6, 6, td->getMPIType(), &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif
  
  template<class T>
  const TypeDescription* fun_getTypeDescription(facedata<T>*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "facedata", true, 
				  &makeMPI_facedata<T>);
    }
    return td;
  }
  
  //__________________________________
  // vertex data
  template <class T> struct vertex {
    T d_vrtx[8];
  };
  
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // Template parameter not used in declaring arguments.
#endif                // This turns off SGI compiler warning.
  template<class T>
  MPI_Datatype makeMPI_vertex()
  {
    ASSERTEQ(sizeof(vertex<T>), sizeof(T)*8);
    const TypeDescription* td = fun_getTypeDescription((T*)0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 8, 8, td->getMPIType(), &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif
  
  template<class T>
  const TypeDescription* fun_getTypeDescription(vertex<T>*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
				  "vertex", true, 
				  &makeMPI_vertex<T>);
    }
    return td;
  }  
  
  const TypeDescription* fun_getTypeDescription(fflux*);    
  const TypeDescription* fun_getTypeDescription(eflux*);
  const TypeDescription* fun_getTypeDescription(cflux*); 

}  // Uintah namespace


//__________________________________
namespace SCIRun {

  template<class T>
  void swapbytes( Uintah::facedata<T>& f) {
    for(int i=0;i<6;i++)
      swapbytes(f.d_data[i]);
  }
  
  template<class T>
  void swapbytes( Uintah::vertex<T>& v) {
    for(int i=0;i<8;i++)
      swapbytes(v.d_vrtx[i]);
  }
  
  void swapbytes( Uintah::fflux& ); 
  void swapbytes( Uintah::eflux& );
  void swapbytes( Uintah::cflux& );

}


#endif
