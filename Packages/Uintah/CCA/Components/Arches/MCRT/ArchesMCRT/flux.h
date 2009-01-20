#ifndef FLUX_H
#define FLUX_H

class flux{
public:
  flux(const int &_surfaceElementNo, const int &_VolElementNo, const int &_TopStartNo,
       const int & _BottomStartNo, const int &_FrontStartNo,
       const int &_BackStartNo, const int &_LeftStartNo,
       const int &_RightStartNo, const int &_sumElementNo,
       const int &_TopBottomNo, const int &_FrontBackNo, const int &_totalElementNo);
  ~flux();

  void net_div_flux(const double *EmissE, const double *AbsorbTerm,
		    const double &ElementAreaTB, const double &ElementAreaFB,
		    const double &ElementAreaLR, const double &ElementVol);
  
  /*
  double EmissSurface(const int &offset_index,
		      const double &ElementArea,
		      const double *RealSurfacePropertyArray);
		       
  double EmissVol(const int &vIndex,
		  const double &ElementVol,
		  const double *VolPropertyArray);
  
  
  void surfaceflux_div(//const int &TopStartNo,
		       // const int &FrontStartNo,
		       // const int &LeftStartNo,
		       //const int &TopBottomNo,
		       //const int &FrontBackNo,
		       // const int &LeftRightNo,
		       //const int &totalElementNo,
		       // const int &Total,
		       //const int &surfaceElementNo,
		       const double *Fij,
		       const double *RealSurfacePropertyArray,
		       const double *VolPropertyArray,
		       const double *emissRayCounter,
		       const double *SurfaceEmissiveEnergy,
		       const double *VolEmissiveEnergy,
		       const double &ElementAreaTB,
		       const double &ElementAreaFB,
		       const double &ElementAreaLR,
		       const double &ElementVol);

  */

  /*
  void norm_error(const double &topexact, const double &bottomexact,
		  const double &frontexact, const double &backexact,
		  const double &leftexact, const double &rightexact,
		  const double &xc, const double &yc, const double &zc);

  */
  
  double *Q_surface;
  double *q_surface;
  double *Qdiv;
  double *qdiv;
private:
  int surfaceElementNo, VolElementNo;
  int TopStartNo,BottomStartNo, FrontStartNo, BackStartNo;
  int  LeftStartNo, RightStartNo, sumElementNo;
  int TopBottomNo, FrontBackNo, totalElementNo;
  static double pi;
  static double SB;

};


#endif
