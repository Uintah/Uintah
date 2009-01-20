#ifndef VolElement_H
#define VolElement_H

class VolElement {
public:
  VolElement();
  VolElement(const int &iIndex,
	     const int &jIndex,
	     const int &kIndex,
	     const int &Ncx_,
	     const int &Ncy_);

  // cell center numbre Nc
  int get_VolIndex();

  void get_limits(const double *X,
		  const double *Y,
		  const double *Z);

  double get_xlow();
  double get_xup();
  double get_ylow();
  double get_yup();
  double get_zlow();
  double get_zup();
  

  double VolumeEmissFlux(const int &vIndex,
			 const double *kl_Vol,
			 const double *T_Vol,
			 const double *a_Vol);
  
  
  
  double VolumeIntensity(const int &vIndex,
			 const double *kl_Vol,
			 const double *T_Vol,
			 const double *a_Vol);
  
  
  
  
  double VolumeEmissFluxBlack(const int &vIndex,
			      const double *T_Vol,
			      const double *a_Vol);
  
  
  
  double VolumeIntensityBlack(const int &vIndex,
			      const double *T_Vol,
			      const double *a_Vol);

  
  void VolSurfaces();
  
  ~VolElement();
  
  //private:
  // assign these variables to be public


  
private:

  double xlow, xup, ylow, yup, zlow, zup;
  int VoliIndex, VoljIndex, VolkIndex;
  int VolIndex;
  int Ncx, Ncy;
};

#endif
