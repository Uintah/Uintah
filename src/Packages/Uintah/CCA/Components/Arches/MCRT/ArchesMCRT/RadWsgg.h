#ifndef RadWsgg_H
#define RadWsgg_H

#include <cmath>

class RadWsgg {

public:
  
  RadWsgg();
  ~RadWsgg();
  
  void WsggkVolwEmiss(const double *CO2,
		      const double *H2O,
		      const int &bands,
		      const double *T_Vol,
		      const double *SFV,
		      const int &VolElementNo,
		      double *kl_Vol, double *wEmiss_Vol);
		  
  void WsggwEmissSurface(const int &surfaceElementNo,
			 const double *T_surface,
			 const int &bands,
			 double *wEmiss_surface);

private:
  double a;
  double b[4];
    
};

#endif
