/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef VolElement_H
#define VolElement_H

class VolElement{
public:
  VolElement();
  VolElement(const int &iIndex,
	     const int &jIndex,
	     const int &kIndex,
	     const int &Ncx_,
	     const int &Ncy_);

  // cell center numbre Nc

  inline
  int get_VolIndex(){    
    return VolIndex;    
  }

  
  inline
  void get_limits(const double *X,
		  const double *Y,
		  const double *Z) {
    
    xlow = X[VoliIndex];
    xup = X[VoliIndex+1];
    
    ylow = Y[VoljIndex];
    yup = Y[VoljIndex+1];
    
    zlow = Z[VolkIndex];
    zup = Z[VolkIndex+1];
        
  }

  
  inline
  double get_xlow(){
    return xlow;
  }


  inline
  double get_xup(){
    return xup;
  }
  
  inline
  double get_ylow(){
    return ylow;
  }


  inline
  double get_yup(){
    return yup;
  }

  inline
  double get_zlow(){
    return zlow;
}


  inline
  double get_zup(){
    return zup;
  }
    

  double VolumeEmissFluxBlack(const int &vIndex,
			      const double *T_Vol,
			      const double *a_Vol);
  
  double VolumeEmissFlux(const int &vIndex,
			 const double *kl_Vol,
			 const double *T_Vol,
			 const double *a_Vol);

  double VolumeIntensityBlack(const int &vIndex,
			      const double *T_Vol,
			      const double *a_Vol);
  
  double VolumeIntensityBlack(const double &TVol,
			      const double &aVol);

  
  double VolumeIntensity(const int &vIndex,
			 const double *kl_Vol,
			 const double *T_Vol,
			 const double *a_Vol);
  
  ~VolElement();
  
  
private:

  double xlow, xup, ylow, yup, zlow, zup;
  int VoliIndex, VoljIndex, VolkIndex;
  int VolIndex;
  int Ncx, Ncy;
};

#endif
