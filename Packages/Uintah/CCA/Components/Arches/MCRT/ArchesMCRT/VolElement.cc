#include <iostream>
#include "VolElement.h"
#include "RNG.h"


using std::cout;
using std::endl;

class RNG;

VolElement::VolElement(){ 
}

VolElement::~VolElement(){
}

void VolElement::get_limits(double *VolTable, const int &vIndex) {
 int index;
 double *p;

 p = VolTable + vIndex * 13;

 index = ( int ) * ( p + 6 ); // look up table's index

 zup = *p;
 zlow = * ( p + 1);
 yup = * ( p + 2 );
 ylow = * ( p + 3 );
 xlow = * ( p + 4 );
 xup = * ( p + 5 );
//  cout << " zup = " << zup << "; zlow = " << zlow << endl;
//  cout << " yup = " << yup << "; ylow = " << ylow << endl;
//  cout << " xup = " << xup << "; xlow = " << xlow << endl;
 
}

void VolElement::get_public_limits(double &_xlow, double &_xup,
				   double &_ylow, double &_yup,
				   double &_zlow, double &_zup) {
  _xlow = xlow;
  _xup = xup;
  _ylow = ylow;
  _yup = yup;
  _zlow = zlow;
  _zup = zup;

}
