#include <iostream.h>
#include "element.h"

double Element::N[8][8];
double Element::dNds[8][8];
double Element::dNdt[8][8];
double Element::dNdu[8][8];

void make_shape()
{

  double gauss_location[2];
  short gauss_pt,i,j,k;
  double s,t,u;
  
  gauss_location[0] = -.57735026918963;
  gauss_location[1] = -gauss_location[0];
  gauss_pt = 0;
  
  for (i=0;i<2;i++) {
    for (j=0;j<2;j++) {
      for (k=0;k<2;k++) {
	
	s = gauss_location[i];
	t = gauss_location[j];
	u = gauss_location[k];
	
	// shape functions evaluated at gauss point
	Element::N[0][gauss_pt]=.125*(1-s)*(1-t)*(1-u);
	Element::N[1][gauss_pt]=.125*(1+s)*(1-t)*(1-u);
	Element::N[2][gauss_pt]=.125*(1+s)*(1+t)*(1-u);
	Element::N[3][gauss_pt]=.125*(1-s)*(1+t)*(1-u);
	Element::N[4][gauss_pt]=.125*(1-s)*(1-t)*(1+u);
	Element::N[5][gauss_pt]=.125*(1+s)*(1-t)*(1+u);
	Element::N[6][gauss_pt]=.125*(1+s)*(1+t)*(1+u);
	Element::N[7][gauss_pt]=.125*(1-s)*(1+t)*(1+u);
	
	// dN/ds evaluated at gauss point
	Element::dNds[0][gauss_pt]=-.125*(1-t)*(1-u);
	Element::dNds[1][gauss_pt]= .125*(1-t)*(1-u);
	Element::dNds[2][gauss_pt]= .125*(1+t)*(1-u);
	Element::dNds[3][gauss_pt]=-.125*(1+t)*(1-u);
	Element::dNds[4][gauss_pt]=-.125*(1-t)*(1+u);
	Element::dNds[5][gauss_pt]= .125*(1-t)*(1+u);
	Element::dNds[6][gauss_pt]= .125*(1+t)*(1+u);
	Element::dNds[7][gauss_pt]=-.125*(1+t)*(1+u);
	
	// dN/dt evaluated at gauss point
	Element::dNdt[0][gauss_pt]=-.125*(1-s)*(1-u);
	Element::dNdt[1][gauss_pt]=-.125*(1+s)*(1-u);
	Element::dNdt[2][gauss_pt]= .125*(1+s)*(1-u);
	Element::dNdt[3][gauss_pt]= .125*(1-s)*(1-u);
	Element::dNdt[4][gauss_pt]=-.125*(1-s)*(1+u);
	Element::dNdt[5][gauss_pt]=-.125*(1+s)*(1+u);
	Element::dNdt[6][gauss_pt]= .125*(1+s)*(1+u);
	Element::dNdt[7][gauss_pt]= .125*(1-s)*(1+u);
	
	// dN/du evaluated at gauss point
	Element::dNdu[0][gauss_pt]=-.125*(1-s)*(1-t);
	Element::dNdu[1][gauss_pt]=-.125*(1+s)*(1-t);
	Element::dNdu[2][gauss_pt]=-.125*(1+s)*(1+t);
	Element::dNdu[3][gauss_pt]=-.125*(1-s)*(1+t);
	Element::dNdu[4][gauss_pt]= .125*(1-s)*(1-t);
	Element::dNdu[5][gauss_pt]= .125*(1+s)*(1-t);
	Element::dNdu[6][gauss_pt]= .125*(1+s)*(1+t);
	Element::dNdu[7][gauss_pt]= .125*(1-s)*(1+t);
	
	gauss_pt++;  // gauss point will count up to 8 points
	
      }
    }
  }
 
  
}



