#include "Surface.h"

const double pi = 3.1415926;

Surface::Surface(){
}

void Surface::getPhi(double &phi, double &random){
  phi = 2 * pi * random;
}


Surface::~Surface(){
}


