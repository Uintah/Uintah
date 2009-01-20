#include "Surface.h"
#include "Consts.h"
#include <iostream>

using std::cout;
using std::endl;

Surface::Surface(){
}

void Surface::getPhi(const double &random){
   phi = 2 * pi * random;

}


Surface::~Surface(){
}


