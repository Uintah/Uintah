#include "Endian.h"

namespace SCIRun {

void swapbytes(short& i){SWAP_2(i);}
void swapbytes(unsigned short& i){SWAP_2(i);}
void swapbytes(int& i){ SWAP_4(i);}
void swapbytes(unsigned int& i){ SWAP_4(i);}
// Temporary for initial compile, must be removed
void swapbytes(long int& i){ SWAP_4(i);}
void swapbytes(long long& i){ SWAP_8(i);}
// /////////////////////////////////////////////
void swapbytes(float& i){SWAP_4(i);}
//void swapbytes(int64_t& i){SWAP_8(i);}
void swapbytes(uint64_t& i){SWAP_8(i);}
void swapbytes(double& i){SWAP_8(i);}
void swapbytes(Point &i){ // probably dangerous, but effective
     double* p = (double *)(&i);
     SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p); }
void swapbytes(Vector &i){ // probably dangerous, but effective
     double* p = (double *)(&i);
     SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p); }

bool isBigEndian()
{
  short i = 0x4321;
  if((*(char *)&i) != 0x21 ){
    return true;
  } else {
    return false;
  }
}

bool isLittleEndian()
{
  return !isBigEndian();
}
 
string endianness()
{
  if( isBigEndian() ){
    return string("big_endian");
  } else {
    return string("little_endian");
  }
}

} // end namespace SCIRun
