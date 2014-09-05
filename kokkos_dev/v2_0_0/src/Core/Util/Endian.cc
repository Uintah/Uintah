#include <Core/Util/Endian.h>

namespace SCIRun {

void swapbytes(int8_t&) { }
void swapbytes(uint8_t&) { }
void swapbytes(int16_t& i) { SWAP_2(i); }
void swapbytes(uint16_t& i) { SWAP_2(i); }
void swapbytes(int32_t& i) { SWAP_4(i); }
void swapbytes(uint32_t& i) { SWAP_4(i); }
void swapbytes(int64_t& i) { SWAP_8(i); }
void swapbytes(uint64_t& i) { SWAP_8(i); }
void swapbytes(float& i){SWAP_4(i);}
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
