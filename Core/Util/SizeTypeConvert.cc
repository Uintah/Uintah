#include <Core/Util/SizeTypeConvert.h>

#include <Core/Util/Endian.h>
#include <Core/Exceptions/InternalError.h>

namespace SCIRun {

unsigned long convertSizeType(uint64_t* ssize, bool swapBytes, int nByteMode)
{
  if (nByteMode == 4) {
    uint32_t size32 = *(uint32_t*)ssize;
    if (swapBytes) swapbytes(size32);
    return (unsigned long)size32;
  }
  else if (nByteMode == 8) {
    uint64_t size64 = *(uint64_t*)ssize;
    if (swapBytes) swapbytes(size64);

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209 // constant controlling expressions (sizeof)
#endif  
    if (sizeof(unsigned long) < 8 && size64 > 0xffffffff)
	throw InternalError("Overflow on 64 to 32 bit conversion");
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1209 
#endif  
  
    return (unsigned long)size64;
  }
  else {
    throw InternalError("Must be 32 or 64 bits");
  }
}

}

