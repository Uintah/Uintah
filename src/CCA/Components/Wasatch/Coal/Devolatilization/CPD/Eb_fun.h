#ifndef Eb_fun_h
#define Eb_fun_h

#include <spatialops/Nebo.h>

namespace CPD{

  template<typename FieldT>
  inline void Eb_fun( FieldT& result, const FieldT& p, const double E0, const double sigma )
  {
    using namespace SpatialOps;
    result <<= E0 - 1.41421356 * sigma * inv_erf( 1.0 - 2.0 * max( min( p, 0.999 ), 1.0e-3 ) );
  }

}  // namespace CPD

#endif // Eb_fun_h
