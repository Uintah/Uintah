#ifndef Wasatch_Operators_h
#define Wasatch_Operators_h

/**
 *  \file Operators.h
 */

namespace SpatialOps{ class OperatorDatabase; }  // forward declaration
namespace Uintah{ class Patch; }

namespace Wasatch{

  /**
   *  \ingroup WasatchOperators
   *  \brief constructs operators for use on the given patch and
   *         stores them in the supplied OperatorDatabase.
   */
  void build_operators( const Uintah::Patch& patch,
                        SpatialOps::OperatorDatabase& opDB );


} // namespace Wasatch

#endif // Wasatch_Operators_h
