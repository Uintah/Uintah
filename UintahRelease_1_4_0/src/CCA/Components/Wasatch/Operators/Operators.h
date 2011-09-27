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
   *
   *  \brief constructs operators for use on the given patch and
   *         stores them in the supplied OperatorDatabase.
   *
   *  \param patch - the Uintah::Patch that the operators are built for.
   *  \param opDB  - the OperatorDatabase to store the operators in.
   *
   *  All supported operators will be constructed and stored in the
   *  supplied OperatorDatabase.  Note that these operators are
   *  associated with the supplied patch only.  Different patches may
   *  have different operators, particularly in the case of AMR.
   */
  void build_operators( const Uintah::Patch& patch,
                        SpatialOps::OperatorDatabase& opDB );


} // namespace Wasatch

#endif // Wasatch_Operators_h
