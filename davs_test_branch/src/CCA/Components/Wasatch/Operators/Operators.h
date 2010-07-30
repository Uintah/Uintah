#ifndef Wasatch_Operators_h
#define Wasatch_Operators_h


namespace SpatialOps{ class OperatorDatabase; }  // forward declaration
namespace Uintah{ class Patch; }

namespace Wasatch{

  void build_operators( const Uintah::Patch& patch,
                        SpatialOps::OperatorDatabase& opDB );


} // namespace Wasatch

#endif // Wasatch_Operators_h
