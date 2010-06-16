#ifndef Wasatch_ParseTools_h
#define Wasatch_ParseTools_h

#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Expr{ class Tag; }

namespace Wasatch{

  Expr::Tag parse_nametag( Uintah::ProblemSpecP param );

} // namespace Wasatch


#endif // Wasatch_ParseTools_h
