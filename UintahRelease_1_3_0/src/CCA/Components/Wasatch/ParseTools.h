#ifndef Wasatch_ParseTools_h
#define Wasatch_ParseTools_h

#include <Core/ProblemSpec/ProblemSpecP.h>

/**
 *  \file ParseTools.h
 *  \defgroup WasatchParser	Wasatch Parsers
 */

// forward declaration
namespace Expr{ class Tag; }

namespace Wasatch{

  /**
   *  \ingroup WasatchParser
   *  \brief Parses a name tag, comprised of a variable name and state.
   *  \param param The parser block for this name tag.
   *  \return the Expr::Tag.
   */
  Expr::Tag parse_nametag( Uintah::ProblemSpecP param );

} // namespace Wasatch


#endif // Wasatch_ParseTools_h
