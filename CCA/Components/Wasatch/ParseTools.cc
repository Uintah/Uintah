//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

//-- Wasatch includes --//
#include "ParseTools.h"

//-- ExprLib includes --//
#include <expression/Tag.h>

#include <string>

namespace Wasatch{

  Expr::Tag
  parse_nametag( Uintah::ProblemSpecP param )
  {
    if( !param ) throw Uintah::ProblemSetupException( "NameTag not found", __FILE__, __LINE__ );

    std::string exprName;
    std::string state;

    param->getAttribute( "name", exprName );
    param->getAttribute( "state", state );

    Expr::Context exprContext(Expr::INVALID_CONTEXT);
    if     ( state.compare("STATE_NONE"   ) == 0 )  exprContext = Expr::STATE_NONE;
    else if( state.compare("STATE_N"      ) == 0 )  exprContext = Expr::STATE_N;
    else if( state.compare("STATE_NP1"    ) == 0 )  exprContext = Expr::STATE_NP1;
    else if( state.compare("CARRY_FORWARD") == 0 )  exprContext = Expr::CARRY_FORWARD;

    return Expr::Tag(exprName, exprContext);
  }

} // namespace Wasatch
