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

    Expr::Tag tag;

    std::string state;
    param->getAttribute( "name", tag.name() );
    param->getAttribute( "state", state );

    if     ( state == "STATE_NONE" )  tag.context() = Expr::STATE_NONE;
    else if( state == "STATE_N"    )  tag.context() = Expr::STATE_N;
    else if( state == "STATE_NP1"  )  tag.context() = Expr::STATE_NP1;
 
    return tag; 
  }

} // namespace Wasatch
