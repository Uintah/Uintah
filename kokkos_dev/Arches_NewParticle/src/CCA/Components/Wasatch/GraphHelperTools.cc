//-- Wasatch Includes --//
#include "GraphHelperTools.h"

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>

namespace Wasatch{

  GraphHelper::GraphHelper( Expr::ExpressionFactory* ef )
    : exprFactory(ef)
  {}

} // namespace Wasatch
