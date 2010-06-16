#ifndef Wasatch_GraphHelperTools_h
#define Wasatch_GraphHelperTools_h

#include <list>
#include <map>
#include <set>

#include <expression/Expr_ExpressionID.h>

namespace Expr{
  class ExpressionBuilder;
  class ExpressionFactory;
  class TransportEquation;
}

namespace Wasatch{

  /**
   *  \enum Category
   *  \brief defines the broad categories for various kinds of tasks.
   */
  enum Category{
    INITIALIZATION,
    TIMESTEP_SELECTION,
    ADVANCE_SOLUTION
  };

  typedef std::list<Expr::TransportEquation*> TransEqns;

  typedef std::set< Expr::ExpressionID > IDSet;

  /**
   *  \struct GraphHelper
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Contains information helpful in constructing graphs from ExprLib
   */
  struct GraphHelper
  {
    Expr::ExpressionFactory* const exprFactory;
    IDSet rootIDs;
    GraphHelper( Expr::ExpressionFactory* ef );
  };

  typedef std::map< Category, GraphHelper* > GraphCategories;

} // namespace Wasatch


#endif // Wasatch_GraphHelperTools_h
