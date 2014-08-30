#ifndef Expr_FieldManager_h
#define Expr_FieldManager_h

#define ENABLE_UINTAH

#include <expression/uintah/UintahFieldManager.h>

namespace Expr{

  template< typename FieldT >
  struct FieldMgrSelector{
    typedef UintahFieldManager<FieldT> type;
  };

}  // namespace Expr

#endif // Expr_FieldManager_h
