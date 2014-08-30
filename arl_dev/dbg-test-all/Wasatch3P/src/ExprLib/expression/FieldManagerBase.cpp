#include <expression/FieldManagerBase.h>
#include <expression/Tag.h>

namespace Expr{


//------------------------------------------------------------------

int
FieldManagerBase::get_name_id()
{
  static int counter=0;
  return ++counter;
}

//--------------------------------------------------------------------

FieldID
FieldManagerBase::register_field( const std::string& fieldName, const Context c )
{
  return register_field( Tag(fieldName,c) );
}

//--------------------------------------------------------------------

} // namespace Expr
