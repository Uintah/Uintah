#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/SField.h>

using namespace SCIRun;

extern "C" {

FieldHandle& testFieldManip(vector<FieldHandle> &)
{
  // Replace this function body with your cool code.
  cout << "hey I was called" << endl;
  static FieldHandle f;
  return f;
}

}
