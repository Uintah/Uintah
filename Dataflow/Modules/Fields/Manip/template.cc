#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
//#include <Core/Datatypes/SField.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

extern "C" {

void execute(vector<FieldHandle>& in, vector<FieldHandle>& out)
{
  cout << "hey I was called" << endl;
  static FieldHandle f;
  return f;
}

}
