#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

extern "C" {

void execute(const vector<FieldHandle>& in, vector<FieldHandle>& out)
{
   enum { number_of_outputs = 1 };   //TODO: replace with the right number
   out.resize( number_of_outputs );

   //TODO: implement your manipulation

   cout << "FieldManip has been executed" << endl; 
}

}
