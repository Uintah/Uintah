#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Matrix.h>
//#include <Core/Datatypes/SMatrix.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

extern "C" {

void execute(const vector<MatrixHandle>& in, vector<MatrixHandle>& out)
{
   enum { number_of_outputs = 1 };   //TODO: replace with the right number
   out.resize( number_of_outputs );

   //TODO: implement your manipulation

   cout << "MatrixManip has been executed" << endl; 
}

}
