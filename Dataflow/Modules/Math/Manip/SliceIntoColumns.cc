#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

extern "C" {

void execute(const vector<MatrixHandle>& in, vector<MatrixHandle>& out)
{
  enum { number_of_columns = 3 };
  ASSERT( in.size() == 1 );
  ASSERT( in[0]->ncols() == number_of_columns );
  ASSERT( out.size() == number_of_columns ); 

  int number_of_rows = in[0]->nrows();
  for( int col = 0; col < number_of_columns; ++col )
  {
    ColumnMatrix* column = new ColumnMatrix( number_of_rows );
    //TODO: for greater efficiency, provide Matrix iterators
    for( int row = 0; row < number_of_rows; ++row )
      column->operator[]( row ) = in[0]->get( row, col );
    out[ col ] = column;
  }
  cout << "MatrixManip has been executed" << endl; 
}

}
