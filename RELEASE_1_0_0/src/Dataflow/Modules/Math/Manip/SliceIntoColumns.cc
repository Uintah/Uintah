/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Util/Assert.h>
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
  cout << "SliceIntoColumns has been executed" << endl; 
}

}
