/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <CCA/Components/Arches/ChemMix/TabProps/LU.h>

//--------------------------------------------------------------------
LU::LU( const int dim, const int bandwidth )
  : dim_( dim ),
    AA_( dim, bandwidth )
{
}
//--------------------------------------------------------------------
LU::~LU()
{
}
//--------------------------------------------------------------------
void
LU::decompose()
{
  // perform the LU decomposition, storing the result in AA
  // this currently does not take advantage of any sparsity.
  for( int j=0; j<dim_; j++){
    for( int i=0; i<j; i++ ){
      double sumterm = AA_(i,j);
      for( int k=0; k<i; k++ ){
        sumterm -= AA_(i,k)*AA_(k,j);
      }
      AA_(i,j) =  sumterm;
    }
    
    for( int i=j; i<dim_; i++ ){
      double sumterm = AA_(i,j);
      for( int k=0; k<j; k++ ) sumterm -= AA_(i,k)*AA_(k,j);
      AA_(i,j) = sumterm;
    }
    
    const double tmp = 1.0/AA_(j,j);
    if( j < dim_-1 ){
      for( int i=j+1; i<dim_; i++ ){
        AA_(i,j) *= tmp;
      }
    }
  }
  isReady_ = true;
}
//--------------------------------------------------------------------
void
LU::back_subs( double * rhs )
{
  if( ! isReady_ ){
    throw std::runtime_error( "LU::back_subs() cannot be executed until LU::decompose() has been called!" );
  }
  // AA_ now contains the LU-decomposition of the original "A" matrix.
  // rhs[0] is untouched for now since L(0,0) = 1.
  // forward substitution:
  for( int i=1; i<dim_; i++ ){
    double sumterm = rhs[i];
    for( int j=0; j<i; j++){
      sumterm -= AA_(i,j)*rhs[j];
    }
    rhs[i] = sumterm;
  }

  // back-substitution:
  rhs[dim_-1] /= AA_(dim_-1,dim_-1);
  for( int i=dim_-2; i>=0; i-- ){
    double sumterm = rhs[i];
    
    for( int j=i+1; j<dim_; j++){
      sumterm -= AA_(i,j)*rhs[j];
      rhs[i] = sumterm / AA_(i,i);
    }
    
  }
}
//--------------------------------------------------------------------
LU::SparseMatrix::SparseMatrix( const int dim,
				const int bandwidth )
  : dim_( dim ),
    band_( bandwidth )
{
  // for now, store densely
  AA_ = new double*[dim];
  for( int i=0; i<dim; i++ ){
    AA_[i] = new double[dim];
    for( int j=0; j<dim; j++ ){
      AA_[i][j]=0.0;
    }
  }
}
//--------------------------------------------------------------------
LU::SparseMatrix::~SparseMatrix()
{
  for( int i=0; i<dim_; i++ ){
    delete [] AA_[i];
  }
  delete [] AA_;
}
//--------------------------------------------------------------------
void
LU::dump()
{
  using std::cout;
  using std::endl;

  for( int i=0; i<dim_; i++ ){
    for( int j=0; j<dim_; j++ ){
      cout << std::setw(9) << std::setprecision(4) << AA_(i,j) << "  ";
    }
    cout << endl;
  }
  cout << "-----------------------------------------------------" << endl;
}
//--------------------------------------------------------------------
