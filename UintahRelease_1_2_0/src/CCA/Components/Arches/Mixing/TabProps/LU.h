/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#ifndef LU_h
#define LU_h

#include <assert.h>

/**
 *  @class LU
 *  @author James C. Sutherland
 *  @date   November, 2005
 *
 *  @brief Supports LU-decompositon for a matrix.
 */
class LU{
 public:
  LU( const int dim, const int bandwidth );
  ~LU();

  inline double& operator ()(const int row, const int col){
    assert( row <= dim_ );
    assert( col <= dim_ );
    isReady_ = false;
    return AA_(row,col);
  };

  // perform the LU-factorization
  void decompose();

  // perform back-substitution given the rhs.
  // Over-writes the rhs with the solution vector.
  void back_subs( double* rhs );

  void dump();

 private:

  LU();  // no default constructor.

  class SparseMatrix{
  public:
    SparseMatrix( const int dim, const int bandwidth );
    ~SparseMatrix();
    inline double& operator ()(const int row, const int col){
      return AA_[row][col];
    };
    void dump();
  private:
    SparseMatrix();
    double **AA_;
    const int dim_, band_;
  };

  const int dim_;
  bool isReady_;
  SparseMatrix AA_;

};

#endif
