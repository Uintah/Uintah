//
//  class Matrix3
//    Matrix3 data type -- holds components of a 3X3 matrix
//
//
//
//    Features:
//      1.  Nearly, all the overloaded operations of the Matrix class,
//      no dynamic memory allocation.

#include "Matrix3.h"
#include <iostream>
using std::cout;
using std::endl;
#include <fstream>
using std::ostream;
#include <stdlib.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Util/FancyAssert.h>
#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif


void Matrix3::set(const int i, const int j, const double value)
{
  // Assign the Matrix3 the value components
   mat3[i-1][j-1] = value;
}

Matrix3 Matrix3::Inverse()
{
  // Return the inverse of a 3x3 matrix
  // This looks ugly but it works -- just for 3x3

  double det;
  Matrix3 inv_matrix(0.0);

  det = this->Determinant();
  if ( det == 0.0 )
  {
    cout << "Singular matrix in matrix inverse..." << endl;
    exit(1);
  }
  else
  {
    inv_matrix(1,1) = (*this)(2,2)*(*this)(3,3) - (*this)(2,3)*(*this)(3,2);
    inv_matrix(1,2) = -(*this)(2,1)*(*this)(3,3) + (*this)(2,3)*(*this)(3,1);
    inv_matrix(1,3) = (*this)(2,1)*(*this)(3,2) - (*this)(2,2)*(*this)(3,1);
    inv_matrix(2,1) = -(*this)(1,2)*(*this)(3,3) + (*this)(1,3)*(*this)(3,2);
    inv_matrix(2,2) = (*this)(1,1)*(*this)(3,3) - (*this)(1,3)*(*this)(3,1);
    inv_matrix(2,3) = -(*this)(1,1)*(*this)(3,2) + (*this)(1,2)*(*this)(3,1);
    inv_matrix(3,1) = (*this)(1,2)*(*this)(2,3) - (*this)(1,3)*(*this)(2,2);
    inv_matrix(3,2) = -(*this)(1,1)*(*this)(2,3) + (*this)(1,3)*(*this)(2,1);
    inv_matrix(3,3) = (*this)(1,1)*(*this)(2,2) - (*this)(1,2)*(*this)(2,1);
 
    inv_matrix = inv_matrix/det;

  }
  return inv_matrix;

} //end Inverse()

ostream & operator << (ostream &out_file, const Matrix3 &m3)
{
  // Overload the output stream << operator

  out_file <<  m3(1,1) << ' ' << m3(1,2) << ' ' << m3(1,3) << endl;
  out_file <<  m3(2,1) << ' ' << m3(2,2) << ' ' << m3(2,3) << endl;
  out_file <<  m3(3,1) << ' ' << m3(3,2) << ' ' << m3(3,3) ;

  return out_file;

}

namespace Uintah {

MPI_Datatype makeMPI_Matrix3()
{
   ASSERTEQ(sizeof(Matrix3), sizeof(double)*9);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 9, 9, MPI_DOUBLE, &mpitype);
   return mpitype;
}

   const TypeDescription* fun_getTypeDescription(Matrix3*)
   {
      static TypeDescription* td = 0;
      if(!td){
	 td = new TypeDescription(TypeDescription::Matrix3, "Matrix3", true,
				  &makeMPI_Matrix3);
      }
      return td;
   }
}

//$Log$
//Revision 1.3  2000/07/27 22:39:45  sparker
//Implemented MPIScheduler
//Added associated support
//
//Revision 1.2  2000/05/20 08:09:12  sparker
//Improved TypeDescription
//Finished I/O
//Use new XML utility libraries
//
//Revision 1.1  2000/03/14 22:12:43  jas
//Initial creation of the utility directory that has old matrix routines
//that will eventually be replaced by the PSE library.
//
//Revision 1.1  2000/02/24 06:11:58  sparker
//Imported homebrew code
//
//Revision 1.1  2000/01/24 22:48:54  sparker
//Stuff may actually work someday...
//
//Revision 1.7  1999/12/20 23:56:41  guilkey
//Worked over the Matrix3 class to do smarter things in methods that return
//a Matrix3.  This results in a dramatic improvement in performance for the
//+ and * operators, which are used heavily in the constitutive models.
//
//Revision 1.6  1999/12/18 19:31:53  guilkey
//Fixed the Matrix3 class so that it now uses a double Mat3[3][3] for
//storage, rather than the 4X4 it used before.  This required adding offsets
//for the access operators.  A future improvement will be to fix all of the
//code so that this isn't necessary.
//
//Revision 1.5  1999/08/18 22:04:25  zhangr
//*** empty log message ***
//
//Revision 1.4  1999/08/17 20:41:02  zhangr
//// - Added two more functions:
//// - overloaded set() to assign the Matrix3 the value components
//// - Added Inverse() to inverse a 3x3 matrix.
//// - r. zhang.
//
//Revision 1.3  1999/05/24 21:11:18  guilkey
//Added Norm() function to return the norm of a Matrix3.
//
//Revision 1.2  1999/02/25 05:52:22  guilkey
//Inlined access operators.
//
//Revision 1.1  1999/02/18 21:13:04  guilkey
//Matrix3 class is a fixed size matrix (3X3).
//
