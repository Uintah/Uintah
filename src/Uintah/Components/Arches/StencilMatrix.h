//
// $Id$
//

#ifndef Uintah_Components_Arches_StencilMatrix_h
#define Uintah_Components_Arches_StencilMatrix_h

/***************************************************************************
CLASS
   StencilMatrix
   
   Class StencilMatrix stores the data for 7 stencils needed for
   Coefficient calculation.

GENERAL INFORMATION
   StencilMatrix.h - declaration of the class
   
   Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)
   
   Creation Date:   June 10, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Detailed description

WARNING
   none
*************************************************************************/

#include <Uintah/Exceptions/InvalidValue.h>
#include <vector>

using std::vector;

namespace Uintah {
namespace ArchesSpace {

template<class T>

class StencilMatrix {

public:

      // GROUP: Constants:
      ////////////////////////////////////////////////////////////////////////
      //
      // Enumerate the names of the variables
      //
      //enum stencilName {AP, AE, AW, AN, AS, AT, AB};

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Construct an instance of a StencilMatrix.
      //
      StencilMatrix();

      ////////////////////////////////////////////////////////////////////////
      //
      // Construct a copy of a StencilMatrix.
      //
      StencilMatrix(const StencilMatrix<T>&);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      //
      // Destructor
      //
      virtual ~StencilMatrix();

      // GROUP:  Operator Overloads
      ////////////////////////////////////////////////////////////////////////
      //
      // Get a reference to the variable at the input index 
      //
      T& operator[](int index); 

private:

      vector<T> d_data;

}; // end Class Source

//********************************************************************
// Default constructor
//********************************************************************
template<class T>
  StencilMatrix<T>::StencilMatrix()
  {
    for (int ii = 0; ii < 7; ii++) d_data.push_back(T());
  }

//********************************************************************
// Copy constructor
//********************************************************************
template<class T>
  StencilMatrix<T>::StencilMatrix(const StencilMatrix<T>& sm)
  {
    for (int ii = 0; ii < 7; ii++) d_data.push_back(sm.d_data[ii]);
  }

//********************************************************************
// Destructor
//********************************************************************
template<class T>
  StencilMatrix<T>::~StencilMatrix()
  {
  }

//********************************************************************
// Get a reference to an object in the vector
//********************************************************************
template<class T>
  T&
  StencilMatrix<T>::operator[](int index)
  {
    if (index < 0 || index > 6) 
      throw InvalidValue("Valid Indices for StencilMatrix are AP,AE,AW,AN,AS,AT and AB ");
    return d_data[index];
  }

}  // End namespace ArchesSpace
}  // End namespace Uintah
#endif  
  
//
// $Log$
// Revision 1.2  2000/07/08 08:03:35  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.1  2000/06/12 21:30:00  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
//
