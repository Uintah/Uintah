/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Uintah_Components_Arches_StencilMatrix_h
#define Uintah_Components_Arches_StencilMatrix_h

/***************************************************************************
CLASS
   StencilMatrix
   
   Class StencilMatrix stores the data for 9 stencils needed for
   Coefficient calculation.

GENERAL INFORMATION
   StencilMatrix.h - declaration of the class
   
   Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)
   
   Creation Date:   June 10, 2000
   
   C-SAFE 
   
   
KEYWORDS


DESCRIPTION
   Detailed description

WARNING
   none
*************************************************************************/

#include <Core/Exceptions/InvalidValue.h>
#include <iostream>

namespace Uintah {

template<class T>
class StencilMatrix {

public:

  // GROUP: Constants:
  ////////////////////////////////////////////////////////////////////////
  // Enumerate the names of the variables
  //enum stencilName {AP, AE, AW, AN, AS, AT, AB};

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a StencilMatrix.
  StencilMatrix();

  ////////////////////////////////////////////////////////////////////////
  // Construct a copy of a StencilMatrix.
  StencilMatrix(const StencilMatrix<T>&);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
  virtual ~StencilMatrix();

  // GROUP:  Operator Overloads
  ////////////////////////////////////////////////////////////////////////
  // Get a reference to the variable at the input index 
  T& operator[](int index); 

private:

  T d_data[9];

}; // end Class Source

//********************************************************************
// Default constructor
//********************************************************************
template<class T>
  StencilMatrix<T>::StencilMatrix()
  {
    //for (int ii = 0; ii < 7; ii++) d_data.push_back(T());
  }

//********************************************************************
// Copy constructor
//********************************************************************
template<class T>
  StencilMatrix<T>::StencilMatrix(const StencilMatrix<T>& sm)
  {
    for (int ii = 0; ii < 9; ii++)
      d_data[ii] = sm.d_data[ii];
      //d_data.push_back(sm.d_data[ii]);
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
    if (index < 0 || index > 9){
      std::cerr << "Invalid Index" << index << std::endl;
      throw InvalidValue("Valid Indices for StencilMatrix are AP,AE,AW,AN,AS,AT and AB ",
                         __FILE__, __LINE__);
    }
    return d_data[index];
  }
} // End namespace Uintah

#endif  
  
