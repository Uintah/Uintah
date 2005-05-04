/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

/*
 *  MissingMath.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */
 
 
#ifndef JGS_CARDIOWAVE_EXTRAMATH_H
#define JGS_CARDIOWAVE_EXTRAMATH_H 1  
    
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/MatrixOperations.h>
 
 namespace SCIRun {
 
 // Somehow a lot of math functions are not needed by the rest of SCIRun,
 // hence I added them here as missing math. Hopefully, one day it is realized
 // functionality is missing to do elementary Scientific computing. Hence, some day
 // this addition may be obsolete.

  // These functions all work on the Matrix class
 
 class ExtraMath {
  public:
  
  // Constructors: not really needed at this pooint
  // but it is always good to define them
  
    ExtraMath();
    virtual ~ExtraMath();
    
  // Calculate Bandwidth
    int computebandwidth(MatrixHandle matH);
    
    void mapmatrix(MatrixHandle im,MatrixHandle& om,MatrixHandle mapping);
    
  // Recursive CutHill-mcKee
    void rcm(MatrixHandle im,MatrixHandle& om,MatrixHandle& mapping,bool calcmapping,MatrixHandle& imapping,bool calcimapping);

  // CutHill-mcKee
    void cm(MatrixHandle im,MatrixHandle& om,MatrixHandle& mapping,bool calcmapping,MatrixHandle& imapping,bool calcimapping);
    
 };
 
 
 } // end namespace

#endif
 