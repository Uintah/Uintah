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

//    File   : LinearAlgebra.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(LinearAlgebra_h)
#define LinearAlgebra_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/MatrixOperations.h>
#include <Core/Math/function.h>

namespace SCIRun {

class LinearAlgebraAlgo : public DynamicAlgoBase
{
public:
  virtual MatrixHandle function0();
  virtual MatrixHandle function1(MatrixHandle A);
  virtual MatrixHandle function2(MatrixHandle A,
				 MatrixHandle B);
  virtual MatrixHandle function3(MatrixHandle A,
				 MatrixHandle B,
				 MatrixHandle C);
  virtual MatrixHandle function4(MatrixHandle A,
				 MatrixHandle B,
				 MatrixHandle C,
				 MatrixHandle D);
  virtual MatrixHandle function5(MatrixHandle A,
				 MatrixHandle B,
				 MatrixHandle C,
				 MatrixHandle D,
				 MatrixHandle E);

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(int matrixcount,
					    string function,
					    int hashoffset);
};


} // end namespace SCIRun

#endif // LinearAlgebra_h
