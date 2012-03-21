/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef Packages_Uintah_CCA_Components_Solvers_MatrixUtil_h
#define Packages_Uintah_CCA_Components_Solvers_MatrixUtil_h

#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

namespace Uintah {
  class SFCXTypes {
  public:
    typedef constSFCXVariable<Stencil7> matrix_type;
    typedef constSFCXVariable<Stencil4> symmetric_matrix_type;    
    typedef constSFCXVariable<double> const_type;
    typedef SFCXVariable<double> sol_type;
  };

  class SFCYTypes {
  public:
    typedef constSFCYVariable<Stencil7> matrix_type;
    typedef constSFCYVariable<Stencil4> symmetric_matrix_type;        
    typedef constSFCYVariable<double> const_type;
    typedef SFCYVariable<double> sol_type;
  };

  class SFCZTypes {
  public:
    typedef constSFCZVariable<Stencil7> matrix_type;
    typedef constSFCZVariable<Stencil4> symmetric_matrix_type;            
    typedef constSFCZVariable<double> const_type;
    typedef SFCZVariable<double> sol_type;
  };

  class CCTypes {
  public:
    typedef constCCVariable<Stencil7> matrix_type;
    typedef constCCVariable<Stencil4> symmetric_matrix_type;            
    typedef constCCVariable<double> const_type;
    typedef CCVariable<double> sol_type;
  };
  
  class NCTypes {
  public:
    typedef constNCVariable<Stencil7> matrix_type;
    typedef constNCVariable<Stencil4> symmetric_matrix_type;
    typedef constNCVariable<double> const_type;
    typedef NCVariable<double> sol_type;
  };
}

#endif // Packages_Uintah_CCA_Components_Solvers_MatrixUtil_h
