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


#ifndef ScrInterfaceImpl_h
#define ScrInterfaceImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <Core/CCA/SSIDL/array.h>

namespace sci_cca {

using SSIDL::array1;

class scrInterfaceImpl : virtual public scrInterface {

public:
  scrInterfaceImpl();
  virtual ~scrInterfaceImpl();

  virtual void exec( int cells,
		     double pressure,
		     double kgcat,
		     double NH3ratio,
		     double NH3,
		     double& flow,
		     double temp,
		     double NO,
		     double N2,
		     double H2O,
		     double O2,
		     double sum_of_all_others,
		     double heat_loss,
		     ::SSIDL::array1< double>& kmol_s,
		     double& noreduction,
		     double& new_temp,
		     double& kmol_s_tot);
};

} // namespace sci_cca

#endif 

