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


#ifndef UINTAH_GRID_BoundCond_H
#define UINTAH_GRID_BoundCond_H

#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <string>

namespace Uintah {
using std::string;
 using namespace SCIRun;
   
/**************************************

CLASS
   BoundCond
   
   
GENERAL INFORMATION

   BoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

 class NoValue {

 public:
   NoValue() {};
   ~NoValue() {};
 };

 template <class T>  class BoundCond : public BoundCondBase {
 public:
   BoundCond() {};

   BoundCond(string var_name, string type, T value) 
     {
       d_variable = var_name;
       d_type__NEW = type;
       d_value = value;
     };
   virtual ~BoundCond() {};
   virtual BoundCond* clone()
   {
     return scinew BoundCond(*this);
   };

   T getValue() const { return d_value;}; 

   
 protected:
   T d_value;

 };


 template <> class BoundCond<NoValue> : public BoundCondBase {


 public:

   BoundCond(string var_name,string type)
     {
       d_variable = var_name;
       d_type__NEW = type;
       d_value = NoValue();
     };

   BoundCond(string var_name)
     {
       d_variable = var_name;
       d_type__NEW = "";
       d_value = NoValue();
     };

   virtual BoundCond* clone()
   {
     return scinew BoundCond(*this);
   };

   
 protected:
   NoValue d_value;

 };
 
} // End namespace Uintah



#endif
