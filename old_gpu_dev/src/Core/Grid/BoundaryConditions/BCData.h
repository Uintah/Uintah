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


#ifndef UINTAH_GRID_BCData_H
#define UINTAH_GRID_BCData_H

#include <map>
#include <string>
#include <vector>
#include <Core/Util/Handle.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>

namespace Uintah {
using std::map;
using std::string;
using std::vector;


/**************************************

CLASS
   BCData
   
   
GENERAL INFORMATION

   BCData.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2001 SCI Group

KEYWORDS
   BoundCondBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class BCData  {
  public:
    BCData();
    ~BCData();
    BCData(const BCData&);
    BCData& operator=(const BCData&);
    bool operator==(const BCData&);
    bool operator<(const BCData&) const;
    void setBCValues(BoundCondBase* bc);
    const BoundCondBase* getBCValues(const string& type) const;
    void print() const;
    bool find(const string& type) const;
    bool find(const string& bc_type,const string& bc_variable) const;
    void combine(BCData& from);
    
   private:
      // The map is for the name of the
    // bc type and then the actual bc data, i.e. 
    // "Velocity", VelocityBoundCond
    vector<BoundCondBase*>  d_BCData;

    
  };
} // End namespace Uintah

#endif


