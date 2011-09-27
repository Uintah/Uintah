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


#include "FieldDiags.h"

#include <string>
#include <map>
#include <list>
#include <numeric>
#include <limits>

using namespace std;
using namespace SCIRun;
using namespace Uintah;
  
FieldDiag::~FieldDiag() {}
  
bool 
FieldDiag::has_mass(DataArchive * da, const Patch * patch, 
                    Uintah::TypeDescription::Type fieldtype,
                    int imat, int index, const IntVector & pt) const
{
  switch(fieldtype) 
    {
    case TypeDescription::NCVariable: {
      NCVariable<double> Mvalue;
      da->query(Mvalue, "g.mass", imat, patch, index);
      return (Mvalue[pt]>numeric_limits<float>::epsilon());
    }
    case TypeDescription::CCVariable: {
      CCVariable<double> Mvalue;
      da->query(Mvalue, "g.mass", imat, patch, index);
      return (Mvalue[pt]>numeric_limits<float>::epsilon());
    }
    default: 
      throw InternalError("Bad type is has_mass()", __FILE__, __LINE__);
    } // end switch

} // end has_mass()

