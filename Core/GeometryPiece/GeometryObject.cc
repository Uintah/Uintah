/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <iostream>

using namespace Uintah;

GeometryObject::GeometryObject(GeometryPieceP piece, ProblemSpecP& ps,
                               list<string>& data) :
  d_piece(piece)
{
   ps->require("res", d_resolution);
   ps->require("velocity", d_initialVel);

   for (list<string>::iterator it = data.begin(); it != data.end();it++){
     double val;
     ps->require(*it,val);
     d_data[*it] = val;
   }
}

void
GeometryObject::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP geom_obj_ps = ps->appendChild("geom_object");
  d_piece->outputProblemSpec(geom_obj_ps);
  
  geom_obj_ps->appendElement("res", d_resolution);
  geom_obj_ps->appendElement("velocity", d_initialVel);
  for (map<string,double>::iterator it = d_data.begin(); 
       it != d_data.end(); it++) {
    geom_obj_ps->appendElement(it->first.c_str(),it->second);
  }
}
