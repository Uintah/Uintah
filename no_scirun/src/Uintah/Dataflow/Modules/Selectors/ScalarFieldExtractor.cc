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


/****************************************
CLASS
    ScalarFieldExtractor

    Visualization control for simulation data that contains
    information on both a regular grid in particle sets.

OVERVIEW TEXT
    This module receives a ParticleGridReader object.  The user
    interface is dynamically created based information provided by the
    ParticleGridReader.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#include "ScalarFieldExtractor.h"
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

 
#include <iostream> 
#include <string>
#include <vector>

using std::cerr;
using std::vector;
using std::string;

using namespace Uintah;
using namespace SCIRun;

  //using DumbScalarField;

  DECLARE_MAKER(ScalarFieldExtractor)

//--------------------------------------------------------------- 
ScalarFieldExtractor::ScalarFieldExtractor(GuiContext* ctx) 
  : FieldExtractor("ScalarFieldExtractor", ctx, "Selectors", "Uintah")
{ 
} 

//------------------------------------------------------------ 
ScalarFieldExtractor::~ScalarFieldExtractor(){} 

//------------------------------------------------------------- 
void
ScalarFieldExtractor::get_vars(vector< string >& names,
                               vector< const Uintah::TypeDescription *>& types)
{
  string command;
  // Set up data to build or rebuild GUI interface
  string sNames("");
  int index = -1;
  bool matches = false;
  // get all of the ScalarField Variables
  for( int i = 0; i < (int)names.size(); i++ ){
    const TypeDescription *td = types[i];
    const TypeDescription *subtype = td->getSubType();
    //  only handle NC and CC Vars
    if( td->getType() ==  TypeDescription::NCVariable ||
        td->getType() ==  TypeDescription::CCVariable ||
        td->getType() ==  TypeDescription::SFCXVariable ||
        td->getType() ==  TypeDescription::SFCYVariable ||
        td->getType() ==  TypeDescription::SFCZVariable )
    {
      // supported scalars double, float, int, long64, long long, short, bool
      if( subtype->getType() == TypeDescription::double_type ||
          subtype->getType() == TypeDescription::float_type ||
          subtype->getType() == TypeDescription::int_type ||
          subtype->getType() == TypeDescription::long64_type ||
          subtype->getType() == TypeDescription::bool_type) {
//        subtype->getType() == TypeDescription::short_int_type ||
        if( sNames.size() != 0 )
          sNames += " ";
        sNames += names[i];
        if( sVar.get() == "" ){ sVar.set( names[i].c_str() ); }
        if( sVar.get() == names[i].c_str()){
          type = td;
          matches = true;
        } else {
          if( index == -1) {index = i;}
        }
      } 
    }
  }

  if( !matches && index != -1 ) {
    sVar.set(names[index].c_str());
    type = types[index];
  }

  // inherited from FieldExtractor
  update_GUI(sVar.get(), sNames);
}


