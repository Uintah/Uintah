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

/*
 *  RescaleColorMap.cc:  Generate Color maps
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Modules/Visualization/RescaleColorMap.h>
#include <Packages/Uintah/Core/Datatypes/DispatchScalar1.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldAlgo.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace Uintah {

/**************************************
CLASS
   RescaleColorMap

   A module that can scale the colormap values to fit the data
   or express a fixed data range.

GENERAL INFORMATION
   RescaleColorMap.h
   Written by:

     Kurt Zimmerman<br>
     Department of Computer Science<br>
     University of Utah<br>
     June 1999

     Copyright (C) 1998 SCI Group

KEYWORDS
   ColorMap, Transfer Function

DESCRIPTION
   This module takes a color map and some data or vector field
   and scales the map to fit that field.

****************************************/

using SCIRun::Field;

class RescaleColorMap : public SCIRun::RescaleColorMap {
public:
          // GROUP:  Constructors:
        ///////////////////////////
        // Constructs an instance of class RescaleColorMap
        // Constructor taking
        //    [in] id as an identifier
  RescaleColorMap(const string& id);

        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
  virtual ~RescaleColorMap();

  virtual void get_minmax(FieldHandle field);
};


extern "C" Module* make_RescaleColorMap(const string& id) {
  return new RescaleColorMap(id);
}

RescaleColorMap::RescaleColorMap( const string& id) :
  SCIRun::RescaleColorMap( id ) 
{
  packageName = "Uintah";
}

RescaleColorMap::~RescaleColorMap() 
{
}

 void RescaleColorMap::get_minmax(FieldHandle field)
{
#if 0
  uintah_dispatch_scalar1(field, dispatch_minmax);
  if( !success_ ){
    SCIRun::RescaleColorMap::get_minmax( field );
  }
#endif
}



} // End namespace Uintah


