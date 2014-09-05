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

#include <Volume/Core/Datatypes/BrickData.h>

namespace Volume {

BrickData::BrickData() :
  nx_(0), ny_(0), nz_(0), nbytes_(0)
{
}
BrickData::~BrickData()
{
//// delete tex;  This was given to us do not delete!  May change.
}

BrickData::BrickData( int nx, int ny, int nz, int nbytes ) :
  nx_(nx), ny_(ny), nz_(nz), nbytes_(nbytes)
{
}  

} // End namespace Volume
