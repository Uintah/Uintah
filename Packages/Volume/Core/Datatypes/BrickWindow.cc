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

#include <Core/Geometry/BBox.h>
#include <Packages/Volume/Core/Datatypes/BrickWindow.h>

namespace Volume {

BrickWindow::BrickWindow(int imin, int jmin, int kmin,
                         int imax, int jmax, int kmax,
                         const BBox& vbox) :
  i_min_(imin), j_min_(jmin), k_min_(kmin),
  i_max_(imax), j_max_(jmax), k_max_(kmax),
  vbox_(vbox)
{
}
void 
BrickWindow::getMinIndex( int& i, int& j, int& k)
{
  i = i_min_;
  j = j_min_;
  k = k_min_;
}

void 
BrickWindow::getMaxIndex( int& i, int& j, int& k)
{
  i = i_max_;
  j = j_max_;
  k = k_max_;
}
  
void  
BrickWindow::getBoundingIndices( int& i_, int& j_, int& k_,
                           int& i, int& j, int& k)
{
  i_ = i_min_;
  j_ = j_min_;
  k_ = k_min_;
  i = i_max_;
  j = j_max_;
  k = k_max_;
}
 

} // End namespace Volume
