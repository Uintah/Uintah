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

#ifndef BRICKWINDOW_H
#define BRICKWINDOW_H


#include <Core/Geometry/BBox.h>

namespace Volume {

/**************************************

CLASS
   BrickWindow
   
   BrickWindow Class for 3D Texturing 

GENERAL INFORMATION

   BrickWindow.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BrickWindow

DESCRIPTION
   BrickWindow class for 3D Texturing.  Stores the texture associated with
   the BrickWindow and the BrickWindows location is space.  For a given view ray,
   min and max ray parameter, and parameter delta, it will create an
   ordered (back to fron) polygon list that can be rendered by a
   volume renderer.

  
WARNING
  
****************************************/

using SCIRun::BBox;

class BrickWindow 
{
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  BrickWindow(int imin, int jmin, int kmin,
              int imax, int jmax, int kmax,
              const BBox& vbox);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~BrickWindow(){};

  // GROUP: Access and Info
  //////////
  // obtain the bounding indices in the orginal data
  void getMinIndex( int& i, int& j, int& k);
  void getMaxIndex( int& i, int& j, int& k);
  void getBoundingIndices( int& i_, int& j_, int& k_,
                           int& i, int& j, int& k); 
  int min_i() { return i_min_; }
  int min_j() { return j_min_; }
  int min_k() { return k_min_; }
  int max_i() { return i_max_; }
  int max_j() { return j_max_; }
  int max_k() { return k_max_; }
  // obtain the bounding box of the BrickWindow
  BBox vbox() const { return vbox_;}

protected:
  int  i_min_, j_min_, k_min_;
  int  i_max_, j_max_, k_max_;
  BBox vbox_;

private:
  BrickWindow(){};

};

} // End namespace Volume

#endif
