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

#ifndef BRICKNODE_H
#define BRICKNODE_H


namespace Volume {

class Brick;
class BrickWindow;


/**************************************

CLASS
   BrickNode
   
   BrickNode Class for 3D Texturing 

GENERAL INFORMATION

   BrickNode.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BrickNode

DESCRIPTION
   BrickNode class for 3D Texturing.  Stores the texture associated with
   the BrickNode and the BrickNodes location is space.  For a given view
   ray, min and max ray parameter, and parameter delta, it will create an
   ordered (back to fron) polygon list that can be rendered by a
   volume renderer.

  
WARNING
  
****************************************/
class BrickNode 
{
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  BrickNode( Brick* brick,
             BrickWindow* bw,
             int index);
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~BrickNode(){};

  // GROUP: Access and Info
  //////////
  // obtain the bounding box of the BrickNode
  Brick* brick() const { return brick_; }
  BrickWindow* brickWindow() const {return brick_window_;}
  int index() const { return idx_; }
protected:
  BrickNode(){};
  Brick *brick_;
  BrickWindow *brick_window_;
  int idx_;

};

} // End namespace Volume

#endif
