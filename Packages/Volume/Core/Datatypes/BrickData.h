/*  The contents of this file are subject to the University of Utah Public
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

#ifndef VOLUME_BRICK_DATA_H
#define VOLUME_BRICK_DATA_H


namespace Volume {

/**************************************

CLASS
   BrickData
   
   Base class for BrickData for 3D Texturing 

GENERAL INFORMATION

   BrickData.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BrickData

DESCRIPTION
   BrickData class for 3D Texturing.  Stores the texture associated with
   the BrickData and the bricks location is space.  For a given view ray,
   min and max ray parameter, and parameter delta, it will create an
   ordered (back to fron) polygon list that can be rendered by a
   volume renderer.

  
WARNING
  
****************************************/
class BrickData 
{
public:
  
  // GROUP: Constructors:
  //////////
  // Constructor
  BrickData( int nx, int ny, int nz, int nbytes );
  
  BrickData();
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~BrickData();
  
  // GROUP: Access and Info
  //////////
  void setSize(int nx, int ny, int nz) { nx_ = nx; ny_ = ny; nz_ = nz;}
  void setBytes( int nbytes ){ nbytes_ = nbytes; }
  
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }
  int nbytes() const { return nbytes_; }

protected:
  int nx_, ny_, nz_;
  int nbytes_;

};


} // End namespace Volume

#endif
 
