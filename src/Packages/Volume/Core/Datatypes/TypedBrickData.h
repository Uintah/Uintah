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

#ifndef VOLUME_TYPED_BRICK_DATA_H
#define VOLUME_TYPED_BRICK_DATA_H


#include <Volume/Core/Datatypes/BrickData.h>

namespace Volume {



/**************************************

CLASS
   TypedBrickData
   
   Base class for TypedBrickData for 3D Texturing 

GENERAL INFORMATION

   TypedBrickData.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TypedBrickData

DESCRIPTION
   TypedBrickData class for 3D Texturing.  Stores the texture associated with
   the TypedBrickData and the bricks location is space.  For a given view ray,
   min and max ray parameter, and parameter delta, it will create an
   ordered (back to fron) polygon list that can be rendered by a
   volume renderer.

  
WARNING
  
****************************************/
template <class T>
class TypedBrickData : public BrickData
{
public:
  // GROUP: Constructors:
  //////////
  // Constructor
  TypedBrickData(int nx, int ny, int nz, int nc, int* nb);

  TypedBrickData();
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~TypedBrickData();

  // GROUP: Access and Info
  ////////// texture is used for loading a texture...
  T* texture(int i) { return &(data_[i][0][0][0]); }
  T*** data(int i) { return data_[i]; }

private:
  T**** data_;
};

template <class T>
TypedBrickData<T>::TypedBrickData(int nx, int ny, int nz, int nc, int* nb) :
  BrickData(nx, ny, nz, nc, nb), data_(0)
{
  data_ = new T***[nc_];
  for (int c=0; c<nc_; c++) {
    data_[c] = new T**[nz_];
    T **p = new T*[nz_ * ny_];
    T *pp = new T[nz_ * ny_ * nx_ * nb_[c]];
    for(int k = 0; k < nz_ * ny_ * nx_ * nb_[c]; k++) pp[k] = 0;
    for(int k = 0; k < nz_; k++) {
      data_[c][k] = p;
      p += ny_;
      for(int j = 0; j < ny_; j++) {
        data_[c][k][j] = pp;
        pp += nx_ * nb_[c];
      }
    }
  }
}

template <class T>
TypedBrickData<T>::TypedBrickData() : data_(0)
{
}
 
template <class T>
TypedBrickData<T>::~TypedBrickData()
{
  if (data_) {
    for (int c=0; c<nc_; c++) {
      delete [] data_[c][0][0];
      delete [] data_[c][0];
      delete [] data_[c];
    }
    delete [] data_;
  }
}

} // End namespace Volume

#endif
