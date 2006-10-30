//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : VolumeSlice.h
//    Author : McKay Davis
//    Date   : Fri Oct 13 15:50:22 2006

#ifndef StandAlone_Apps_LEXOV_VolumeSlice_h
#define StandAlone_Apps_LEXOV_VolumeSlice_h

#include <Core/Geometry/Plane.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/NrrdBitmaskOutline.h>
#include <map>
#include <vector>


namespace SCIRun {

class Painter;
class NrrdVolume;
class SliceWindow;

struct VolumeSlice {
  // For LockingHandle
  Mutex                 lock;
  int                   ref_cnt;

  VolumeSlice(NrrdVolume *, 
              const Plane &,
              NrrdDataHandle nrrd=0);
  
  void                  bind();
  void                  draw();
  unsigned int          axis();
  void                  set_tex_dirty();
  const Plane &         get_plane() { return plane_; }

  NrrdVolume *          volume_;
  NrrdDataHandle        nrrd_handle_;
  
  bool                  tex_dirty_;
  
  Point                 pos_;
  Vector                xdir_;
  Vector                ydir_;
  
  NrrdBitmaskOutlineHandle              outline_;
  ColorMappedNrrdTextureObjHandle       texture_;
  GeomHandle                            geom_texture_;

private:
  void                  extract_nrrd_slice_from_volume();
  Plane                 plane_;



};

typedef LockingHandle<VolumeSlice> VolumeSliceHandle;
typedef std::vector<VolumeSliceHandle> VolumeSlices_t;
typedef map<string, VolumeSlices_t> VolumeSliceGroups_t;    
typedef map<int, VolumeSlices_t> VolumeSlicesAlongAxis_t;
typedef map<int, VolumeSlicesAlongAxis_t> VolumeSliceCache_t;


}

#endif
  
