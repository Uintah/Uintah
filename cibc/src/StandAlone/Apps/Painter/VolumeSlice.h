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
#include <map>
#include <vector>


namespace SCIRun {

class Painter;
class NrrdVolume;
class SliceWindow;

struct VolumeSlice {
  VolumeSlice(Painter *, NrrdVolume *, SliceWindow * window,
              Point &p, Vector &normal);
  
  void                  bind();
  void                  draw();
  unsigned int          axis();
  void                  set_tex_dirty();
  void                  set_coords();

  Painter *             painter_;
  NrrdVolume *          volume_;
  SliceWindow *         window_;
  
  bool                  nrrd_dirty_;
  bool                  tex_dirty_;
  
  Point                 pos_;
  Vector                xdir_;
  Vector                ydir_;
  Plane                 plane_;
  
  ColorMappedNrrdTextureObjHandle     texture_;  
};

typedef std::vector<VolumeSlice *>              VolumeSlices;
typedef std::vector<VolumeSlices>		NrrdVolumeSlices;
typedef std::map<NrrdVolume *, VolumeSlice*>    VolumeSliceMap;

}

#endif
  
