//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
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
//    File   : NrrdTextureBuilderAlgo.h
//    Author : Milan Ikits
//    Date   : Fri Jul 16 02:48:14 2004

#ifndef Volume_NrrdTextureBuilderAlgo_h
#define Volume_NrrdTextureBuilderAlgo_h

#include <vector>
#include <Core/Datatypes/Datatype.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Core/Datatypes/NrrdData.h>

using SCIRun::BBox;

namespace Volume {

class NrrdTextureBuilderAlgo
{
public:
  NrrdTextureBuilderAlgo();
  ~NrrdTextureBuilderAlgo();

  void build(TextureHandle texture,
             SCIRun::NrrdDataHandle vnrrd, SCIRun::NrrdDataHandle gnrrd,
             int card_mem);

protected:
  void build_bricks(std::vector<Brick*>& bricks, int nx, int ny, int nz,
                    int nc, int* nb, const BBox& bbox, int brick_mem);
  void fill_brick(Brick* brick, Nrrd* nv_nrrd, Nrrd* gm_nrrd,
                  int ni, int nj, int nk);
};

} // namespace Volume

#endif // Volume_NrrdTextureBuilderAlgo_h
