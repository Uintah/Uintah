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
#include <Core/Volume/Texture.h>
#include <Core/Volume/TextureBrick.h>
#include <Core/Datatypes/NrrdData.h>

namespace SCIRun {

class NrrdTextureBuilderAlgo
{
public:
  NrrdTextureBuilderAlgo();
  ~NrrdTextureBuilderAlgo();

  bool build(ProgressReporter *report,
	     TextureHandle texture,
             const NrrdDataHandle &vnrrd,
	     const NrrdDataHandle &gnrrd,
             int card_mem);

protected:

  void fill_brick(TextureBrickHandle &brick, const NrrdDataHandle &v_nrrd,
		  const NrrdDataHandle &gm_nrrd, int ni, int nj, int nk);
};

} // namespace SCIRun

#endif // Volume_NrrdTextureBuilderAlgo_h
