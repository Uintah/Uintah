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

/*
 *  GLTexture3DBuilder:
 *  Written by:
 *   Kurt Zimmerman
 *   SCI Institute
 *   University of Utah
 *   Jan 2004
 *
 *  Copyright (C) 2003 SCI Group
 */


#ifndef NrrdTextureBuilderAlgo_h
#define NrrdTextureBuilderAlgo_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Containers/BinaryTree.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

#include <Core/Util/DynamicLoader.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Core/Datatypes/TypedBrickData.h>
#include <Packages/Volume/Core/Datatypes/BrickWindow.h>

#include <Packages/Teem/Core/Datatypes/NrrdData.h>

namespace Volume {

using std::ostringstream;

using SCIRun::Transform;
using SCIRun::BBox;
using SCITeem::NrrdData;
using SCITeem::NrrdDataHandle;
class BrickNode;

class NrrdTextureBuilderAlgo {
public:
  NrrdTextureBuilderAlgo();
  ~NrrdTextureBuilderAlgo();

  Texture* build(NrrdDataHandle, NrrdDataHandle, int);

protected:
  int ni_, nj_, nk_;
  int nc_, nb_[2];
  int vi_, vj_, vk_;
  Transform transform_;
  BBox bbox_;

  void computeDivisions(int nx, int ny, int nz, int nb,
                        int& max_tex, int& sx, int& sy, int& sz);
  void buildBricks(BinaryTree<BrickNode *>*& tree,
                   NrrdDataHandle nrrd, NrrdDataHandle nrrd,
		   int max_tex, int sx, int sy, int sz, int nc, int* nb);
  BinaryTree<BrickNode*>* buildTree(int& mi, int& mj, int& mk,
                                    int& ni, int& nj, int& nk, int nc, int* nb,
                                    BBox bbox,
                                    int& mvi, int &mvj, int& mvk,
                                    int& vi, int& vj, int& vk, BBox vbox,
                                    const double& di, const double& dj,
                                    const double& dk, const int& max_tex, 
                                    int axis, int& index);
  void fillTree(BinaryTree<BrickNode*>* tree,
                NrrdDataHandle vfield,
                NrrdDataHandle gfield);
  void filldata(BrickData* bdata,
                BrickWindow* bw,
                NrrdDataHandle vfield,
                NrrdDataHandle gfield);
};

} // end namespace Volume

#endif // NrrdTextureBuilderAlgo_h
