#ifndef SCIRun_GatherPoints_H
#define SCIRun_GatherPoints_H
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

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/PointCloudMesh.h>

namespace SCIRun {

class GatherPointsAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle src, PointCloudMeshHandle pcmH) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class GatherPointsAlgoT : public GatherPointsAlgo
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, PointCloudMeshHandle pcmH);
};

template <class MESH>
void 
GatherPointsAlgoT<MESH>::execute(MeshHandle mesh_h, PointCloudMeshHandle pcmH)
{
  typedef typename MESH::Node::iterator node_iter_type;

  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  node_iter_type ni; mesh->begin(ni);
  node_iter_type nie; mesh->end(nie);
  while (ni != nie)
  {
    Point p;
    mesh->get_center(p, *ni);
    pcmH->add_node(p);
    ++ni;
  }
}

} // End namespace SCIRun

#endif
