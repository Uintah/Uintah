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

//  SharedNodes.h - A collection of nodes, potentially shared by other collections 
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/Nodes.h>


class SharedNodes:public Nodes
{
public:
  SharedNodes();
  SharedNodes(const SharedNodes&);
  SharedNodes(GeomHandle, vector<int>, const string); 
  ~SharedNodes();

  void add(vector<int>);
  // ...

private:
  Bbox& getBox();
  void setBox(const Bbox&);
  Point &p(int index) {return nodes->p(idx[index])};
  
  // translates local index to master index held in nodes
  vector<int> idx;

  // the node group that holds the nodes this object points to
  GeomHandle nodes;

}
