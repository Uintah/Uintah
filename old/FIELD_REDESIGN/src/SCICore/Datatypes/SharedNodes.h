//  SharedNodes.h - A collection of nodes, potentially shared by other collections 
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Nodes.h>


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
