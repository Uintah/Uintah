//  Nodes.h - A collection of nodes 
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


class Nodes:public Datatype
{
public:
  Nodes();
  Nodes(vector<GNode>, const string) 
  Nodes(const Nodes&);

  ~Nodes();

  void add(const GNode&);
  // ...
private:
  vector<GNode> nodes;

  virtual Bbox &getBbox();
  virtual void setBbox(const Bbox&);
  // returns the point that corresponds to index 
  virtual Point &p(int index);
  
}

