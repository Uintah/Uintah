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

//  Nodes.h - A collection of nodes 
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
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

