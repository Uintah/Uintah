//  TetMeshGeom.cc - A group of Tets in 3 space
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/TetMeshGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID TetMeshGeom::type_id("TetMeshGeom", "Datatype", 0);

DebugStream TetMeshGeom::dbg("TetMeshGeom", true);

TetMeshGeom::TetMeshGeom(const vector<NodeSimp>& inodes, const vector<TetSimp>& itets):
  has_neighbors(0)
{
  d_node = inodes;
  tets = itets;
}

TetMeshGeom::~TetMeshGeom(){
}

void TetMeshGeom::set_tets(const vector<TetSimp>& itets){
  tets.clear();
  tets = itets;
}


void TetMeshGeom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
