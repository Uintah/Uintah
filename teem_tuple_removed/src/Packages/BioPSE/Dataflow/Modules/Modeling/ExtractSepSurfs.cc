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
 *  ExtractSepSurfs.cc:  Extract the Separating Surfaces from a Segmented Field
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/BioPSE/Core/Datatypes/SegLatVolField.h>
#include <Packages/BioPSE/Core/Datatypes/SepSurf.h>
#include <Core/GuiInterface/GuiVar.h>
#include <map>
#include <queue>
#include <iostream>

using std::queue;

namespace BioPSE {

using namespace SCIRun;

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned, int> simplex_hash_type;
#else
  typedef map<unsigned, int> simplex_hash_type;
#endif

class ExtractSepSurfs : public Module {
private:
#if 0
  int bldPMask(Array1<int>& cells, Array1<int>& thin);
  void bldSplits(SegLatVolField* ch,
		 Array1<LatVolMesh::Node::index_type>& splits,
		 Array1<int>& splT, Array1<int>& thin);
#endif
  int getNode(int ii, int jj, int kk, int n,
	      SegLatVolField& field, const Array1<int>& splT, 
	      const Array1<int>& splI, simplex_hash_type &hash,
	      SepSurf& surf);
#if 0
  int getEdge(int ii, int jj, int kk, int o, 
	      SepSurf& surf, int p1, int p2,
	      simplex_hash_type &ehash);
#endif
//  void setFaces(int e1, int e2, int e4, int e5, int e3, 
//		int p3, int p1, int p2, int p4, 
//		SurfTree& surf, int comp, int bcomp);
  void setFaces(int p3, int p1, int p2, int p4, 
		SepSurf& surf, int comp, int bcomp);
  void buildSurfs(SegLatVolField &field, SepSurf &surf);
public:
  ExtractSepSurfs(GuiContext *ctx);
  virtual ~ExtractSepSurfs();
  virtual void execute();
};

DECLARE_MAKER(ExtractSepSurfs)

ExtractSepSurfs::ExtractSepSurfs(GuiContext *ctx)
  : Module("ExtractSepSurfs", ctx, Filter, "Modeling", "BioPSE")
{
}

ExtractSepSurfs::~ExtractSepSurfs()
{
}


#if 0
// for a cube of 8 cells, determine the dominant type and see if it is
// split into two separated groups.
// if so, determine the list of faces that bound each group.
// for each face in each group, determine which p index touches the
// center node of the 8 cells, and add it to the list for that group

int ExtractSepSurfs::bldPMask(Array1<int>& cells, Array1<int>& thin) {
  int pMask=0;
    
  // figure out the dominant type of the cells
  Array1<int> valTypes;
  Array1<int> valCnt;
  int i,j;
  for (i=0; i<8; i++) {
    for (j=0; j<valTypes.size(); j++)
      if (cells[i] == valTypes[j]) {valCnt[j]=valCnt[j]+1; break;}
    if (j==valTypes.size()) {valTypes.add(cells[i]); valCnt.add(1);}
  }

  // if there aren't 2 types, return 0
  if (valTypes.size()<2) return 0;

  int dType=valTypes[0];
  int dNum=valCnt[0];
  for (i=1; i<valTypes.size(); i++) 
    if (valCnt[i]>dNum) {
      dNum=valCnt[i];
      dType=valTypes[i];
    }

  // if dominant type has less than 4 cells or more than 6 cells, return 0
  if (dNum<4 || dNum>6) return 0;

  if (dNum==4 && valTypes.size()==2)
    if (thin[valTypes[0]] > thin[valTypes[1]]) dType=valTypes[0];
    else dType=valTypes[1];

  int dMask=0;
  for (i=0; i<8; i++)
    if (cells[i] == dType) dMask |= (1<<i);

  //    cerr << "dMask="<<dMask<<"\n";
  //    cerr << "dType="<<dType<<"\n";

  // build mask of dominant type
  if ((dMask & 0xc3) == 0xc3) { // 11000011
    if (cells[2]!=dType) pMask |= (1<<3 | 1<<1); // 0/2 2/(3,6)
    if (cells[3]!=dType) pMask |= (1<<5 | 1<<1 | 1<<7); // 1/3 2/3 3/7
    return pMask;
  }
  if ((dMask & 0x3c) == 0x3c) { // 00111100
    if (cells[0]!=dType) pMask |= (1<<3); // 0/(1,2,4)
    if (cells[1]!=dType) pMask |= (1<<3 | 1<<5); // 0/1 1/(3,5)
    return pMask;
  }
  if ((dMask & 0xa5) == 0xa5) { // 10100101
    if (cells[1]!=dType) pMask |= (1<<3 | 1<<5); // 0/1 1/(3,5)
    if (cells[3]!=dType) pMask |= (1<<5 | 1<<1 | 1<<7); // 1/3  2/3  3/7
    return pMask;
  }
  if ((dMask & 0x5a) == 0x5a) { // 01011010
    if (cells[0]!=dType) pMask |= (1<<3); // 0/(1,2,4)
    if (cells[2]!=dType) pMask |= (1<<3 | 1<<1); // 0/2 2/(3,6)
    return pMask;
  }
  if ((dMask & 0x99) == 0x99) { // 10011001
    if (cells[1]!=dType) pMask |= (1<<3 | 1<<5); // 0/1 1/(3,5)
    if (cells[5]!=dType) pMask |= (1<<5 | 1<<4 | 1<<6); // 1/5 4/5 5/7
    return pMask;
  }
  if ((dMask & 0x66) == 0x66) { // 01100110
    if (cells[0]!=dType) pMask |= (1<<3); // 0/(1,2,4)
    if (cells[4]!=dType) pMask |= (1<<3 | 1<<4); // 0/4 4/(5,6)
    return pMask;
  }
  return 0;
}

void ExtractSepSurfs::bldSplits(SegLatVolField* ch,
				Array1<LatVolMesh::Node::index_type>& splits,
				Array1<int>& splT, Array1<int>& thin) {
  int i,j,k;
  Array1<int> cells(8);

  for (i=1; i<ch->fdata().dim1(); i++)
    for (j=1; j<ch->fdata().dim2(); j++)
      for (k=1; k<ch->fdata().dim3(); k++) {
	cells[0]=ch->fdata()(i,j,k);
	cells[1]=ch->fdata()(i-1,j,k);
	cells[2]=ch->fdata()(i,j-1,k);
	cells[3]=ch->fdata()(i-1,j-1,k);
	cells[4]=ch->fdata()(i,j,k-1);
	cells[5]=ch->fdata()i-1,j,k-1);
	cells[6]=ch->fdata()(i,j-1,k-1);
	cells[7]=ch->fdata()(i-1,j-1,k-1);
	int pMask=bldPMask(cells, thin);
	if (pMask) {
	  splits.add(LatVolMesh::Node::index_type(i,j,k));
	  splT.add(pMask);
	}
      }
//  cerr << "Here are the splits:\n";
//  for (i=0; i<splits.size(); i++)
//    cerr << "   node = ("<<splits[i].i_ <<", "<<splits[i].j_<<", "<<splits[i].k_<<"), type = "<<(int)splT[i]<<"\n";
//  cerr << "DONE!\n";
}
#endif

int ExtractSepSurfs::getNode(int ii, int jj, int kk, int n,
			     SegLatVolField& field, const Array1<int>& splT, 
			     const Array1<int>& splI, simplex_hash_type& hash,
			     SepSurf& surf) {
  int h;
  LatVolMesh::Node::index_type nidx(field.get_typed_mesh().get_rep(),ii,jj,kk);
  
//  cerr << "Getting node ("<<ii<<","<<jj<<","<<kk<<") n="<<n;
  unsigned pidx = (unsigned)(nidx);
  simplex_hash_type::iterator iter = hash.find(pidx);
  if (iter == hash.end()) {
    hash[pidx] = surf.nodes.size();
    h = surf.nodes.size();
    Point p;
    field.get_typed_mesh()->get_center(p, nidx);
    surf.nodes.add(surf.get_typed_mesh()->add_point(p));
  } else {
    h = (*iter).second;
    if (h<0) {
      h = -h-1;
      if (splT[h] & (1<<n)) 
	h = splI[h];
      else
	h = splI[h]+1;
    }
  }
//  cerr << " h="<<h<<"\n";
  return h;
}

#if 0
int ExtractSepSurfs::getEdge(int ii, int jj, int kk, int o, 
			     SepSurf& surf, int p1, int p2,
			     simplex_hash_type &ehash) {
  int h;
  unsigned eidx=(ii<<21)+(jj<<12)+(kk<<3)+o;
  simplex_hash_type::iterator iter = ehash.find(eidx);
  if (iter == ehash.end()) {
    ehash[eidx] = surf.edges.size();
    h=surf.edges.size();
    surf.edges.add(new TSEdge(p1, p2));
    surf.edgeI.resize(surf.edgeI.size()+1);
  }
  else {
    h = (*iter).second;
  }
  return h;
}
#endif

//void ExtractSepSurfs::setFaces(int e1, int e2, int e4, int e5, int e3, 
//			       int p3, int p1, int p2, int p4, 
//			       SepSurf& surf, int comp, int bcomp) {
void ExtractSepSurfs::setFaces(int p3, int p1, int p2, int p4, 
			       SepSurf& surf, int comp, int bcomp) {

  int iii=surf.faces.size();
//  Array1<int> edges(3);
//  Array1<int> orient(3);
//  surf.edgeI[e2].faces.add(iii);
//  surf.edgeI[e3].faces.add(iii);
//  surf.edgeI[e1].faces.add(iii);
//  surf.faces.add(new TSElement(p3, p1, p2));
//  edges[0]=e2; edges[1]=e3; edges[2]=e1;
//  orient[0]=1; orient[1]=0; orient[2]=1;
//  surf.faceI.resize(surf.faceI.size()+2);
//  surf.faceI[iii].edges = edges; 
//  surf.faceI[iii].edgeOrient = orient;
//  surf.edgeI[e5].faces.add(iii+1);
//  surf.edgeI[e3].faces.add(iii+1);
//  surf.edgeI[e4].faces.add(iii+1);
//  surf.faces.add(new TSElement(p2, p4, p3));
//  edges[0]=e5; edges[1]=e3; edges[2]=e4;
//  orient[0]=0; orient[1]=1; orient[2]=0;
//  surf.faceI[iii+1].edges = edges; 
//  surf.faceI[iii+1].edgeOrient = orient;

  surf.faces.add(surf.get_typed_mesh()->add_quad(p1, p2, p4, p3));
  surf.faceI.resize(surf.faceI.size()+1);

  surf.faceI[iii].surfIdx.add(bcomp);
  surf.faceI[iii].surfOrient.add(1);
 // surf.faceI[iii+1].surfIdx.add(bcomp);
 // surf.faceI[iii+1].surfOrient.add(1);
    
  surf.faceI[iii].surfIdx.add(comp);
  surf.faceI[iii].surfOrient.add(0);
 // surf.faceI[iii+1].surfIdx.add(comp);
 // surf.faceI[iii+1].surfOrient.add(0);		    
    
  surf.surfI[bcomp].faces.add(iii);
 // surf.surfI[bcomp].faces.add(iii+1);
  surf.surfI[bcomp].faceOrient.add(1);
 // surf.surfI[bcomp].faceOrient.add(1);
  surf.surfI[comp].faces.add(iii);
 // surf.surfI[comp].faces.add(iii+1);
  surf.surfI[comp].faceOrient.add(0);
 // surf.surfI[comp].faceOrient.add(0);

}

// each component will correspond to one surface -- the outer surface
// which bounds it.
// starting from air(0), flood fill to find all components which touch air.
// as each component is found, we mark it as visited, and push the bounding
// voxel location and direction onto 
// from all of those 
void ExtractSepSurfs::buildSurfs(SegLatVolField &field, SepSurf &surf) {
  Array1<Array1<int> > touches;
  touches.resize(field.ncomps());
  Array1<int> is(3), js(3), ks(3);
  is[0]=0; is[1]=1; is[2]=1;
  int i,j,k,ii,jj,kk;

  // find out which components "touch" each other
  for (i=1; i<field.fdata().dim1(); i++, is[0]++, is[1]++, is[2]++) {
    js[0]=1; js[1]=0; js[2]=1;
    for (j=1; j<field.fdata().dim2(); j++, js[0]++, js[1]++, js[2]++) {
      ks[0]=1; ks[1]=1; ks[2]=0;
      for (k=1; k<field.fdata().dim3(); k++, ks[0]++, ks[1]++, ks[2]++) {
	int comp=field.fdata()(i,j,k);
	for (int nbr=0; nbr<3; nbr++) {
	  ii=is[nbr]; jj=js[nbr]; kk=ks[nbr];
	  int bcomp=field.fdata()(ii,jj,kk);
	  if (bcomp != comp) {
	    int found=0;
	    for (int f=0; f<touches[comp].size() && !found; f++)
	      if (touches[comp][f] == bcomp) found=1;
	    if (!found) {
	      touches[comp].add(bcomp);
	      touches[bcomp].add(comp);
	    }
	  }
	}
      }	
    }
	
  }

#if 0
  cerr << "\n";
  for (i=0; i<touches.size(); i++) {
    if (touches[i].size() != 0) {
      cerr << "Component "<<i<<" touches: ";
      for (int j=0; j<touches[i].size(); j++)
	cerr << touches[i][j]<<" ";
      cerr << "\n";
    }
  }
#endif

  Array1<int> visited(field.ncomps());
  Array1<int> queued(field.ncomps());
  surf.surfI.resize(field.ncomps());
  for (i=0; i<field.ncomps(); i++) surf.surfI[i].outer=-1;
  visited.initialize(0);
  queued.initialize(0);

  for (i=0; i<touches.size(); i++) if (touches[i].size() == 0) visited[i]=1;
  queue<pair<int,int> > q;
  queue<pair<int,int> > kids;

  int air=field.fdata()(0,0,0);
  // visit 0 -- push all of its nbrs
  visited[air]=1;
  queued[air]=1;


  // we'll enqueue each component index, along with which
  // component is outside of it.

  for (i=0; i<touches[air].size(); i++) {
    q.push(pair<int,int>(air,touches[air][i]));;
    visited[touches[air][i]]=queued[touches[air][i]]=1;
  }

  // go in one "level" at a time -- push everyone you touch into the
  // "kids" queue.  when there isn't anyone else, move everyone from
  // that queue into the main one.  continue till there's no one left.

  while (!q.empty()) {
    while (!q.empty()) {
      pair<int,int> qitem = q.front();
      q.pop();
      int outer=qitem.first;
      int inner=qitem.second;
      //	    cerr << "outer="<<outer<<"   inner="<<inner<<"\n";
      surf.surfI[inner].outer=outer;
      for (i=0; i<touches[inner].size(); i++) {
	// if another component it touches has been visited
	int nbr=touches[inner][i];
	if (!visited[nbr]) {
	  if (queued[nbr]) {	// noone should have queued -- must be
	    // crack-connected
	    //			cerr << "why am i here??\n";
	    q.push(pair<int,int>(outer,nbr));
	    visited[nbr]=1;
	  } else {
	    queued[nbr]=1;
	    kids.push(pair<int,int>(inner,nbr));
	  }
	}
      }
    }
    while (!kids.empty()) {
      pair<int,int> qitem=kids.front();
      kids.pop();
      int outer=qitem.first;
      int inner=qitem.second;
      visited[inner]=1;
      q.push(pair<int,int>(outer,inner));
    }
  }
  // make sure everything's cool

#if 0
  cerr << "Outer boundaries...\n";
  for (i=0; i<surf.surfI.size(); i++) {
    if (surf.surfI[i].outer != -1) {
      surf.surfI[surf.surfI[i].outer].inner.add(i);
  	    cerr << "  Surface "<<i<<" is bounded by surface "<<surf.surfI[i].outer<<"\n";
    }
  }

  cerr << "Inner boundaries...\n";
  for (i=0; i<surf.surfI.size(); i++) {
    if (surf.surfI[i].inner.size() != 0) {
      cerr << "  Surface "<<i<<" bounds surfaces: ";
      for (int j=0; j<surf.surfI[i].inner.size(); j++)
	cerr << surf.surfI[i].inner[j] << " ";
      cerr << "\n";
    }
  }
#endif

  int p1, p2, p3, p4, p5, p6, p7, bcomp;
//  int e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12;

  simplex_hash_type hash;
//  simplex_hash_type ehash;
//  BBox bbox = field.get_typed_mesh()->get_bounding_box();
//  Vector v((bbox.max()-bbox.min())*.5);
//  v.x(v.x()/(field.fdata().dim1()-1));
//  v.y(v.y()/(field.fdata().dim2()-1));
//  v.z(v.z()/(field.fdata().dim3()-1));

  Array1<LatVolMesh::Node::index_type> splits;
  Array1<int> splT;
  Array1<int> splI;


#if 0
  // DAVE - figure out why this makes holes in the vein datasets (sl2.sr)
  //    bldSplits(field.getTypeFld(), splits, splT, field.thin);

  // first add in all of the nodes that need to be split
  for (i=0; i<splits.size(); i++) {
    LatVolMesh::Node::index_type idx=splits[i];
    int ii=idx.x-1;
    int jj=idx.y-1;
    int kk=idx.z-1;
    int pidx=(ii<<20)+(jj<<10)+kk;
    if (hash->find(pidx) == hash->end()) {
      (*hash)[pidx] = -(i+1);
    } 
    // else cerr << "ExtractSepSurfs - why would I be here??\n";
    splI.add(surf.nodes.size());
    surf.nodes.add(field.get_point(ii,jj,kk)+v);
    surf.nodes.add(field.get_point(ii,jj,kk)+v);
  }
#endif

  // for each cell, look at the negative neighbors (down, back, left)
  // if they're of a different material, build the triangles
  // for each triangle, hash the nodes -- if they don't exist, add em
  // we also need to build the edges -- we'll hash these too.

  for (i=1; i<field.fdata().dim1(); i++)
    for (j=1; j<field.fdata().dim2(); j++)
      for (k=1; k<field.fdata().dim3(); k++) {
	int comp=field.fdata()(i,j,k);
	bcomp=field.fdata()(i-1,j,k);
	if (bcomp != comp) {
	  p3=getNode(i-1,j-1,k-1,3,field,splT,splI,hash,surf);
	  p1=getNode(i-1,j  ,k-1,1,field,splT,splI,hash,surf);
	  p2=getNode(i-1,j  ,k  ,2,field,splT,splI,hash,surf);
	  p4=getNode(i-1,j-1,k  ,4,field,splT,splI,hash,surf);

//	  e1=getEdge(i-1,j-1,k-1,0,surf,p3,p1,ehash);
//	  e3=getEdge(i-1,j-1,k-1,3,surf,p3,p2,ehash);
//	  e5=getEdge(i-1,j-1,k-1,1,surf,p3,p4,ehash);
//	  e2=getEdge(i-1,j,  k-1,1,surf,p1,p2,ehash);
//	  e4=getEdge(i-1,j-1,k  ,0,surf,p4,p2,ehash);

//	  setFaces(e1,e2,e4,e5,e3,p3,p1,p2,p4,surf,comp,bcomp);
	  setFaces(p3,p1,p2,p4,surf,comp,bcomp);
	}
	bcomp=field.fdata()(i,j-1,k);
	if (bcomp != comp) {
	  p3=getNode(i-1,j-1,k-1,3,field,splT,splI,hash,surf);
	  p5=getNode(i  ,j-1,k-1,5,field,splT,splI,hash,surf);
	  p6=getNode(i  ,j-1,k  ,6,field,splT,splI,hash,surf);
	  p4=getNode(i-1,j-1,k  ,4,field,splT,splI,hash,surf);

//	  e5=getEdge(i-1,j-1,k-1,1,surf,p3,p4,ehash);
//	  e7=getEdge(i-1,j-1,k-1,4,surf,p3,p6,ehash);
//	  e6=getEdge(i-1,j-1,k-1,2,surf,p3,p5,ehash);
//	  e9=getEdge(i-1,j-1,k  ,2,surf,p4,p6,ehash);
//	  e8=getEdge(i  ,j-1,k-1,1,surf,p5,p6,ehash);

//	  setFaces(e5,e9,e8,e6,e7,p3,p4,p6,p5,surf,comp,bcomp);
	  setFaces(p3,p4,p6,p5,surf,comp,bcomp);
	}
	bcomp=field.fdata()(i,j,k-1);
	if (bcomp != comp) {
	  p3=getNode(i-1,j-1,k-1,3,field,splT,splI,hash,surf);
	  p5=getNode(i  ,j-1,k-1,5,field,splT,splI,hash,surf);
	  p7=getNode(i  ,j  ,k-1,7,field,splT,splI,hash,surf);
	  p1=getNode(i-1,j  ,k-1,1,field,splT,splI,hash,surf);

//	  e1 =getEdge(i-1,j-1,k-1,0,surf,p3,p1,ehash);
//	  e11=getEdge(i-1,j-1,k-1,5,surf,p3,p7,ehash);
//	  e6 =getEdge(i-1,j-1,k-1,2,surf,p3,p5,ehash);
//	  e12=getEdge(i-1,j  ,k-1,2,surf,p1,p7,ehash);
//	  e10=getEdge(i  ,j-1,k-1,0,surf,p5,p7,ehash);

//	  setFaces(e6,e10,e12,e1,e11,p3,p5,p7,p1,surf,comp,bcomp);
	  setFaces(p3,p5,p7,p1,surf,comp,bcomp);
	}
      }

  for (i=0; i<field.ncomps(); i++) {
    surf.surfI[i].matl=field.compMatl(i);
    surf.surfI[i].size=field.compSize(i);
  }

#if 0
  int bigGreyIdx=-1;
  int bigGreySize=-1;
  int bigWhiteIdx=-1;
  int bigWhiteSize=-1;
  for (i=0; i<field.comps.size(); i++) {
    if (field.comps[i]) {
      int thisType=field.get_type(field.comps[i]);
      int thisSize=field.get_size(field.comps[i]);
      if (thisType==4 && thisSize>bigGreySize) {
	bigGreyIdx=i;
	bigGreySize=thisSize;
      } else if (thisType==5 && thisSize>bigWhiteSize) {
	bigWhiteIdx=i;
	bigWhiteSize=thisSize;
      }
    }
    else surf.surfI[i].matl=-1;
    cerr << "surf.matl["<<i<<"]="<<surf.surfI[i].matl<<"\n";
  }
  if (surf.surfI.size()>1 && surf.surfI[1].outer == 0)
    surf.surfI[1].name="scalpAll";
  surf.surfI[0].name="scalp";
  if (bigGreyIdx != -1) {
    surf.surfI[bigGreyIdx].name="cortex";
    cerr << "**** Biggest grey matter (material 4) is region "<<bigGreyIdx<<"\n";

    if (surf.surfI[bigGreyIdx].inner.size()) {
      cerr << "**** WARNING: this region contains inner regions!\n";
    }
  } else {
    cerr << "No grey matter (material 4) found.\n";
  }
  if (bigWhiteIdx != -1) {
    cerr << "**** Biggest white matter (material 5) is region "<<bigWhiteIdx<<"\n";
    if (surf.surfI[bigWhiteIdx].inner.size()) {
      cerr << "**** WARNING: this region contains inner regions!\n";
    }
  } else {
    cerr << "No white matter (material 5) found.\n";
  }

#endif
  surf.bldNodeInfo();
}

void ExtractSepSurfs::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("SegField");
  if (!ifp) {
    error("Unable to initialize port 'SegFld'.");
    return;
  }
  FieldOPort *ofp = (FieldOPort *)get_oport("SepSurf");
  if (!ofp) {
    error("Unable to initialize port 'SepSurf'.");
    return;
  }

  FieldHandle ifieldH;
  if (!ifp->get(ifieldH) || !ifieldH.get_rep()) {
    error("No input data");
    return;
  }
  SegLatVolField *slvf = dynamic_cast<SegLatVolField *>(ifieldH.get_rep());
  if (!slvf) {
    error("Input field was not a SegLatVolField");
    return;
  }

  QuadSurfMeshHandle qsmH = new QuadSurfMesh;
  SepSurf *surf = new SepSurf(qsmH);
  buildSurfs(*slvf, *surf);
  FieldHandle fH(surf);
  ofp->send(fH);
}

} // End namespace BioPSE
