
/*
 *  TopoSurfTree.cc: Tree of non-manifold bounding surfaces
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1997
  *
 *  Copyright (C) 1997 SCI Group
 */

#include <iostream>
using std::cerr;

#include <queue>
using std::queue;

#include <Core/Util/Assert.h>
#include <Core/Containers/Array2.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Packages/DaveW/Core/Datatypes/General/TopoSurfTree.h>
#include <Core/Malloc/Allocator.h>
#include <math.h>

namespace DaveW {
using SCIRun::Pio;

static Persistent* make_TopoSurfTree()
{
  return scinew TopoSurfTree;
}

PersistentTypeID TopoSurfTree::type_id("TopoSurfTree", "Surface", make_TopoSurfTree);

TopoSurfTree::TopoSurfTree(Representation r)
  : SurfTree(r)
{
}

TopoSurfTree::TopoSurfTree(const TopoSurfTree& copy, Representation)
  : SurfTree(copy)
{
  NOT_FINISHED("TopoSurfTree::TopoSurfTree");
}

TopoSurfTree::~TopoSurfTree() {
}	

Surface* TopoSurfTree::clone()
{
  return scinew TopoSurfTree(*this);
}

void TopoSurfTree::BldTopoInfo() {
  TopoEntity unused;
  unused.type=NONE;
  unused.idx=-1;
  topoNodes.resize(nodes.size());
  topoNodes.initialize(unused);
  topoEdges.resize(edges.size());
  topoEdges.initialize(unused);
  topoFaces.resize(faces.size());
  topoFaces.initialize(unused);
  BldPatches();
  BldWires();
  BldJunctions();
  BldOrientations();
}

void order(Array1<int>& a) {
  int swap=1;
  int tmp;
  while (swap) {
    swap=0;
    for (int i=0; i<a.size()-1; i++)
      if (a[i]>a[i+1]) {
	tmp=a[i];
	a[i]=a[i+1];
	a[i+1]=tmp;
	swap=1;
      }
  }
}

// tmpPatchIdx[patchHash][faceCtr]=faceG
//	patchHash is a hashed value based on the region ID on eiher side
//	faceCtr is a count of all the faces this patch contains (0-n)
//	faceG is the actual global face # 

// tmpPatchIdxOrient[patchHash][surfCtr][faceCtr]=orientB
//	patchHash is a hashed value based on the region ID on eiher side
//	surfCtr is a surface counting idx (0-n).  the global surf # this
//		corresponds to can be found via patchRegions[patchG][surfIdx]
//	faceIdx is a count of all the faces this patch contains (0-n)
//	orientB is a binary orientation flag

// patchRegions[patchG][regionCtr]=surfG
//	patchG is the true patch number
//	regionCtr just indexes the regions the patch separates
//	surfG is the global index of the neighboring region/surface

// allFaceRegions[faceG][surfCtr]=surfG
//	faceG is the true face number
//	surfCtr is a counter of all the surfaces this face touches
//	surfG is the global index of each region/surface this face neighbors

struct SrchLst {
  Array1<int> items;
  Array1<int> lookup;
  Array1<Array1<int> > orient;
  Array1<SrchLst *> next;
  void *operator new(size_t);
  void operator delete(void*, size_t);
};

static TrivialAllocator SrchLst_alloc(sizeof(SrchLst));

void* SrchLst::operator new(size_t)
{
  return SrchLst_alloc.alloc();
}

void SrchLst::operator delete(void* rp, size_t)
{
  SrchLst_alloc.free(rp);
}

void TopoSurfTree::addPatchAndDescend(SrchLst* curr, 
				      Array1<Array1<int> > &tmpPatches, 
				      Array1<Array1<Array1<int> > > &tmpPatchesOrient, 
				      Array1<int> &visList) {
  if (curr->items.size()) {
    int currSize=patchI.size();
    patchI.resize(currSize+1);
    patchI[currSize].bdryEdgeFace.resize(edgeI.size());
    patchI[currSize].bdryEdgeFace.initialize(-1);
    patches.resize(currSize+1);
    tmpPatches.resize(currSize+1);
    tmpPatches[currSize]=curr->items;
    patchI[currSize].facesOrient.resize(curr->orient.size());
    tmpPatchesOrient.resize(currSize+1);
    tmpPatchesOrient[currSize]=curr->orient;
    patchI[currSize].regions=visList;
    for (int j=0; j<curr->items.size(); j++) {
      faceI[curr->items[j]].patchIdx=currSize;
      faceI[curr->items[j]].patchEntry=j;
    }
  }
  for (int j=0; j<curr->next.size(); j++) {
    visList.add(curr->lookup[j]);
    addPatchAndDescend(curr->next[j],tmpPatches,tmpPatchesOrient,visList);
    visList.resize(visList.size()-1);
  }
}

void TopoSurfTree::addWireAndDescend(SrchLst* curr, 
				     Array1<Array1<int> > &tmpWires, 
				     Array1<Array1<Array1<int> > > &tmpWiresOrient, 
				     Array1<int> &visList) {
  if (curr->items.size()) {
    int currSize=wireI.size();
    wireI.resize(currSize+1);
    wires.resize(currSize+1);
    tmpWires.resize(currSize+1);
    tmpWiresOrient.resize(currSize+1);
    tmpWiresOrient[currSize]=curr->orient;
    tmpWires[currSize]=curr->items;
    wireI[currSize].patches=visList;
    for (int j=0; j<curr->items.size(); j++) {
      edgeI[curr->items[j]].wireIdx=currSize;
      edgeI[curr->items[j]].wireEntry=j;
    }
  }
  for (int j=0; j<curr->next.size(); j++) {
    visList.add(curr->lookup[j]);
    addWireAndDescend(curr->next[j], tmpWires, tmpWiresOrient, visList);
    visList.resize(visList.size()-1);
  }
}

void TopoSurfTree::addJunctionAndDescend(SrchLst* curr, Array1<int> &visList) {
  for (int j=0; j<curr->items.size(); j++) {
    int currSize=junctionI.size();
    junctions.resize(currSize+1);
    junctions[currSize]=curr->items[j];
    junctionI.resize(currSize+1);
    junctionI[currSize].wires=visList;
  }
  for (int j=0; j<curr->next.size(); j++) {
    visList.add(curr->lookup[j]);
    addJunctionAndDescend(curr->next[j], visList);
    visList.resize(visList.size()-1);
  }
}

void descendAndFree(SrchLst *curr) {
  for (int i=0; i<curr->next.size(); i++) {
    descendAndFree(curr->next[i]);
  }
  delete(curr);
}

// we'll build the connected components based on the nbrs list
void buildCCs(const Array1<Array1<int> > &nbrs, Array1<Array1<int> > &/*CCs*/)
{
  Array1<int> visited(nbrs.size());
  visited.initialize(0);
}

void TopoSurfTree::BldPatches() {
  int i,j,k,l;

  cerr << "We have "<<faces.size()<<" faces.\n";
  if (!faces.size()) return;

  // make a list of what regions a faces bounds
  Array1<Array1<int> > allFaceRegions(faces.size());
  //    cerr << "Membership.size()="<<allFaceRegions.size()<<"\n";
  for (i=0; i<surfI.size(); i++)
    for (j=0; j<surfI[i].faces.size(); j++)
      allFaceRegions[surfI[i].faces[j]].add(i);

  // sort all of the lists from above, and find the maximum number of
  // surfaces any face belongs to
  for (i=0; i<allFaceRegions.size(); i++)
    if (allFaceRegions[i].size() > 1) {
      order(allFaceRegions[i]);
    }

  // create a recursive array structure that can be descended by
  // indexing all of the regions a face touches (allFaceRegions[])
  // the list of faces for a patch is stored in an items[][] array
  // there is also an array of orientations for a patch, orient[][][], 
  //   indexed by the patch, then the region, and then the face

  SrchLst *tmpPatchIdx = new SrchLst;

  cerr << "allFaceRegions.size()="<<allFaceRegions.size()<<"\n";
  for (i=0; i<allFaceRegions.size(); i++) {
    //	cerr << "Face="<<i<<"\n";
    //	if ((i%1000)==0) cerr << i <<" ";
    //	cerr << "  **** LOOKING IT UP!\n";

    // initially, point to the top level array

    SrchLst *currLst=tmpPatchIdx;

    // for each region, recursively descend

    for (j=0; j<allFaceRegions[i].size(); j++) {

      // reg is the next region in the list for this face

      int reg=allFaceRegions[i][j];

      // search through the lookup array to find which next[] corresponds
      //   to the region we want to descend to.
      // if we get to the end of the list, allocate a new one and add 
      //   reg to the end of the lookup array

      for (k=0; k<currLst->lookup.size(); k++) 
	if (currLst->lookup[k] == reg) break;
      if (k==currLst->lookup.size()) {
	currLst->lookup.add(reg);
	currLst->next.add(new SrchLst);
      }

      // k has the index for region reg - now descend

      currLst=currLst->next[k];
    }

    // currLst now has the right patch (we've decended through all
    //   of the regions) -- now store this face's info in the items[][]
    //   and orient[][][] arrays

    // if the orient[][] array hasn't been allocated yet (first face
    //   of the patch) - allocate it big enough for all of the regions

    if (!currLst->orient.size())
      currLst->orient.resize(j);

    // add this face to the list for the patch

    currLst->items.add(i);

    // for all of the regions this face bounds, find this face in the
    //   list and add its orientation information to orient[][]

    for (j=0; j<allFaceRegions[i].size(); j++) {

      for (k=0; k<faceI[i].surfIdx.size(); k++)
	if (allFaceRegions[i][j] == faceI[i].surfIdx[k]) break;
      currLst->orient[j].add(faceI[i].surfOrient[k]);



#if 0
      //          cerr << "     j="<<j<<"\n";
      for (k=0; k<surfI[allFaceRegions[i][j]].faces.size(); k++) {
	//              cerr << "       k="<<k<<"\n";
	if (surfI[allFaceRegions[i][j]].faces[k] == i) {
	  //                  cerr <<"         idx="<<idx<<" j="<<j<<" k="<<k<<" or="<<surfI[j].faceOrient[k]<<"\n";
	  currLst->orient[j].add(surfI[allFaceRegions[i][j]].faceOrient[k]);
	  break;
	}
      }
#endif


    }
  }

  // these will temporarily hold the faces for each patch.  
  // since a patch should be a single connected component, these 
  // tmpPatches will be split up below - then we'll build new patches
  // as needed and store the tmp data in the correct place

  Array1<Array1<int> > tmpPatches;
  Array1<Array1<Array1<int> > > tmpPatchesOrient;

  cerr << "Allocating and storing temporary patch values... ";

  //    cerr << "\nPatchRegions.size()="<<patchRegions.size()<<"\n";
    
  // recursively descend the tmpPatchIdx srchList -- add all items
  // to the tmpPatches and tmpPatchesOrient lists

  Array1<int> descend;
  addPatchAndDescend(tmpPatchIdx, tmpPatches, tmpPatchesOrient, descend);
  descendAndFree(tmpPatchIdx);

#if 0
  cerr << "patches.size()="<<patches.size()<<"\n";
  for (i=0; i<patchBdryEdgeFace.size(); i++)
    for (j=0; j<patchBdryEdgeFace[i].size(); j++)
      cerr << "patchBdryEdgeFace["<<i<<"]["<<j<<"]="<<patchBdryEdgeFace[i][j]<<"\n";
  for (i=0; i<tmpPatchesOrient.size(); i++)
    for (j=0; j<tmpPatchesOrient[i].size(); j++)
      for (k=0; k<tmpPatchesOrient[i][j].size(); k++)
	cerr << "tmpPatchesOrient["<<i<<"]["<<j<<"]["<<k<<"]="<<tmpPatchesOrient[i][j][k]<<"\n";
  for (i=0; i<tmpPatches.size(); i++)
    for (j=0; j<tmpPatches[i].size(); j++)
      cerr << "tmpPatches["<<i<<"]["<<j<<"]="<<tmpPatches[i][j]<<"\n";
  for (i=0; i<patchRegions.size(); i++)
    for (j=0; j<patchRegions[i].size(); j++)
      cerr << "patchRegions["<<i<<"]["<<j<<"]="<<patchRegions[i][j]<<"\n";
  for (i=0; i<faceI.size(); i++) {
    cerr << "faceI["<<i<<"].patchIdx="<<faceI[i].patchIdx<<"\n";
    cerr << "faceI["<<i<<"].patchEntry="<<faceI[i].patchEntry<<"\n";
  }    
#endif

  cerr << "done.\n";
  // now we have to do connectivity searches on each patch -- make sure
  // it's one connected component.  If it isn't, break it into separate
  // patches.

  cerr << "Doing connectivity search\n";

  int numPatches=patches.size();
  int currPatchIdx;
  for (i=0; i<numPatches; i++) {
    //	cerr << " patch "<<i<<"/"<<numPatches<<"... ";
    currPatchIdx=i;

    // make a list of all the faces attached to each edge of this patch

    Array1<Array1<int> > localEdgeFaces(edges.size());
    for (j=0; j<tmpPatches[i].size(); j++)
      for (k=0; k<faceI[tmpPatches[i][j]].edges.size(); k++)
	localEdgeFaces[faceI[tmpPatches[i][j]].edges[k]].add(tmpPatches[i][j]);

#if 0
    for (j=0; j<localEdgeFaces.size(); j++)
      for (k=0; k<localEdgeFaces[j].size(); k++)
	cerr << "localEdgeFaces["<<j<<"]["<<k<<"]="<<localEdgeFaces[j][k]<<"\n";
#endif
    // make a list of all the faces on this patch we have to visit

    Array1<int> visited(faces.size());
    visited.initialize(1);
    for (j=0; j<tmpPatches[i].size(); j++)
      visited[tmpPatches[i][j]]=0;

    //	for (j=0; j<visited.size(); j++)
    //	    cerr << "visited["<<j<<"]="<<visited[j]<<"\n";

    for (j=0; j<tmpPatches[i].size(); j++) {
      int ff=tmpPatches[i][j];

      // find a remaining face and make a queue from it

      if (!visited[ff]) {
	//		cerr << "\n  Starting from face "<<ff<<"...\n";
	// if this is a new (unallocated) component, allocate it

	if (currPatchIdx == patches.size()) {
	  //		    cerr << "Adding another patch!\n";
	  int currSize=patches.size();
	  patches.resize(currSize+1);
	  patchI.resize(currSize+1);
	  patchI[currSize].facesOrient.resize(patchI[i].regions.size());
	  patchI[currSize].bdryEdgeFace.resize(edgeI.size());
	  patchI[currSize].bdryEdgeFace.initialize(-1);
	}

	queue<int> q;
	q.push(ff);
	while (!q.empty()) {
	  int c = q.front();
	  q.pop();

	  // put in the edge-neighbor faces that haven't been visited

	  for (k=0; k<faceI[c].edges.size(); k++) {
			
	    // eidx is an edge of the face we're looking at

	    int eidx=faceI[c].edges[k];

	    if (localEdgeFaces[eidx].size() == 1) // boundary!
	      patchI[currPatchIdx].bdryEdgeFace[eidx]=c;
			    
	    for (l=0; l<localEdgeFaces[eidx].size(); l++) {
			    
	      // face[fidx] will store a face neighbor of face[c]

	      int fidx=localEdgeFaces[eidx][l];
	      if (!visited[fidx]) {
		q.push(fidx);
		visited[fidx]=1;
		//				cerr << "     ...found face "<<fidx<<"\n";
		int tmpIdx=faceI[fidx].patchIdx;
		int tmpEntry=faceI[fidx].patchEntry;
		faceI[fidx].patchIdx=currPatchIdx;
		faceI[fidx].patchEntry=
		  patchI[currPatchIdx].faces.size();
		patchI[currPatchIdx].faces.add(
					       tmpPatches[tmpIdx][tmpEntry]);
		for (int m=0; m<patchI[i].regions.size(); m++) {
		  patchI[currPatchIdx].facesOrient[m].add(tmpPatchesOrient[tmpIdx][m][tmpEntry]);
		}
	      }
	    }
	  }
	}

	// patchRegions wasn't stored in a temp array - just need to 
	// copy it for each connected component

	if (currPatchIdx != i)
	  patchI[currPatchIdx].regions=patchI[i].regions;

	// set currPatchIdx to point at a new patch - allocation will
	// take place later, when we find the first face of this patch

	currPatchIdx=patches.size();
      }
    }   
    //	cerr << "done.\n\n\n";
  }

#if 0
  for (i=0; i<patches.size(); i++) 
    for (j=0; j<patches[i].size(); j++)
      cerr <<"patches["<<i<<"]["<<j<<"]="<<patches[i][j]<<"\n";

  for (i=0; i<patchesOrient.size(); i++) 
    for (j=0; j<patchesOrient[i].size(); j++)
      for (k=0; k<patchesOrient[i][j].size(); k++)
	cerr <<"patchesOrient["<<i<<"]["<<j<<"]["<<k<<"]="<<patchesOrient[i][j][k]<<"\n";

  for (i=0; i<patchRegions.size(); i++)
    for (j=0; j<patchRegions[i].size(); j++)
      cerr << "patchRegions["<<i<<"]["<<j<<"]="<<patchRegions[i][j]<<"\n";

  cerr << "Non (-1) patchBdryEdgeFaces...\n";
  for (i=0; i<patchBdryEdgeFace.size(); i++) 
    for (j=0; j<patchBdryEdgeFace[i].size(); j++)
      if (patchBdryEdgeFace[i][j] != -1) 
	cerr << "patchBdryEdgeFace["<<i<<"]["<<j<<"]="<<patchBdryEdgeFace[i][j]<<"\n";
#endif

  // now - for all of the regions, figure out which patches are inside
  //    and which are outside

  // for each patch, we have all the regions that it's part of
  // so it's easy to "invert" that such that for each region, we have
  //   a list of the patches that are part of it.
    
  Array1<Array1<int> > regionPatches(surfI.size());

  // for a particular patch in a particular regions, what region is on
  //   the other side of it
  Array1<Array1<int> > regionPatchRegOpp(surfI.size());
    
  // a lookup table so we can rapidly see if a region is directly inside 
  //   another region
  Array2<int> regionInside(surfI.size(), surfI.size());
  regionInside.initialize(0);
  for (i=0; i<surfI.size(); i++) 
    for (j=0; j<surfI[i].inner.size(); j++) 
      regionInside(i,surfI[i].inner[j])=1;
	
  for (i=0; i<patchI.size(); i++)
    for (j=0; j<patchI[i].regions.size(); j++) {
      int reg=patchI[i].regions[j];
      regionPatches[reg].add(i);
      for (k=0; k<patchI[i].regions.size(); k++)
	if (j!=k) regionPatchRegOpp[reg].add(j);
    }

  // the difficulty is figuring out which ones are inside vs outside
  //   and for the inside ones, clustering them.
  // we can figure out inside by looking at the other region the patch
  //   borders -- if our region is the "outside" of that region, then
  //   it's an inside patch.
  // but since "ouside" is a bit sloppy, instead we'll look at inside...

  // regionInsidePatches will hold the unorganized set of inside patches
  Array1<Array1<int> > regionInsidePatches(surfI.size());
    
  for (i=0; i<regionPatches.size(); i++) {
    for (j=0; j<regionPatches[i].size(); j++) {
      int reg=regionPatchRegOpp[i][j];
      if (regionInside(i,reg)) 
	regionInsidePatches[i].add(regionPatches[i][j]);
      else 
	regions[i].outerPatches.add(regionPatches[i][j]);
    }
  }

  // now we have to break up each region of regionInsidePatches into
  //   connected components
  Array1<int> regionInsidePatchesTab(patches.size());
  for (i=0; i<regionInsidePatches.size(); i++) {

    // we'll use a lookup table (rather than traversing an array) to 
    //   check membership
    regionInsidePatchesTab.initialize(0);
    for (j=0; j<regionInsidePatches[i].size(); j++)
      regionInsidePatchesTab[regionInsidePatches[i][j]]=1;

    // build up the nbr info to send our connected-component algorithm...
    Array1<Array1<int> > nbrs(patches.size());
    for (j=0; j<regionInsidePatches[i].size(); j++) {
      int p=regionInsidePatches[i][j];
      for (k=0; k<patchI[p].regions.size(); k++)
	if (i != patchI[p].regions[k])
	  nbrs[j].add(patchI[p].regions[k]);
    }

    Array1<Array1<int> > CCs;
    buildCCs(nbrs, CCs);
    regions[i].innerPatches.resize(CCs.size());
    for (j=0; j<CCs.size(); j++) 
      for (k=0; k<CCs[j].size(); k++)
	if (regionInsidePatchesTab[CCs[j][k]])
	  regions[i].innerPatches[j].add(CCs[j][k]);
  }

  // we can do clustering through connected component analysis of the
  //   patches.  first figure out the neighbors of every region (iterate
  //   through the patches to build this).  then seed a queue with one 
  //   of the inside regions and push all of the unvisited neighbors onto 
  //   the queue.  go until the q is empty.  everyone we popped off, who
  //   is also part one of the "patchRegions" is part of a single shell.
  //   if there's an inner region who hasn't been visited, seed the q
  //   with them and start over.  continue until there aren't any inner
  //   regions left.

  // we'll straighten out orientations later...

  // add all of the nodes, edges and faces - and make them part of a PATCH

  for (i=0; i<patchI.size(); i++)
    for (j=0; j<patchI[i].faces.size(); j++) {
      int f=patchI[i].faces[j];
      topoFaces[f].type = PATCH;
      topoFaces[f].idx = i;
      for (k=0; k<faceI[f].edges.size(); k++) {
	int e=faceI[f].edges[k];
	topoEdges[e].type = PATCH;
	topoEdges[e].idx = i;
	int n=edges[faceI[f].edges[k]]->i1;
	topoNodes[n].type = PATCH;
	topoNodes[n].idx = i;
	n=edges[faceI[f].edges[k]]->i2;
	topoNodes[n].type = PATCH;
	topoNodes[n].idx = i;
      }
    }
  cerr << "Done with BuildPatches!!\n";
}

void TopoSurfTree::BldWires() {
  // make an allEdgePatches arrays the size of the edges list
  // for each face in each patch, add its patch idx to all of the 
  //    allEdgePatches[edge] edges it contains
  // for any allEdgePatches[edge] edge that has more than two patches, 
  //    is an edge that's part of a wire
  // find the highest number of patches any edge is part of and allocate
  //    tmpTypeMembers and tmpOrientMembers arrays
  // make up an idx for each edge that's "wired" based on the patches 
  //    it's part of and insert the edge into the tmpTypeMembers list
  //    and insert its orientations into the tmpOrientMembers array
  // for every non-empty tmpTypeMembers array, build the: wires, 
  //    wiresOrient and wiresPatches arrays
  int i,j,k;

  cerr << "We have "<<edges.size()<<" edges.\n";
  // make a list of what patches each edge is attached to
  Array1<Array1<int> > allEdgePatches(edges.size());
    
  for (i=0; i<patches.size(); i++)
    for (j=0; j<edges.size(); j++)
      if (patchI[i].bdryEdgeFace[j] != -1)
	allEdgePatches[j].add(i);
  int maxFacesOnAnEdge=1;

  // sort all of the lists from above, and find the maximum number of
  // faces any edge belongs to
  for (i=0; i<allEdgePatches.size(); i++)
    if (allEdgePatches[i].size() > 1) {
      if (allEdgePatches[i].size() > maxFacesOnAnEdge)
	maxFacesOnAnEdge=allEdgePatches[i].size();
      order(allEdgePatches[i]);
    }

  if (maxFacesOnAnEdge<2) return;

  // create a recursive array structure that can be descended by
  // indexing all of the patches an edge touches (allEdgePatches[])
  // the list of edges for a wire is stored in an items[][] array
  // there is also an array of orientations for wire, orient[][][], 
  //   indexed by the wire, then the patch, and then the edge

  SrchLst *tmpWireIdx = new SrchLst;

  cerr << "AllEdgePatches.size()="<<allEdgePatches.size()<<"\n";
  for (i=0; i<allEdgePatches.size(); i++) {

    // this has to happen somewhere... might as well be here
    edgeI[i].wireIdx=-1;
    edgeI[i].wireEntry=-1;

    if (!allEdgePatches[i].size()) continue; // only use bdry edges

    // initially, point to the top level array

    SrchLst *currLst=tmpWireIdx;

    // for each patch, recursively descend

    for (j=0; j<allEdgePatches[i].size(); j++) {

      // ptch is the next patch in the list for this edge

      int ptch=allEdgePatches[i][j];

      // search through the lookup array to find which next[] corresponds
      //   to the patch we want to descend to.
      // if we get to the end of the list, allocate a new one and add 
      //   reg to the end of the lookup array

      for (k=0; k<currLst->lookup.size(); k++) 
	if (currLst->lookup[k] == ptch) break;
      if (k==currLst->lookup.size()) {
	currLst->lookup.add(ptch);
	currLst->next.add(new SrchLst);
      }

      // k has the index for patch ptch - now descend

      currLst=currLst->next[k];
    }

    // currLst now has the right wire (we've decended through all
    //   of the patches) -- now store this edge's info in the items[][]
    //   and orient[][][] arrays

    // if the orient[][] array hasn't been allocated yet (first edge
    //   of the wire) - allocate it big enough for all of the regions

    if (!currLst->orient.size())
      currLst->orient.resize(j);

    // add this edge to the list for the wire

    currLst->items.add(i);

    // for all of the patches this edge bounds, find this edge in the
    //   list and add its orientation information to orient[][]

    for (j=0; j<allEdgePatches[i].size(); j++) {
      int fidx=patchI[j].bdryEdgeFace[i];
      if (fidx == -1) continue;
      for (k=0; k<faceI[fidx].edges.size(); k++) {
	if (faceI[fidx].edges[k] == i) {
	  currLst->orient[j].add(faceI[fidx].edgeOrient[k]);
	  break;
	}
      }
    }
  }

  // these will temporarily hold the edges for each wire.
  // since a wire should be a single connected component, these
  // tmpWires will be split up below - then we'll build new wires
  // as needed and store the tmp data in the correct place

  Array1<Array1<int> > tmpWires;
  Array1<Array1<Array1<int> > > tmpWiresOrient;

  // recursively descend the tmpWireIdx srchList -- add all items
  // to the tmpWires and tmpWiresOrient lists

  Array1<int> descend;
  addWireAndDescend(tmpWireIdx, tmpWires, tmpWiresOrient, descend);
  descendAndFree(tmpWireIdx);

  // now we have to do connectivity searches on each wire -- make sure
  // it's one connected component.  If it isn't, break it into separate
  // patches.

#if 0
  for (i=0; i<tmpWiresOrient.size(); i++)
    for (j=0; j<tmpWiresOrient[i].size(); j++)
      for (k=0; k<tmpWiresOrient[i][j].size(); k++)
	cerr << "tmpWiresOrient["<<i<<"]["<<j<<"]["<<k<<"]="<<tmpWiresOrient[i][j][k]<<"\n";
  for (i=0; i<tmpWires.size(); i++)
    for (j=0; j<tmpWires[i].size(); j++)
      cerr << "tmpWires["<<i<<"]["<<j<<"]="<<tmpWires[i][j]<<"\n";
  for (i=0; i<wirePatches.size(); i++)
    for (j=0; j<wirePatches[i].size(); j++)
      cerr << "wirePatches["<<i<<"]["<<j<<"]="<<wirePatches[i][j]<<"\n";
  for (i=0; i<edgeI.size(); i++) {
    cerr << "edgeI["<<i<<"].wireIdx="<<edgeI[i].wireIdx<<"\n";
    cerr << "edgeI["<<i<<"].wireEntry="<<edgeI[i].wireEntry<<"\n";
  }    
#endif
  cerr << "Doing connectivity search\n";

  int numWires=wires.size();
  int currWireIdx;
  for (i=0; i<numWires; i++) {
    //	cerr << " wire "<<i<<"/"<<numWires<<"... ";
    currWireIdx=i;

    // make a list of all the edges attached to each node of this wire

    Array1<Array1<int> > localNodeEdges(nodes.size());
    for (j=0; j<tmpWires[i].size(); j++) {
      localNodeEdges[edges[tmpWires[i][j]]->i1].add(tmpWires[i][j]);
      localNodeEdges[edges[tmpWires[i][j]]->i2].add(tmpWires[i][j]);
    }

#if 0
    for (j=0; j<localNodeEdges.size(); j++)
      for (k=0; k<localNodeEdges[j].size(); k++)
	cerr << "localNodeEdges["<<j<<"]["<<k<<"]="<<localNodeEdges[j][k]<<"\n";
#endif

    // make a list of all the edges on this wire we have to visit

    Array1<int> visited(edges.size());
    visited.initialize(1);
    for (j=0; j<tmpWires[i].size(); j++)
      visited[tmpWires[i][j]]=0;

    for (j=0; j<tmpWires[i].size(); j++) {
      Array1<int> nodesSeen;
      int ee=tmpWires[i][j];

      // find a remaining edge and make a queue from it

      if (!visited[ee]) {
	//		cerr << "\n  Starting from edge "<<ee<<"...\n";

	// if this is a new (unallocated) component, allocate it

	if (currWireIdx == wires.size()) {
	  //		    cerr << "Adding another wire!\n";
	  int currSize=wires.size();
	  wires.resize(currSize+1);
	  wireI.resize(currSize+1);
	}

	queue<int> q;
	q.push(ee);
	while (!q.empty()) {
	  int c = q.front();
	  q.pop();

	  // put in the node-neighbor edges that haven't been visited
	  // nidx is a node the face we're looking at

	  int nidx=edges[c]->i1;
	  nodesSeen.add(nidx);
	  //		    cerr << "   edge("<<c<<") - examining node("<<nidx<<")\n";
	  for (k=0; k<localNodeEdges[nidx].size(); k++) {
			
	    // edge[eidx] will store an edge neighbor of edge[c]
			
	    int eidx=localNodeEdges[nidx][k];
	    //			cerr << "       edge "<<eidx<<" is a neighbor\n";
	    if (!visited[eidx]) {
	      q.push(eidx);
	      visited[eidx]=1;
	      //			    cerr << "     ...found edge "<<eidx<<"\n";
	      int tmpIdx=edgeI[eidx].wireIdx;
	      int tmpEntry=edgeI[eidx].wireEntry;
	      edgeI[eidx].wireIdx=currWireIdx;
	      edgeI[eidx].wireEntry=wireI[currWireIdx].edges.size();
	      wireI[currWireIdx].edges.add(tmpWires[tmpIdx][tmpEntry]);
	      wireI[currWireIdx].edgesOrient.add(tmpWires[tmpIdx][tmpEntry]);
	    }
	  }

	  // nidx is a node the face we're looking at
	  nidx=edges[c]->i2;
	  nodesSeen.add(nidx);
	  //		    cerr << "   edge("<<c<<") - examining node("<<nidx<<")\n";
	  for (k=0; k<localNodeEdges[nidx].size(); k++) {
			
	    // edge[eidx] will store an edge neighbor of edge[c]
			
	    int eidx=localNodeEdges[nidx][k];
	    //			cerr << "       edge "<<eidx<<" is a neighbor\n";
	    if (!visited[eidx]) {
	      q.push(eidx);
	      visited[eidx]=1;
	      //			    cerr << "      ...found edge "<<eidx<<"\n";
	      int tmpIdx=edgeI[eidx].wireIdx;
	      int tmpEntry=edgeI[eidx].wireEntry;
	      edgeI[eidx].wireIdx=currWireIdx;
	      edgeI[eidx].wireEntry=wireI[currWireIdx].edges.size();
	      wireI[currWireIdx].edges.add(tmpWires[tmpIdx][tmpEntry]);
	      wireI[currWireIdx].edgesOrient.add(tmpWires[tmpIdx][tmpEntry]);
	    }
	  }		    
	}

	// now that we have a single connected component, find the
	// junctions of this wire since we have the localNodeEdges
	// information around now...

	for (k=0; k<nodesSeen.size(); k++) {
	  if (localNodeEdges[nodesSeen[k]].size() == 1) {
	    wires[currWireIdx].nodes.add(nodesSeen[k]);
	    //			cerr << "wireBdryNodes["<<currWireIdx<<"].add("<<nodesSeen[k]<<")\n";
	  }
	}
	if (wires[currWireIdx].nodes.size() == 0) {
	  //		    cerr << "junctionlesWires.add("<<nodesSeen[0]<<")\n";
	  wires[currWireIdx].nodes.add(nodesSeen[0]);
	}

	// wirePatches wasn't stored in a temp array - just need to
	// copy it for each connected component

	if (currWireIdx != i)
	  wireI[currWireIdx].patches=wireI[i].patches;

	// set currWireIdx to point at a new wire - allocation will
	// take place later, when we find the first edge of this wire

	currWireIdx=wires.size();
      }
    }   
  }


  // now - for each patch, figure out which wires bound it

  // go through all the wires, and for each one, add it to the patches
  //   it bounds.

  // sort the wires, so they're head-to-tail according to the nodes of the
  //   wires.

  // now, look for any patches without wires -- we have to add a 
  //   wire to those -- just choose a triangle and make its edges a wire

  // once again, we'll handle orientations later...

  for (i=0; i<wireI.size(); i++) {
    for (j=0; j<wireI[i].edges.size(); j++) {
      topoEdges[wireI[i].edges[j]].type = WIRE;
      topoEdges[wireI[i].edges[j]].idx = i;
    }
    for (j=0; i<wires[i].nodes.size(); i++) {
      topoNodes[wires[i].nodes[j]].type = WIRE;
      topoNodes[wires[i].nodes[j]].idx = i;
    }	    
  }
  cerr << "Done with BuildWires!!\n";

}

void TopoSurfTree::BldJunctions() {
  // make an allNodeWires array the size of the nodes list
  // for each edge in each wire, add its wire idx to both of the 
  //    allNodeWires[node] nodes it contains
  // for any allNodeWires[node] node that has more than two wires, 
  //    is a node that's part of a junction
  // find the highest number of junctions any wire is part of and allocate
  //    tmpTypeMembers and tmpOrientMembers arrays
  // make up an idx for each node that's "junctioned" based on the wires
  //    it's part of and insert the node into the tmpTypeMembers list
  // for every non-empty tmpTypeMembers array, build the: junctions
  //    and junctionsWires arrays
  int i,j,k;

  cerr << "We have "<<nodes.size()<<" nodes.\n";
  if (!wires.size()) return;

  // make a list of what wires each node is attached to
  Array1<Array1<int> > allNodeWires(nodes.size());

  //    cerr << "Here are the wireBdryNodes...\n";
  for (i=0; i<wires.size(); i++)
    for (j=0; j<wires[i].nodes.size(); j++) {
      //	    cerr << "   wireBdryNodes["<<i<<"]["<<j<<"]="<<wireBdryNodes[i][j]<<"\n";
      allNodeWires[wires[i].nodes[j]].add(i);
    }

  // sort all of the lists from above, and find the maximum number of
  // wires a node belongs to
  for (i=0; i<allNodeWires.size(); i++)
    if (allNodeWires[i].size() > 1) {
      order(allNodeWires[i]);
    }

  // create a recursive array structure that can be descended by
  // indexing all of the wires a node touches (allNodeWires[])
  // the list of nodes for a junction is stored in an items[][] array

  SrchLst *tmpJunctionIdx = new SrchLst;

  cerr << "AllNodeWires.size()="<<allNodeWires.size()<<"\n";
  for (i=0; i<allNodeWires.size(); i++) {
    //	cerr << "  **** LOOKING IT UP!\n";
    if (!allNodeWires[i].size()) continue;

    // initially, point to the top level array

    SrchLst *currLst=tmpJunctionIdx;

    // for each wire, recursively descend

    for (j=0; j<allNodeWires[i].size(); j++) {

      // wr is the next wire in the list for this node

      int wr=allNodeWires[i][j];

      // search through the lookup array to find which next[] corresponds
      //   to the wire we want to descend to.
      // if we get to the end of the list, allocate a new one and add 
      //   wr to the end of the lookup array

      for (k=0; k<currLst->lookup.size(); k++) 
	if (currLst->lookup[k] == wr) break;
      if (k==currLst->lookup.size()) {
	currLst->lookup.add(wr);
	currLst->next.add(new SrchLst);
      }

      // k has the index for wire wr - now descend

      currLst=currLst->next[k];
    }

    // currLst now has the right junction (we've decended through all
    //   of the wires) -- now store this node's info in the items[][] array

    currLst->items.add(i);
  }

  // recursively descend the tmpJunctionsIdx srchList -- add all items
  // to the junctions and junctionWires lists

  Array1<int> descend;
  addJunctionAndDescend(tmpJunctionIdx, descend);
  descendAndFree(tmpJunctionIdx);

#if 0
  cerr << "Junctions.size() = "<<junctions.size()<<"\n";
  for (i=0; i<junctions.size(); i++)
    cerr << "junctions["<<i<<"] = "<<junctions[i]<<"\n";

  cerr << "JunctionlessWires.size() = "<<junctionlessWires.size()<<"\n";
  for (i=0; i<junctionlessWires.size(); i++)
    cerr << "junctionlessWires["<<i<<"] = "<<junctionlessWires[i]<<"\n";
#endif

  // make a list of all the wires and note which ones have junctions
  // any that don't have junctions still need to be added

  Array1<int> visited(wires.size());
  visited.initialize(0);
  for (i=0; i<junctionI.size(); i++)
    for (j=0; j<junctionI[i].wires.size(); j++)
      visited[junctionI[i].wires[j]]=1;

  for (i=0; i<visited.size(); i++)
    if (!visited[i]) {
      junctions.add(wires[i].nodes[0]);
      junctionI.resize(junctionI.size()+1);
      junctionI[junctionI.size()-1].wires.add(i);
    }

  // for each of the wires, figure out which junctions bound it.

  for (i=0; i<junctions.size(); i++) {
    topoNodes[junctions[i]].type = JUNCTION;
    topoNodes[junctions[i]].idx = i;
  }

  cerr << "Done with BuildJunctions!!\n";
}

void TopoSurfTree::BldOrientations() {
}

GeomObj* TopoSurfTree::get_obj(const ColorMapHandle&)
{
  NOT_FINISHED("TopoSurfTree::get_obj");
  return 0;
}

#define TopoSurfTree_VERSION 1

void TopoSurfTree::io(Piostream& stream) {
  /*  int version=*/stream.begin_class("TopoSurfTree", TopoSurfTree_VERSION);
  SurfTree::io(stream);		    
  Pio(stream, regions);
  Pio(stream, patches);
  Pio(stream, patchI);
  Pio(stream, wires);
  Pio(stream, wireI);
  Pio(stream, junctions);
  Pio(stream, junctionI);
  Pio(stream, topoNodes);
  Pio(stream, topoEdges);
  Pio(stream, topoFaces);
  stream.end_class();
}

} // End namespace DaveW

namespace SCIRun {

void Pio(Piostream& stream, DaveW::TopoEntity& te)
{
  stream.begin_cheap_delim();
  int* typep=(int*)&(te.type);
  Pio(stream, *typep);
  Pio(stream, te.idx);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, DaveW::Patch& p) {
  stream.begin_cheap_delim();
  Pio(stream, p.wires);
  Pio(stream, p.wiresOrient);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, DaveW::PatchInfo& pi) {
  stream.begin_cheap_delim();
  Pio(stream, pi.faces);
  Pio(stream, pi.facesOrient);
  Pio(stream, pi.regions);
  Pio(stream, pi.bdryEdgeFace);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, DaveW::Wire& w) {
  stream.begin_cheap_delim();
  Pio(stream, w.nodes);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, DaveW::WireInfo& wi) {
  stream.begin_cheap_delim();
  Pio(stream, wi.edges);
  Pio(stream, wi.edgesOrient);
  Pio(stream, wi.patches);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, DaveW::JunctionInfo& ji) {
  stream.begin_cheap_delim();
  Pio(stream, ji.wires);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, DaveW::Region& r) {
  stream.begin_cheap_delim();
  Pio(stream, r.outerPatches);
  Pio(stream, r.innerPatches);
  stream.end_cheap_delim();
}

} // End namespace SCIRun



