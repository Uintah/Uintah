
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
#include <iostream.h>
#include <Classlib/Assert.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Classlib/TrivialAllocator.h>
#include <Datatypes/TopoSurfTree.h>
#include <Malloc/Allocator.h>
#include <math.h>
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
    BldPatches();
    BldWires();
    BldJunctions();
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
	int currSize=patchRegions.size();
	patchBdryEdgeFace.resize(currSize+1);
	patchBdryEdgeFace[currSize].resize(edgeI.size());
	patchBdryEdgeFace[currSize].initialize(-1);
	patches.resize(currSize+1);
	tmpPatches.resize(currSize+1);
	tmpPatches[currSize]=curr->items;
	patchesOrient.resize(currSize+1);
	patchesOrient[currSize].resize(curr->orient.size());
	tmpPatchesOrient.resize(currSize+1);
	tmpPatchesOrient[currSize]=curr->orient;
	patchRegions.add(visList);
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
	int currSize=wirePatches.size();
	wires.resize(currSize+1);
	tmpWires.resize(currSize+1);
	wireBdryNodes.resize(currSize+1);
	wiresOrient.resize(currSize+1);
	tmpWiresOrient.resize(currSize+1);
	tmpWiresOrient[currSize]=curr->orient;
	tmpWires[currSize]=curr->items;
	wirePatches.add(visList);
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
	int currSize=junctionWires.size();
	junctions.resize(currSize+1);
	junctions[currSize]=curr->items[j];
	junctionWires.add(visList);
    }
    for (j=0; j<curr->next.size(); j++) {
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

void TopoSurfTree::BldPatches() {
    int i,j,k,l;

    faceToPatch.resize(faces.size());
    faceToPatch.initialize(-1);

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
    //
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

    cerr << "\nPatchRegions.size()="<<patchRegions.size()<<"\n";
    
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
		    patchesOrient.resize(currSize+1);
		    patchesOrient[currSize].resize(patchRegions[i].size());
		    patchBdryEdgeFace.resize(currSize+1);
		    patchBdryEdgeFace[currSize].resize(edgeI.size());
		    patchBdryEdgeFace[currSize].initialize(-1);
		}

		Queue<int> q;
		q.append(ff);
		while (!q.is_empty()) {
		    int c=q.pop();

		    // put in the edge-neighbor faces that haven't been visited

		    for (k=0; k<faceI[c].edges.size(); k++) {
			
			// eidx is an edge of the face we're looking at

			int eidx=faceI[c].edges[k];

			if (localEdgeFaces[eidx].size() == 1) // boundary!
			    patchBdryEdgeFace[currPatchIdx][eidx]=c;
			    
			for (l=0; l<localEdgeFaces[eidx].size(); l++) {
			    
			    // face[fidx] will store a face neighbor of face[c]

			    int fidx=localEdgeFaces[eidx][l];
			    if (!visited[fidx]) {
				q.append(fidx);
				visited[fidx]=1;
//				cerr << "     ...found face "<<fidx<<"\n";
				int tmpIdx=faceI[fidx].patchIdx;
				int tmpEntry=faceI[fidx].patchEntry;
				faceI[fidx].patchIdx=currPatchIdx;
				faceI[fidx].patchEntry=
				    patches[currPatchIdx].size();
				patches[currPatchIdx].add(
					      tmpPatches[tmpIdx][tmpEntry]);
				for (int m=0; m<patchRegions[i].size(); m++) {
				    patchesOrient[currPatchIdx][m].add(tmpPatchesOrient[tmpIdx][m][tmpEntry]);
				}
			    }
			}
		    }
		}

		// patchRegions wasn't stored in a temp array - just need to 
		// copy it for each connected component

		if (currPatchIdx != i)
		    patchRegions.add(patchRegions[i]);

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

    cerr << "Done with BuildPatches!!\n";

    for (i=0; i<patches.size(); i++)
	for (j=0; j<patches[i].size(); j++)
	    faceToPatch[patches[i][j]] = i;
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

    edgeToWire.resize(edges.size());
    edgeToWire.initialize(-1);

    cerr << "We have "<<edges.size()<<" edges.\n";
    // make a list of what patches each edge is attached to
    Array1<Array1<int> > allEdgePatches(edges.size());
    
    for (i=0; i<patches.size(); i++)
	for (j=0; j<edges.size(); j++)
	    if (patchBdryEdgeFace[i][j] != -1)
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
    //
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
	    int fidx=patchBdryEdgeFace[j][i];
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
		    wiresOrient.resize(currSize+1);
		    wireBdryNodes.resize(currSize+1);
		}

		Queue<int> q;
		q.append(ee);
		while (!q.is_empty()) {
		    int c=q.pop();

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
			    q.append(eidx);
			    visited[eidx]=1;
//			    cerr << "     ...found edge "<<eidx<<"\n";
			    int tmpIdx=edgeI[eidx].wireIdx;
			    int tmpEntry=edgeI[eidx].wireEntry;
			    edgeI[eidx].wireIdx=currWireIdx;
			    edgeI[eidx].wireEntry=wires[currWireIdx].size();
			    wires[currWireIdx].add(tmpWires[tmpIdx][tmpEntry]);
			    wiresOrient[currWireIdx].add(tmpWires[tmpIdx][tmpEntry]);
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
			    q.append(eidx);
			    visited[eidx]=1;
//			    cerr << "      ...found edge "<<eidx<<"\n";
			    int tmpIdx=edgeI[eidx].wireIdx;
			    int tmpEntry=edgeI[eidx].wireEntry;
			    edgeI[eidx].wireIdx=currWireIdx;
			    edgeI[eidx].wireEntry=wires[currWireIdx].size();
			    wires[currWireIdx].add(tmpWires[tmpIdx][tmpEntry]);
			    wiresOrient[currWireIdx].add(tmpWires[tmpIdx][tmpEntry]);
			}
		    }		    
		}

		// now that we have a single connected component, find the
		// junctions of this wire since we have the localNodeEdges
		// information around now...

		for (k=0; k<nodesSeen.size(); k++) {
		    if (localNodeEdges[nodesSeen[k]].size() == 1) {
			wireBdryNodes[currWireIdx].add(nodesSeen[k]);
//			cerr << "wireBdryNodes["<<currWireIdx<<"].add("<<nodesSeen[k]<<")\n";
		    }
		}
		if (wireBdryNodes[currWireIdx].size() == 0) {
//		    cerr << "junctionlesWires.add("<<nodesSeen[0]<<")\n";
		    junctionlessWires.add(nodesSeen[0]);
		}

		// wirePatches wasn't stored in a temp array - just need to
		// copy it for each connected component

		if (currWireIdx != i)
		    wirePatches.add(wirePatches[i]);

		// set currWireIdx to point at a new wire - allocation will
		// take place later, when we find the first edge of this wire

		currWireIdx=wires.size();
	    }
	}   
    }

    for (i=0; i<wires.size(); i++)
	for (j=0; j<wires[i].size(); j++)
	    edgeToWire[wires[i][j]] = i;

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

    nodeToJunction.resize(nodes.size());
    nodeToJunction.initialize(-1);

    cerr << "We have "<<nodes.size()<<" nodes.\n";
    if (!wires.size()) return;

    // make a list of what wires each node is attached to
    Array1<Array1<int> > allNodeWires(nodes.size());

//    cerr << "Here are the wireBdryNodes...\n";
    for (i=0; i<wireBdryNodes.size(); i++)
	for (j=0; j<wireBdryNodes[i].size(); j++) {
//	    cerr << "   wireBdryNodes["<<i<<"]["<<j<<"]="<<wireBdryNodes[i][j]<<"\n";
	    allNodeWires[wireBdryNodes[i][j]].add(i);
	}

    // sort all of the lists from above, and find the maximum number of
    // wires a node belongs to
    for (i=0; i<allNodeWires.size(); i++)
	if (allNodeWires[i].size() > 1) {
	    order(allNodeWires[i]);
	}

    // create a recursive array structure that can be descended by
    // indexing all of the wires a node touches (allNodeWires[])
    //
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

    for (i=0; i<junctions.size(); i++)
	nodeToJunction[junctions[i]] = i;

    cerr << "Done with BuildJunctions!!\n";
}

GeomObj* TopoSurfTree::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("TopoSurfTree::get_obj");
    return 0;
}

#define TopoSurfTree_VERSION 1

void TopoSurfTree::io(Piostream& stream) {
    int version=stream.begin_class("TopoSurfTree", TopoSurfTree_VERSION);
    SurfTree::io(stream);		    
    Pio(stream, patches);	// indices into the SurfTree elems
    Pio(stream, patchesOrient);
    Pio(stream, patchRegions);
    Pio(stream, wires);		// indices into the SurfTree edges
    Pio(stream, wiresOrient);
    Pio(stream, wirePatches);
    Pio(stream, junctions);		// indices into the SurfTree nodes
    Pio(stream, junctionWires);
    Pio(stream, junctionlessWires);
    Pio(stream, junctions);
    Pio(stream, faceToPatch);
    Pio(stream, edgeToWire);
    Pio(stream, nodeToJunction);
    stream.end_class();
}











#if 0
// call this before outputting persistently.  that way we have the Arrays
// for VTK Decimage algorithm.
void TopoSurfTree::SurfsToTypes() {
    int i,j;

    cerr << "We have "<<faces.size()<<" faces.\n";
    // make a list of what surfaces each faces is attached to
    Array1<Array1<int> > allFaceRegions(faces.size());
//    cerr << "EdgesOnANode.size()="<<allFaceRegions.size()<<"\n";
    for (i=0; i<surfI.size(); i++)
	for (j=0; j<surfI.faces[i].size(); j++)
	    allFaceRegions[surfI[i].faces[j]].add(i);
    int maxWiresOnANode=1;

    // sort all of the lists from above, and find the maximum number of
    // surfaces any face belongs to
    for (i=0; i<allFaceRegions.size(); i++)
	if (allFaceRegions[i].size() > 1) {
	    if (allFaceRegions[i].size() > maxWiresOnANode)
		maxWiresOnANode=allFaceRegions[i].size();
	    order(allFaceRegions[i]);
	}
    int sz=pow(surfEls.size(), maxWiresOnANode);
    cerr << "allocating "<<maxWiresOnANode<<" levels with "<< surfEls.size()<< " types (total="<<sz<<").\n";

    // allocate all combinations of the maximum number of surfaces
    // from the lists of which surfaces each face belong to,
    // construct a list of faces which belong for each surface
    // combination

    Array1 <Array1<int> > tmpJunctions(pow(surfEls.size(), maxWiresOnANode));
    cerr << "EdgesOnANode.size()="<<allFaceRegions.size()<<"\n";
    for (i=0; i<allFaceRegions.size(); i++) {
//	cerr << "  **** LOOKING IT UP!\n";
	int idx=getIdx(allFaceRegions[i], surfEls.size());
//	cerr << "this faces has index: "<<idx<<"\n";
	tmpJunctions[idx].add(i);
    }
    typeSurfs.resize(0);
    typeIds.resize(faces.size());
    for (i=0; i<tmpJunctions.size(); i++) {
	// if there are any faces of this combination type...
	if (tmpJunctions[i].size()) {
	    // find out what surfaces there were	
//	    cerr << "found "<<tmpJunctions[i].size()<<" faces of type "<<i<<"\n";
	    Array1<int> surfList;
	    getList(surfList, i, surfEls.size());
	    int currSize=typeSurfs.size();
	    typeSurfs.resize(currSize+1);
	    typeSurfs[currSize].resize(0);
//	    cerr << "here's the array: ";
	    for (j=0; j<tmpJunctions[i].size(); j++) {
//		cerr << tmpJunctions[i][j]<<" ";
		typeIds[tmpJunctions[i][j]]=currSize;
	    }
//	    cerr << "\n";
//	    cerr << "copying array ";
//	    cerr << "starting to add faces...";
	    for (j=0; j<surfList.size(); j++) {
		typeSurfs[currSize].add(surfList[j]);
//		cerr << ".";
	    }
//	    cerr << "   done!\n";
	}
    }
    cerr << "done with SurfsToTypes!!\n";
}

// call this after VTK Decimate has changed the Faces and nodes -- need
// to rebuild typeSurfs information
void TopoSurfTree::TypesToSurfs() {
    int i,j;
//    cerr << "building surfs from types...\n";
    for (i=0; i<surfEls.size(); i++) {
	surfEls[i].resize(0);
	surfOrient[i].resize(0);
    }
//    cerr << "typeSurfs.size() = "<<typeSurfs.size()<<"\n";
//    cerr << "surfEls.size() = "<<surfEls.size()<<"\n";
    for (i=0; i<typeIds.size(); i++) {
//	cerr << "working on typeSurfs["<<typeIds[i]<<"\n";
	for (j=0; j<typeSurfs[typeIds[i]].size(); j++) {
//	    cerr << "adding "<<i<<" to surfEls["<<typeSurfs[typeIds[i]][j]<<"]\n";
	    surfEls[typeSurfs[typeIds[i]][j]].add(i);
	    surfOrient[typeSurfs[typeIds[i]][j]].add(1);
	}
    }
}
#endif














