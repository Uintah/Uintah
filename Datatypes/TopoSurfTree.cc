
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

inline int getIdx(const Array1<int>& a, int size) {
    int val=0;
    for (int k=a.size()-1; k>=0; k--)
	val = val*size + a[k];
    return val;
}

void getList(Array1<int>& surfList, int i, int size) {
    int valid=1;
//    cerr << "starting list...\n";
    while (valid) {
	surfList.add(i%size);
//	cerr << "just added "<<i%size<<" to the list.\n";
	if (i >= size) valid=1; else valid=0;
	i /= size;
    }
 //   cerr << "done with list.\n";
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

// allFaceSurfaces[faceG][surfCtr]=surfG
//	faceG is the true face number
//	surfCtr is a counter of all the surfaces this face touches
//	surfG is the global index of each region/surface this face neighbors



void TopoSurfTree::BldPatches() {
    int i,j,k,l;

    cerr << "We have "<<faces.size()<<" faces.\n";
    if (!faces.size()) return;

    // make a list of what surfaces each faces is attached to
    Array1<Array1<int> > allFaceSurfaces(faces.size());
//    cerr << "Membership.size()="<<allFaceSurfaces.size()<<"\n";
    for (i=0; i<surfI.size(); i++)
	for (j=0; j<surfI[i].faces.size(); j++)
	    allFaceSurfaces[surfI[i].faces[j]].add(i);
    int maxSurfacesOnAFace=1;

    // sort all of the lists from above, and find the maximum number of
    // surfaces any face belongs to
    for (i=0; i<allFaceSurfaces.size(); i++)
	if (allFaceSurfaces[i].size() > 1) {
	    if (allFaceSurfaces[i].size() > maxSurfacesOnAFace)
		maxSurfacesOnAFace=allFaceSurfaces[i].size();
	    order(allFaceSurfaces[i]);
	}
    int sz=pow(surfI.size(), maxSurfacesOnAFace);
    cerr << "allocating "<<maxSurfacesOnAFace<<" levels with "<< surfI.size()<< " types (total="<<sz<<").\n";

    // allocate all combinations of the maximum number of surfaces
    // from the lists of which surfaces each face belong to,
    // construct a list of faces which belong for each surface
    // combination

    // a better way to do this is to HASH on each surface(regions) --
    // each hashed key should find another hash table, as well as a
    // list of faces (we'll need a new struct for this).
    // NOTE - we should use this multi-level hashing technique in
    // SegFldToSurfTree when we hash the faces and edges.

    Array1 <Array1<int> > tmpPatchIdx(pow(surfI.size(), maxSurfacesOnAFace));
    Array1 <Array1 <Array1<int> > > tmpPatchIdxOrient(pow(surfI.size(), maxSurfacesOnAFace));
    cerr << "Membership.size()="<<allFaceSurfaces.size()<<"\n";
    for (i=0; i<allFaceSurfaces.size(); i++) {
//	cerr << "Face="<<i<<"\n";
//	if ((i%1000)==0) cerr << i <<" ";
//	cerr << "  **** LOOKING IT UP!\n";
	int idx=getIdx(allFaceSurfaces[i], surfI.size());
//	cerr << "   idx="<<idx<<"\n";
//	cerr << "this faces has index: "<<idx<<"\n";
	tmpPatchIdx[idx].add(i);
	Array1<int> surfList;
	getList(surfList, idx, surfI.size());
	if (!tmpPatchIdxOrient[idx].size()) 
	    tmpPatchIdxOrient[idx].resize(surfList.size());
	// for each surf in the list, we have to look up face i, find its
	// orientation and add it to 
	for (j=0; j<surfList.size(); j++) {
//	    cerr << "     j="<<j<<"\n";
	    for (k=0; k<surfI[surfList[j]].faces.size(); k++) {
//		cerr << "       k="<<k<<"\n";
		if (surfI[surfList[j]].faces[k] == i) {
//		    cerr <<"         idx="<<idx<<" j="<<j<<" k="<<k<<" or="<<surfI[j].faceOrient[k]<<"\n";
		    tmpPatchIdxOrient[idx][j].add(surfI[surfList[j]].faceOrient[k]);
		    break;
		}
	    }
	}
    }

//#if 0
    for (i=0; i<tmpPatchIdxOrient.size(); i++)
	for (j=0; j<tmpPatchIdxOrient[i].size(); j++)
	    for (k=0; k<tmpPatchIdxOrient[i][j].size(); k++)
		cerr << "tmpPatchIdxOrient["<<i<<"]["<<j<<"]["<<k<<"] = "<<tmpPatchIdxOrient[i][j][k]<<"\n";
    cerr << "\n";
    for (i=0; i<tmpPatchIdx.size(); i++)
	for (j=0; j<tmpPatchIdx[i].size(); j++)
	    cerr << "tmpPatchIdx["<<i<<"]["<<j<<"] = "<<tmpPatchIdx[i][j]<<"\n";
    cerr << "\n";
//#endif

    // these will temporarily hold the faces for each patch.
    // since a patch should be a single connected component, these
    // tmpPatches will be split up below - then we'll build new patches
    // as needed and store the tmp data in the correct place

    Array1<Array1<int> > tmpPatches;
    Array1<Array1<Array1<int> > > tmpPatchesOrient;

    cerr << "Allocating and storing temporary patch values... ";

    cerr << "\nPatchRegions.size()="<<patchRegions.size()<<"\n";
    for (i=0; i<tmpPatchIdx.size(); i++) {

	// if there are any faces of this combination type...

	if (tmpPatchIdx[i].size()) {

	    // find out what surfaces there were	

	    Array1<int> surfList;
	    getList(surfList, i, surfI.size());
	    int currSize=patchRegions.size();
	    patchBdryEdgeFace.resize(currSize+1);
	    patchBdryEdgeFace[currSize].resize(edgeI.size());
	    patchBdryEdgeFace[currSize].initialize(-1);
	    patches.resize(currSize+1);
	    tmpPatches.resize(currSize+1);
	    patchesOrient.resize(currSize+1);
	    patchesOrient[currSize].resize(tmpPatchIdxOrient[i].size());
	    tmpPatchesOrient.resize(currSize+1);
	    tmpPatchesOrient[currSize]=tmpPatchIdxOrient[i];
	    tmpPatches[currSize]=tmpPatchIdx[i];
	    patchRegions.add(surfList);
	    for (j=0; j<tmpPatchIdx[i].size(); j++) {
		faceI[tmpPatchIdx[i][j]].patchIdx=currSize;
		faceI[tmpPatchIdx[i][j]].patchEntry=j;
	    }
	}
    }

//#if 0
    cerr << "patches.size()="<<patches.size()<<"\n";
    for (i=0; i<patchBdryEdgeFace.size(); i++)
	for (j=0; j<patchBdryEdgeFace[i].size(); j++)
//	    cerr << "patchBdryEdgeFace["<<i<<"]["<<j<<"]="<<patchBdryEdgeFace[i][j]<<"\n";
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
//#endif

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

	Array1<Array1<int> > patchEdgeFaces(edges.size());
	for (j=0; j<tmpPatches[i].size(); j++)
	    for (k=0; k<faceI[tmpPatches[i][j]].edges.size(); k++)
		patchEdgeFaces[faceI[tmpPatches[i][j]].edges[k]].add(tmpPatches[i][j]);

//#if 0
	for (j=0; j<patchEdgeFaces.size(); j++)
	    for (k=0; k<patchEdgeFaces[j].size(); k++)
		cerr << "patchEdgeFaces["<<j<<"]["<<k<<"]="<<patchEdgeFaces[j][k]<<"\n";
//#endif
	// make a list of all the faces on this patch we have to visit

	Array1<int> visited(faces.size());
	for (j=0; j<tmpPatches[i].size(); j++)
	    visited[tmpPatches[i][j]]=0;

//	for (j=0; j<visited.size(); j++)
//	    cerr << "visited["<<i<<"]="<<visited[i]<<"\n";

	for (j=0; j<visited.size(); j++) {

	    // find a remaining face and make a queue from it

	    if (!visited[j]) {
//		cerr << "\n  Visiting node "<<j<<"\n";
		// if this is a new (unallocated) component, allocate it

		if (currPatchIdx == patches.size()) {
		    int currSize=patches.size();
		    patches.resize(currSize+1);
		    patchesOrient.resize(currSize+1);
		    patchesOrient[currSize].resize(patchRegions[i].size());
		    patchBdryEdgeFace.resize(currSize+1);
		    patchBdryEdgeFace[currSize].resize(edgeI.size());
		    patchBdryEdgeFace[currSize].initialize(-1);
		}

		Queue<int> q;
		q.append(j);
		while (!q.is_empty()) {
		    int c=q.pop();

		    // put in the edge-neighbor faces that haven't been visited

		    for (k=0; k<faceI[c].edges.size(); k++) {
			
			// eidx is an edge of the face we're looking at

			int eidx=faceI[c].edges[k];

			if (patchEdgeFaces[eidx].size() == 1) // boundary!
			    patchBdryEdgeFace[currPatchIdx][eidx]=c;
			    
			for (l=0; l<patchEdgeFaces[eidx].size(); l++) {
			    
			    // face[fidx] will store a face neighbor of face[c]

			    int fidx=patchEdgeFaces[eidx][l];
			    if (!visited[fidx]) {
				q.append(fidx);
				visited[fidx]=1;
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
	    }

	    // patchRegions wasn't stored in a temp array - just need to copy
	    // it for each connected component

	    if (currPatchIdx != i)
		patchRegions.add(patchRegions[i]);

	    // set currPatchIdx to point at a new patch - allocation will take
	    // place later, when we find the first triangle of this patch

	    currPatchIdx=patches.size();
	}   
	cerr << "done.\n\n\n";
    }

//#if 0
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
//#endif

    cerr << "Non (-1) patchBdryEdgeFaces...\n";
    for (i=0; i<patchBdryEdgeFace.size(); i++) 
	for (j=0; j<patchBdryEdgeFace[i].size(); j++)
	    if (patchBdryEdgeFace[i][j] != -1) 
		cerr << "patchBdryEdgeFace["<<i<<"]["<<j<<"]="<<patchBdryEdgeFace[i][j]<<"\n";



    cerr << "Done with BuildPatches!!\n";
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
    int i,j;

    cerr << "We have "<<nodes.size()<<" nodes.\n";
    if (!wires.size()) return;

    // make a list of what patches each edge is attached to
    Array1<Array1<int> > allNodeWires(nodes.size());

    for (i=0; i<wires.size(); i++)
	for (j=0; j<wires[i].size(); j++) {
	    allNodeWires[edges[wires[i][j]]->i1].add(i);
	    allNodeWires[edges[wires[i][j]]->i1].add(i);
	}
    int maxMembership=1;

    // sort all of the lists from above, and find the maximum number of
    // wires and node belongs to
    for (i=0; i<allNodeWires.size(); i++)
	if (allNodeWires[i].size() > 1) {
	    if (allNodeWires[i].size() > maxMembership)
		maxMembership=allNodeWires[i].size();
	    order(allNodeWires[i]);
	}

    int sz=pow(wires.size(), maxMembership);
    cerr << "allocating "<<maxMembership<<" levels with "<< wires.size()<< " types (total="<<sz<<").\n";

    // allocate all combinations of the maximum number of wires
    // from the lists of which wires each node belong to,
    // construct a list of nodes which belong for each wire
    // combination

    // a better way to do this is to HASH on each wire --
    // each hashed key should find another hash table, as well as a
    // list of nodes (we'll need a new struct for this).

    Array1 <Array1<int> > tmpTypeMembers(pow(wires.size(), maxMembership));
    cerr << "AllNodeWires.size()="<<allNodeWires.size()<<"\n";
    for (i=0; i<allNodeWires.size(); i++) {
//	cerr << "  **** LOOKING IT UP!\n";
	int idx=getIdx(allNodeWires[i], wires.size());
//	cerr << "this faces has index: "<<idx<<"\n";
	tmpTypeMembers[idx].add(i);
    }

    for (i=0; i<tmpTypeMembers.size(); i++) {

	// if there are any nodes of this combination type...

	if (tmpTypeMembers[i].size()) {

	    // find out what wires there were	

	    Array1<int> wireList;
	    getList(wireList, i, wires.size());
	    int currSize=junctionWires.size();
	    junctions.resize(currSize+1);
	    junctionWires.add(wireList);
	    for (j=0; j<tmpTypeMembers[i].size(); j++)
		junctions[currSize]=tmpTypeMembers[i][j];
	}
    }

    Array1<int> wireVisited(wires.size());
    wireVisited.initialize(0);
    
    for (i=0; i<junctionWires.size(); i++) 
	for (j=0; j<junctionWires[i].size(); j++)
	    wireVisited[junctionWires[i][j]]=1;
    for (i=0; i<wireVisited.size(); i++)
	if (!wireVisited[i]) junctionlessWires.add(i);

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
//    cerr << "Membership.size()="<<allFaceRegions.size()<<"\n";
    for (i=0; i<surfI.size(); i++)
	for (j=0; j<surfI.faces[i].size(); j++)
	    allFaceRegions[surfI[i].faces[j]].add(i);
    int maxMembership=1;

    // sort all of the lists from above, and find the maximum number of
    // surfaces any face belongs to
    for (i=0; i<allFaceRegions.size(); i++)
	if (allFaceRegions[i].size() > 1) {
	    if (allFaceRegions[i].size() > maxMembership)
		maxMembership=allFaceRegions[i].size();
	    order(allFaceRegions[i]);
	}
    int sz=pow(surfEls.size(), maxMembership);
    cerr << "allocating "<<maxMembership<<" levels with "<< surfEls.size()<< " types (total="<<sz<<").\n";

    // allocate all combinations of the maximum number of surfaces
    // from the lists of which surfaces each face belong to,
    // construct a list of faces which belong for each surface
    // combination

    Array1 <Array1<int> > tmpTypeMembers(pow(surfEls.size(), maxMembership));
    cerr << "Membership.size()="<<allFaceRegions.size()<<"\n";
    for (i=0; i<allFaceRegions.size(); i++) {
//	cerr << "  **** LOOKING IT UP!\n";
	int idx=getIdx(allFaceRegions[i], surfEls.size());
//	cerr << "this faces has index: "<<idx<<"\n";
	tmpTypeMembers[idx].add(i);
    }
    typeSurfs.resize(0);
    typeIds.resize(faces.size());
    for (i=0; i<tmpTypeMembers.size(); i++) {
	// if there are any faces of this combination type...
	if (tmpTypeMembers[i].size()) {
	    // find out what surfaces there were	
//	    cerr << "found "<<tmpTypeMembers[i].size()<<" faces of type "<<i<<"\n";
	    Array1<int> surfList;
	    getList(surfList, i, surfEls.size());
	    int currSize=typeSurfs.size();
	    typeSurfs.resize(currSize+1);
	    typeSurfs[currSize].resize(0);
//	    cerr << "here's the array: ";
	    for (j=0; j<tmpTypeMembers[i].size(); j++) {
//		cerr << tmpTypeMembers[i][j]<<" ";
		typeIds[tmpTypeMembers[i][j]]=currSize;
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

    int sz=pow(patches.size(), maxFacesOnAnEdge);
    cerr << "allocating "<<maxFacesOnAnEdge<<" levels with "<< patches.size()<< " types (total="<<sz<<").\n";

    // allocate all combinations of the maximum number of patches
    // from the lists of which patches each edge belong to,
    // construct a list of edges which belong for each patch
    // combination

    // a better way to do this is to HASH on each patch --
    // each hashed key should find another hash table, as well as a
    // list of edges (we'll need a new struct for this).
    // NOTE - we should use this multi-level hashing technique in
    // SegFldToSurfTree when we hash the faces and edges.

    Array1 <Array1<int> > tmpWireIdx(pow(patches.size(), maxFacesOnAnEdge));
    Array1 <Array1 <Array1<int> > > tmpWireIdxOrient(pow(patches.size(), 
							 maxFacesOnAnEdge));
    cerr << "AllEdgePatches.size()="<<allEdgePatches.size()<<"\n";
    for (i=0; i<allEdgePatches.size(); i++) {
	// cerr << "  **** LOOKING IT UP!\n";
	int idx=getIdx(allEdgePatches[i], patches.size());
	// cerr << "this faces has index: "<<idx<<"\n";
	tmpWireIdx[idx].add(i);
	Array1<int> patchList;
	getList(patchList, idx, patches.size());
	tmpWireIdxOrient[idx].resize(patchList.size());

	// for each patch in the list, we have to look up edge i, find its
	// orientation and add it to tmpWireIdxOrient
	for (j=0; j<patchList.size(); j++) {
	    int fidx=patchBdryEdgeFace[j][i];
	    for (k=0; k<faceI[fidx].edges.size(); k++) {
		if (faceI[fidx].edges[k] == i) {
		    tmpWireIdxOrient[idx][j].add(faceI[fidx].edgeOrient[k]);
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

    for (i=0; i<tmpWireIdx.size(); i++) {

	// if there are any faces of this combination type...

	if (tmpWireIdx[i].size()) {

	    // find out what patches there were	

	    Array1<int> patchList;
	    getList(patchList, i, patches.size());
	    int currSize=wirePatches.size();
	    wires.resize(currSize+1);
	    tmpWires.resize(currSize+1);
	    wiresOrient.resize(currSize+1);
	    tmpWiresOrient.resize(currSize+1);
	    tmpWiresOrient[currSize]=tmpWireIdxOrient[i];
	    tmpWires[currSize]=tmpWireIdx[i];
	    wirePatches.add(patchList);
	    for (j=0; j<tmpWireIdx[i].size(); j++) {
		edgeI[tmpWireIdx[i][j]].wireIdx=currSize;
		edgeI[tmpWireIdx[i][j]].wireEntry=j;
	    }
	}
    }

    // now we have to do connectivity searches on each wire -- make sure
    // it's one connected component.  If it isn't, break it into separate
    // patches.

    int numWires=wires.size();
    int currWireIdx;
    for (i=0; i<numWires; i++) {
	currWireIdx=i;

	// make a list of all the edges attached to each node of this wire

	Array1<Array1<int> > wireNodeEdges(nodes.size());
	for (j=0; j<wires[i].size(); j++) {
	    wireNodeEdges[edges[wires[i][j]]->i1].add(wires[i][j]);
	    wireNodeEdges[edges[wires[i][j]]->i2].add(wires[i][j]);
	}

	// make a list of all the edges on this wire we have to visit

	Array1<int> visited(edges.size());
	for (j=0; j<edges.size(); j++) 
	    visited[j]=-1;
	for (j=0; j<tmpWires[i].size(); j++)
	    visited[tmpWires[i][j]]=0;
	for (j=0; j<visited.size(); j++) {

	    // find a remaining edge and make a queue from it

	    if (!visited[j]) {

		// if this is a new (unallocated) component, allocate it

		if (currWireIdx == wires.size()) {
		    int currSize=wires.size();
		    wires.resize(currSize+1);
		    wiresOrient.resize(currSize+1);
		}

		Queue<int> q;
		q.append(j);
		while (!q.is_empty()) {
		    int c=q.pop();

		    // put in the node-neighbor edges that haven't been visited
		    // nidx is a node the face we're looking at

		    int nidx=edges[c]->i1;
		    for (k=0; k<wireNodeEdges[nidx].size(); k++) {
			
			// edge[eidx] will store an edge neighbor of edge[c]
			
			int eidx=wireNodeEdges[nidx][k];

			if (!visited[eidx]) {
			    q.append(eidx);
			    visited[eidx]=1;
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
		    for (k=0; k<wireNodeEdges[nidx].size(); k++) {
			
			// edge[eidx] will store an edge neighbor of edge[c]
			
			int eidx=wireNodeEdges[nidx][k];

			if (!visited[eidx]) {
			    q.append(eidx);
			    visited[eidx]=1;
			    int tmpIdx=edgeI[eidx].wireIdx;
			    int tmpEntry=edgeI[eidx].wireEntry;
			    edgeI[eidx].wireIdx=currWireIdx;
			    edgeI[eidx].wireEntry=wires[currWireIdx].size();
			    wires[currWireIdx].add(
					   tmpWires[tmpIdx][tmpEntry]);
			    wiresOrient[currWireIdx].add(
					   tmpWires[tmpIdx][tmpEntry]);
			}
		    }		    
		}
	    }

	    // wirePatches wasn't stored in a temp array - just need to copy
	    // it for each connected component

	    if (currWireIdx != i)
		wirePatches.add(wirePatches[i]);

	    // set currWireIdx to point at a new wire - allocation will take
	    // place later, when we find the first edge of this wire

	    currWireIdx=wires.size();
	}   
    }
    cerr << "Done with BuildWires!!\n";
}

