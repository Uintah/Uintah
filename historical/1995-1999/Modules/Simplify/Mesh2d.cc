/*
 * 2d mesh class - manifold surface
 * Peter-Pike Sloan
 */

#include "Mesh2d.h"

#include <iostream.h>

// this does cyclic rempaing for vertices...

static int vremap[6] = {0,1,2,0,1,2};

static int oppMask[8][3] = { {-1,-1,-1}, // 0 is invalid...
			     {0,1,2},
			     {0,2,1},
			     {1,0,2},
			     {1,2,0},
			     {2,0,1},
			     {2,1,0},
			     {-1,-1,-1}, };

void Mesh2d::GetFaceRing(int nid, Array1<int> &res)
{
  int curFace = nodeFace[nid];
  int lastFace = curFace;
  int finalFace=lastFace;

  int done=0;

  // ok, lets make sure we are in this face!
  
  for(int ii=0;ii<3;ii++) {
    if (topology[curFace].v[ii] == nid)
      done=1;
  }
  
  if (!done) {
    cerr << nid << " Face ring sucks!\n";
    return;
  }

  done=0;

  while(!done) {
    res.add(curFace);

    // go to a neighbor...
    done=1;
    int bdry=0;
    for(int i=0;i<3;i++) {
      if (topology[curFace].v[i] != nid) {
	// any neighbor is ok, as long as it
	// isn't the last face, and doesn't
	// point to -1 boundary curve...
	int testFace = topology[curFace].f[i];
	if (testFace > -1){ /* can't be a boundary */
	  if ((testFace != lastFace) &&  /* can't go into last face */
	      (testFace != finalFace)) { /* if we get here we are done */
	    lastFace = curFace;
	    curFace = testFace;
	    i=4;
	    done=0; // go into next face...
	  }
	} else {
	  bdry=1;
	}
      }
    }

    if (done && bdry) { // we ran into a boundary?
    
      curFace = finalFace;
      // start back from the other spot...
      for(i=2;i>=0;i--) {
	if (topology[curFace].v[i] != nid) {

	  int testFace = topology[curFace].f[i];

	  if ((testFace > -1) && /* can't be a boundary */
	      (testFace != lastFace) &&  /* can't go into last face */
	      (testFace != finalFace)) { /* if we get here we are done */
	    lastFace = curFace;
	    curFace = testFace;
	    i=-1;
	    done=0; // go into next face...
	    // make sure it isn't already here...

	    for(int j=0;j<res.size();j++) {
	      if (res[j] == curFace) {
		done =1;
		j = res.size() + 1;// break out
	      }
	    }

	  }
	  
	}
      }
    }

  }
}

void Mesh2d::GetFaceRing2(int nid, Array1<int> &res)
{
  int curFace = nodeFace[nid]; // start
  int lastFace = curFace;

  int done=0;

  int ringI=-1;

  for(int ii=0;ii<3;ii++) { // this could also be stored...
    if (topology[curFace].v[ii] == nid)
      ringI = ii;
  }

  static int startMaps[2][2] = { {2,1}, {1,2} };

  int oppI=vremap[ringI+2];
  int checkI = vremap[ringI+1];
#if 0
  if (oppI == checkI) {
    cerr << "Bad from Start!\n";
    return;
  }
  if ((oppI == ringI) || (checkI == ringI)) {
    cerr << ringI << " " << oppI << " " << checkI << " WOah2 - bad from start!\n";
    return;
  }
#endif

  int bdryHit=0;

  res.add(curFace); // add the first one...
  while (!done) {
    // see if you have finished...

    if (topology[curFace].f[checkI] == -1) {
      if (bdryHit)
	return; // come back full circle...

      bdryHit = 1;
      curFace = lastFace; // start over...

      oppI = vremap[ringI + 1];
      checkI = vremap[ringI + 2];

      //cerr << nid << " Flip ->\n";

    } else {
      if (topology[curFace].f[checkI] == lastFace) {
	// full circle!  Just return...
	return;
      }

      // this is the base case

      // get the remap info...
      // since we are making no assumptions about ordering
      // we have to check 2 vertices...

      int remapID = topology[curFace].getFlags(checkI);

      curFace = topology[curFace].f[checkI]; // bump the face...

      res.add(curFace); // add the new face...

      // oppI is next checkI, index 3 is next oppI
      // oppI is either +1 or +2

      // it can be 1,-1,2 or -2
      // 1 and -2 are +1, -1 and 2 are + 2

      static int oppIRemap[5] = {0,1,-1,0,1};

      int deltI = (oppI - checkI);
      int dval = oppIRemap[deltI + 2];
#if 0
      if (deltI == 0) { 
	cerr << nid << " " << remapID << " Woah, dval is zero!\n";
	return;
      }
#endif
      checkI = oppMask[remapID][dval];
      oppI = oppMask[remapID][2]; // always opposite vertex...
#if 0
      if ((topology[curFace].v[checkI] == nid) || 
	  (topology[curFace].v[oppI] == nid)) {
	cerr << nid << " " << topology[curFace].v[checkI];
	cerr << " " << deltI << " Woah - using nid?\n";
	return;
      }
#endif
    }
  }

}

int FindListIntersect(Array1<int> &la, Array1<int> &lb, int me)
{
  int ca=0,cb=0;

  while((ca < la.size()) && (cb < lb.size())) {
    if ((la[ca] == lb[cb]) && (la[ca] != me))
      return la[ca];

    if (la[ca] < lb[cb]) {
      ca++; // go to next index
    } else {
      cb++; // b was smaller - so increment it...
    }
  }
  
  return -1;
}

void Mesh2d::CreateFaceNeighbors()
{
  // all we have are vertex ids - build the
  // face neighbors now...

  for(int i=0;i<topology.size();i++) {
    for(int j=0;j<3;j++) {
      nodeFaces[topology[i].v[j]].add(i);
      nodeFace[topology[i].v[j]] = i;
    }
  }

  // above built the face rings for each node...

  for(i=0;i<topology.size();i++) {
    for(int j=0;j<3;j++) {
      if (topology[i].f[j] == -2) {
	int nbr = FindListIntersect(nodeFaces[topology[i].v[vremap[j+1]]],
				    nodeFaces[topology[i].v[vremap[j+2]]],
				    i); // can't be yourself...
	topology[i].f[j] = nbr;
      } else {
	cerr << "Face already done!\n";
      }
    }
  }

  // you also have to compute the bitflags using the face neighbors...
#if 0
  for(i=0;i<topology.size();i++) {
    for(int j=0;j<3;j++) {
      int nbr = topology[i].f[j];
      if (nbr != -1) { // there is a neighbor...
	int p1 = topology[i].v[vremap[j+1]];
	int p2 = topology[i].v[vremap[j+2]];

	int p1r,p2r;

	for(int k=0;k<3;k++) {
	  if (topology[nbr].v[k] == p1)
	    p1r = k;
	  if (topology[nbr].v[k] == p2)
	    p2r = k;
	}

	int id = p1r*2 +1; // sets the base
	if (p1r?(p2r>0):(p2r>1)) id++; // increments if neccesary...

	if (!id || (id > 6)) {
	  cerr << "Bad ID!\n";
	}

	topology[i].setFlags(j,id);

      }
    }
  }
#endif
  Validate();

}

void Mesh2d::Validate() 
{
  // now validate the shit

  Array1<int> test;
  int nbadsize=0;
  int nbadmatch=0;
  int badfaceinring=0;

  cerr << "Validating...\n";

  for(int i=0;i<nodeFace.size();i++) {
    if (pt_flags[i] == NODE_IS_VALID) {
      test.resize(0);
      GetFaceRing(i,test);
#if 0      
      if (test.size() != nodeFaces[i].size()) {
	nbadsize++;
      } else {
	int rdone=0;
	for(int j=0;j<test.size();j++) {
	  int havematch=0;
	  for(int k=0;k<nodeFaces[i].size();k++) {
	    if (test[j] == (nodeFaces[i])[k]) havematch = 1;
	  }
	  if (!havematch)
	    rdone = 1;
	}
	
	if (rdone)
	  nbadmatch++;
      }
#else
      // make sure all of the ring have this vertex...

      int ngood=0;
      for(int j=0;j<test.size();j++) {
	if (topology[test[j]].flags) {
	  for(int k=0;k<3;k++) {
	    if (topology[test[j]].v[k] == i)
	      ngood++;
	  }
	} else {
	  badfaceinring++;
	}
      }

      if (ngood != test.size())
	nbadsize++;
#endif
    }
  }

  // also look for face neighbors...

  int ndeadfaceneighbors=0;
  int nbadfaceneighbors=0;
  int faceusingbadvert=0;

  int badbits=0;

#if 0

  for(i=0;i<topology.size();i++) {
    if (topology[i].flags&ELEM_IS_VALID) {
      for(int j=0;j<3;j++) {
	int wf = topology[i].f[j];
	if (wf != -1 && 
	    !(topology[wf].flags&ELEM_IS_VALID)) {
	  ++ndeadfaceneighbors;
	} else {
	  if (wf != -1) { // find this guy...
	    int foundme=0;
	    for(int k=0;k<3;k++) {
	      foundme += (topology[wf].f[k] == i);
	    }
	    if (foundme != 1) {
	      ++nbadfaceneighbors;
	      cerr << foundme << " badfaceneighbor " << wf << " " << i << endl;
	    }
	  }
	}

	if (pt_flags[topology[i].v[j]] == 0) {
	  ++faceusingbadvert;
	  cerr << i << " Using bad vert! " << topology[i].v[j] << endl;
	}

	// also check the bitflags...
#if 0
	if (wf != -1) {
	  int flgs = topology[i].getFlags(j);

	  //cerr << i << " " << flgs << " Validate...\n";

	  if (!flgs | flgs > 6) {
	    cerr << i << " Bad Flags!\n";
	  }

	  if (topology[wf].v[oppMask[flgs][0]] != topology[i].v[vremap[j+1]]) {
	    cerr << "Woah bad mojo!\n";
	  }

	  if (topology[wf].v[oppMask[flgs][1]] != topology[i].v[vremap[j+2]]) {
	    cerr << "Woah bad mojo 2!\n";
	  }
	  
	}
#endif

      }
    }
  }
#endif
  cerr << nbadsize << " " << badfaceinring << endl;
  cerr << ndeadfaceneighbors << " " << nbadfaceneighbors << endl;
  cerr << " out of: " << nodeFace.size() << endl;

}

void Mesh2d::Init(TriSurface* ts, int DoConnect)
{
  points = ts->points; // just copy the points...

  MeshTop2d cur;
  cur.f[0] =   cur.f[1] =   cur.f[2] = -2;
  cur.flags = ELEM_IS_VALID;
  
  topology.resize(0); // zero everything out...

  pt_flags.resize(points.size());
  pt_flags.initialize(1);

  for(int i=0;i<ts->elements.size();i++) {
    if (ts->elements[i]) {
      
      cur.v[0] = ts->elements[i]->i1;
      cur.v[1] = ts->elements[i]->i2;
      cur.v[2] = ts->elements[i]->i3;

      topology.add(cur);
    }
  }

  nodeFace.resize(points.size());
  nodeFaces.resize(points.size()); // also clear out
  for(i=0;i<points.size();i++) nodeFaces[i].resize(0);

  if (DoConnect)
    CreateFaceNeighbors(); // generate the connectivity...
}

void Mesh2d::Dump(TriSurface *ts)
{
  ts->points.resize(0);
  ts->elements.resize(0);

  // add the points
  
  Array1<int> pt_remap;
  pt_remap.setsize(points.size());

  for(int i=0;i<points.size();i++) {
    if (pt_flags[i] == NODE_IS_VALID) {
      pt_remap[i] = ts->points.size();
      ts->points.add(points[i]);
    }
  }

  for(i=0;i<topology.size();i++) {
    if (topology[i].flags) { // face is valid...
      ts->elements.add(new TSElement(pt_remap[topology[i].v[0]],
				     pt_remap[topology[i].v[1]],
				     pt_remap[topology[i].v[2]]));
    }
  }
}

// angleweighted or just "averaged"

void Mesh2d::ComputeNormals(int angleweighted)
{
  cerr << "Starting normals...\n";

  fnormals.setsize(topology.size());
  vnormals.setsize(points.size());

  double maxv=-1;
  for(int i=0;i<topology.size();i++) {
    Vector v0,v1;

    v0 = points[topology[i].v[1]] - points[topology[i].v[0]];
    v1 = points[topology[i].v[2]] - points[topology[i].v[0]];

    fnormals[i] = Cross(v0,v1);
    double val = fnormals[i].normalize();
    if (val > maxv) maxv = val;
  }

  cerr << maxv << " Done with face normals!\n";
  maxv=-1;
  Array1<int> faces;
  for(int j=0;j<points.size();j++) {
    GetFaceRing(j,faces);
    
    if (faces.size()) {

      Vector fv(0,0,0);
      
      for(i=0;i<faces.size();i++) {
	fv += fnormals[faces[i]];
      }
      double val = fv.normalize();
      vnormals[j] = fv;
      if (val > maxv) maxv = val;
      faces.resize(0);// clear it out...
    } else {
      vnormals[j] = Vector(0,0,1);
    }
  }
  cerr << maxv << " Vertex normals finished!\n";
}
