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
 *  CVRTItoTetVolFieldPot.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TetVolField.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/Array1.h>
#include <iostream>
#include <fstream>
#include <stack>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;
using std::stack;

using namespace SCIRun;

void sort_and_add(int a, int b, int c, int d, Array1<int>& sorted) {
  sorted.resize(4);
  sorted[0]=a; sorted[1]=b; sorted[2]=c; sorted[3]=d;
  int i, j, tmp;
  for (i=0; i<3; i++) {
    for (j=i+1; j<4; j++) {
      if (sorted[i] > sorted[j]) {
	tmp = sorted[i]; sorted[i]=sorted[j]; sorted[j]=tmp;
      }
    }
  }
}

int
main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: "<<argv[0]<<" OPXJfile fieldout\n";
    return 0;
  }

  ifstream obxj(argv[1]);
  char *buff = scinew char[500];
  int nnodes=0;
  int nelems=0;

  char* element = scinew char[15];
  char* Xi = scinew char[15];
  char* Xj = scinew char[15];
  char* values = scinew char[15];
  int nodeIdx, elemIdx;
  double lx, ly, lz, gx, gy, gz, val;
  int i, j;
  Array1<Array1<Point> > tmpNodes;
  Array1<Array1<double> > tmpVals;
  buff[0]=0;
  obxj.getline(buff, 500);
  while (buff[0] != '[') {
    if (buff[0] == 0) { obxj.getline(buff, 500); continue; }
    if (buff[1] == 'E') {
      if (4 != sscanf(buff, "%s %d %s %s", element, &elemIdx, Xj, values)) {
	cerr << "Error reading element "<<nelems+1<<" -- "<<buff<<"\n";
	return 0;
      }
      tmpNodes.resize(nelems+1);
      tmpVals.resize(nelems+1);
      for (i=1; i<9; i++) {
	obxj.getline(buff, 500);
	if (10 != sscanf(buff, "%d %s %lf %lf %lf %s %lf %lf %lf %lf", &nodeIdx, Xi, &lx, &ly, &lz, Xj, &gx, &gy, &gz, &val)) {
	  cerr << "Error reading element "<<nelems+1<<", node "<<i<<" -- "<<buff<<"\n";
	  return 0;
	}
	tmpNodes[nelems].add(Point(gz, gy, gz));
	tmpVals[nelems].add(val);
      }
      Point tmpP;
      double dP;

      tmpP = tmpNodes[nelems][2]; dP = tmpVals[nelems][2];
      tmpNodes[nelems][2]=tmpNodes[nelems][3]; 
      tmpVals[nelems][2]=tmpVals[nelems][3];
      tmpNodes[nelems][3]=tmpP; tmpVals[nelems][3]=dP;

      tmpP = tmpNodes[nelems][6]; dP = tmpVals[nelems][6];
      tmpNodes[nelems][6]=tmpNodes[nelems][7]; 
      tmpVals[nelems][6]=tmpVals[nelems][7];
      tmpNodes[nelems][7]=tmpP; tmpVals[nelems][7]=dP;
    }
    nelems++;
    obxj.getline(buff, 500);
  }
  cerr << "Read "<<nelems<<" elements.\n";

  buff[0] = ' ';
  int n1, n2, n3, n4, n5, n6, n7, n8;
  int nelems2 = 0;
  Array1<Array1<int> > nodeList;
  while (buff[1] == '[') {
    buff[1] = ' ';
    if (8 != sscanf(buff, "%d %d %d %d %d %d %d %d", &n1, &n2, &n3, &n4, &n5, &n6, &n7, &n8)) {
      cerr << "Error reading node list -- "<<buff<<"\n";
      return 0;
    }
    if (n1 == 0) break;
    nodeList.resize(nelems2+1);
    nodeList[nelems2].add(n1-1);
    nodeList[nelems2].add(n2-1);
    nodeList[nelems2].add(n4-1);
    nodeList[nelems2].add(n3-1);
    nodeList[nelems2].add(n5-1);
    nodeList[nelems2].add(n6-1);
    nodeList[nelems2].add(n8-1);
    nodeList[nelems2].add(n7-1);
    nelems2++;
    obxj.getline(buff, 500);
  }
  if (nelems != nelems2) {
    cerr << "Error - found "<<nelems2<<" nodes lists for "<<nelems<<" elements.\n";
    return 0;
  }

  cerr << "Got all of the nodes.\n";
  for (i=0; i<nelems; i++) {
    for (j=0; j<8; j++) {
      if (nodeList[i][j] > nnodes) nnodes = nodeList[i][j];
    }
  }
  nnodes++;

  cerr << "There were "<<nnodes<<" distinct nodes.\n";
  Array1<Point> nodes(nnodes);
  Array1<double> vals(nnodes);
  for (i=0; i<nelems; i++) {
    for (j=0; j<8; j++) {
      nodes[nodeList[i][j]] = tmpNodes[i][j];
      vals[nodeList[i][j]] = tmpVals[i][j];
    }
  }

  int yy,zz;
  for (yy=0; yy<nodes.size()-1; yy++) {
    for (zz=yy+1; zz<nodes.size(); zz++) {
      if ((nodes[yy]-nodes[zz]).length2() < 0.0000001) {
	cerr << "ERROR -- NODES "<<yy<< " and "<<zz<<" are the same!\n";
      }
    }
  }

  Array1<int> parity(nelems);
  Array1<int> visited(nelems);
  visited.initialize(0);
  parity.initialize(-1);
  Array1<Array1<Array1<int> > > cell_faces(nelems);
  for (i=0; i<nelems; i++) {
    cell_faces[i].resize(6);
    sort_and_add(nodeList[i][0], nodeList[i][1], 
		 nodeList[i][2], nodeList[i][3], cell_faces[i][0]);
    sort_and_add(nodeList[i][4], nodeList[i][5], 
		 nodeList[i][6], nodeList[i][7], cell_faces[i][1]);
    sort_and_add(nodeList[i][0], nodeList[i][2], 
		 nodeList[i][4], nodeList[i][6], cell_faces[i][2]);
    sort_and_add(nodeList[i][1], nodeList[i][3], 
		 nodeList[i][5], nodeList[i][7], cell_faces[i][3]);
    sort_and_add(nodeList[i][0], nodeList[i][1], 
		 nodeList[i][4], nodeList[i][5], cell_faces[i][4]);
    sort_and_add(nodeList[i][2], nodeList[i][3], 
		 nodeList[i][6], nodeList[i][7], cell_faces[i][5]);
  }

  cerr << "Here are the cells:\n";
  int cell, face, node;
  for (cell=0; cell<cell_faces.size(); cell++) {
    cerr << "  Cell "<<cell<<"\n";
    for (face=0; face<6; face++) {
      cerr <<"    Face "<<face<<": ";
      for (node=0; node<4; node++) {
	cerr <<cell_faces[cell][face][node]<<" ";
      }
      cerr << "\n";
    }
  }

  parity[0] = 0;
  stack<int> s;
  s.push(0);
  cerr << "pushed 0 -- parity=0 \n";
  while(!s.empty()) {
    int curr_elem = s.top(); 
    s.pop();
    if (visited[curr_elem]) continue;
    int nbr_parity = !parity[curr_elem];
    cerr << "popped "<<curr_elem<<" -- nbr parity="<<nbr_parity<<"\n";
    visited[curr_elem] = 1;
    for (i=0; i<6; i++) {
      for (j=0; j<nelems; j++) {
	if (curr_elem==j || visited[j]) continue;

	for (int k=0; k<6; k++) {
	  int l=0;
	  for (; l<4; l++) {
	    if (cell_faces[curr_elem][i][l] != cell_faces[j][k][l]) break;
	  }
	  if (l==4) {
	    if (parity[j] == -1) {
	      parity[j] = nbr_parity;
	      cerr << "pushing nbr elem "<<j<<" w/ parity "<<nbr_parity<<"\n";
	      s.push(j);
	    } else if (parity[j] != nbr_parity) { // paranoid check
	      cerr << "Error -- got different parities for element "<<j<<"\n";
	      return 0;
	    } // else this is a neighbor we've already pushed - check next face
	  }
	}
      }
    }
  }
  for (i=0; i<nelems; i++) {
    if (visited[i] == 0 || parity[i] == -1) {
      cerr << "Error -- didn't set parity (or visit) all elements.\n";
      return 0;
    }
  }

  TetVolMesh *tvm = new TetVolMesh();
  for (i=0; i<nnodes; i++) tvm->add_point(nodes[i]);

  for (i=0; i<nelems; i++) {
    if (parity[i]) {
      tvm->add_tet(nodeList[i][0], nodeList[i][1], 
		   nodeList[i][2], nodeList[i][5]);
      tvm->add_tet(nodeList[i][0], nodeList[i][2], 
		   nodeList[i][3], nodeList[i][7]);
      tvm->add_tet(nodeList[i][0], nodeList[i][2], 
		   nodeList[i][5], nodeList[i][7]);
      tvm->add_tet(nodeList[i][0], nodeList[i][4], 
		   nodeList[i][5], nodeList[i][7]);
      tvm->add_tet(nodeList[i][2], nodeList[i][5], 
		   nodeList[i][6], nodeList[i][7]);
    } else {
      tvm->add_tet(nodeList[i][1], nodeList[i][0], 
		   nodeList[i][3], nodeList[i][4]);
      tvm->add_tet(nodeList[i][1], nodeList[i][3], 
		   nodeList[i][2], nodeList[i][6]);
      tvm->add_tet(nodeList[i][1], nodeList[i][3], 
		   nodeList[i][4], nodeList[i][6]);
      tvm->add_tet(nodeList[i][1], nodeList[i][5], 
		   nodeList[i][4], nodeList[i][6]);
      tvm->add_tet(nodeList[i][3], nodeList[i][4], 
		   nodeList[i][7], nodeList[i][6]);
    }
  }

  TetVolMeshHandle tvmH(tvm);
  TetVolField<double> *tv = scinew TetVolField<double>(tvmH, Field::NODE);

  for (i=0; i<nnodes; i++)
    tv->fdata()[i]=vals[i];
  FieldHandle tvH(tv);

  TextPiostream out_stream(argv[1], Piostream::Write);
  Pio(out_stream, tvH);
  return 0;
}
