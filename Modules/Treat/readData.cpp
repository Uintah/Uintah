#include<fstream.h>
#include<math.h>
#include"list.h"
#include "Array1.h"
#include"node.h"
#include"element.h"
#include"property.h"
#include <stdlib.h>

int Node::numberOfBCS = 0;

int readData(Array1<Node*> &nodeList, Array1<Element*> &elementList,
	     Property &propertyList, char* sarcname ) {
  
// returns 0 on error

  double val[8];
  unsigned long dum;
  int valint[8];
  ifstream fin;
  int i,j;
  int direchlet, reduced;
  char dummy[20];

  // read in CONTROL file

  cout << "CONTROL file..." << endl;
  fin.open("CONTROL");
  if (!fin) {
    cout << "Unable to open CONTROL for reading.\n";
    return(0);
  }
  fin >> dummy >> valint[0];
  propertyList.setSS(valint[0]);
  fin >> dummy >> valint[0];
  propertyList.setSymmetric(valint[0]);
  fin >> dummy >> val[0];
  propertyList.setTsubA(val[0]);
  fin >> dummy >> val[0];
  propertyList.setTheta(val[0]);
  fin >> dummy >> val[0];
  propertyList.setDeltaT(val[0]);
  fin >> dummy >> valint[0];
  propertyList.setTimeSteps(valint[0]);
  fin >> dummy >> valint[0];
  propertyList.setWriteEvery(valint[0]);
  fin >> dummy >> valint[0];
  propertyList.setPeriod(valint[0]);
  fin >> dummy >> val[0];
  propertyList.setEPS(val[0]);

  fin.close();
  
  // read in node file

  cout << "NODES file..." << endl;
  fin.open("NODES");
  if (!fin) {
    cout << "Unable to open NODES for reading.\n";
    return(0);
  }
  while(1) { 
    fin >> dum >> val[0] >> val[1] >> val[2];
    if (fin.eof()) break;
    nodeList.add(new Node); // add this Node to the linked list "node"
  }
  fin.close();
  fin.open("NODES");
  i=0;
  while(1) {
    fin >> dum >> val[0] >> val[1] >> val[2];
    if (fin.eof()) break;
    nodeList[i]->setNum(dum-1);
    nodeList[i]->setX(val[0]); 
    nodeList[i]->setY(val[1]);
    nodeList[i]->setZ(val[2]);
    nodeList[i]->setBoundType(-1); // initializes boundary value
    i++;
  }
  fin.close();
  
  // read in SARCF file ########################

  cout << "SARCF file..." << endl;
  fin.open(sarcname);
  if (!fin) {
      cout << sarcname << "\n";
    cout << "Unable to open SARCF for reading.\n";
    return(0);
  }
  i=0;
  for (j=0;j<nodeList.size();j++) {
    fin >> dum >> val[0];
    nodeList[i]->setSarc(val[0]);
    i++;
  }
  fin.close();

  // read in OUTFILE file #######################

  cout << "OUTFILE file..." << endl;
  fin.open("OUTFILE");
  if (!fin) {
    cout << "Unable to open OUTFILE for reading.\n";
    return(0);
  }
  i=0;
  for (j=0;j<nodeList.size();j++) {
    fin >> dum >> val[0];
    val[0]=0;
    nodeList[i]->setTemp(val[0]);
    nodeList[i]->setTempPast(val[0]);
    i++;
  }
  fin.close();

  // read in VELOCITY file #######################

  cout << "VELOCITY file..." << endl;
  fin.open("VELOCITY");
  if (!fin) {
    cout << "Unable to open VELOCITY for reading.\n";
    return(0);
  }
  i=0;
  for (j=0;j<nodeList.size();j++) {
    fin >> dum >> val[0] >> val[1] >> val[2];
    nodeList[i]->setVX(val[0]);
    nodeList[i]->setVY(val[1]);
    nodeList[i]->setVZ(val[2]);
    i++;
  }
  fin.close();

  // read in BOUNDF file ##########################

  cout << "BOUNDF file.." << endl;
  fin.open("BOUNDF");
  if (!fin) {
    cout << "Unable to open BOUNDF for reading.\n";
    return(0);
  }
  i=0;
  while(1) {
    fin >> dum >> valint[0] >> val[0];
    if (fin.eof()) break;
    if (dum > nodeList.size() ) {
      cout << "Illegal Boundary Value:  " << dum << endl;
      cout << "Node set upper bound of: " << nodeList.size() - 1 << endl;
      return(0);
    }
    nodeList[dum-1]->setBoundType(valint[0]);
    nodeList[dum-1]->setBoundVal(val[0]);
    nodeList[dum-1]->setTemp(val[0]);
    nodeList[dum-1]->setTempPast(val[0]);
    i++;
  }
  fin.close();

  // make global to reduced degree of freedom mapping here
  direchlet = -1;
  reduced = 0;
  for (int ii=0;ii<nodeList.size();ii++) {
    if (nodeList[ii]->getBoundType() < 0 ) {
      nodeList[ii]->setRdof(reduced++);
    } else {
      nodeList[ii]->setRdof(direchlet--);
    }
  }
  Node::numberOfBCS = -(direchlet+1);
  
  
  // read in ELEMS file ###########################

  cout << "ELEMS file..." << endl;
  fin.open("ELEMS");
  if (!fin) {
    cout << "Unable to open ELEMS for reading.\n";
    return(0);
  }
  while(1) {
    fin >> dum >> valint[0] >> valint[1] >> valint[2] >> valint[3] >>
      valint[4] >> valint[5] >> valint[6] >> valint[7];
    if (fin.eof()) break;
    elementList.add(new Element);
  }
  fin.close();
  fin.open("ELEMS");
  i=0;
  while(1) {
    fin >> dum >> valint[0] >> valint[1] >> valint[2] >> valint[3] >>
      valint[4] >> valint[5] >> valint[6] >> valint[7];
    if (fin.eof()) break;
    if ( valint[7] > 0) { // hex elements
      for (j=0;j<8;j++) {
	elementList[i]->setNode(j,nodeList[valint[j]-1]);
      }
    } else { // tetrahedral elements
      elementList[i]->setNode(0,nodeList[valint[0]-1]);
      elementList[i]->setNode(1,nodeList[valint[0]-1]);
      elementList[i]->setNode(2,nodeList[valint[0]-1]);
      elementList[i]->setNode(3,nodeList[valint[0]-1]);
      elementList[i]->setNode(4,nodeList[valint[1]-1]);
      elementList[i]->setNode(5,nodeList[valint[1]-1]);
      elementList[i]->setNode(6,nodeList[valint[2]-1]);
      elementList[i]->setNode(7,nodeList[valint[3]-1]);
    }
    i++;
  }
  fin.close();

  // read in ALPHA file ###########################

  cout << "ALPHAF file.." << endl;
  fin.open("ALPHAF");
  if (!fin) {
    cout << "Unable to open ALPHAF for reading.\n";
    return(0);
  }
  i=0;
  for (j=0;j<elementList.size();j++) {
    fin >> dum >> val[0];
    elementList[i]->setAlpha(val[0]);
    i++;
  }
  fin.close();


  // read in PERFF file ##########################

  cout << "PERFF file.." << endl;
  fin.open("PERFF");
  if (!fin) {
    cout << "Unable to open PERFF for reading.\n";
    return(0);
  }
  i=0;
  for (j=0;j<elementList.size();j++) {
    fin >> dum >> val[0];
    elementList[i]->setPerf(val[0]);
    i++;
  }
  fin.close();

  return(1);

}
