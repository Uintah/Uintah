/* main function for fem3dc
   last modified 3 Feb 1997
   
   */

#include<fstream.h>
#include"list.h"
#include "Array1.h"
#include"property.h"
#include"block.h"
#include"node.h"
#include"element.h"
#include"shape.h"
#include"readData.h"
#include"sparseMatrix.h"
#include <stdlib.h>
#include <stdio.h>

void main()
{

  Array1<Node*> nodeList;
  Array1<Element*> elementList;
  Property propertyList;
  ofstream fout;
  int i,j,solve,ss, maxit, num_time_steps, iteration_total, writeEvery;
  double eps,residual,current_residual;
  char str[20];
  
  // ----------------------------------------------------------
  // Begin Program
  // ----------------------------------------------------------

  // ----------------------------------------------------------
  // read data files
  // ----------------------------------------------------------

  cout << "Reading Data Files " << endl;
  if (!readData(nodeList, elementList, propertyList)) {
    cout << "Error in reading data files. Exiting program." << endl;
    exit(1);
  } else {
    cout << "All data files read successfully." << endl;
  }

  cout << "Number of nodes:   " << nodeList.size() << endl;
  cout << "Number of element: " << elementList.size() << endl;
 
  // make element shape functions, attach to Element class
  cout << "Creating element shape functions..." << endl;
  Element::makeShape();
  
  // ----------------------------------------------------------
  // loop through elements and create stiffness for each element
  // ----------------------------------------------------------

  cout << "Creating stiffness for each element." << endl;
  for (i=0;i<elementList.size();i++) {
    elementList[i]->makeStiff();
  }
  
  // ----------------------------------------------------------
  // assemble the global mass and stiffness matrix
  // ----------------------------------------------------------
  
  cout << "Assembling Global System..." << endl;
  SparseMatrix globalSystem(nodeList, elementList, propertyList);
    
  eps =  propertyList.getEPS();
  maxit = nodeList.size();
  num_time_steps = propertyList.getTimeSteps();
  iteration_total = 0;
  ss = propertyList.getSS();
  writeEvery = propertyList.getWriteEvery();

  // ----------------------------------------------------------
  // now march in time or solve for steady state solution
  // ----------------------------------------------------------
  
  do {

    if (!ss) {
      cout << "Working on step: " << iteration_total + 1
	   << " of " << num_time_steps << endl;
    }
    
    // ----------------------------------------------------------
    // Make the right hand side vector
    // ----------------------------------------------------------
    
    cout << "Making Right hand Side..." << endl;
    globalSystem.makeRhs(nodeList,iteration_total);

    // ----------------------------------------------------------
    // Apply Boundary Conditions
    // ----------------------------------------------------------
    
    cout << "Applying boundary Conditions..." << endl;
    globalSystem.direchlet(nodeList);

    // ----------------------------------------------------------
    // Solve Linear System
    // ----------------------------------------------------------
    
    cout << "Solving System..." << endl;
    do {
      if (propertyList.getSymmetric()) {
	solve = globalSystem.solveCG(maxit, eps);
      } else {
	solve = globalSystem.solveQMR(maxit, eps);
      }
    } while ( solve == 3 );
    
    // ----------------------------------------------------------
    // Write A Matrix (left hand side) to file
    // ----------------------------------------------------------

    cout << "\nWriting A matrix... " << endl;
    globalSystem.writeMatrix(2);
    
    // ----------------------------------------------------------
    // Write Solution to File
    // ----------------------------------------------------------

    cout << "\nWriting solution... " << endl;
    if (ss) {
      fout.open("OUTFILE");
    } else {
      if (!(iteration_total % writeEvery)) {
	sprintf(str,"OUTFILE.%d",iteration_total+1);
	fout.open(str);
      }
    }
    if (ss || !(iteration_total % writeEvery)) {
      for (i=0;i<nodeList.size();i++) {
	fout << i+1 << " " <<  nodeList[i]->getTemp() << endl;
      }
    }
    fout.close();

    // ----------------------------------------------------------
    // compute residual for transient cases
    // ----------------------------------------------------------

    if (!ss) {
      residual = 0;
      for (i=0;i<nodeList.size();i++) {
	current_residual = fabs(nodeList[i]->getTemp() -
				nodeList[i]->getTempPast());
	if (fabs(nodeList[i]->getTemp()) > 0) {
	  current_residual = current_residual/fabs(nodeList[i]->getTemp());
	}
	if (current_residual > residual) residual = current_residual;
	nodeList[i]->setTempPast(nodeList[i]->getTemp());
      }
    }
    
    iteration_total ++;

    if (!ss) {
      cout << "residual: " << residual << endl;
    }
    
  } while ( (!ss) && (iteration_total < num_time_steps) && (residual > eps) );

  cout << "\n\nDone with FEM " << endl;
  
} // end of main






