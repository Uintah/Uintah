#ifndef _SPARSE_    /* include this file 1 time only */

#define _SPARSE_

#include<math.h>
#include "Array1.h"
#include "block.h"
#include "property.h"
#include "node.h"
#include "element.h"

class TreatFEM;

class SparseMatrix {

public:
  
  SparseMatrix(Array1<Node*> &nodeList, Array1<Element*> &elementList, Property property); // constructor
  ~SparseMatrix(); // destructor
  void addIndex(int row, int column);
  int findSparseColumn(int row, int column);
  void addMass(int row, int column, double mass);
  void addStiff(int row, int column, double stiff);
  double getA(int row, int column);
  double getAsparse(int row, int column);
  void writeMatrix(int val);
  void printMatrix(int val);
  void makeRhs(Array1<Node*> &nodeList, int iteration_total);
  void direchlet(Array1<Node*> &nodeList);
  int solveQMR(int maxit, double eps);
  int solveCG(int maxit,double eps);
  bool solveSCIRun(TreatFEM* module);
  void mult(const double* x, double* b);
  void multTran(const double* x, double* b);
  int size() {
      return numberRDOF;
  }
private:
  
  int numberOfRows;
  int numberRDOF;
  Array1<Block*> *matrix;  // matrix is an array of arrays
  double *rhs;
  double *rbcv;
  double *rrhs;
  double theta,tsuba,deltaT;
  int period,timeSteps;
  int ss,symmetric;
  Array1<Node*>& nodeList;

};

#endif
