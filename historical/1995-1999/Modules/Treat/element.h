#ifndef _ELEMENT_    /* include this file 1 time only */

#define _ELEMENT_

#include"node.h"

#define Element tElement

class Element
{
public:

  // member functions in other files
  static void makeShape();
  void makeStiff();

  // accessor member functions
  
  inline double getN(const int i, const int j) const {
    return N[i][j];
  }
  inline double getdNds(const int i, const int j) const {
    return dNds[i][j];
  }
  inline double getdNdt(const int i, const int j) const {
    return dNdt[i][j];
  }
  inline double getdNdu(const int i, const int j) const {
    return dNdu[i][j];
  }
  void setNode(const int i, Node *mynode) { 
    node[i]=mynode;
  }
  Node *getNode(const int i) {
    return node[i];
  }
  void setPerf(double val) { perf = val; }
  void setAlpha(double val) { alpha = val; }
  void setMassStiff(int i, int j, double val) { massStiff[i][j] = val; }
  void setThermStiff(int i, int j, double val) { thermStiff[i][j] = val; }
  double getPerf() const { return perf; }
  double getAlpha() const { return alpha; }
  double getMassStiff(int i, int j) { return massStiff[i][j]; }
  double getThermStiff(int i, int j) { return thermStiff[i][j]; }
  double getVolume() { return volume; }

private:
  
  static double N[8][8];
  static double dNds[8][8];
  static double dNdt[8][8];
  static double dNdu[8][8];
  Node *node[8];
  
  double perf;
  double alpha;
  double massStiff[8][8];
  double thermStiff[8][8];
  double volume;
  
};

#endif











