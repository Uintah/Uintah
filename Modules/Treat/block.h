#ifndef _BLOCK_    /* include this file 1 time only */

#define _BLOCK_

class Block {

public:

  Block(int myIndex, double myMass, double myStiff);
  ~Block();
  
  // accessor functions
  
  void setIndex(int myIndex) { index = myIndex; }
  void addMass(double myMass) { mass += myMass; }
  void addStiff(double myStiff) { stiff += myStiff; }
  int getIndex() { return index; }
  double getMass() { return mass; }
  double getStiff() { return stiff; }
  
private:
  
  int index;
  double mass;
  double stiff;

};

#endif





