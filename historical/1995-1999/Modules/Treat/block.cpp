#include"block.h"

// constructor
Block::Block(int myIndex, double myMass, double myStiff) {
  
  index = myIndex;
  mass = myMass;
  stiff = myStiff;
  
}

// destructor
Block::~Block()
{
}
