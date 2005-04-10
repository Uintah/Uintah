#ifndef __BOUNDARY_CONDITIONS_H__
#define __BOUNDARY_CONDITIONS_H__

class BoundaryConditions {
 public:
  virtual void loadBoundaryConditions() = 0;
  virtual void displacementBoundaryConditions() = 0;
  virtual void heatFluxBoundaryConditions() = 0;
};


#endif //__BOUNDARY_CONDITIONS_H__


// $Log$
// Revision 1.2  2000/03/15 21:58:20  jas
// Added logging and put guards in.
//
