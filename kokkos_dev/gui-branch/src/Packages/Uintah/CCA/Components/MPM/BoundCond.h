#ifndef __BOUND_COND_H__
#define __BOUND_COND_H__

class BoundCond {
private:

  int boundaryObject,boundaryPiece,boundarySurf;
  double boundaryForce[4];

public:

  BoundCond();
  ~BoundCond();

  void setBC(int obj,int piece,int surf,double force[4]);
  int	getBCObj();
  int	getBCPiece();
  int	getBCSurf();
  double getBCForce(int i);

};

#endif // __BOUND_COND_H__

