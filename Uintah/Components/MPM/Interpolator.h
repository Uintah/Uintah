#ifndef __INTERPOLATOR_H__
#define __INTERPOLATOR_H__

class Interpolator {
 public:
  static void fromGrid(ParticleSet &p);
  static void toGrid(ParticleSet &p);
  static void div();
  static void grad();


};

#endif __INTERPOLATOR_H__

// $Log$
// Revision 1.1  2000/03/15 21:58:21  jas
// Added logging and put guards in.
//
