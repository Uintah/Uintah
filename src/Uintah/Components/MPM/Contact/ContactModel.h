#ifndef __CONTACT_MODEL_H__
#define __CONTACT_MODEL_H__

class ContactModel {
 public:
  void computeSurfaceNormals();
  void computeTractions();
  virtual void exchangeMomentumInterpolated() = 0;
  virtual void exchangeMomentumIntegrated() = 0;
};

#endif __CONTACT_MODEL_H__

// $Log$
// Revision 1.2  2000/03/15 21:58:22  jas
// Added logging and put guards in.
//
