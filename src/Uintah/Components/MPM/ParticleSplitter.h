#ifndef __PARTICLE_SPLITTER_H__
#define __PARTICLE_SPLITTER_H__

class ParticleSplitter {
  virtual void splitParticles() = 0;
  virtual void mergeParticles() = 0;
};

#endif __PARTICLE_SPLITTER_H__

// $Log$
// Revision 1.2  2000/03/15 21:58:21  jas
// Added logging and put guards in.
//

