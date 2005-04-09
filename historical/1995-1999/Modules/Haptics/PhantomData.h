/*
 * PhantomData.h:  Contains the PhantomXYZ and PhantomUVW classes inherited
 *		   from the VoidStar class.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Modules_Haptics_PhantomData_h
#define SCI_Modules_Haptics_PhantomData_h 1

#include <Datatypes/VoidStar.h>

// Phantom Position class definition
// based heavily on dweinste's VoidStar.cc in his own work/Datatypes dir
// ImageXYZ

class PhantomXYZ : public VoidStar {
public: 
    Vector position;
    Semaphore sem;
    Semaphore Esem;  // ideally I would have a registry -- anyone who wanted
             // to receive position could sign up. Hack for now: each one
          // gets its own hard-coded semaphore LATER
    CrowdMonitor updateLock; 
public:
    PhantomXYZ();
    PhantomXYZ(const PhantomXYZ& copy);
    PhantomXYZ(const Vector& xyz);
    virtual  ~PhantomXYZ();
    virtual  VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

// Phantom Force class definition
// based heavily on dweinste's VoidStar.cc in his own work/Datatypes dir
// ImageXYZ

class PhantomUVW : public VoidStar {
public: 
    Vector force;
    Semaphore sem;
    CrowdMonitor updateLock; 
public:
    PhantomUVW();
    PhantomUVW(const PhantomUVW& copy);
    PhantomUVW(const Vector& uvw);
    virtual  ~PhantomUVW();
    virtual  VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
