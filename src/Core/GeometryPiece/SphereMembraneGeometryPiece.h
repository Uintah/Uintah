/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __SPHERE_MEMBRANE_GEOMETRY_OBJECT_H__
#define __SPHERE_MEMBRANE_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <Core/Geometry/Point.h>

namespace Uintah {

/**************************************
        
CLASS
   SphereMembraneGeometryPiece
        
   Creates a sphere from the xml input file description.
        
GENERAL INFORMATION
        
   SphereMembraneGeometryPiece.h
        
   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah
        
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
        
 
        
KEYWORDS
   SphereMembraneGeometryPiece  BoundingBox inside
        
DESCRIPTION
   Creates a sphere from the xml input file description.
   Requires five inputs: origin, radius and thickness as well as
   num_lat and num_long.  These last two indicate how many lines of
   latitude and longitude there are that are made up by particles.
   There are methods for checking if a point is inside the sphere
   and also for determining the bounding box for the sphere.
   The input form looks like this:
       <sphere_membrane>
         <origin>[0.,0.,0.]</origin>
         <radius>2.0</radius>
         <thickness>0.1</thickness>
         <num_lat>20</num_lat>
         <num_long>40</num_long>
       </sphere_membrane>
        
        
WARNING
        
****************************************/

class SphereMembraneGeometryPiece : public GeometryPiece {
         
public:
  //////////
  //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds a sphere.
  SphereMembraneGeometryPiece(ProblemSpecP &);
         
  //////////
  // Destructor
  virtual ~SphereMembraneGeometryPiece();

  static const string TYPE_NAME;
  virtual std::string getType() const { return TYPE_NAME; }

  /// Make a clone
  virtual GeometryPieceP clone() const;
         
  //////////
  // Determines whether a point is inside the sphere. 
  virtual bool inside(const Point &p) const;
         
  //////////
  // Returns the bounding box surrounding the box.
  virtual Box getBoundingBox() const;

  int returnParticleCount(const Patch* patch);

  int createParticles(const Patch* patch,
                      ParticleVariable<Point>&  pos,
                      ParticleVariable<double>& vol,
                      ParticleVariable<Vector>& pt1,
                      ParticleVariable<Vector>& pt2,
                      ParticleVariable<Vector>& pn,
                      ParticleVariable<Matrix3>& psize,
                      particleIndex start);


private:
  virtual void outputHelper( ProblemSpecP & ps ) const;

  Point  d_origin;
  double d_radius;
  double d_h;
  double d_numLat;
  double d_numLong;
};

} // End namespace Uintah

#endif // __SPHERE_MEMBRANE_GEOMETRY_PIECE_H__
