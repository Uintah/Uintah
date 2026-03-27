/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

//  TriangleMaterial.cc

#include <CCA/Components/MPM/Triangle/TriangleMaterial.h>
#include <Core/Geometry/IntVector.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <string>

using namespace std;
using namespace Uintah;

// Standard Constructor
TriangleMaterial::TriangleMaterial(ProblemSpecP& ps, MaterialManagerP& ss,
                                                           MPMFlags* flags)
  : Material(ps), d_triangle(0)
{
  d_lb = scinew MPMLabel();
  // The standard set of initializations needed
  standardInitialization(ps,flags);
  
  d_triangle = scinew Triangle(this,flags,ss);
}

void
TriangleMaterial::standardInitialization(ProblemSpecP& ps, MPMFlags* flags)

{
  ps->require("associated_material", d_associated_material);
  ps->require("triangle_filename",   d_triangle_filename);
  ps->get("filename",                d_filename);
  ps->getWithDefault("include_rotation", d_includeRotation, false);

  if(d_filename!="") {
    std::ifstream is(d_filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR: opening MPM rigid motion file '"+d_filename+"'\nFailed to find profile file",
                                  __FILE__, __LINE__);
    }

    double t0(-1.e9);
    if(d_includeRotation){
      while(is) {
        double t1;
        double vx, vy, vz, ox, oy, oz, wx, wy, wz;
        is >> t1 >> vx >> vy >> vz >> ox >> oy >> oz >> wx >> wy >> wz;
        if(is) {
         if(t1<=t0){
           throw ProblemSetupException("ERROR: profile file is not monotomically increasing", __FILE__, __LINE__);
         }
         d_vel_profile.push_back(std::pair<double,Vector>(t1,Vector(vx,vy,vz)));
         d_rot_profile.push_back(std::pair<double,Vector>(t1,Vector(wx,wy,wz)));
         d_ori_profile.push_back(std::pair<double,Vector>(t1,Vector(ox,oy,oz)));
        }
        t0 = t1;
      }
    } else {
      while(is) {
        double t1;
        double vx, vy, vz;
        is >> t1 >> vx >> vy >> vz;
        if(is) {
            if(t1<=t0){
              throw ProblemSetupException("ERROR: profile file is not monotomically increasing", __FILE__, __LINE__);
            }
            d_vel_profile.push_back( std::pair<double,Vector>(t1,Vector(vx,vy,vz)) );
        }
        t0 = t1;
      }
    }
    if(d_vel_profile.size()<2) {
        throw ProblemSetupException("ERROR: Failed to generate valid velocity profile", __FILE__, __LINE__);
    }
  }

}

// Default constructor
TriangleMaterial::TriangleMaterial() : d_triangle(0)
{
  d_lb = scinew MPMLabel();
}

TriangleMaterial::~TriangleMaterial()
{
  delete d_lb;
  delete d_triangle;
}

void TriangleMaterial::registerParticleState(
          std::vector<std::vector<const VarLabel*> > &TriangleState,
          std::vector<std::vector<const VarLabel*> > &TriangleState_preReloc)
{
  TriangleState.push_back         (d_triangle->returnTriangleState());
  TriangleState_preReloc.push_back(d_triangle->returnTriangleStatePreReloc());
}

ProblemSpecP TriangleMaterial::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP triangle_ps = ps->appendChild("Triangle");

  triangle_ps->appendElement("associated_material", d_associated_material);
  triangle_ps->appendElement("triangle_filename",   d_triangle_filename);
  triangle_ps->appendElement("filename",            d_filename);
  triangle_ps->appendElement("include_rotation",    d_includeRotation);

  return triangle_ps;
}

void
TriangleMaterial::copyWithoutGeom(ProblemSpecP& ps,
                                  const TriangleMaterial* mat, 
                                  MPMFlags* flags)
{
  d_triangle_filename = mat->d_triangle_filename;
}

Triangle* TriangleMaterial::getTriangle()
{
  return  d_triangle;
}

int TriangleMaterial::getAssociatedMaterial() const
{
  return d_associated_material;
}

string TriangleMaterial::getTriangleFilename() const
{
  return d_triangle_filename;
}

Vector TriangleMaterial::findVelFromProfile(double time)
{
  Vector value = findValFromProfile(time, d_vel_profile);
  return value;
}

Vector TriangleMaterial::findRotFromProfile(double time)
{
  Vector value = findValFromProfile(time, d_rot_profile);
  return value;
}

Vector TriangleMaterial::findOriFromProfile(double time)
{
  Vector value = findValFromProfile(time, d_ori_profile);
  return value;
}

int TriangleMaterial::getProfileSize()
{
  return d_vel_profile.size();
}

// find velocity from table of values
Vector TriangleMaterial::findValFromProfile(double t,
                               vector<pair<double, Vector> > profile) const
{
  int smin = 0;
  int smax = (int)(profile.size())-1;
  double tmin = profile[0].first;
  double tmax = profile[smax].first;
  if(t<=tmin) {
      return profile[0].second;
  }
  else if(t>=tmax) {
      return profile[smax].second;
  }
  else {
      // bisection search on table
      // could probably speed this up by keeping copy of last successful
      // search, and looking at that point and a couple to the right
      //
      while (smax>smin+1) {
          int smid = (smin+smax)/2;
          if(d_vel_profile[smid].first<t){
            smin = smid;
          }
          else{
            smax = smid;
          }
      }
      double l  = (profile[smin+1].first-profile[smin].first);
      double xi = (t-profile[smin].first)/l;
      double vx = xi*profile[smin+1].second[0]+(1-xi)*profile[smin].second[0];
      double vy = xi*profile[smin+1].second[1]+(1-xi)*profile[smin].second[1];
      double vz = xi*profile[smin+1].second[2]+(1-xi)*profile[smin].second[2];
      return Vector(vx,vy,vz);
    }
}
