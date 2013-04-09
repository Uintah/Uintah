/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <vector>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;


const string SmoothGeomPiece::TYPE_NAME = "smooth_geom";

SmoothGeomPiece::SmoothGeomPiece()
{
  d_dx = 1.0;
}

SmoothGeomPiece::~SmoothGeomPiece()
{
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle locations */
//////////////////////////////////////////////////////////////////////
vector<Point>* 
SmoothGeomPiece::getPoints()
{
  return &d_points;
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle volumes */
//////////////////////////////////////////////////////////////////////
vector<double>* 
SmoothGeomPiece::getVolume()
{
  return &d_volume;
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle temperatures */
//////////////////////////////////////////////////////////////////////
vector<double>* 
SmoothGeomPiece::getTemperature()
{
  return &d_temperature;
}
//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle color*/
//////////////////////////////////////////////////////////////////////
vector<double>* 
SmoothGeomPiece::getColors()
{
  return &d_color;
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle forces */
//////////////////////////////////////////////////////////////////////
vector<Vector>* 
SmoothGeomPiece::getForces()
{
  return &d_forces;
}

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle fiber directions */
//////////////////////////////////////////////////////////////////////
vector<Vector>* 
SmoothGeomPiece::getFiberDirs()
{
  return &d_fiberdirs;
}

////////////////////////////////////////////////////////////////////// // gcd adds
/* Returns the vector containing the set of particle velocity components */
//////////////////////////////////////////////////////////////////////
vector<Vector>* 
SmoothGeomPiece::getVelocity()
{
  return &d_velocity;
}                                                  // end gcd add

//////////////////////////////////////////////////////////////////////
/* Returns the vector containing the set of particle size tensor */
//////////////////////////////////////////////////////////////////////
vector<Matrix3>* 
SmoothGeomPiece::getSize()
{
  return &d_size;
}

//////////////////////////////////////////////////////////////////////
/* Deletes the vector containing the set of particle locations */
//////////////////////////////////////////////////////////////////////
void 
SmoothGeomPiece::deletePoints()
{
  d_points.clear();
}

//////////////////////////////////////////////////////////////////////
/* Deletes the vector containing the set of particle volumes */
//////////////////////////////////////////////////////////////////////
void 
SmoothGeomPiece::deleteVolume()
{
  d_volume.clear();
}

//////////////////////////////////////////////////////////////////////
/* Deletes the vector containing the set of particle sizes */
//////////////////////////////////////////////////////////////////////
void 
SmoothGeomPiece::deleteSizes()
{
  d_size.clear();
}

//////////////////////////////////////////////////////////////////////
/* Deletes the vector containing the set of particle temperatures */
//////////////////////////////////////////////////////////////////////
void 
SmoothGeomPiece::deleteTemperature()
{
  d_temperature.clear();
}

void 
SmoothGeomPiece::writePoints(const string& f_name, const string& var)
{
  std::cout << "Not implemented : " << f_name << "." << var 
            << " output " << std::endl;
}

int 
SmoothGeomPiece::returnPointCount() const
{
  return d_points.size();
}

void 
SmoothGeomPiece::setParticleSpacing(double dx)
{
  d_dx = dx;
}
