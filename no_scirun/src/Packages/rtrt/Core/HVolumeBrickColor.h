/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef HVOLUMEBRICKCOLOR_H
#define HVOLUMEBRICKCOLOR_H 1

#include <Packages/rtrt/Core/Material.h>

#include <Core/Thread/WorkQueue.h>
#include <Core/Geometry/Point.h>

#include <cstdlib>

namespace rtrt {

using SCIRun::WorkQueue;

class HVolumeBrickColor : public Material {
protected:
    Point min;
    Vector datadiag;
    Vector sdiag;
    double dt;
    unsigned int nx,ny,nz;
    unsigned char* indata;
    unsigned char* blockdata;
    unsigned long* xidx;
    unsigned long* yidx;
    unsigned long* zidx;
    WorkQueue work;
    void brickit(int);
    char* filebase;
    double Ka, Kd, Ks;
    int specpow;
    double refl;
    bool grid;
    bool nn;
    friend class HVolumeBrickColorDpy;
public:
    HVolumeBrickColor(char* filebase, int np, double Ka, double Kd,
		      double Ks, int specpow, double refl,
		      double dt=0);
    virtual ~HVolumeBrickColor();
    virtual void shade(Color& result, const Ray& ray,
		       const HitInfo& hit, int depth,
		       double atten, const Color& accumcolor,
		       Context* cx);
};

} // end namespace rtrt

#endif
