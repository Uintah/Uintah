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



#include <Packages/rtrt/Core/Turbulence.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Assert.h>

using namespace rtrt;
using namespace SCIRun;

Turbulence::Turbulence(int _noctaves,double _a, double _s,
						 int seed, int tblsize)
	: noise(seed, tblsize)
{
    noctaves=_noctaves;
    a=_a;
    s=_s;
}

Turbulence::Turbulence(const Turbulence& t)
	: noise(t.noise)
{
    noctaves=t.noctaves;
    a=t.a;
    s=t.s;
}

double Turbulence::operator()(const Point &p)
{
    Vector vv(p.x(), p.y(), p.z());
    double turb=0.0;
    double scale=1.0;
    for(int i=0;i<noctaves;i++){
	turb+=Abs(scale*noise(vv));
	scale*=a;
	vv*=s;
    }
    return turb;
}

double Turbulence::operator()(const Point& p, double m)
{
    ASSERT(m>0);
    Vector vv(p.x(), p.y(), p.z());
    double turb=0.0;
    double scale=1.0;
    double m2=m*2;
    while(scale > m2){
	turb+=scale*noise(vv);
	vv*=s;
	scale*=a;
    }

    // Gradual fade out
    double w=(scale/m)-1;
    w=Clamp(w,0.0, 1.0);
    turb+=w*scale*noise(vv);
    return turb;
}

Vector Turbulence::dturb(const Point& p, double m)
{
    ASSERT(m>0);
    Vector vv(p.x(), p.y(), p.z());
    Vector dt(0,0,0);
    double scale=1.0;
    double m2=m*2;
    while(scale > m2){
	dt+=noise.dnoise(vv)*scale;
	vv*=s;
	scale*=a;
    }

    // Gradual fade out
    double w=(scale/m)-1;
    w=Clamp(w,0.0,1.0);
    dt+=noise.dnoise(vv)*(w*scale);
    return dt;
}

Vector Turbulence::dturb(const Point& p, double m, double& n)
{
    ASSERT(m>0);
    Vector vv(p.x(), p.y(), p.z());
    Vector dt(0,0,0);
    n=0;
    double scale=1.0;
    double m2=m*2;
    while(scale>m2){
	double dn;
	dt+=noise.dnoise(vv, dn)*scale;
	n+=dn*scale;
	vv*=s;
	scale*=a;
    }

    // Gradual fade out
    double w=(scale/m)-1;
    w=Clamp(w,0.0,1.0);
    double dn;
    dt+=noise.dnoise(vv, dn)*(w*scale);
    n+=dn*w*scale;
    return dt;
}
