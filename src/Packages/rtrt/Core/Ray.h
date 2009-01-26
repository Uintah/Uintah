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



#ifndef RAY_H
#define RAY_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Transform;

class Object; 

class Ray {
    Point  _origin;
    Vector _direction;

public:
    Object* already_tested[4];

    inline void reset()
      {
	already_tested[0] = 
	  already_tested[1] = 
	  already_tested[2] = 
	  already_tested[3] = 0;
      } 
    inline Ray() {
      reset();
    }
    inline Ray(const Point& origin, const Vector& dir) : _origin(origin), 
      _direction(dir)
    {
      reset();
    }
	
    inline const Point& origin() const {
	return _origin;
    }
    inline const Vector& direction() const {
	return _direction;
    }
    inline void set_origin(const Point& p) {
      reset();
      _origin = p;
    }
    inline void set_direction(const Vector& v) {
      reset();
      _direction = v;
    }
    inline Point eval(const double t) const {
      return _origin + t * _direction;
    }
  
    inline Ray transform(Transform *t) const
        {
            Ray tray;
            
            t->unproject(_origin,tray._origin);
            t->unproject(_direction,tray._direction);

            return tray;

        }
    inline void transform(Transform *t, Ray& tray) const
        {
	  tray.reset();
	  t->unproject(_origin,tray._origin);
	  t->unproject(_direction,tray._direction);
        }

    inline void transform_inplace(Transform *t)
        {
	  reset();
	  t->unproject_inplace(_origin);
	  t->unproject_inplace(_direction);
        }
};

} // end namespace rtrt

#endif
