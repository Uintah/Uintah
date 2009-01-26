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


/*
  AmbientOcclusion - This lays a grid on top of an object and then
  computes the amount of ambient light which reaches the bounding box
  of the object.

  Author: James Bigler
  Date: June 21, 2004

*/

#include <Packages/rtrt/Core/Array3.h>
#include <Packages/rtrt/Core/Array1.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Thread/Runnable.h>

namespace rtrt {

  class Object;
  class Background;
  class PerProcessorContext;
  class DepthStats;

  using SCIRun::Point;
  using SCIRun::Vector;
  using SCIRun::Runnable;
  
// This contains all the common data to be shared by all rendering
// threads.
class AmbientOccContext {
public:
  AmbientOccContext(Object* geom,
                    const Point& min, const Point& max,
                    int nx, int ny, int nz,
                    int num_samples,
                    Background* bg);
  
  // This is the object for which we will be computing the ambient
  // occlusion for.
  Object *geometry;

  // The extents of where the grid will be placed.
  Point min, max;

  // The resolution of the grid
  int nx, ny, nz;

  // The Background
  Background *background;

  // These are the values we compute
  Array3<float> ambient_values;

  // The set of directions to render for each node
  Array1<Vector> directions;

  // Stuff for PerProcessorContext
  int pp_size;
  int pp_scratchsize;
  
};

class AmbientOccWorker: public Runnable {
  AmbientOccContext *aoc;
  
  // These are the nodes which will be rendered by this thread.
  // [minx, maxx), [miny, maxy), [minz, maxz)
  int minx, miny, minz;
  int maxx, maxy, maxz; // Reminder: These are non inclusive.

  // Used for rendering
  PerProcessorContext* ppc;
  DepthStats* depth_stats;
    
public:
  AmbientOccWorker(AmbientOccContext* aoc);
  ~AmbientOccWorker();

  // Allocated memory for ambient_values and checks the input
  // parameters.
  int setWork(int minx_in, int miny_in, int minz_in,
              int maxx_in, int maxy_in, int maxz_in);

  // This is the function that does all the work.
  void run();
};


} // end namespace rtrt
