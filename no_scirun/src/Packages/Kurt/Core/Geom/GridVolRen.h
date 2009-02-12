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


#ifndef GRID_VOL_REN_H
#define GRID_VOL_REN_H

#include <Packages/Kurt/Core/Geom/BrickGrid.h>
#include <Core/Geometry/Polygon.h>
#include <vector>

namespace Kurt {
using std::vector;
using SCIRun::Polygon;



class GridVolRen 
{
public:

  GridVolRen();
  virtual ~GridVolRen(){}
  void SetColorMap( unsigned char *cmap ){
    cmap_ = cmap; }
  void SetNewBricks(bool nv) { newbricks_ = nv; }
  void SetInterp( bool interp) {interp_ = interp; }
  void SetControlPoint( const Point& p){ controlPoint = p; }
  void SetX(bool b){ if(b){drawView = false;} drawX = b; }
  void SetY(bool b){ if(b){drawView = false;} drawY = b; }
  void SetZ(bool b){ if(b){drawView = false;} drawZ = b; }
  void SetView(bool b){ if(b){drawX=false; drawY=false; drawZ=false;}
                        drawView = b; }
  virtual void draw(const BrickGrid& bg, int slices);
  virtual void drawWireFrame(const BrickGrid& bg);
  void Reload(){reload_ = (unsigned char *)1;}
  //  void SetDrawInfo( ){}

protected:

  void setAlpha(const Brick& brick);
  void computeView(Ray&);
  void loadColorMap();
  void loadTexture( Brick& brick);
  void makeTextureMatrix(const Brick& brick);
  void enableTexCoords();
  void enableBlend();
  void drawPolys( vector<Polygon *> polys);
  void disableTexCoords();
  void disableBlend();
  void drawWireFrame(const Brick& brick);

  GLuint* trexName;
  vector<GLuint> textureNames;
  unsigned char *reload_;
  unsigned char *cmap_;
  bool interp_, newbricks_;
  Point controlPoint;
  bool drawX, drawY, drawZ, drawView;


};

} // end namespace Kurt
#endif
