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
