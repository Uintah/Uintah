#ifndef GRID_VOL_REN_H
#define GRID_VOL_REN_H

#include <Packages/Kurt/Core/Geom/BrickGrid.h>
#include <Core/Datatypes/Polygon.h>
#include <vector>
using std::vector;


namespace Kurt {

using SCIRun::Polygon;



class GridVolRen 
{
public:

  GridVolRen();
  ~GridVolRen(){}
  void SetColorMap( unsigned char *cmap ){
    cmap_ = cmap; }
  void SetInterp( bool interp) {interp_ = interp; }
  void draw(const BrickGrid& bg,int slices);
  void drawWireFrame(const BrickGrid& bg);
  void Reload(){reload_ = (unsigned char *)1;}

private:

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
  unsigned char *reload_;
  unsigned char *cmap_;
  bool interp_;
};

#endif
} // end namespace Kurt
