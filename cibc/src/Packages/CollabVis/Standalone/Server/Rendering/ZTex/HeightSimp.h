//////////////////////////////////////////////////////////////////////
// HeightSimp.cpp - Simplify a height field into a mesh.
//
// David K. McAllister, September 1999.

#ifndef _heightsimp_h
#define _heightsimp_h

#include <Rendering/ZTex/SimpMesh.h>

#include <Rendering/ZTex/macros.h>
#include <Rendering/ZTex/ZImage.h>

namespace SemotusVisum {
namespace Rendering {

// C has the following bits:
// 0 - Top vert edge exists.
// 1 - Left horiz edge exists.
// 2 - Bottom vert edge exists.
// 3 - Right horiz edge exists.
// 4 - Left diag exists.
// 5 - Right diag exists.
// 6 - Top tri exists.
// 7 - Bottom tri exists.
// C = 0 is empty space.
// C = 0xc0 is part of a larger tri.
// Lev == 0 is full res.

#define MAX_LEVELS 32
/* PIGFUCKER! THIS ONLY WORKS ON SGI! */
#ifdef __sgi
#define MAXRANGE 0xffffff00
#define NORANGE  0xffffff00
#else
#define MAXRANGE 0xffffffff
#define NORANGE  0xffffffff
#endif



#define RNGCONV (1. / double(MAXRANGE))

union fui {
  unsigned int i;
  float f;
};

class HeightSimp
{
  ZImage ZBuf;
  Image *Levels[MAX_LEVELS];
  int lev, levcnt;
  int MaxErr, SkinDiff;
  SimpMesh *Me;
  unsigned char *Prv; // The pixels of the previous level.
  Vertex ** VertP;

  enum Flags
  {
    TOPV = 1,
    LEFTH = 2,
    BOTTOMV = 4,
    RIGHTH = 8,
    LEFTD = 0x10,
    RIGHTD = 0x20,
    HIGH = 0x40,
    LOW = 0x80,
    
    LEFTD_SOLID = 0xd0,
    RIGHTD_SOLID = 0xe0,

    DX = 0x30,
    DX0 = 0x3f,
    HL = 0xc0/*,*/
  };
  
  inline int P0(const int i)
  {
    return i << 1;
  }
  
  inline int P1(const int i)
  {
    return (i << 1)+1;
  }
  
  inline int P0(const int x, const int y, const int xo=0, const int yo=0)
  {
    return ((y << 1) + yo) * Levels[lev-1]->wid + ((x << 1) + xo);
  }
  
  inline unsigned char P(const int x, const int y, const int xo=0, const int yo=0)
  {
    return Prv[((y << 1) + yo) * Levels[lev-1]->wid + ((x << 1) + xo)];
  }
  
  inline int M(const int i)
  {
    return i << lev;
  }
  
  inline int M1(const int x)
  {
    return (x << lev) + (1 << (lev-1));
  }
  
  // Return index of this point in the full-res map.
  inline int M(const int x, const int y, const int xo=0, const int yo=0)
  {
    return ((y << lev) + (yo << (lev-1))) * ZBuf.wid + (x << lev) +
      (xo << (lev-1));
  }
  
  // Return Z value of this parent point in the full-res map.
  inline unsigned int Z(const int x, const int y, const int xo=0,
			const int yo=0)
  {
    int xc = (x << lev) + (xo << (lev-1));
    int yc = (y << lev) + (yo << (lev-1));

    ASSERT1(xc < ZBuf.wid && yc < ZBuf.hgt);
 
    if(xc>= ZBuf.wid || yc >= ZBuf.hgt) return NORANGE;

    return ZBuf[yc * ZBuf.wid + xc];
  }
  
  // Return cooked Z value of this point in the full-res map.
  inline unsigned int Zf(const int x, const int y, const int xo=0,
			 const int yo=0)
  {
    int xc = (x << lev) + (xo << (lev-1));
    int yc = (y << lev) + (yo << (lev-1));
    
    ASSERT1(xc < ZBuf.wid && yc < ZBuf.hgt);
 
    if(xc>= ZBuf.wid || yc >= ZBuf.hgt) return NORANGE;
    
    return ZBuf[yc * ZBuf.wid + xc] >> 21;
  }

  // return floating point z value of this point in the full-res map.
  inline double Zdbl(const int x, const int y) {
    
    ASSERT1(x < ZBuf.wid && y < ZBuf.hgt);

    fui fi;
    //fi.i = ((ZBuf[y * ZBuf.wid + x] >> 8) | 0x3f000000) & 0x3f7fffff;
    fi.f = float(ZBuf[y * ZBuf.wid + x]) * (1./4294967295.);
    if (fi.f > 1.0 || fi.f < 0.0) {
      DEBUG(fi.i);
      DEBUG(fi.f);
      cerr << endl;
    }

    return double(fi.f);
  }

  // Tell whether these three points are colinear enough to be simplified.
  // Return true if the edge is in the ocean.
  inline bool StraightEnough(const unsigned int y0,
    const unsigned int y1, const unsigned int y2, const int x) {
    
    if(y1 == NORANGE && (y0 == NORANGE || y2 == NORANGE))
      return true;
    
    unsigned int med = ((y0>>4)+(y2>>4))<<3;
    int err = abs((long)(med - y1));
    int relerr = err / x;

    // fprintf(stderr, "%08x %08x %08x %08x %08x %08x %d %d %s\n",
    // y0,y1,y2,med,err,relerr,err,relerr, relerr<MaxErr?"*":"");

    return relerr < MaxErr;
  }

  Image *SimplifyLevel(Image &Prev);
  Image *MakeLevel0();
  void QTreeToMesh(const int x, const int y, const int lev);
  void MakeTri(const int x0, const int y0, const int x1, const int y1, const int x2, const int y2);

public:
  inline HeightSimp() : lev(0), VertP(NULL) {}

  inline HeightSimp(ZImage &_ZBuf, int _MaxErr = 0, int _SkinDiff = 0x10000000) :
    ZBuf(_ZBuf), lev(0), levcnt(0), MaxErr(_MaxErr), SkinDiff(_SkinDiff),
    VertP(NULL) {}

  SimpMesh *HFSimp();
};

} // namespace Modules {y
} // namespace SemotusVisum {

#endif // _heightsimp_h
