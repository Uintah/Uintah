//////////////////////////////////////////////////////////////////////
// HeightSimp.cpp - Simplify a height field into a mesh.
// David K. McAllister, September 1999.

#include <Packages/Remote/Tools/Util/Assert.h>
#include <Packages/Remote/Dataflow/Modules/remoteSalmon/HeightSimp.h>
//int TCnt = 0;

using namespace Remote::Tools;

namespace Remote {
//----------------------------------------------------------------------
Image *HeightSimp::SimplifyLevel(Image &Prev) {
				// Create a level by simplifying the
				// previous level.  The size of this
				// level must completely enclose the
				// previous level.
  int wid = Prev.wid>>1; if((wid<<1) < Prev.wid) wid++;
  int hgt = Prev.hgt>>1; if((hgt<<1) < Prev.hgt) hgt++;

  Prv = Prev.Pix;

  cerr << levcnt << " Simplifying to: " << wid << "x" << hgt << endl;
  Image *_Cur = new Image(wid, hgt, 1);
  Image &Cur = *_Cur;

				// Fill in the level with the default
				// data.
  memset(Cur.Pix, 0x0f, Cur.size);

				// Step through the new level making a
				// simplified version of the previous
				// level.

  int x,y;
				// Do top and bottom rows.
  for(y=hgt-1, x=0; x<wid; x++) {
    if((((P(x,0) & DX0) == LEFTD) || (!P(x,0))) &&
       (((P(x,0,1,0) & DX0) == RIGHTD) || (!P(x,0,1,0))) &&
       ((P(x,0) & HIGH) == (P(x,0,1,0) & HIGH)) &&
       M(x+1) < ZBuf.wid &&
       StraightEnough(Z(x,0), Z(x,0,1,0), Z(x+1,0), 1<<lev)) {
      // cerr << "OT ";
      Cur.ch(x,0) &= (unsigned char)(~TOPV);
    }
    
    if((((P(x,y,0,1) & DX0) == RIGHTD) || (!P(x,y,0,1))) &&
       (((P(x,y,1,1) & DX0) == LEFTD) || (!P(x,y,1,1))) &&
       ((P(x,y,0,1) & LOW) == (P(x,y,1,1) & LOW)) &&
       M(x+1) < ZBuf.wid && M(y+1) < ZBuf.hgt &&
       StraightEnough(Z(x,y+1), Z(x,y+1,1,0), Z(x+1,y+1), 1<<lev)) {
      // cerr << "OB ";
      Cur.ch(x,y) &= (unsigned char)(~BOTTOMV);
    }
  }
  
  for(y=0; y<hgt; y++) {
				// Do left and right edges.
    if((((P(0,y) & DX0) == LEFTD) || (!P(0,y))) &&
       (((P(0,y,0,1) & DX0) == RIGHTD) || (!P(0,y,0,1))) &&
       ((P(0,y) & LOW) == (P(0,y,0,1) & HIGH)) &&
       M(y+1) < ZBuf.hgt &&
       StraightEnough(Z(0,y), Z(0,y,0,1), Z(0,y+1), 1<<lev)) {
      // cerr << "OL ";
      Cur.ch(0,y) &= (unsigned char)(~LEFTH);
    }

    x=wid-1;

    if((((P(x,y,1,0) & DX0) == RIGHTD) || (!P(x,y,1,0))) &&
       (((P(x,y,1,1) & DX0) == LEFTD) || (!P(x,y,1,1))) &&
       ((P(x,y,1,0) & LOW) == (P(x,y,1,1) & HIGH)) &&
       M(x+1) < ZBuf.wid && M(y+1) < ZBuf.hgt &&
       StraightEnough(Z(x+1,y), Z(x+1,y,0,1), Z(x+1,y+1), 1<<lev)) {
      // cerr << "OR ";
      Cur.ch(x,y) &= (unsigned char)(~RIGHTH);
    }
    
    for(x=0; x<wid; x++) {
				// This is an optimization. Works the
				// same without it.
      if((!P(x,y,0,0)) && (!P(x,y,1,0)) && (!P(x,y,0,1) &&
	(!P(x,y,1,1)))) continue;
      
      if(x<wid-1)
	if((((P(x,y,1,0) & DX0) == RIGHTD) || (!P(x,y,1,0))) &&
	   (((P(x,y,1,1) & DX0) == LEFTD) || (!P(x,y,1,1))) &&
	   // PRINT(P(x,y,1,0), P(x,y,1,1)) && 
	   (((P(x,y,1,0) & LOW)>>1) == (P(x,y,1,1) & HIGH)) &&
	   (((P(x+1,y) & DX0) == LEFTD) || (!P(x+1,y))) &&
	   (((P(x+1,y,0,1) & DX0) == RIGHTD) || (!P(x+1,y,0,1))) &&
	   (((P(x+1,y) & LOW)>>1) == (P(x+1,y,0,1) & HIGH)) &&
	   M(x+1) < ZBuf.wid && M(y+1) < ZBuf.hgt &&
	   StraightEnough(Z(x+1,y), Z(x+1,y,0,1), Z(x+1,y+1), 1<<lev)) {
	  // cerr << "LR ";
	  Cur.ch(x,y) &= (unsigned char)(~RIGHTH);
	  Cur.ch(x+1,y) &= (unsigned char)(~LEFTH);
	}
      
      if(y<hgt-1)
	if((((P(x,y,0,1) & DX0) == RIGHTD) || (!P(x,y,0,1))) &&
	   (((P(x,y,1,1) & DX0) == LEFTD) || (!P(x,y,1,1))) &&
	   ((P(x,y,0,1) & LOW) == (P(x,y,1,1) & LOW)) &&
	   (((P(x,y+1) & DX0) == LEFTD) || (!P(x,y+1))) &&
	   (((P(x,y+1,1,0) & DX0) == RIGHTD) || (!P(x,y+1,1,0))) &&
	   ((P(x,y+1) & HIGH) == (P(x,y+1,1,0) & HIGH)) &&
	   M(x+1) < ZBuf.wid && M(y+1) < ZBuf.hgt &&
	   StraightEnough(Z(x,y+1), Z(x,y+1,1,0), Z(x+1,y+1), 1<<lev)) {
	  // cerr << "TB ";
	  Cur.ch(x,y) &= (unsigned char)(~BOTTOMV);
	  Cur.ch(x,y+1) &= (unsigned char)(~TOPV);
	}
      
				// We have now removed all of (x,y)'s
				// edges that we can.  See if we can
				// polish it off.
      unsigned char &C = Cur.ch(x,y);
      if(!C) {
				// All 4 kids are either 0 or diag.
	unsigned char pl = P(x,y,0,0) & LOW;
	unsigned char ph = P(x,y,0,0) & HIGH;
	if((x+y) & 1)
	  {
				// right
	    if(((P(x,y,1,0) & LOW) == pl) &&
	       ((P(x,y,0,1) & LOW) == pl) &&
	       (((P(x,y,1,1) & HIGH)<<1) == pl) &&
	       ((P(x,y,1,1) & LOW) == pl) &&
	       ((P(x,y,0,0) & HIGH) == ph) &&
	       (((P(x,y,0,0) & LOW)>>1) == ph) &&
	       ((P(x,y,1,0) & HIGH) == ph) &&
	       ((P(x,y,0,1) & HIGH) == ph) &&
	       ((pl == 0 && ph == 0) || 
		StraightEnough(Z(x+1,y), Z(x,y,1,1),
			       Z(x,y+1), (1<<lev)*M_SQRT2)))
	      C = P(x,y,1,0);
	    else
	      C = DX;
	  }
	else
	  {
	    // left
	    // fprintf(stderr, "Left: 0x%02x 0x%02x 0x%02x 0x%02x ",
	    // int(P(x,y,0,0)), int(P(x,y,1,0)), int(P(x,y,0,1)), int(P(x,y,1,1)));
	    if(((P(x,y,0,0) & LOW) == pl) &&
	       ((P(x,y,0,1) & LOW) == pl) &&
	       (((P(x,y,0,1) & HIGH)<<1) == pl) &&
	       ((P(x,y,1,1) & LOW) == pl) &&
	       ((P(x,y,0,0) & HIGH) == ph) &&
	       (((P(x,y,1,0) & LOW)>>1) == ph) &&
	       ((P(x,y,1,0) & HIGH) == ph) &&
	       ((P(x,y,1,1) & HIGH) == ph) &&
	       ((pl == 0 && ph == 0) || 
		StraightEnough(Z(x,y), Z(x,y,1,1), Z(x+1,y+1),
			       (1<<lev)*M_SQRT2)))
	      C = P(x,y);
	    else
	      C = DX;
	  }
	// fprintf(stderr, "Res: %d,%d 0x%02x ", M(x), M(y), int(C));
      }
      
      ASSERT1(!((C & 0x0f) && (C & 0xf0)));
      ASSERT1(!(((C & (LEFTD | RIGHTD)) == (LEFTD | RIGHTD)) && 
	(C & (HIGH | LOW))));
      ASSERT1(C != LEFTD);
      ASSERT1(C != RIGHTD);
      
      // cerr << endl;
    }
  }
  
  return _Cur;
}


#if 1
//----------------------------------------------------------------------
Image *HeightSimp::MakeLevel0() {
  int zwid = ZBuf.wid, zhgt = ZBuf.hgt;
  int wid = zwid-1, hgt = zhgt-1;

  Image *_Cur = new Image(wid, hgt, 1);
  Image &Cur = *_Cur;

  int x;

				// Makes every square be 0, RIGHTD or
				// LEFTD and HIGH and/or LOW.
  for(int y=0; y<hgt; y++) {
    unsigned int *ZZ = &ZBuf(0,y);
    unsigned char *C = &Cur.ch(0,y);
    
    for(x=0; x<wid; x++, C++, ZZ++) {
      *C = 0;
      
				// Is LEFT
      if((*ZZ != NORANGE) && (*(ZZ+zwid+1) != NORANGE))	{
	*C |= (*(ZZ+zwid) != NORANGE) ? (LEFTD | LOW) : 0;
	*C |= (*(ZZ+1) != NORANGE) ? (LEFTD | HIGH) : 0;
      }
      
      x++;
      if(x>=wid)
	continue;
      C++;
      ZZ++;

      *C = 0;
      
				// Is RIGHT
      if((*(ZZ+zwid) != NORANGE) && (*(ZZ+1) != NORANGE)) {
	*C |= (*(ZZ+zwid+1) != NORANGE) ? (RIGHTD | LOW) : 0;
	*C |= (*ZZ != NORANGE) ? (RIGHTD | HIGH) : 0;
      }
    }
    
    y++;
    if(y>=hgt)
      continue;
    
    ZZ = &ZBuf(0,y);
    C = &Cur.ch(0,y);
    
    for(x=0; x<wid; x++, C++, ZZ++) {
      *C = 0;
      
				// Is RIGHT
      if((*(ZZ+zwid) != NORANGE) && (*(ZZ+1) != NORANGE)) {
	*C |= (*(ZZ+zwid+1) != NORANGE) ? (RIGHTD | LOW) : 0;
	*C |= (*ZZ != NORANGE) ? (RIGHTD | HIGH) : 0;
      }
      
      x++;
      if(x>=wid)
	continue;
      C++;
      ZZ++;

      *C = 0;
      
				// Is LEFT
      if((*ZZ != NORANGE) && (*(ZZ+zwid+1) != NORANGE)) {
	*C |= (*(ZZ+zwid) != NORANGE) ? (LEFTD | LOW) : 0;
	*C |= (*(ZZ+1) != NORANGE) ? (LEFTD | HIGH) : 0;
      }
    }
  }
  return _Cur;
}
#else
//----------------------------------------------------------------------
Image *HeightSimp::MakeLevel0() {
  int wid = ZBuf.wid-1, hgt = ZBuf.hgt-1;

  Image *_Cur = new Image(wid, hgt, 1);
  Image &Cur = *_Cur;

				// Makes every square be 0, RIGHTD or
				// LEFTD and HIGH and/or LOW.
  for(int y=0; y<hgt; y++) {
    for(int x=0; x<wid; x++) {
      unsigned char &C = Cur.ch(x,y);
      C = 0;
      
      if((x+y) & 1) {
				// Is RIGHT
	if(ZBuf(x,y+1) != NORANGE && ZBuf(x+1,y) != NORANGE) {
	  if(ZBuf(x+1,y+1) != NORANGE)
	    C |= (RIGHTD | LOW);
	  if(ZBuf(x,y) != NORANGE)
	    C |= (RIGHTD | HIGH);
	}
      }
      else {
				// Is LEFT
	if(ZBuf(x,y) != NORANGE && ZBuf(x+1,y+1) != NORANGE) {
	  if(ZBuf(x,y+1) != NORANGE)
	    C |= (LEFTD | LOW);
	  if(ZBuf(x+1,y) != NORANGE)
	    C |= (LEFTD | HIGH);
	}
      }
      
      ASSERT1(!(C & 0x0f));
      ASSERT1(!(((C & (LEFTD | RIGHTD)) == (LEFTD | RIGHTD)) && 
	(C & (HIGH | LOW))));
      ASSERT1(C != LEFTD);
      ASSERT1(C != RIGHTD);
    }
  }
  return _Cur;
}
#endif

//----------------------------------------------------------------------
// Add a tri with the given coords into the mesh.
void HeightSimp::MakeTri(const int x0, const int y0, const int x1,
			 const int y1, const int x2, const int y2) {
  
  int i0 = y0*ZBuf.wid+x0;
  int i1 = y1*ZBuf.wid+x1;
  int i2 = y2*ZBuf.wid+x2;

  // XXX return;
  if((ZBuf[i0] == NORANGE) || (ZBuf[i1] == NORANGE) || (ZBuf[i2] == NORANGE))
    return;
  
  Vertex *v0 = VertP[i0]; 
  if(v0 == NULL) v0 = VertP[i0] = 
		   Me->AddVertex(Vector(x0, y0,
		     //ZBuf[i0] >> 21
		     Zdbl(x0, y0)
					));
  Vertex *v1 = VertP[i1]; 
  if(v1 == NULL) v1 = VertP[i1] = 
		   Me->AddVertex(Vector(x1, y1,
		     //ZBuf[i1] >> 21
		     Zdbl(x1, y1)
					));
  Vertex *v2 = VertP[i2]; 
  if(v2 == NULL) v2 = VertP[i2] = 
		   Me->AddVertex(Vector(x2, y2,
		     //ZBuf[i2] >> 21
		     Zdbl(x2, y2)
					));
  
  Edge *e0 = Me->FindEdge(v0, v1);
  Edge *e1 = Me->FindEdge(v1, v2);
  Edge *e2 = Me->FindEdge(v2, v0);
 
  Me->AddFace(v0, v1, v2, e0, e1, e2);
}

//---------------------------------------------------------------------- 
// Add the triangles of this square to the mesh.
void HeightSimp::QTreeToMesh(const int x, const int y, const int clv) {
  lev = clv;

  if(x >= Levels[clv]->wid || y >= Levels[clv]->hgt)
    return;

  unsigned char C = Levels[clv]->ch(x,y);
  // fprintf(stderr, "0x%02x ", C); cerr << clv << ": " << x << "," << y << " " << M(x)<<","<<M(y)<<endl;

  ASSERT1(!((C & 0x0f) && (C & 0xf0)));
  ASSERT1(!(((C & (LEFTD | RIGHTD)) == (LEFTD | RIGHTD)) &&
    (C & (HIGH | LOW))));
  ASSERT1(C != LEFTD);
  ASSERT1(C != RIGHTD);
  
				// First check all the leaf cases.
  if(C == 0)
    return;			// Ocean
  //TCnt++;
  if(C & HL) {
    if(C & LEFTD) {
				// Has a left cut.
      if(C & HIGH)
	MakeTri(M(x), M(y), M(x+1), M(y+1), M(x+1), M(y));
      if(C & LOW)
	MakeTri(M(x), M(y), M(x), M(y+1), M(x+1), M(y+1));
    }
    else if(C & RIGHTD) {
				// Has a right cut.
      if(C & HIGH)
	MakeTri(M(x), M(y), M(x), M(y+1), M(x+1), M(y));
      if(C & LOW)
	MakeTri(M(x), M(y+1), M(x+1), M(y+1), M(x+1), M(y));
    }
  }
  else {
				// Comes in here if 00110000 or
				// 0000xxxx
    
      ASSERT1(((C & 0xf0)==DX) || (C & 0x0f));
      ASSERT1(clv > 0);
      
      if(!(C & TOPV)) {
	MakeTri(M(x), M(y), M1(x), M1(y), M(x+1), M(y)); // T
	if(C & RIGHTH) 
	  MakeTri(M(x+1), M(y), M1(x), M1(y), M(x+1), M1(y)); // RT
	if(C & LEFTH) 
	  MakeTri(M(x), M1(y), M1(x), M1(y), M(x), M(y)); // LT
      }
      
      if(!(C & LEFTH)) {
	MakeTri(M(x), M(y+1), M1(x), M1(y), M(x), M(y)); // L
	if(C & TOPV) 
	  MakeTri(M(x), M(y), M1(x), M1(y), M1(x), M(y)); // TL
	if(C & BOTTOMV) 
	  MakeTri(M1(x), M(y+1), M1(x), M1(y), M(x), M(y+1)); // BL
      }
      
      if(!(C & BOTTOMV)) {
	MakeTri(M(x+1), M(y+1), M1(x), M1(y), M(x), M(y+1)); // B
	if(C & RIGHTH) 
	  MakeTri(M(x+1), M1(y), M1(x), M1(y), M(x+1), M(y+1)); // RB
	if(C & LEFTH) 
	  MakeTri(M(x), M(y+1), M1(x), M1(y), M(x), M1(y)); // LB
      }
      
      if(!(C & RIGHTH)) {
	MakeTri(M(x+1), M(y), M1(x), M1(y), M(x+1), M(y+1)); // R
	if(C & TOPV) 
	  MakeTri(M1(x), M(y), M1(x), M1(y), M(x+1), M(y)); // TR
	if(C & BOTTOMV)
	  MakeTri(M(x+1), M(y+1), M1(x), M1(y), M1(x), M(y+1)); // BR
      }
      
      if((C & TOPV) && (C & LEFTH)) QTreeToMesh(P0(x), P0(y), clv-1); // TL
      if((C & TOPV) && (C & RIGHTH)) QTreeToMesh(P1(x), P0(y), clv-1); // TR
      if((C & BOTTOMV) && (C & LEFTH)) QTreeToMesh(P0(x), P1(y), clv-1); // BL
      if((C & BOTTOMV) && (C & RIGHTH)) QTreeToMesh(P1(x), P1(y), clv-1); // BR
  }
}

//----------------------------------------------------------------------
// Create a mesh from a height field.
SimpMesh *HeightSimp::HFSimp()
{
  cerr << "Simplifying a Z Buffer of " << ZBuf.wid << "x" << ZBuf.hgt << endl;

  // for(int i=0; i<100; i++)  // XXX
  {
    levcnt = 0;
    Levels[levcnt] = MakeLevel0();

    while(Levels[levcnt]->wid > 1 || Levels[levcnt]->hgt > 1)
      {
	levcnt++;
	lev = levcnt;
	Levels[levcnt] = SimplifyLevel(*Levels[levcnt-1]);
      }
    levcnt++;
  }

  // for(int i=0; i<10; i++) // XXX
  {
    cerr << "Making mesh." << endl;
    Me = new SimpMesh;

    VertP = new Vertex*[ZBuf.size];
    memset(VertP, 0, ZBuf.size * sizeof(Vertex *));

    QTreeToMesh(0, 0, levcnt-1);

    delete [] VertP;
  }

  for(int j=0; j<levcnt; j++)
    delete Levels[j];
  //cerr << "tcount:" << TCnt;
  cerr << ", zbuf.size:" << ZBuf.size << endl;
  return Me;
}
} // End namespace Remote


