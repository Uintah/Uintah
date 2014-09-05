#include <Packages/rtrt/Core/TimeVaryingCheapCaustics.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

Persistent* TVCC_maker() {
  return new TimeVaryingCheapCaustics();
}

// initialize the static member type_id
PersistentTypeID TimeVaryingCheapCaustics::type_id("TimeVaryingCheapCaustics", 
						   "Persistent", 
						   TVCC_maker);


TimeVaryingCheapCaustics::TimeVaryingCheapCaustics(char *fileNames, 
						   int numFiles, 
						   Point minPoint, 
						   Vector uAxis, 
						   Vector vAxis, 
						   Color lightColor, 
						   float timeScale, 
						   float tilingFactor) :
  minPoint(minPoint), 
  uAxis(uAxis), 
  vAxis(vAxis), 
  lightColor(lightColor),
  timeScale(timeScale), 
  tilingFactor(tilingFactor), 
  numFiles(numFiles)
{
  totalTime = timeScale*numFiles;
  projAxis = Cross(uAxis,vAxis);
  projAxis.normalize();
  uVecLen = uAxis.normalize();
  vVecLen = vAxis.normalize();
  uSizePerTile = uVecLen/tilingFactor;
  vSizePerTile = vVecLen/tilingFactor;

  caustics.resize( numFiles );

  float *currArrayPtr;
  char *dataPtr;
  char buf[1024];
  int xDim=1, yDim=1;
  FILE *currFile;
  for (int i=0;i<numFiles;i++)
    {
      char buf2[1024];
      int xDimTmp, yDimTmp, maxVal;
      sprintf( buf, fileNames, i );
      currFile = fopen( buf, "r" );
      if (!currFile)
	{
	  std::cerr << "Error opening file \"" << buf <<"\"!  Exiting...\n";
	  exit(0);
	}
      fgets( buf2, 1024, currFile );
      if (buf2[0] != 'P' || buf2[1] != '5')
	{
	  std::cerr << "Error!  File \""<<buf<<"\" is not a valid PGM file!\n";
	  exit(0);
	}
      do
	{
	  fgets( buf2, 1024, currFile );
	} while ( buf2[0] == '#' );
      sscanf( buf2, "%d %d", &xDimTmp, &yDimTmp);
      if (i==0 && xDimTmp > 0 && yDimTmp > 0)
	{
	  uDim = xDim = xDimTmp; vDim = yDim=yDimTmp;
	  dataPtr = (char *)malloc(sizeof(char)*xDim*yDim);
	  if (!dataPtr)
	    {
	      std::cerr << "Error!  Unable to allocate temporary memory!\n";
	      exit(0);
	    }
	}
      else if ( xDimTmp!=xDim || yDimTmp != yDim )
	{
	  std::cerr << "Error!  Image \""<<buf<<"\" doesn't have the same dimensions as others\n";
	  exit(0);
	}

      fgets( buf2, 1024, currFile );  
      sscanf( buf2, "%d", &maxVal );
      if (maxVal > 255)
	{
	  std::cerr << "Error!  Bad file \""<<buf<<"\".  Binary formats has maxVal > 255!\n";
	  exit(0);
	}

      caustics[i] = new Array2< float >(xDim,yDim);
      currArrayPtr = caustics[i]->get_dataptr();
      int totalRead = fread( (void *)dataPtr, sizeof( char ), xDim*yDim, currFile );
      if (totalRead != xDim*yDim)
	{
	  std::cerr << "Error!  File \""<<buf<<"\" too short!\n";
	}
      for (int j=0;j<xDim*yDim;j++)
	currArrayPtr[j] = dataPtr[j]/(float)maxVal;

      fclose( currFile );

    }

  free( dataPtr );
}

TimeVaryingCheapCaustics::~TimeVaryingCheapCaustics()
{
}

Color TimeVaryingCheapCaustics::GetCausticColor( Point atPoint, float currTime )
{
  float useTime;

  /* if we override the time by passing in currTime, use that, else use
  ** the value set using SetCurrentTime() 
  */
  if (currTime<0) useTime = currentTime; else useTime=currTime;
  
  /* find u,v coords of this point (remember, using ortho. projection) */
  Vector toAtPoint = atPoint-minPoint;
  float uCoord = Dot(toAtPoint,uAxis);
  float vCoord = Dot(toAtPoint,vAxis);
  
  uCoord = fmodf( uCoord, uSizePerTile )/uSizePerTile;
  vCoord = fmodf( vCoord, vSizePerTile )/vSizePerTile;
  
  if (uCoord < 0) uCoord += 1;
  if (vCoord < 0) vCoord += 1;

  int u = uCoord*uDim;
  int v = vCoord*vDim;

  /* gives a time value between 0 & number images */
  useTime = fmodf( useTime, totalTime)/timeScale;
  int timeIdx1 = useTime, timeIdx2;
  timeIdx2 = timeIdx1 % numFiles;

  float pct2 = useTime-timeIdx1;
  float pct1 = 1-pct2;

  if( timeIdx1 == numFiles )
    {
      cout << "WARNING: TimeVaryingCheapCaustics WOULD HAVE DIED!";
      timeIdx1--;
    }


  float color = pct1 * (*caustics[timeIdx1])(u,v) + 
    pct2 * (*caustics[timeIdx2])(u,v);
  return color*lightColor;
}

const int TIMEVARYINGCHEAPCAUSTICS_VERSION = 1;

void 
TimeVaryingCheapCaustics::io(SCIRun::Piostream &str)
{
  str.begin_class("TimeVaryingCheapCaustics", 
		  TIMEVARYINGCHEAPCAUSTICS_VERSION);
  SCIRun::Pio(str, caustics);
  SCIRun::Pio(str, timeScale);
  SCIRun::Pio(str, tilingFactor);
  SCIRun::Pio(str, currentTime);
  SCIRun::Pio(str, lightColor);
  SCIRun::Pio(str, minPoint);
  SCIRun::Pio(str, uAxis);
  SCIRun::Pio(str, vAxis);
  SCIRun::Pio(str, uVecLen);
  SCIRun::Pio(str, vVecLen);
  SCIRun::Pio(str, uDim);
  SCIRun::Pio(str, vDim);
  SCIRun::Pio(str, projAxis);
  SCIRun::Pio(str, totalTime);
  SCIRun::Pio(str, uSizePerTile);
  SCIRun::Pio(str, vSizePerTile);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::TimeVaryingCheapCaustics*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::TimeVaryingCheapCaustics::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::TimeVaryingCheapCaustics*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
