/*
 *
 * Util: Implementations for non-inlined utility functions
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */
#include <Util/stringUtil.h>
#include <Util/Point.h>
#include <Util/Timer.h>

#include <Malloc/Allocator.h>
#include <Logging/Log.h>
#include <Network/dataItem.h>
#include <Message/Handshake.h>
#include <Compression/Compressors.h>
#include <UI/uiHelper.h>
#include <UI/UserInterface.h>
#include <Rendering/Renderer.h>
#include <Rendering/ImageRenderer.h>
#include <Rendering/GeometryRenderer.h>
#include <Rendering/ZTexRenderer.h>
#include <Util/ClientProperties.h>

#include <sys/times.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>

using namespace std;

namespace SemotusVisum {

/********************************************************************
 *                  STRING
 ********************************************************************/
int strcasecmp( const string &s1, const string &s2 ) {
  return ::strncasecmp( s1.data(), s2.data(), s1.length() );
}

string mkString( const unsigned int i ) {  
  char c[20];
  snprintf( c, 20, "%u", i );
  return string(c);
}

string mkString( const int i ) {
  char c[20];
  snprintf( c, 20, "%d", i );
  return string(c);
}
string mkString( const double d ) {
  char c[20];
  snprintf( c, 20, "%f", d );
  return string(c);
}
string mkString( const void * v ) {
  char c[20];
  snprintf( c, 20, "0x%x", (unsigned int)v );
  return string(c);
}

char * toChar( const string &s ) {
  char * c = scinew char[ s.length() + 1 ];
  memset( c, 0, s.length() + 1 );
  strncpy( c, s.data(), s.length() );
  return c;
}

int atoi( const string& s ) {
  char * b = toChar( s );
  int returnval = ::atoi( b );
  delete b;
  return returnval;
}

double atof( const string& s ) {
  char * b = toChar( s );
  double returnval = ::atof( b );
  delete b;
  return returnval;
}


bool startsWith( const string &haystack, const string &needle ) {
  if ( haystack.length() < needle.length() )
    return false;
  for ( unsigned i = 0; i < needle.length(); i++ )
    if ( haystack[i] != needle[i] )
      return false;
  return true;
}

bool endsWith( const string &haystack, const string &needle ) {
  if ( haystack.length() < needle.length() )
    return false;
  unsigned start = haystack.length() - needle.length();
  for ( unsigned i = 0; i < needle.length(); i++ )
    if ( haystack[ start+i ] != needle[i] )
      return false;
  return true;
}

/********************************************************************
 *                  POINT
 ********************************************************************/
Point3d Max(const Point3d& p1, const Point3d& p2)
{

  double x=Max(p1.x, p2.x);
  double y=Max(p1.y, p2.y);
  double z=Max(p1.z, p2.z);
  return Point3d(x,y,z);
}
Point3d Min(const Point3d& p1, const Point3d& p2)
{

  double x=Min(p1.x, p2.x);
  double y=Min(p1.y, p2.y);
  double z=Min(p1.z, p2.z);
  return Point3d(x,y,z);
}

Point3d project(float mat[4][4], const Point3d& p)
{
  return Point3d(mat[0][0]*p.x+mat[0][1]*p.y+mat[0][2]*p.z+mat[0][3],
		 mat[1][0]*p.x+mat[1][1]*p.y+mat[1][2]*p.z+mat[1][3],
		 mat[2][0]*p.x+mat[2][1]*p.y+mat[2][2]*p.z+mat[2][3],
		 mat[3][0]*p.x+mat[3][1]*p.y+mat[3][2]*p.z+mat[3][3]);
}



/********************************************************************
 *                  CLIENT PROPERTIES
 ********************************************************************/

vector<string>
ClientProperties::imageFormats;
vector<string>
ClientProperties::compressionFormats;
vector<string>
ClientProperties::serverRenderers;
vector<string>
ClientProperties::transferModes;

const char * const
ClientProperties::renderers[] = { ImageRenderer::name,
				  GeometryRenderer::name,
				  ZTexRenderer::name,
				  0 };

const char * const
ClientProperties::pimageFormats[] = { "INT_RGB",
				      "INT_ARGB",
				      "BYTE_GRAY",
				      "USHORT_GRAY",
				      "INT_ABGR",
				      "INT_BGR",
				      0 };


void
ClientProperties::setServerProperties( void * obj, MessageData *input ) {
  Log::log( DEBUG, "Setting server properties..." );
  
  Handshake *h = (Handshake *)(input->message);

    // Image formats
    vector<string> &l = h->getImageFormats();

    imageFormats.clear();
    for ( int i = 0; i < (int)l.size(); i++ )
      imageFormats.push_back( l[i] );
    
    
    // Compression formats
    l = h->getCompressionFormats();
    bool found=false;
    for ( int i = 0; i < (int)l.size(); i++ ) {
      found = false;
      for ( int j = 0; j < NUM_COMPRESSORS; j++ )
	if ( !strcasecmp( l[i], compressionMethods[j] ) ) {
	  Log::log( MESSAGE, string("Adding compression method ") +
		    compressionMethods[j] );
	  compressionFormats.push_back( compressionMethods[j] );
	  found=true;
	  break;
	}
      if ( !found ) 
	Log::log( MESSAGE, "Unable to utilize server compression method " +
		  l[i] );
    }

    // Transfer Modes

    transferModes.push_back("\"PTP (default)\"");
    transferModes.push_back("\"Reliable Multicast\"");
    transferModes.push_back("\"IP Multicast\"");
    
    /*
    l = h->getTransferModes();
    bool found=false;
    for ( int i = 0; i < (int)l.size(); i++ ) {
      found = false;
      for ( int j = 0; j < NUM_TRANSFER_MODES; j++ )
	if ( !strcasecmp( l[i], compressionMethods[j] ) ) {
	  Log::log( MESSAGE, string("Adding transfer mode ") +
		    transferMethods[j] );
	  transferModes.push_back( transferMethods[j] );
	  found=true;
	  break;
	}
      if ( !found ) 
	Log::log( MESSAGE, "Unable to utilize server transfer mode " +
		  l[i] );
    }
    */

    // Tell the ui manager that we have these compression formats.
    //LocalUIManager.getInstance().enableCompression();

    // Viewing methods
    l = h->getViewingMethods();
    
    for ( int i = 0; i < (int)l.size(); i+=2 /* Name/version */ ) {
      found = false;
      for ( unsigned j = 0; renderers[ j ] != 0; j++ )
	if ( !strcasecmp( l[i], renderers[j] ) ) {
	  Log::log( MESSAGE, string("Adding rendering method ") +
		    renderers[j] );
	  serverRenderers.push_back( renderers[j] );
	  found = true;
	  break;
	}
      if ( !found ) 
	Log::log( MESSAGE, "Unable to utilize server rendering method " +
		  l[i] );
    }
    
    Log::log( DEBUG, "Done setting server properties." );
    UserInterface::getHelper()->updateProperties();
  }

Handshake *
ClientProperties::mkHandshake() {
  int i;
  
  Handshake *h = scinew Handshake();
  
  h->setClientName( clientName );
  h->setClientRev ( revision   );
    
  // Image Formats
  for (i = 0; pimageFormats[i] != 0; i++) {
    h->addImageFormat( pimageFormats[ i ] );
  }
  
  // Build a list of renderers FIXME - add list of renderer name/versions
  // but base it on class members
  for ( i = 0; renderers[i] != 0; i++ )
    h->addViewingMethod( renderers[i], "1.0" );
  
  // Compression formats
  for ( i = 0; i < NUM_COMPRESSORS; i++ ) {
    h->addCompressionFormat( compressionMethods[ i ] );
  }
  
  // Finish and send the message
  h->finish();
  
  return h;
}

/********************************************************************
 *                  TIMER
 ********************************************************************/


#ifdef CLK_TCK
extern "C" long _sysconf(int);
#define CLOCK_INTERVAL CLK_TCK
#else
#include <sys/param.h>
#define CLOCK_INTERVAL HZ
#endif

static double ci=0;

Timer::Timer()
{
    state=Stopped;
    total_time=0;
}

Timer::~Timer()
{
  if(state != Stopped){
    Log::log( WARNING, "Timer destroyed while it was running" );
  }
}

void
Timer::start()
{
    if(state == Stopped){
	start_time=get_time();
	state=Running;
    } else {
      Log::log( WARNING, "Timer started while it was already running" );
    }
}

void
Timer::stop()
{
  if(state == Stopped){
    Log::log( WARNING, "Timer stopped while it was already stopped" );
  } else {
    state=Stopped;
    double t=get_time();
    total_time+=t-start_time;
  }
}

void
Timer::add(double t) {
  start_time -= t;
}

void
Timer::clear()
{
  if(state == Stopped){
    total_time=0;
  } else {
    Log::log( WARNING, "Timer cleared while it was running" );
    total_time=0;
    start_time=get_time();
  }
}

double
Timer::time()
{
  if(state == Running){
    double t=get_time();
    return t-start_time+total_time;
  } else {
    return total_time;
  }
}


double
CPUTimer::get_time()
{
  struct tms buffer;
  times(&buffer);
  double cpu_time=
    double(buffer.tms_utime+buffer.tms_stime)/double(CLOCK_INTERVAL);
  return cpu_time;
}

WallClockTimer::WallClockTimer()
{
  if(ci==0)
    ci=1./double(CLOCK_INTERVAL);
}

double
WallClockTimer::get_time()
{
  struct tms buffer;
  double time=double(times(&buffer))*ci;
  return time;
}

WallClockTimer::~WallClockTimer()
{
}

/* Convenience macro */
#ifndef timersub
#define timersub(a, b, result)                                                \
  do {                                                                        \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                             \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                          \
    if ((result)->tv_usec < 0) {                                              \
      --(result)->tv_sec;                                                     \
      (result)->tv_usec += 1000000;                                           \
    }                                                                         \
  } while (0)
#endif

PreciseTimer::PreciseTimer()
{
}

double
PreciseTimer::get_time()
{
  struct timeval _start;
  gettimeofday( &_start,NULL );

  return _start.tv_sec + (double)_start.tv_usec / 1000000.0;
}

PreciseTimer::~PreciseTimer()
{
}


CPUTimer::~CPUTimer()
{
}

TimeThrottle::TimeThrottle()
{
}

TimeThrottle::~TimeThrottle()
{
}

void
TimeThrottle::wait_for_time(double endtime)
{
  if(endtime==0)
    return;
  double time_now=time();
  double delta=endtime-time_now;
  if(delta <=0)
    return;
#ifdef __sgi
  int nticks=delta*CLOCK_INTERVAL;
  if(nticks<1)return;
  if(delta > 10){
    cerr << "WARNING: delta=" << delta << endl;
  }
  sginap(nticks);
#else
  /* FIXME - not finished! Use select() */
  //NOT_FINISHED("TimeThrottle::wait_for_time");
#endif
}

}
