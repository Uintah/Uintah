
//------------------------------
// message.h
// David Hart
// SCI group, University of Utah
// Copyright November 1999
// All Rights Reserved
//------------------------------

#ifndef __MESSAGE_H__
#define __MESSAGE_H__

#define VR_PORT 15150
#define VR_NUMPORTS 5

namespace Remote {
enum vrMessage {
  

  //----------------------------------------------------------------------
  
				// request view to change to a new
				// poition & direction
  
				// format: m 1 2 3 4 5 6 7 8 9
				// 1,2,3: float - (x,y,z) eye position
				// 4,5,6: float - (x,y,z) view
				// direction
				// 7,8,9: float - (x,y,z) up vector
  VR_SETPOS,

  //----------------------------------------------------------------------

				// request the current viewing
				// direction information
  
				// format: m
  VR_GETPOS,
  
  //----------------------------------------------------------------------

				// send an image and geometry from the
				// depth buffer
  
				// format: m 1 2 3 4 5
				// 1,2: float - width, height
  VR_SETSCENE,

  //----------------------------------------------------------------------

				// request an images and geometry from
				// the depth buffer
  
				// format: m 1 2 3 4 5 6 7 8 9
				// 1,2,3: float - (x,y,z) eye position
				// 4,5,6: float - (x,y,z) view
				// direction
				// 7,8,9: float - (x,y,z) up vector
  VR_GETSCENE,

  //----------------------------------------------------------------------

				// receive geometry

				// format: m
  VR_SETGEOM,
  
  //----------------------------------------------------------------------

				// request the real scene geometry
				// from the server
  
				// format: m 1 2 3 4 5 6 7 8 9
				// 1,2,3: float - (x,y,z) eye position
				// 4,5,6: float - (x,y,z) view
				// direction
				// 7,8,9: float - (x,y,z) up vector
  VR_GETGEOM,

  //----------------------------------------------------------------------

				// acknowledge the end of a message

				// format: m
  VR_ENDMESSAGE,

  //----------------------------------------------------------------------

				// request a shutdown
				// format: m
  VR_QUIT

  //----------------------------------------------------------------------
  
};

} // End namespace Remote


#endif // __MESSAGE_H__
