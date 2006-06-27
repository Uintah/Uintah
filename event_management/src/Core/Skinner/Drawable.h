//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Drawable.h
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:16 2006

#ifndef SKINNER_DRAWABLE_H
#define SKINNER_DRAWABLE_H

#include <Core/Events/Tools/BaseTool.h>
#include <Core/Skinner/RectRegion.h>
#include <sci_gl.h>

#include <vector>
#include <map>
#include <string>

using std::vector;
using std::map;
using std::string;
using std::pair;

namespace SCIRun {
  namespace Skinner {
    class Variables;
    typedef pair<double, double> MinMax;

    class Drawable : public BaseTool {
    public:
      Drawable (Variables *);
      virtual ~Drawable();

      virtual propagation_state_e       process_event(event_handle_t);
      virtual string                    get_id() const;

      virtual RectRegion &              region();
      virtual MinMax                    minmax(unsigned int);

      //      void                              set_vars(Variables *vars);
      Variables *                       get_vars();
    protected:
      //      typedef vector<CallbackFunc_t> CallbackSet_t;
      //      typedef map<EventManager *, CallbackSet_t> Callbacks_t;
      //      void                      process_before_callbacks(EventState &);
      //      void                      process_after_callbacks(EventState &);

      RectRegion                        region_;
    private:
      Variables *                       variables_;
      //      Callbacks_t               callbacks_;
    };
  
    typedef map<string, string> KeyValMap_t;
    typedef vector<Drawable *> Drawables_t;
    
    typedef Drawable * DrawableMakerFunc_t(Skinner::Variables *,
                                           const Drawables_t &,
                                           void *);


    typedef pair<GLenum, GLenum> blendfunc_t;

    static MinMax INFINITE_MINMAX(AIR_NEG_INF, AIR_POS_INF);
    static MinMax SPRING_MINMAX(0, AIR_POS_INF);


  }
}
#endif

