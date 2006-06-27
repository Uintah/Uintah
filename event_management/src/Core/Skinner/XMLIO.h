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
//  
//    File   : XMLIO.h
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:01:25 2006
#ifndef SKINNER_XMLIO_H
#define SKINNER_XMLIO_H

#include <Core/Skinner/Drawable.h>
#include <libxml/xmlreader.h>

#include <string>
#include <map>

using std::string;
using std::map;
using std::pair;


namespace SCIRun {
  namespace Skinner {
    class Variables;

    class XMLIO {
    public:
      static Drawables_t        load(const string &filename);
      static void               register_maker(const string &,
                                               DrawableMakerFunc_t *,
                                               void *data = 0);
      template<class T>
      static void               register_maker(void *data=0) 
      {
        register_maker(T::class_name(), &T::maker, data);
      }

      
    private:
      // purely static class, dont allow creation
      XMLIO();
      XMLIO(const XMLIO &);
      virtual ~XMLIO();

      typedef map<string, xmlNodePtr> string_node_map_t;
      typedef pair<DrawableMakerFunc_t *, void *> DrawableMaker_t;
      typedef map<string, DrawableMaker_t> DrawableMakerMap_t;


      static Drawables_t        eval_skinner_node       (const xmlNodePtr,
                                                         const string &id);
      static void               eval_definition_node    (const xmlNodePtr,
                                                         string_node_map_t &);

      static Drawable *        eval_object_node        (const xmlNodePtr,
                                                        Variables *variables,
                                                        string_node_map_t &,
                                                        bool inst_def = false);

      static void              eval_var_node           (const xmlNodePtr,
                                                         Variables *);

      static DrawableMakerMap_t makers_;
    
    }; // end class XMLIO
  } // end namespace Skinner
} // end namespace SCIRun


#endif // #define Skinner_XMLIO_H
