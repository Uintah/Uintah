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
#include <Core/Skinner/Skinner.h>

#include <string>
#include <map>
#include <list>
#include <vector>

using std::string;
using std::map;
using std::pair;
using std::list;
using std::vector;


#include <Core/Skinner/share.h>
namespace SCIRun {
  namespace Skinner {
    class Variables;
    class Root;

    class XMLIO {
    public:
      static Root *             load(const string &filename, Root *inroot = 0);
      template<class T>
      static void               register_maker() 
      {
        register_maker(T::class_name(), &T::maker);
      }

      
    private:
      // purely static class, dont allow creation
      XMLIO();
      XMLIO(const XMLIO &);
      virtual ~XMLIO();

      typedef map<string, xmlNodePtr> string_node_map_t;

      typedef vector<string_node_map_t> definition_nodes_t;

      typedef map<string, DrawableMakerFunc_t *> DrawableMakerMap_t;
      typedef vector<xmlNodePtr> merged_nodes_t;


      static xmlNodePtr find_definition(definition_nodes_t &,
                                        const string &classname);
      
      static void eval_merged_object_nodes_and_push_definitions
      (merged_nodes_t &, definition_nodes_t &);
                                 


      
      SCISHARE static void        register_maker(const string &,
                                        DrawableMakerFunc_t *);
      static Root *      eval_skinner_node(const xmlNodePtr,
                                           const string &id,
                                           Root *root);
      static void        eval_definition_node(const xmlNodePtr,
                                              string_node_map_t &);
      
      static Drawable *  eval_object_node(const xmlNodePtr,
                                          Variables *variables,
                                          definition_nodes_t &,
                                          SignalCatcher::TreeOfCatchers_t &,
                                          Root *);

      static void        eval_signal_node(const xmlNodePtr,
                                          Drawable *,
                                          SignalThrower::SignalToAllCatchers_t &,
                                          SignalCatcher::TreeOfCatchers_t &);
                                          

      static void        eval_var_node(const xmlNodePtr,
                                       Variables *,
                                       bool override_propagate=false);

      static DrawableMakerMap_t makers_;
    
    }; // end class XMLIO
  } // end namespace Skinner
} // end namespace SCIRun


#endif // #define Skinner_XMLIO_H
