/**
 * \file ManagerTypes.h
 * \author Devin Robison
 *
 * Copyright (c) 2011 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <string>

#ifndef EXPR_MEMMANAGER
#define EXPR_MEMMANAGER
namespace Expr{

  /**
   *  \enum MemoryManager
   *  \brief Enumerates the supported memory management strategies.
   */
   enum MemoryManager {
     MEM_EXTERNAL, 	///< External memory management - controlled elsewhere.
     // jcs note that MEM_STATIC is not yet supported, and would take some effort to get in place.
     //          if we enable MEM_STATIC, then look very carefully at VertexProperty::set_is_persistent().
     //     MEM_STATIC,    	///< Internal memory management with persistent fields (no re-use).
     MEM_DYNAMIC,	///< Internal memory management with dynamic fields (may be released for reuse)
     MEM_STATIC_GPU,	///< static allocation on GPU
     MEM_DYNAMIC_GPU,	///< dynamic allocation on GPU
     UNKNOWN		///< error state
   };


   inline std::string get_memory_manager_description( const MemoryManager m )
   {
     std::string name;
     switch ( m ){
       case MEM_EXTERNAL   : name = "External-Framework";           break;
//       case MEM_STATIC     : name = "Local, Persistent";            break;
       case MEM_DYNAMIC    : name = "Local, Non-Persistent";        break;
       case MEM_STATIC_GPU : name = "External-GPU, Persistent";     break;
       case MEM_DYNAMIC_GPU: name = "External-GPU, Non-Persistent"; break;
       default             : name = "Unknown or Invalid";           break;
     }
     return name;
   }

   template<typename T>
   inline T& operator<<( T& t, const MemoryManager m ){
     t << get_memory_manager_description(m);
     return t;
   }


} // namespace Expr
#endif
