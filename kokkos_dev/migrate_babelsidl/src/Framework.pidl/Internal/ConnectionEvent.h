/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
 *  ConnectionEvent.h: Implementation of the CCA
 *                     ConnectionEvent interface for SCIRun
 *
 *  Written by:
 *   Ayla Khan
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   October 2004
 *
 *  Copyright (C) 2004 SCI Institute
 *
 */

#ifndef Framework_ConnectionEvent_h
#define Framework_ConnectionEvent_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Framework/Internal/InternalComponentModel.h>
#include <Framework/Internal/InternalComponentInstance.h>

namespace SCIRun {

class ConnectionEvent : public sci::cca::ports::ConnectionEvent
{
public:
    ConnectionEvent(sci::cca::ports::EventType type, sci::cca::TypeMap::pointer tm);
    virtual ~ConnectionEvent();

    /** Returns the event type from the EventType enumeration. */
    virtual sci::cca::ports::EventType getEventType();

    /** Returns the property map for the connection, which should contain
     * at least the following keys and values:
     * UserName, string
     * UsesPort, string
     * ProviderName, string
     * ProvidesPort, string
     */
    // type info???
    // TODO: how to lay out keys and values in doxygen?
    virtual sci::cca::TypeMap::pointer getPortInfo();

private:
    sci::cca::ports::EventType type;
    sci::cca::TypeMap::pointer tm;
};

}

#endif
