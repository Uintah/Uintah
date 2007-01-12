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
 *  DTMessage.cc defines the message structure used in the data transmitter
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#include <iostream>
#include <string.h>
#include <Core/CCA/DT/DTMessage.h>
#include <Core/CCA/DT/DTAddress.h>

using namespace SCIRun;

#define DISPLAY_MSG

DTMessage::~DTMessage()
{
  if (autofree && buf != NULL) {
    delete [] buf;
  }
}


void
DTMessage::display()
{
#ifdef DISPLAY_MSG
  //char *str=new char[length];
  //strncpy(str, buf+sizeof(int), length-sizeof(int));
  //str[length-sizeof(int)]='\0';
  std::cout << "DTMessage:\n"
            << "\t recver=" << (long) recver << "\n"
            << "\t sender=" << (long) sender << "\n"
            << "\t fr_addr=" << fr_addr.getIP() << "/" << fr_addr.getPort() << "\n"
            << "\t to_addr=" << to_addr.getIP() << "/" << to_addr.getPort() << "\n"
            << "\t length=" << length << "\n"
    //<<"\t tag="<<tag<<"\n"
            << "\t offset=" << offset << "\n"
            << "\t buf(id)=" << *((int*)(this->buf))
            << std::endl;
#endif
}
