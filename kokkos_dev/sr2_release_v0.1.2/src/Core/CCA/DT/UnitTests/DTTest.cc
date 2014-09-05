/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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

#include <Core/CCA/DT/UnitTests/DTTest.h>

#include <Core/Exceptions/InternalError.h>

#include <cppunit/TestAssert.h>
#include <cppunit/portability/Stream.h>

#include <iostream>
#include <exception>

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( DTTest );

DTTest::~DTTest()
{
}

void DTTest::setUp()
{
  spDataTransmitter = new DataTransmitter;
  spDataTransmitter->run();
  epDataTransmitter = new DataTransmitter;
  epDataTransmitter->run();

  ep = new DTPoint(epDataTransmitter);
  sp = new DTPoint(spDataTransmitter);
  //ep->service = ::service;
}

void DTTest::tearDown()
{
  spDataTransmitter->exit();
  epDataTransmitter->exit();
}

void DTTest::testPingPong()
{
  DTMessage* xmsg;
  int* msg;

  try {
    msg = (int*) malloc(sizeof(int));
    (*msg) = 13;
    
    DTMessage *wmsg = new DTMessage;
    wmsg->buf = (char*)msg;
    wmsg->length = sizeof(int);
    wmsg->autofree = true;
    wmsg->recver = ep;
    wmsg->to_addr= epDataTransmitter->getAddress();

    tag = sp->putInitialMessage(wmsg);
    
    xmsg = ep->getMessage(tag);

  } catch (const SCIRun::Exception& e2) {
    CPPUNIT_FAIL(e2.message());
  } catch (std::exception& e) {
    CPPUNIT_FAIL(e.what());
  }
  

  CPPUNIT_ASSERT_EQUAL(xmsg->length, (int)(sizeof(int)));
  CPPUNIT_ASSERT_EQUAL((int)(*(xmsg->buf)), 13);
  CPPUNIT_ASSERT(xmsg->fr_addr == spDataTransmitter->getAddress());
}

