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

//prototype ptolemy server
//by oscar barney 


#ifndef Ptolemy_Core_Comm_PtolemyServer_h
#define Ptolemy_Core_Comm_PtolemyServer_h

//NOTE: took out JNIUTILS so this may mess some header up

#include <Core/Thread/Runnable.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Dataflow/Network/Network.h>
#include <Core/Thread/Semaphore.h>

using namespace SCIRun;

class PtolemyServer : public Runnable 
{
	public:
		PtolemyServer(TCLInterface *tclInt, Network *n) : gui(tclInt), net(n) {}
		virtual ~PtolemyServer() {}
		void run();
		static Semaphore& iterSem();
		void processItrRequest(int sockfd);

	private:
		static bool iter_callback(void *data);	
		string Iterate(vector<string> doOnce, int size1, vector<string> iterate, int size2, int numParams, string picPath, string picFormat);
		void quit(int sockfd, int listenfd);
		TCLInterface *gui;
		Network *net;
		
};

class QuitSCIRun : public Runnable {
	public:
		QuitSCIRun() {}
		virtual ~QuitSCIRun() {}
		void run();
};

class LoadNet : public Runnable {
	public:
		LoadNet(TCLInterface *tclInt) : gui(tclInt) {}
		virtual ~LoadNet() {}
		void run();
	private:
		TCLInterface *gui;
};

#endif
