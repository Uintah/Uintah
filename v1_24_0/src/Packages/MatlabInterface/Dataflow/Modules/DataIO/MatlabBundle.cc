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
 *  MatlabBundle.cc:
 *
 *  Written by:
 *   Jeroen Stinstra
 *
 */


#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/BundlePort.h>
#include <Core/SystemCall/TempFileManager.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabconverter.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>
#include <Packages/MatlabInterface/Services/MatlabEngine.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/ICom/IComAddress.h>
#include <Core/ICom/IComPacket.h>
#include <Core/Services/ServiceClient.h>
#include <Core/Services/Service.h>
#include <Core/Services/ServiceBase.h>
#include <Core/Services/FileTransferClient.h>
#include <Core/ICom/IComSocket.h>
#include <Core/Thread/CleanupManager.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h> 
 
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#pragma set woff 1209 
#endif 
 
namespace MatlabIO {

using namespace SCIRun;

class MatlabBundle;
class MatlabBundleEngineThreadInfo;
class MatlabBundleEngineThread;

typedef LockingHandle<MatlabBundleEngineThreadInfo> MatlabBundleEngineThreadInfoHandle;

class MatlabBundleEngineThread : public Runnable, public ServiceBase 
{
  public:
	MatlabBundleEngineThread(ServiceClientHandle serv_handle,MatlabBundleEngineThreadInfoHandle info_handle);
    virtual ~MatlabBundleEngineThread();
	void run();

  private:
	ServiceClientHandle serv_handle_;
	MatlabBundleEngineThreadInfoHandle info_handle_;
};

class MatlabBundleEngineThreadInfo 
{
  public:
	MatlabBundleEngineThreadInfo();
	virtual ~MatlabBundleEngineThreadInfo();

	void dolock();
	void unlock();
	
  public:
	Mutex				lock;
	int					ref_cnt;
	
  std::string         output_cmd_;
  GuiInterface*       gui_;

	ConditionVariable   wait_code_done_;
	bool				code_done_;
	bool				code_success_;
	std::string	code_error_;
	
	ConditionVariable   wait_exit_;
	bool				exit_;
	bool				passed_test_;

};


class MatlabBundle : public Module, public ServiceBase 
{
  
  public:
    // Constructor
	MatlabBundle(GuiContext* ctx);

    // Destructor
	virtual ~MatlabBundle();
	
	// Std functions for each module
	// execute():
	//   Execute the module and put data on the output port
	
	virtual void execute();
  virtual void presave();
	
  virtual void tcl_command(GuiArgs& args, void* userdata);

  static std::string totclstring(std::string &instring);
	std::vector<std::string>	converttcllist(std::string str);
	void	update_status(std::string text);
  
  private:
  
  bool	open_matlab_engine();
	bool	close_matlab_engine();
	
	bool	create_temp_directory();
	bool	delete_temp_directory();

	bool	save_input_matrices();
	bool	load_output_matrices();
	
	bool	generate_matlab_code();
	bool	send_matlab_job();
  bool  send_input(std::string str);

	bool	synchronise_input();
	
  private:
  
	 enum { NUM_BUNDLE_PORTS = 5 };
  
	// Temp directory for writing files coming from the 
	// the matlab engine
	
	std::string temp_directory_;
	
	// GUI variables
	
	// Names of matrices
	GuiString   input_bundle_name_;
	GuiString   input_bundle_array_;
	GuiString   output_bundle_name_;
	GuiString   output_bundle_pnrrds_;
	GuiString   output_bundle_pbundles_;
  
	// Fields per port
	std::vector<std::string>   input_bundle_name_list_;
	std::vector<std::string>   input_bundle_name_list_old_;
	std::vector<std::string>   input_bundle_array_list_;
	std::vector<std::string>   output_bundle_name_list_;
	std::vector<std::string>   output_bundle_pnrrds_list_;
	std::vector<std::string>   output_bundle_pbundles_list_;

  std::vector<int> input_bundle_generation_old_;

	std::string	matlab_code_list_;
	
	// Ports for input and output
	BundleIPort*		bundle_iport_[NUM_BUNDLE_PORTS];
	BundleOPort*		bundle_oport_[NUM_BUNDLE_PORTS];
	
	BundleHandle		bundle_handle_[NUM_BUNDLE_PORTS];

	std::string		input_bundle_matfile_[NUM_BUNDLE_PORTS];
	std::string		output_bundle_matfile_[NUM_BUNDLE_PORTS];
	
	// Internet connectivity stuff
	GuiString   inet_address_;
	GuiString   inet_port_;
	GuiString   inet_passwd_;
	GuiString   inet_session_;
	
	std::string inet_address_old_;
	std::string inet_port_old_;
	std::string inet_passwd_old_;
	std::string inet_session_old_;
    	
	// The tempfilemanager
	TempFileManager tfmanager_;
	
	// MatlabBundle converter class
	matlabconverter translate_;
	
	std::string		mfile_;
	
	GuiString		matlab_code_;
  GuiString   matlab_code_file_;
	GuiString		matlab_var_;
	GuiString		matlab_add_output_;
	GuiString		matlab_update_status_;
    
	ServiceClientHandle             matlab_engine_;
  FileTransferClientHandle        file_transfer_;
	MatlabBundleEngineThreadInfoHandle	thread_info_;
    
  bool            need_file_transfer_;
  std::string     remote_tempdir_;
  std::string     inputstring_;
    
  public:
    static void cleanup_callback(void *data);
};


MatlabBundleEngineThreadInfo::MatlabBundleEngineThreadInfo() :
	lock("MatlabBundleEngineInfo lock"),
	ref_cnt(0),
  gui_(0),
	wait_code_done_("MatlabBundleEngineInfo condition variable code"),
	code_done_(false),
	code_success_(false),
	wait_exit_("MatlabBundleEngineInfo condition variable exit"),
	exit_(false),
	passed_test_(false)
{
}

MatlabBundleEngineThreadInfo::~MatlabBundleEngineThreadInfo()
{
}

inline void MatlabBundleEngineThreadInfo::dolock()
{
	lock.lock();
}

inline void MatlabBundleEngineThreadInfo::unlock()
{
	lock.unlock();
}

MatlabBundleEngineThread::MatlabBundleEngineThread(ServiceClientHandle serv_handle, MatlabBundleEngineThreadInfoHandle info_handle) :
	serv_handle_(serv_handle),
	info_handle_(info_handle)
{
}

MatlabBundleEngineThread::~MatlabBundleEngineThread()
{
}

void MatlabBundleEngineThread::run()
{
	IComPacketHandle	packet;
	bool				done = false;
	while(!done)
	{
		if(!(serv_handle_->recv(packet)))
		{
			info_handle_->dolock();
			if (info_handle_->exit_ == true) 
			{
				// It crashed as result of closing of connection
				// Anyway, the module was destroyed so it should not
				// matter anymore
				info_handle_->wait_code_done_.conditionBroadcast();
				info_handle_->wait_exit_.conditionBroadcast();
				info_handle_->unlock();
				return;
			}
			info_handle_->code_done_ = true;
			info_handle_->code_success_ = false;
			info_handle_->wait_code_done_.conditionBroadcast();
			info_handle_->exit_ = true;
			info_handle_->wait_exit_.conditionBroadcast();
 			info_handle_->unlock();

			done = true;
			continue;
		}

        info_handle_->dolock();       

        if (info_handle_->exit_ == true)
        {
			info_handle_->wait_exit_.conditionBroadcast();
            info_handle_->unlock();
            return;
        } 
		
		switch (packet->gettag())
		{
			case TAG_STDO:
                {
                    std::string str;
                    if (packet->getparam1() < 0) str = "STDOUT END";
                    else str = packet->getstring();
                    std::string cmd = info_handle_->output_cmd_ + " \"" + MatlabBundle::totclstring(str) + "\""; 
                    info_handle_->unlock();       
                    info_handle_->gui_->lock();
                    info_handle_->gui_->execute(cmd);                     
                    info_handle_->gui_->unlock();
                }
				break;
			case TAG_STDE:
                {
                    std::string str;
                    if (packet->getparam1() < 0) str = "STDERR END";
                    else str = packet->getstring();
                    std::string cmd = info_handle_->output_cmd_ + " \"STDERR: " + MatlabBundle::totclstring(str) + "\""; 
                    info_handle_->unlock();       
                    info_handle_->gui_->lock();
                    info_handle_->gui_->execute(cmd);                     
                    info_handle_->gui_->unlock();
                }
				break;
			case TAG_END_:
			case TAG_EXIT:
				info_handle_->code_done_ = true;
				info_handle_->code_success_ = false;
				info_handle_->wait_code_done_.conditionBroadcast();
				info_handle_->exit_ = true;
				info_handle_->wait_exit_.conditionBroadcast();
				done = true;
                info_handle_->unlock();       
			break;
			case TAG_MCODE_SUCCESS:
				info_handle_->code_done_ = true;
				info_handle_->code_success_ = true;
				info_handle_->wait_code_done_.conditionBroadcast();				
                info_handle_->unlock();       
			break;
			case TAG_MCODE_ERROR:
				info_handle_->code_done_ = true;
				info_handle_->code_success_ = false;
				info_handle_->code_error_ = packet->getstring();
				info_handle_->wait_code_done_.conditionBroadcast();				
                info_handle_->unlock();       
			break;
            default:
                info_handle_->unlock();       
		}
	}
}

DECLARE_MAKER(MatlabBundle)

MatlabBundle::MatlabBundle(GuiContext *context) :
  Module("MatlabBundle", context, Filter, "DataIO", "MatlabInterface"), 
  input_bundle_name_(context->subVar("input-bundle-name")),
  input_bundle_array_(context->subVar("input-bundle-array")),
  output_bundle_name_(context->subVar("output-bundle-name")),
  output_bundle_pnrrds_(context->subVar("output-bundle-pnrrds")),
  output_bundle_pbundles_(context->subVar("output-bundle-pbundles")),
  inet_address_(context->subVar("inet-address")),
  inet_port_(context->subVar("inet-port")),
  inet_passwd_(context->subVar("inet-passwd")),
  inet_session_(context->subVar("inet-session")),
  matlab_code_(context->subVar("matlab-code")),
  matlab_code_file_(context->subVar("matlab-code-file")),
  matlab_var_(context->subVar("matlab-var")),
  matlab_add_output_(context->subVar("matlab-add-output")),
  matlab_update_status_(context->subVar("matlab-update-status")),
  need_file_transfer_(false)
{

	// find the input and output ports
	
	int portnum = 0;
	for (int p = 0; p<NUM_BUNDLE_PORTS; p++)  bundle_iport_[p] = static_cast<BundleIPort *>(get_iport(portnum++));

	portnum = 0;
	for (int p = 0; p<NUM_BUNDLE_PORTS; p++)  bundle_oport_[p] = static_cast<BundleOPort *>(get_oport(portnum++));

  input_bundle_name_list_.resize(NUM_BUNDLE_PORTS);
	input_bundle_name_list_old_.resize(NUM_BUNDLE_PORTS);

	input_bundle_generation_old_.resize(NUM_BUNDLE_PORTS);
    for (int p = 0; p<NUM_BUNDLE_PORTS; p++)  input_bundle_generation_old_[p] = -1;

  CleanupManager::add_callback(MatlabBundle::cleanup_callback,reinterpret_cast<void *>(this));

}


// Function for cleaning up
// matlab modules
void MatlabBundle::cleanup_callback(void *data)
{
 
    MatlabBundle* ptr = reinterpret_cast<MatlabBundle *>(data);
    // We just want to make sure that the matlab engine is released and 
    // any temp dirs are cleaned up
    ptr->close_matlab_engine();
    ptr->delete_temp_directory();
}

MatlabBundle::~MatlabBundle()
{
    // Again if we registered a module for destruction and we are removing it
    // we need to unregister
    CleanupManager::invoke_remove_callback(MatlabBundle::cleanup_callback,
					   reinterpret_cast<void *>(this));
}


void	MatlabBundle::update_status(std::string text)
{
	std::string cmd = matlab_update_status_.get() + " \"" + totclstring(text) + "\"";
	gui->execute(cmd);
}

// converttcllist:
// converts a TCL formatted list into a STL array
// of strings

std::vector<std::string> MatlabBundle::converttcllist(std::string str)
{
	std::string result;
	std::vector<std::string> list(0);
	long lengthlist = 0;
	
	// Yeah, it is TCL dependent:
	// TCL::llength determines the length of the list
	gui->lock();
	gui->eval("llength { "+str + " }",result);	
	istringstream iss(result);
	iss >> lengthlist;
	gui->unlock();
	if (lengthlist < 0) return(list);
	
	list.resize(lengthlist);
	gui->lock();
	for (long p = 0;p<lengthlist;p++)
	{
		ostringstream oss;
		// TCL dependency:
		// TCL::lindex retrieves the p th element from the list
		oss << "lindex { " << str <<  " } " << p;
		gui->eval(oss.str(),result);
		list[p] = result;
	}
	gui->unlock();
	return(list);
}


bool MatlabBundle::synchronise_input()
{

	gui->execute(id+" Synchronise");
	ctx->reset();

	std::string str;

	str = input_bundle_name_.get(); input_bundle_name_list_ = converttcllist(str);
	str = input_bundle_array_.get(); input_bundle_array_list_ = converttcllist(str);
	str = output_bundle_name_.get(); output_bundle_name_list_ = converttcllist(str);
	str = output_bundle_pnrrds_.get(); output_bundle_pnrrds_list_ = converttcllist(str);
	str = output_bundle_pbundles_.get(); output_bundle_pbundles_list_ = converttcllist(str);

        gui->execute(id + " update_text"); // update matlab_code_ before use.
	matlab_code_list_ = matlab_code_.get(); 
	
	return(true);
}


void MatlabBundle::execute()
{
	// Synchronise input: translate TCL lists into C++ STL lists
	if (!(synchronise_input()))
	{
		error("MatlabBundle: Could not retreive GUI input");
		return;
	}

	// If we haven't created a temporary directory yet
	// we open one to store all temp files in
	if (!(create_temp_directory()))
	{
		error("MatlabBundle: Could not create temporary directory");
		return;
	}

	if (!(open_matlab_engine()))
	{
		error("MatlabBundle: Could not open matlab engine");
		return;
	}

	if (!(save_input_matrices()))
	{
		error("MatlabBundle: Could not create the input matrices");
		return;
	}

	if (!(generate_matlab_code()))
	{
		error("MatlabBundle: Could not create m-file code for matlabengine");
		return;
	}	

	if (!send_matlab_job())
	{
	   error("MatlabBundle: MatlabBundle returned an error or Matlab could not be launched");
	   return;
	}
	
	if (!load_output_matrices())
	{
		error("MatlabBundle: Could not load matrices that matlab generated");
		return;
	}
}

bool MatlabBundle::send_matlab_job()
{
	IComPacketHandle packet = scinew IComPacket;
	
	if (packet.get_rep() == 0)
	{
		error("MatlabBundle: Could not create packet");
		return(false);
	}

	thread_info_->dolock();
	thread_info_->code_done_ = false;
	thread_info_->unlock();
	
	packet->settag(TAG_MCODE);
  std::string mfilename = mfile_.substr(0,mfile_.size()-2);
	packet->setstring(file_transfer_->remote_file(mfilename)); // strip the .m
	matlab_engine_->send(packet);

	thread_info_->dolock();
	if (!thread_info_->code_done_)
	{
		thread_info_->wait_code_done_.wait(thread_info_->lock);
	}
	bool success = thread_info_->code_success_;
	bool exitcond = thread_info_->exit_;
	if (!success) 
	{
		if (exitcond)
		{
			error("MatlabBundle: the matlab engine crashed or did not start: "+ thread_info_->code_error_);
      error("MatlabBundle: possible causes are:");
			error("(1) matlab code failed in such a way that the engine was no able to catch the error (e.g. failure mex of code)");
			error("(2) matlab crashed and the matlab engine detected an end of communication of the matlab process");
			error("(3) temparory files could not be created or are corrupted");
			error("(4) improper matlab version, use matlab V5 or higher, currently matlab V5-V7 are supported");
		}
		else
		{
			error("MatlabBundle: matlab code failed: "+thread_info_->code_error_);
			error("MatlabBundle: Detected an error in the Matlab code, the matlab engine is still running and caught the exception");
			error("MatlabBundle: Please check the matlab code in the GUI and try again. The output window in the GUI should contain the reported error message generated by matlab");            
		}
		thread_info_->code_done_ = false;
		thread_info_->unlock();
		return(false);
	}
	thread_info_->code_done_ = false;
	thread_info_->unlock();
	
	return(success);
}



void MatlabBundle::presave()
{
  gui->execute(id + " update_text"); // update matlab_code_ before use.
}



bool MatlabBundle::send_input(std::string str)
{
	IComPacketHandle packet = scinew IComPacket;
	
  if (matlab_engine_.get_rep() == 0) return(true);
  
	if (packet.get_rep() == 0)
	{
		error("MatlabBundle: Could not create packet");
		return(false);
	}
	
	packet->settag(TAG_INPUT);
	packet->setstring(str); 
  
	matlab_engine_->send(packet);
	
	return(true);
}

bool MatlabBundle::open_matlab_engine()
{

	std::string inetaddress = inet_address_.get();
	std::string inetport = inet_port_.get();
	std::string passwd = inet_passwd_.get();
	std::string session = inet_session_.get();

	// Check whether some touched the engine settings
	// if so close the engine and record the current settings
	if ((inetaddress != inet_address_old_)||(inetport != inet_port_old_)||(passwd != inet_passwd_old_)||(session != inet_session_old_))
	{
		close_matlab_engine();
		inet_address_old_ = inetaddress;
		inet_port_old_ = inetport;
		inet_passwd_old_ = passwd;
		inet_session_old_ = session;
	}

	if (!(matlab_engine_.get_rep()))
	{
    
		IComAddress address;
		if (inetaddress == "")
		{
			address.setaddress("internal","servicemanager");
		}
		else
		{
			address.setaddress("scirun",inetaddress,inetport);
		}
		
		int sessionnum = 0;
		std::istringstream iss(session);
		iss >> sessionnum;

		// Inform the impatient user we are still working for him
		update_status("Please wait while launching matlab, this may take a few minutes ....\n");
		
		matlab_engine_ = scinew ServiceClient();
		if(!(matlab_engine_->open(address,"matlabengine",sessionnum,passwd)))
		{
			error(std::string("MatlabBundle: Could not open matlab engine (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("MatlabBundle: Make sure the matlab engine has not been disabled in $HOME/SCIRun/services/matlabengine.rc"));
			error(std::string("MatlabBundle: Check remote address information, or leave all fields except 'session' blank to connect to local matlab engine"));
			error(std::string("MatlabBundle: If using matlab engine on local machine start engine with '-eai' option"));
			
			matlab_engine_ = 0;
			return(false);
        }
        
        file_transfer_ = scinew FileTransferClient();
        if(!(file_transfer_->open(address,"matlabenginefiletransfer",sessionnum,passwd)))
		{
            matlab_engine_->close();

			error(std::string("MatlabBundle: Could not open matlab engine file transfer service (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("MatlabBundle: Make sure the matlab engine file transfer service has not been disabled in $HOME/SCIRun/services/matlabengine.rc"));
			error(std::string("MatlabBundle: Check remote address information, or leave all fields except 'session' blank to connect to local matlab engine"));
			error(std::string("MatlabBundle: If using matlab engine on local machine start engine with '-eai' option"));
			
            matlab_engine_ = 0;
			file_transfer_ = 0;
			return(false);
        }
        
		IComPacketHandle packet;
		if(!(matlab_engine_->recv(packet)))
		{
            matlab_engine_->close();
            file_transfer_->close();
			error(std::string("MatlabBundle: Could not get answer from matlab engine (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("MatlabBundle: This is an internal communication error, make sure that the portnumber is correct"));
			error(std::string("MatlabBundle: If address information is correct, this most probably points to a bug in the SCIRun software"));

			matlab_engine_ = 0;
			file_transfer_ = 0;

			return(false);	
		}
		
		if (packet->gettag() == TAG_MERROR)
		{
			matlab_engine_->close();
            file_transfer_->close();

			error(std::string("MatlabBundle: Matlab engine returned an error (error=") + packet->getstring() + std::string(")"));
			error(std::string("MatlabBundle: Please check whether '$HOME/SCIRun/services/matlabengine.rc' has been setup properly"));
			error(std::string("MatlabBundle: Edit the 'startmatlab=' line to start matlab properly"));
			error(std::string("MatlabBundle: If you running matlab remotely, this file must be edited on the machine running matlab"));

			matlab_engine_ = 0;
			file_transfer_ = 0;

			return(false);					
		}

		thread_info_ = scinew MatlabBundleEngineThreadInfo();
		if (thread_info_.get_rep() == 0)
		{
			matlab_engine_->close();
            file_transfer_->close();

			error(std::string("MatlabBundle: Could not create thread information object"));
			matlab_engine_ = 0;
			file_transfer_ = 0;

			return(false);		
		}


        need_file_transfer_ = false;
        std::string localid;
        std::string remoteid;
        file_transfer_->get_local_homedirid(localid);
        file_transfer_->get_remote_homedirid(remoteid);
        if (localid != remoteid)
        {
            need_file_transfer_ = true;
            if(!(file_transfer_->create_remote_tempdir("matlab-engine.XXXXXX",remote_tempdir_)))
            {
                matlab_engine_->close();
                file_transfer_->close();

                error(std::string("MatlabBundle: Could not create remote temporary directory"));
                matlab_engine_ = 0;
                file_transfer_ = 0;
                return(false);		
            }
            file_transfer_->set_local_dir(temp_directory_);
            file_transfer_->set_remote_dir(remote_tempdir_);
        }
        else
        {
            // Although they might share a home directory
            // This directory can be mounted at different trees
            // Hence we translate between both. MatlabBundle does not like
            // the use of $HOME
            file_transfer_->set_local_dir(temp_directory_);
            std::string tempdir = temp_directory_;
            file_transfer_->translate_scirun_tempdir(tempdir);
            file_transfer_->set_remote_dir(tempdir);
        }

		
		thread_info_->gui_ = gui;
		thread_info_->output_cmd_ = matlab_add_output_.get(); 
		// By cloning the object, it will have the same fields and sockets, but the socket
		// and error handling will be separate. As the thread will invoke its own instructions
		// it is better to have a separate copy. Besides, the socket obejct will point to the
		// same underlying socket. Hence only the error handling part will be duplicated
		ServiceClientHandle matlab_engine_copy = matlab_engine_->clone();
		MatlabBundleEngineThread* enginethread = scinew MatlabBundleEngineThread(matlab_engine_copy,thread_info_);
		if (enginethread == 0)
		{
			matlab_engine_->close();
            file_transfer_->close();

			matlab_engine_ = 0;
			file_transfer_ = 0;

			error(std::string("MatlabBundle: Could not create thread object"));
			return(false);
		}
		
		Thread* thread = scinew Thread(enginethread,"MatlabBundle module thread");
		if (thread == 0)
		{
			delete enginethread;
			matlab_engine_->close();
            file_transfer_->close();

			matlab_engine_ = 0;
			file_transfer_ = 0;

			error(std::string("MatlabBundle: Could not create thread"));
			return(false);	
		}
		thread->detach();
	
		int sessionn = packet->getparam1();
		matlab_engine_->setsession(sessionn);
            	
        std::string sharehomedir = "yes";
        if (need_file_transfer_) sharehomedir = "no";
               
		std::string status = "Matlab engine running\n\nmatlabengine version: " + matlab_engine_->getversion() + "\nmatlabengine address: " +
			matlab_engine_->getremoteaddress() + "\nmatlabengine session: " + matlab_engine_->getsession() + "\nmatlabengine filetransfer version :" +
            file_transfer_->getversion() + "\nshared home directory: " + sharehomedir + "\nlocal temp directory: " + file_transfer_->local_file("") +
            "\nremote temp directory: " + file_transfer_->remote_file("") + "\n";
		update_status(status);
	}

	return(true);
}


bool MatlabBundle::close_matlab_engine()
{

	if (matlab_engine_.get_rep()) 
	{
		matlab_engine_->close();
		matlab_engine_ = 0;
	}

	if (file_transfer_.get_rep()) 
	{
		file_transfer_->close();
		file_transfer_ = 0;
	}

	if (thread_info_.get_rep())
	{
		thread_info_->dolock();
        if (thread_info_->exit_ == false)
        {
            thread_info_->exit_ = true;
            thread_info_->wait_exit_.conditionBroadcast();
            thread_info_->wait_exit_.wait(thread_info_->lock);
        }
		thread_info_->unlock();
		thread_info_ = 0;
	}
	
	
	return(true);
}


bool MatlabBundle::load_output_matrices()
{
	try
	{

		for (int p = 0; p < NUM_BUNDLE_PORTS; p++)
		{
			// Test whether the bundle port exists
			if (bundle_oport_[p] == 0) continue;
			if (bundle_oport_[p]->nconnections() == 0) continue;
			if (output_bundle_name_list_[p] == "") continue;
			if (output_bundle_matfile_[p] == "") continue;
		
			matlabfile mf;
			matlabarray ma;
			try
			{
        if (need_file_transfer_) file_transfer_->get_file(file_transfer_->remote_file(output_bundle_matfile_[p]),file_transfer_->local_file(output_bundle_matfile_[p]));

				mf.open(file_transfer_->local_file(output_bundle_matfile_[p]),"r");
				ma = mf.getmatlabarray(output_bundle_name_list_[p]);
				mf.close();
			}
			catch(...)
			{
				error("MatlabBundle: Could not read output matrix");
				continue;
			}
			
			if (ma.isempty())
			{
				error("MatlabBundle: Could not read output matrix");
				continue;
			}
			
			BundleHandle handle;
      std::string info;
      
      translate_.prefersciobjects();
      translate_.prefermatrices();
     	if (output_bundle_pnrrds_list_[p] == "prefer nrrds") translate_.prefernrrds();
     	if (output_bundle_pbundles_list_[p] == "prefer bundles") translate_.preferbundles();
			if (translate_.sciBundleCompatible(ma,info,static_cast<SCIRun::Module *>(this))) translate_.mlArrayTOsciBundle(ma,handle,static_cast<SCIRun::Module *>(this));
			bundle_oport_[p]->send(handle);
		}
	}
	catch(...)
	{
		return(false);
	}
	return(true);
}



bool MatlabBundle::generate_matlab_code()
{
	try
	{
		std::ofstream m_file;
		
		mfile_ = std::string("scirun_code.m");
    std::string filename = file_transfer_->local_file(mfile_);
		m_file.open(filename.c_str(),std::ios::app);

		m_file << matlab_code_list_ << "\n";

		for (int p = 0; p < NUM_BUNDLE_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (bundle_oport_[p] == 0) continue;
			if (output_bundle_name_list_[p] == "") continue;
		
			ostringstream oss;
			oss << "output_bundle" << p << ".mat";
			output_bundle_matfile_[p] = oss.str();
			std::string cmd;
			cmd = "if exist('" + output_bundle_name_list_[p] + "','var'), save " + file_transfer_->remote_file(output_bundle_matfile_[p]) + " " + output_bundle_name_list_[p] + "; end\n";
			m_file << cmd;
		}
		

		m_file.close();
    if (need_file_transfer_) file_transfer_->put_file(file_transfer_->local_file(mfile_),file_transfer_->remote_file(mfile_));
        
	}
	catch(...)
	{
		return(false);
	}

	return(true);
}



bool MatlabBundle::save_input_matrices()
{

	try
	{

		std::ofstream m_file; 
		std::string loadcmd;

		mfile_ = std::string("scirun_code.m");
        std::string filename = file_transfer_->local_file(mfile_);

		m_file.open(filename.c_str(),std::ios::out);

		for (int p = 0; p < NUM_BUNDLE_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (bundle_iport_[p] == 0) continue;
			if (bundle_iport_[p]->nconnections() == 0) continue;
			
			BundleHandle	handle = 0;
			bundle_iport_[p]->get(handle);
			// if there is no data
			if (handle.get_rep() == 0) 
			{
				// we do not need the old file any more so delete it
				input_bundle_matfile_[p].clear();
				continue;
			}
			// if the data as the same before
			// do nothing
			if ((handle == bundle_handle_[p])&&(input_bundle_name_list_[p]==input_bundle_name_list_old_[p])&&(bundle_handle_[p]->generation == input_bundle_generation_old_[p]))
			{
				// this one was not created again
				// hence we do not need to translate it again
				// with big datasets this should improve performance
				loadcmd = "load " + file_transfer_->remote_file(input_bundle_matfile_[p]) + ";\n";
				m_file << loadcmd;

				continue;
			}
			
			bundle_handle_[p] = handle;
			
			// Create a new filename for the input matrix
			ostringstream oss;
			oss << "input_bundle" << p << ".mat";
			input_bundle_matfile_[p] = oss.str();
			
			matlabfile mf;
			matlabarray ma;

			mf.open(file_transfer_->local_file(input_bundle_matfile_[p]),"w");
			mf.setheadertext("Matlab V5 compatible file generated by SCIRun [module MatlabBundle version 1.0]");

			translate_.converttostructmatrix();
			if (input_bundle_array_list_[p] == "numeric array") translate_.converttonumericmatrix();
			translate_.sciBundleTOmlArray(handle,ma,static_cast<SCIRun::Module *>(this));
			
			mf.putmatlabarray(ma,input_bundle_name_list_[p]);
			mf.close();
			
			loadcmd = "load " + file_transfer_->remote_file(input_bundle_matfile_[p]) + ";\n";
			m_file << loadcmd;
            
      if (need_file_transfer_) 
      {
          if(!(file_transfer_->put_file(file_transfer_->local_file(input_bundle_matfile_[p]),file_transfer_->remote_file(input_bundle_matfile_[p]))))
          {
              error("MatlabBundle: Could not transfer file");
              std::string err = "Error :" + file_transfer_->geterror();
              error(err);
              return(false);
          }
          
      }
      input_bundle_name_list_old_[p] = input_bundle_name_list_[p];
      input_bundle_generation_old_[p] = handle->generation;            
    }
	}
	catch (matlabfile::could_not_open_file)
	{   // Could not open the temporary file
		error("MatlabBundle: Could not open temporary matlab file");
		return(false);
	}
	catch (matlabfile::io_error)
	{   // IO error from ferror
		error("MatlabBundle: IO error");
		return(false);		
	}
	catch (matlabfile::matfileerror) 
	{   // All other errors are classified as internal
		// matfileerrror is the base class on which all
		// other exceptions are based.
		error("MatlabBundle: Internal error in writer");
		return(false);		
	}
	
	return(true);
}


bool MatlabBundle::create_temp_directory()
{
	if (temp_directory_ == "")
	{
		return(tfmanager_.create_tempdir("matlab-engine.XXXXXX",temp_directory_));
	}
	return(true);
}


bool MatlabBundle::delete_temp_directory()
{
    if(temp_directory_ != "") tfmanager_.delete_tempdir(temp_directory_);
	temp_directory_ = "";
	return(true);
}

std::string MatlabBundle::totclstring(std::string &instring)
{
	int strsize = instring.size();
	int specchar = 0;
	for (int p = 0; p < strsize; p++)
		if ((instring[p]=='\n')||(instring[p]=='\t')||(instring[p]=='\b')||(instring[p]=='\r')||(instring[p]=='{')||(instring[p]=='}')
				||(instring[p]=='[')||(instring[p]==']')||(instring[p]=='\\')||(instring[p]=='$')||(instring[p]=='"')) specchar++;
	
	std::string newstring;
	newstring.resize(strsize+specchar);
	int q = 0;
	for (int p = 0; p < strsize; p++)
	{
		if (instring[p]=='\n') { newstring[q++] = '\\'; newstring[q++] = 'n'; continue; }
		if (instring[p]=='\t') { newstring[q++] = '\\'; newstring[q++] = 't'; continue; }
		if (instring[p]=='\b') { newstring[q++] = '\\'; newstring[q++] = 'b'; continue; }
		if (instring[p]=='\r') { newstring[q++] = '\\'; newstring[q++] = 'r'; continue; }
		if (instring[p]=='{')  { newstring[q++] = '\\'; newstring[q++] = '{'; continue; }
		if (instring[p]=='}')  { newstring[q++] = '\\'; newstring[q++] = '}'; continue; }
		if (instring[p]=='[')  { newstring[q++] = '\\'; newstring[q++] = '['; continue; }
		if (instring[p]==']')  { newstring[q++] = '\\'; newstring[q++] = ']'; continue; }
		if (instring[p]=='\\') { newstring[q++] = '\\'; newstring[q++] = '\\'; continue; }
		if (instring[p]=='$')  { newstring[q++] = '\\'; newstring[q++] = '$'; continue; }
		if (instring[p]=='"')  { newstring[q++] = '\\'; newstring[q++] = '"'; continue; }
		newstring[q++] = instring[p];
	}
	
	return(newstring);
}

void MatlabBundle::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() > 1)
  {
    if (args[1] == "keystroke")
    {
        std::string str = args[2];
        if (str.size() == 1)
        {
            if (str[0] == '\r') str[0] = '\n';
            
            if (str[0] == '\b')
            {
              inputstring_ = inputstring_.substr(0,(inputstring_.size()-1));            
            }
            else
            {
              inputstring_ += str;
            }
            
            if (str[0] == '\n')
            {
              if(!(send_input(inputstring_)))
              {
                  error("Matlab: Could not close matlab engine");
                  return;
              }
              inputstring_ = "";
            }
        }
        else
        {
            std::string key = args[3];
            if (key == "Enter") 
            {
              str = "\n";
              inputstring_ += str;
            }
            else if (key == "BackSpace") 
            {
              inputstring_ = inputstring_.substr(0,(inputstring_.size()-1));
            }
            else if (key == "Tab") str = "\t";
            else if (key == "Return") str ="\r";
            
            if (str.size() == 1)
            {
                if (str[0] == '\n')
                {
                  if(!(send_input(inputstring_)))
                  {
                      error("Matlab: Could not close matlab engine");
                      return;
                  }
                  inputstring_ = "";
                }
            }    
            
        }
        return;
    }

    if (args[1] == "disconnect")
    {
        ctx->reset();
        if(!(close_matlab_engine()))
        {
            error("Matlab: Could not close matlab engine");
            return;
        }
        update_status("Matlab engine not running\n");
        return;
    }


    if (args[1] == "connect")
    {
        
        if(!(close_matlab_engine()))
        {
            error("MatlabBundle: Could not close matlab engine");
            return;
        }

        update_status("Matlab engine not running\n");

        // Synchronise input: translate TCL lists into C++ STL lists
        if (!(synchronise_input()))
        {
            error("MatlabBundle: Could not retreive GUI input");
            return;
        }

        // If we haven't created a temporary directory yet
        // we open one to store all temp files in
        if (!(create_temp_directory()))
        {
            error("MatlabBundle: Could not create temporary directory");
            return;
        }

        if (!(open_matlab_engine()))
        {
            error("MatlabBundle: Could not open matlab engine");
            return;
        }
        return;
    }

  }

  Module::tcl_command(args, userdata);
}

} // End namespace MatlabBundleInterface

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#pragma reset woff 1209 
#endif
