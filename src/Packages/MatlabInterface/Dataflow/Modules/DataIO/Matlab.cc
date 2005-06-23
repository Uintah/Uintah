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
 *  Matlab.cc:
 *
 *  Written by:
 *   Jeroen Stinstra
 *
 */




#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/NrrdPort.h>
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

class Matlab;
class MatlabEngineThreadInfo;
class MatlabEngineThread;

typedef LockingHandle<MatlabEngineThreadInfo> MatlabEngineThreadInfoHandle;

class MatlabEngineThread : public Runnable, public ServiceBase 
{
  public:
	MatlabEngineThread(ServiceClientHandle serv_handle,MatlabEngineThreadInfoHandle info_handle);
    virtual ~MatlabEngineThread();
	void run();

  private:
	ServiceClientHandle serv_handle_;
	MatlabEngineThreadInfoHandle info_handle_;
};

class MatlabEngineThreadInfo 
{
  public:
	MatlabEngineThreadInfo();
	virtual ~MatlabEngineThreadInfo();

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
	std::string			code_error_;
	
	ConditionVariable   wait_exit_;
	bool				exit_;
	bool				passed_test_;

};


class Matlab : public Module, public ServiceBase 
{
  
  public:
    // Constructor
	Matlab(GuiContext* ctx);

    // Destructor
	virtual ~Matlab();
	
	// Std functions for each module
	// execute():
	//   Execute the module and put data on the output port
	
	virtual void execute();
        virtual void presave();
  
        virtual void tcl_command(GuiArgs& args, void* userdata);

    
	static matlabarray::mitype			convertdataformat(std::string dataformat);
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
    bool    send_input(std::string str);

	
	bool	synchronise_input();
	
  private:
  
	 enum { NUM_MATRIX_PORTS = 5 };
	 enum { NUM_FIELD_PORTS = 3 };
	 enum { NUM_NRRD_PORTS = 3 };
  
	// Temp directory for writing files coming from the 
	// the matlab engine
	
	std::string temp_directory_;
	
	// GUI variables
	
	// Names of matrices
	GuiString   input_matrix_name_;
	GuiString   input_field_name_;
	GuiString   input_nrrd_name_;
	GuiString   input_matrix_type_;
	GuiString   input_nrrd_type_;
	GuiString   input_matrix_array_;
	GuiString   input_field_array_;
	GuiString   input_nrrd_array_;
	GuiString   output_matrix_name_;
	GuiString   output_field_name_;
	GuiString   output_nrrd_name_;


	// Fields per port
	std::vector<std::string>   input_matrix_name_list_;
	std::vector<std::string>   input_matrix_name_list_old_;
	std::vector<std::string>   input_field_name_list_;
	std::vector<std::string>   input_field_name_list_old_;
	std::vector<std::string>   input_nrrd_name_list_;
	std::vector<std::string>   input_nrrd_name_list_old_;
	std::vector<std::string>   input_matrix_type_list_;
	std::vector<std::string>   input_nrrd_type_list_;
	std::vector<std::string>   input_matrix_array_list_;
	std::vector<std::string>   input_field_array_list_;
	std::vector<std::string>   input_nrrd_array_list_;
	std::vector<std::string>   output_matrix_name_list_;
	std::vector<std::string>   output_field_name_list_;
	std::vector<std::string>   output_nrrd_name_list_;

    std::vector<int> input_matrix_generation_old_;
    std::vector<int> input_field_generation_old_;
    std::vector<int> input_nrrd_generation_old_;

	std::string	matlab_code_list_;
	
	// Ports for input and output
	MatrixIPort*	matrix_iport_[NUM_MATRIX_PORTS];
	MatrixOPort*	matrix_oport_[NUM_MATRIX_PORTS];
	
	FieldIPort*		field_iport_[NUM_FIELD_PORTS];
	FieldOPort*		field_oport_[NUM_FIELD_PORTS];
	
	NrrdIPort*		nrrd_iport_[NUM_NRRD_PORTS];
	NrrdOPort*		nrrd_oport_[NUM_NRRD_PORTS];
	
	MatrixHandle	matrix_handle_[NUM_MATRIX_PORTS];
	FieldHandle		field_handle_[NUM_FIELD_PORTS];
	NrrdDataHandle  nrrd_handle_[NUM_NRRD_PORTS];

	std::string		input_matrix_matfile_[NUM_MATRIX_PORTS];
	std::string		input_field_matfile_[NUM_FIELD_PORTS];
	std::string		input_nrrd_matfile_[NUM_NRRD_PORTS];
	
	std::string		output_matrix_matfile_[NUM_MATRIX_PORTS];
	std::string		output_field_matfile_[NUM_FIELD_PORTS];
	std::string		output_nrrd_matfile_[NUM_NRRD_PORTS];
	
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
	
	// Matlab converter class
	matlabconverter translate_;
	
	std::string		mfile_;
	
	GuiString		matlab_code_;
  GuiString   matlab_code_file_;
	GuiString		matlab_var_;
	GuiString		matlab_add_output_;
	GuiString		matlab_update_status_;
    
	ServiceClientHandle				matlab_engine_;
  FileTransferClientHandle        file_transfer_;
	MatlabEngineThreadInfoHandle	thread_info_;
    
  bool            need_file_transfer_;
  std::string     remote_tempdir_;
    
  std::string     inputstring_;
     
  public:
    static void cleanup_callback(void *data);
};


MatlabEngineThreadInfo::MatlabEngineThreadInfo() :
	lock("MatlabEngineInfo lock"),
	ref_cnt(0),
    gui_(0),
	wait_code_done_("MatlabEngineInfo condition variable code"),
	code_done_(false),
	code_success_(false),
	wait_exit_("MatlabEngineInfo condition variable exit"),
	exit_(false),
	passed_test_(false)
{
}

MatlabEngineThreadInfo::~MatlabEngineThreadInfo()
{
}

inline void MatlabEngineThreadInfo::dolock()
{
	lock.lock();
}

inline void MatlabEngineThreadInfo::unlock()
{
	lock.unlock();
}

MatlabEngineThread::MatlabEngineThread(ServiceClientHandle serv_handle, MatlabEngineThreadInfoHandle info_handle) :
	serv_handle_(serv_handle),
	info_handle_(info_handle)
{
}

MatlabEngineThread::~MatlabEngineThread()
{
}

void MatlabEngineThread::run()
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
                    std::string cmd = info_handle_->output_cmd_ + " \"" + Matlab::totclstring(str) + "\""; 
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
                    std::string cmd = info_handle_->output_cmd_ + " \"STDERR: " + Matlab::totclstring(str) + "\""; 
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


DECLARE_MAKER(Matlab)

Matlab::Matlab(GuiContext *context) :
  Module("Matlab", context, Filter, "DataIO", "MatlabInterface"), 
  input_matrix_name_(context->subVar("input-matrix-name")),
  input_field_name_(context->subVar("input-field-name")),
  input_nrrd_name_(context->subVar("input-nrrd-name")),
  input_matrix_type_(context->subVar("input-matrix-type")),
  input_nrrd_type_(context->subVar("input-nrrd-type")),
  input_matrix_array_(context->subVar("input-matrix-array")),
  input_field_array_(context->subVar("input-field-array")),
  input_nrrd_array_(context->subVar("input-nrrd-array")),
  output_matrix_name_(context->subVar("output-matrix-name")),
  output_field_name_(context->subVar("output-field-name")),
  output_nrrd_name_(context->subVar("output-nrrd-name")),
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
	for (int p = 0; p<NUM_MATRIX_PORTS; p++)  matrix_iport_[p] = static_cast<MatrixIPort *>(get_iport(portnum++));
	for (int p = 0; p<NUM_FIELD_PORTS; p++)  field_iport_[p] = static_cast<FieldIPort *>(get_iport(portnum++));
	for (int p = 0; p<NUM_NRRD_PORTS; p++)  nrrd_iport_[p] = static_cast<NrrdIPort *>(get_iport(portnum++));

	portnum = 0;
	for (int p = 0; p<NUM_MATRIX_PORTS; p++)  matrix_oport_[p] = static_cast<MatrixOPort *>(get_oport(portnum++));
	for (int p = 0; p<NUM_FIELD_PORTS; p++)  field_oport_[p] = static_cast<FieldOPort *>(get_oport(portnum++));
	for (int p = 0; p<NUM_NRRD_PORTS; p++)  nrrd_oport_[p] = static_cast<NrrdOPort *>(get_oport(portnum++));

	input_matrix_name_list_.resize(NUM_MATRIX_PORTS);
	input_matrix_name_list_old_.resize(NUM_MATRIX_PORTS);
    input_field_name_list_.resize(NUM_FIELD_PORTS);
	input_field_name_list_old_.resize(NUM_FIELD_PORTS);
	input_nrrd_name_list_.resize(NUM_NRRD_PORTS);
	input_nrrd_name_list_old_.resize(NUM_NRRD_PORTS);

	input_matrix_generation_old_.resize(NUM_MATRIX_PORTS);
    for (int p = 0; p<NUM_MATRIX_PORTS; p++)  input_matrix_generation_old_[p] = -1;
	input_field_generation_old_.resize(NUM_MATRIX_PORTS);
    for (int p = 0; p<NUM_FIELD_PORTS; p++)  input_field_generation_old_[p] = -1;
	input_nrrd_generation_old_.resize(NUM_MATRIX_PORTS);
    for (int p = 0; p<NUM_NRRD_PORTS; p++)  input_nrrd_generation_old_[p] = -1;

  CleanupManager::add_callback(Matlab::cleanup_callback,reinterpret_cast<void *>(this));


}


// Function for cleaning up
// matlab modules
void Matlab::cleanup_callback(void *data)
{
 
    Matlab* ptr = reinterpret_cast<Matlab *>(data);
    // We just want to make sure that the matlab engine is released and 
    // any temp dirs are cleaned up
    ptr->close_matlab_engine();
    ptr->delete_temp_directory();
    
}

Matlab::~Matlab()
{
    // Again if we registered a module for destruction and we are removing it
    // we need to unregister
    CleanupManager::invoke_remove_callback(Matlab::cleanup_callback,
					   reinterpret_cast<void *>(this));
}


void	Matlab::update_status(std::string text)
{
	std::string cmd = matlab_update_status_.get() + " \"" + totclstring(text) + "\"";
	gui->execute(cmd);
}

matlabarray::mitype Matlab::convertdataformat(std::string dataformat)
{
	matlabarray::mitype type = matlabarray::miUNKNOWN;
	if (dataformat == "same as data")  { type = matlabarray::miSAMEASDATA; }
	else if (dataformat == "double")   { type = matlabarray::miDOUBLE; }
	else if (dataformat == "single")   { type = matlabarray::miSINGLE; }
	else if (dataformat == "uint64")   { type = matlabarray::miUINT64; }
	else if (dataformat == "int64")    { type = matlabarray::miINT64; }
	else if (dataformat == "uint32")   { type = matlabarray::miUINT32; }
	else if (dataformat == "int32")    { type = matlabarray::miINT32; }
	else if (dataformat == "uint16")   { type = matlabarray::miUINT16; }
	else if (dataformat == "int16")    { type = matlabarray::miINT16; }
	else if (dataformat == "uint8")    { type = matlabarray::miUINT8; }
	else if (dataformat == "int8")     { type = matlabarray::miINT8; }
	return (type);
}

// converttcllist:
// converts a TCL formatted list into a STL array
// of strings

std::vector<std::string> Matlab::converttcllist(std::string str)
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


bool Matlab::synchronise_input()
{

	gui->execute(id+" Synchronise");
	ctx->reset();

	std::string str;
	str = input_matrix_name_.get(); input_matrix_name_list_ = converttcllist(str);
	str = input_matrix_type_.get(); input_matrix_type_list_ = converttcllist(str);
	str = input_matrix_array_.get(); input_matrix_array_list_ = converttcllist(str);
	str = output_matrix_name_.get(); output_matrix_name_list_ = converttcllist(str);

	str = input_field_name_.get(); input_field_name_list_ = converttcllist(str);
	str = input_field_array_.get(); input_field_array_list_ = converttcllist(str);
	str = output_field_name_.get(); output_field_name_list_ = converttcllist(str);

	str = input_nrrd_name_.get(); input_nrrd_name_list_ = converttcllist(str);
	str = input_nrrd_type_.get(); input_nrrd_type_list_ = converttcllist(str);
	str = input_nrrd_array_.get(); input_nrrd_array_list_ = converttcllist(str);
	str = output_nrrd_name_.get(); output_nrrd_name_list_ = converttcllist(str);

        gui->execute(id + " update_text"); // update matlab_code_ before use.
	matlab_code_list_ = matlab_code_.get(); 
	
	return(true);
}


void Matlab::execute()
{
	// Synchronise input: translate TCL lists into C++ STL lists
	if (!(synchronise_input()))
	{
		error("Matlab: Could not retreive GUI input");
		return;
	}

	// If we haven't created a temporary directory yet
	// we open one to store all temp files in
	if (!(create_temp_directory()))
	{
		error("Matlab: Could not create temporary directory");
		return;
	}

	if (!(open_matlab_engine()))
	{
		error("Matlab: Could not open matlab engine");
		return;
	}

	if (!(save_input_matrices()))
	{
		error("Matlab: Could not create the input matrices");
		return;
	}

	if (!(generate_matlab_code()))
	{
		error("Matlab: Could not create m-file code for matlabengine");
		return;
	}	

	
	if (!send_matlab_job())
	{
	   error("Matlab: Matlab returned an error or Matlab could not be launched");
	   return;
	}
	
	if (!load_output_matrices())
	{
		error("Matlab: Could not load matrices that matlab generated");
		return;
	}
}


void Matlab::presave()
{
  gui->execute(id + " update_text");  // update matlab-code before saving.
}

bool Matlab::send_matlab_job()
{
	IComPacketHandle packet = scinew IComPacket;
	
	if (packet.get_rep() == 0)
	{
		error("Matlab: Could not create packet");
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
			error("Matlab: the matlab engine crashed or did not start: "+ thread_info_->code_error_);
            error("Matlab: possible causes are:");
			error("(1) matlab code failed in such a way that the engine was no able to catch the error (e.g. failure mex of code)");
			error("(2) matlab crashed and the matlab engine detected an end of communication of the matlab process");
			error("(3) temparory files could not be created or are corrupted");
			error("(4) improper matlab version, use matlab V5 or higher, currently matlab V5-V7 are supported");
		}
		else
		{
			error("Matlab: matlab code failed: "+thread_info_->code_error_);
			error("Matlab: Detected an error in the Matlab code, the matlab engine is still running and caught the exception");
			error("Matlab: Please check the matlab code in the GUI and try again. The output window in the GUI should contain the reported error message generated by matlab");            
		}
		thread_info_->code_done_ = false;
		thread_info_->unlock();
		return(false);
	}
	thread_info_->code_done_ = false;
	thread_info_->unlock();
	
	return(success);
}



bool Matlab::send_input(std::string str)
{
	IComPacketHandle packet = scinew IComPacket;
	
  if (matlab_engine_.get_rep() == 0) return(true);
    
	if (packet.get_rep() == 0)
	{
		error("Matlab: Could not create packet");
		return(false);
	}
	
	packet->settag(TAG_INPUT);
	packet->setstring(str); 
	matlab_engine_->send(packet);
	
	return(true);
}




bool Matlab::open_matlab_engine()
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
			error(std::string("Matlab: Could not open matlab engine (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("Matlab: Make sure the matlab engine has not been disabled in $HOME/SCIRun/services/matlabengine.rc"));
			error(std::string("Matlab: Check remote address information, or leave all fields except 'session' blank to connect to local matlab engine"));
			error(std::string("Matlab: If using matlab engine on local machine start engine with '-eai' option"));
			
			matlab_engine_ = 0;
			return(false);
        }
        
        file_transfer_ = scinew FileTransferClient();
        if(!(file_transfer_->open(address,"matlabenginefiletransfer",sessionnum,passwd)))
		{
            matlab_engine_->close();

			error(std::string("Matlab: Could not open matlab engine file transfer service (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("Matlab: Make sure the matlab engine file transfer service has not been disabled in $HOME/SCIRun/services/matlabengine.rc"));
			error(std::string("Matlab: Check remote address information, or leave all fields except 'session' blank to connect to local matlab engine"));
			error(std::string("Matlab: If using matlab engine on local machine start engine with '-eai' option"));
			
            matlab_engine_ = 0;
			file_transfer_ = 0;
			return(false);
        }
        
		IComPacketHandle packet;
		if(!(matlab_engine_->recv(packet)))
		{
            matlab_engine_->close();
            file_transfer_->close();
			error(std::string("Matlab: Could not get answer from matlab engine (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("Matlab: This is an internal communication error, make sure that the portnumber is correct"));
			error(std::string("Matlab: If address information is correct, this most probably points to a bug in the SCIRun software"));

			matlab_engine_ = 0;
			file_transfer_ = 0;

			return(false);	
		}
		
		if (packet->gettag() == TAG_MERROR)
		{
			matlab_engine_->close();
            file_transfer_->close();

			error(std::string("Matlab: Matlab engine returned an error (error=") + packet->getstring() + std::string(")"));
			error(std::string("Matlab: Please check whether '$HOME/SCIRun/services/matlabengine.rc' has been setup properly"));
			error(std::string("Matlab: Edit the 'startmatlab=' line to start matlab properly"));
			error(std::string("Matlab: If you running matlab remotely, this file must be edited on the machine running matlab"));

			matlab_engine_ = 0;
			file_transfer_ = 0;

			return(false);					
		}

		thread_info_ = scinew MatlabEngineThreadInfo();
		if (thread_info_.get_rep() == 0)
		{
			matlab_engine_->close();
            file_transfer_->close();

			error(std::string("Matlab: Could not create thread information object"));
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

                error(std::string("Matlab: Could not create remote temporary directory"));
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
            // Hence we translate between both. Matlab does not like
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
		MatlabEngineThread* enginethread = scinew MatlabEngineThread(matlab_engine_copy,thread_info_);
		if (enginethread == 0)
		{
			matlab_engine_->close();
            file_transfer_->close();

			matlab_engine_ = 0;
			file_transfer_ = 0;

			error(std::string("Matlab: Could not create thread object"));
			return(false);
		}
		
		Thread* thread = scinew Thread(enginethread,"Matlab module thread");
		if (thread == 0)
		{
			delete enginethread;
			matlab_engine_->close();
            file_transfer_->close();

			matlab_engine_ = 0;
			file_transfer_ = 0;

			error(std::string("Matlab: Could not create thread"));
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


bool Matlab::close_matlab_engine()
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


bool Matlab::load_output_matrices()
{
	try
	{
		for (int p = 0; p < NUM_MATRIX_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (matrix_oport_[p] == 0) continue;
			if (matrix_oport_[p]->nconnections() == 0) continue;
			if (output_matrix_name_list_[p] == "") continue;
			if (output_matrix_matfile_[p] == "") continue;
		
			matlabfile mf;
			matlabarray ma;
			try
			{
                if (need_file_transfer_) file_transfer_->get_file(file_transfer_->remote_file(output_matrix_matfile_[p]),file_transfer_->local_file(output_matrix_matfile_[p]));
				mf.open(file_transfer_->local_file(output_matrix_matfile_[p]),"r");
				ma = mf.getmatlabarray(output_matrix_name_list_[p]);
				mf.close();
			}
			catch(...)
			{
				error("Matlab: Could not read output matrix");
				continue;
			}
			
			if (ma.isempty())
			{
				error("Matlab: Could not read output matrix");
				continue;
			}
			
			MatrixHandle handle;
            std::string info;
			if (translate_.sciMatrixCompatible(ma,info,static_cast<SCIRun::Module *>(this))) translate_.mlArrayTOsciMatrix(ma,handle,static_cast<SCIRun::Module *>(this));
			matrix_oport_[p]->send(handle);
		}


		for (int p = 0; p < NUM_FIELD_PORTS; p++)
		{
			// Test whether the field port exists
			if (field_oport_[p] == 0) continue;
			if (field_oport_[p]->nconnections() == 0) continue;
			if (output_field_name_list_[p] == "") continue;
			if (output_field_matfile_[p] == "") continue;
		
			matlabfile mf;
			matlabarray ma;
			try
			{
                if (need_file_transfer_) file_transfer_->get_file(file_transfer_->remote_file(output_field_matfile_[p]),file_transfer_->local_file(output_field_matfile_[p]));

				mf.open(file_transfer_->local_file(output_field_matfile_[p]),"r");
				ma = mf.getmatlabarray(output_field_name_list_[p]);
				mf.close();
			}
			catch(...)
			{
				error("Matlab: Could not read output matrix");
				continue;
			}
			
			if (ma.isempty())
			{
				error("Matlab: Could not read output matrix");
				continue;
			}
			
			FieldHandle handle;
            std::string info;
			if (translate_.sciFieldCompatible(ma,info,static_cast<SCIRun::Module *>(this))) translate_.mlArrayTOsciField(ma,handle,static_cast<SCIRun::Module *>(this));
			field_oport_[p]->send(handle);
		}


		for (int p = 0; p < NUM_NRRD_PORTS; p++)
		{
			// Test whether the nrrd port exists
			if (nrrd_oport_[p] == 0) continue;
			if (nrrd_oport_[p]->nconnections() == 0) continue;
			if (output_nrrd_name_list_[p] == "") continue;
			if (output_nrrd_matfile_[p] == "") continue;
		
			matlabfile mf;
			matlabarray ma;
			try
			{
                if (need_file_transfer_) file_transfer_->get_file(file_transfer_->remote_file(output_nrrd_matfile_[p]),file_transfer_->local_file(output_nrrd_matfile_[p]));
				mf.open(file_transfer_->local_file(output_nrrd_matfile_[p]),"r");
				ma = mf.getmatlabarray(output_nrrd_name_list_[p]);
				mf.close();
			}
			catch(...)
			{
				error("Matlab: Could not read output matrix");
				continue;
			}
			
			if (ma.isempty())
			{
				error("Matlab: Could not read output matrix");
				continue;
			}
			
			NrrdDataHandle handle;
            std::string info;
			if (translate_.sciNrrdDataCompatible(ma,info,static_cast<SCIRun::Module *>(this))) translate_.mlArrayTOsciNrrdData(ma,handle,static_cast<SCIRun::Module *>(this));
			nrrd_oport_[p]->send(handle);
		}

	}
	catch(...)
	{
		return(false);
	}
	return(true);
}



bool Matlab::generate_matlab_code()
{
	try
	{
		std::ofstream m_file;
		
		mfile_ = std::string("scirun_code.m");
        std::string filename = file_transfer_->local_file(mfile_);
		m_file.open(filename.c_str(),std::ios::app);

		m_file << matlab_code_list_ << "\n";
		for (int p = 0; p < NUM_MATRIX_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (matrix_oport_[p] == 0) continue;
			if (output_matrix_name_list_[p] == "") continue;

			ostringstream oss;
			oss << "output_matrix" << p << ".mat";
			output_matrix_matfile_[p] = oss.str();
			std::string cmd;
			cmd = "if exist('" + output_matrix_name_list_[p] + "','var'), save " + file_transfer_->remote_file(output_matrix_matfile_[p]) + " " + output_matrix_name_list_[p] + "; end\n";
			m_file << cmd;
		}

		for (int p = 0; p < NUM_FIELD_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (field_oport_[p] == 0) continue;
			if (output_field_name_list_[p] == "") continue;
		
			ostringstream oss;
			oss << "output_field" << p << ".mat";
			output_field_matfile_[p] = oss.str();
			std::string cmd;
			cmd = "if exist('" + output_field_name_list_[p] + "','var'), save " + file_transfer_->remote_file(output_field_matfile_[p]) + " " + output_field_name_list_[p] + "; end\n";
			m_file << cmd;
		}
		
		for (int p = 0; p < NUM_NRRD_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (nrrd_oport_[p] == 0) continue;
			if (output_nrrd_name_list_[p] == "") continue;
		
			ostringstream oss;
			oss << "output_nrrd" << p << ".mat";
			output_nrrd_matfile_[p] = temp_directory_ + oss.str();
			std::string cmd;
			cmd = "if exist('" + output_nrrd_name_list_[p] + "','var'), save " + file_transfer_->remote_file(output_nrrd_matfile_[p]) + " " + output_nrrd_name_list_[p] + "; end\n";
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



bool Matlab::save_input_matrices()
{

	try
	{

		std::ofstream m_file; 
		std::string loadcmd;

		mfile_ = std::string("scirun_code.m");
        std::string filename = file_transfer_->local_file(mfile_);

		m_file.open(filename.c_str(),std::ios::out);

		for (int p = 0; p < NUM_MATRIX_PORTS; p++)
		{
				
			// Test whether the matrix port exists
			if (matrix_iport_[p] == 0) continue;
			if (matrix_iport_[p]->nconnections() == 0) continue;

			MatrixHandle	handle = 0;
			matrix_iport_[p]->get(handle);
			// if there is no data
			if (handle.get_rep() == 0)  
			{
				// we do not need the old file any more so delete it
				input_matrix_matfile_[p] = "";
				continue;
			}
			// if the data as the same before
			// do nothing
			if ((handle == matrix_handle_[p])&&(input_matrix_name_list_[p]==input_matrix_name_list_old_[p])&&(matrix_handle_[p]->generation == input_matrix_generation_old_[p]))
			{
				// this one was not created again
				// hence we do not need to translate it again
				// with big datasets this should improve performance
				loadcmd = "load " + file_transfer_->remote_file(input_matrix_matfile_[p]) + ";\n";
				m_file << loadcmd;
				
				continue;
			}
			
			matrix_handle_[p] = handle;
			
			// Create a new filename for the input matrix
			ostringstream oss;
			oss << "input_matrix" << p << ".mat";
			input_matrix_matfile_[p] = oss.str();
			
			matlabfile mf;
			matlabarray ma;

     		mf.open(file_transfer_->local_file(input_matrix_matfile_[p]),"w");
			mf.setheadertext("Matlab V5 compatible file generated by SCIRun [module Matlab version 1.0]");

			translate_.converttostructmatrix();
			if (input_matrix_array_list_[p] == "numeric array") translate_.converttonumericmatrix();
			translate_.setdatatype(convertdataformat(input_matrix_type_list_[p]));
			translate_.sciMatrixTOmlArray(handle,ma,static_cast<SCIRun::Module *>(this));

			mf.putmatlabarray(ma,input_matrix_name_list_[p]);
			mf.close();
			
			loadcmd = "load " + file_transfer_->remote_file(input_matrix_matfile_[p]) + ";\n";
			m_file << loadcmd;
            
            
            if (need_file_transfer_) 
            {
                if(!(file_transfer_->put_file(file_transfer_->local_file(input_matrix_matfile_[p]),file_transfer_->remote_file(input_matrix_matfile_[p]))))
                {
                    error("Matlab: Could not transfer file");
                    std::string err = "Error :" + file_transfer_->geterror();
                    error(err);
                    return(false);
                }
                
            }
            input_matrix_name_list_old_[p] = input_matrix_name_list_[p];
            input_matrix_generation_old_[p] = handle->generation;

		}

		for (int p = 0; p < NUM_FIELD_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (field_iport_[p] == 0) continue;
			if (field_iport_[p]->nconnections() == 0) continue;
			
			FieldHandle	handle = 0;
			field_iport_[p]->get(handle);
			// if there is no data
			if (handle.get_rep() == 0) 
			{
				// we do not need the old file any more so delete it
                                input_field_matfile_[p] = "";
				continue;
			}
			// if the data as the same before
			// do nothing
			if ((handle == field_handle_[p])&&(input_field_name_list_[p]==input_field_name_list_old_[p])&&(field_handle_[p]->generation == input_field_generation_old_[p]))
			{
				// this one was not created again
				// hence we do not need to translate it again
				// with big datasets this should improve performance
				loadcmd = "load " + file_transfer_->remote_file(input_field_matfile_[p]) + ";\n";
				m_file << loadcmd;

				continue;
			}
			
			field_handle_[p] = handle;
			
			// Create a new filename for the input matrix
			ostringstream oss;
			oss << "input_field" << p << ".mat";
			input_field_matfile_[p] = oss.str();
			
			matlabfile mf;
			matlabarray ma;

			mf.open(file_transfer_->local_file(input_field_matfile_[p]),"w");
			mf.setheadertext("Matlab V5 compatible file generated by SCIRun [module Matlab version 1.0]");

			translate_.converttostructmatrix();
			if (input_field_array_list_[p] == "numeric array") translate_.converttonumericmatrix();
			translate_.sciFieldTOmlArray(handle,ma,static_cast<SCIRun::Module *>(this));
			
			mf.putmatlabarray(ma,input_field_name_list_[p]);
			mf.close();
			
			loadcmd = "load " + file_transfer_->remote_file(input_field_matfile_[p]) + ";\n";
			m_file << loadcmd;
            
            if (need_file_transfer_) 
            {
                if(!(file_transfer_->put_file(file_transfer_->local_file(input_field_matfile_[p]),file_transfer_->remote_file(input_field_matfile_[p]))))
                {
                    error("Matlab: Could not transfer file");
                    std::string err = "Error :" + file_transfer_->geterror();
                    error(err);
                    return(false);
                }
                
            }
            input_field_name_list_old_[p] = input_field_name_list_[p];
            input_field_generation_old_[p] = handle->generation;            
        }

		for (int p = 0; p < NUM_NRRD_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (nrrd_iport_[p] == 0) continue;
			if (nrrd_iport_[p]->nconnections() == 0) continue;

			NrrdDataHandle	handle = 0;
			nrrd_iport_[p]->get(handle);
			// if there is no data
			if (handle.get_rep() == 0) 
			{
				// we do not need the old file any more so delete it
                                input_nrrd_matfile_[p] = "";
				continue;
			}
			// if the data as the same before
			// do nothing
			if ((handle == nrrd_handle_[p])&&(input_nrrd_name_list_[p]==input_nrrd_name_list_old_[p])&&(nrrd_handle_[p]->generation == input_nrrd_generation_old_[p]))
			{
				// this one was not created again
				// hence we do not need to translate it again
				// with big datasets this should improve performance
				loadcmd = "load " + file_transfer_->remote_file(input_nrrd_matfile_[p]) + ";\n";
				m_file << loadcmd;
				
				continue;
			}
			
			nrrd_handle_[p] = handle;
			
			// Create a new filename for the input matrix
			ostringstream oss;
			oss << "input_nrrd" << p << ".mat";
			input_nrrd_matfile_[p] = oss.str();
			
			matlabfile mf;
			matlabarray ma;

			mf.open(file_transfer_->local_file(input_nrrd_matfile_[p]),"w");
			mf.setheadertext("Matlab V5 compatible file generated by SCIRun [module Matlab version 1.0]");
		
			translate_.converttostructmatrix();
			if (input_nrrd_array_list_[p] == "numeric array") translate_.converttonumericmatrix();
			translate_.setdatatype(convertdataformat(input_nrrd_type_list_[p]));	
			translate_.sciNrrdDataTOmlArray(handle,ma,static_cast<SCIRun::Module *>(this));
			mf.putmatlabarray(ma,input_nrrd_name_list_[p]);
			mf.close();

			loadcmd = "load " + file_transfer_->remote_file(input_nrrd_matfile_[p]) + ";\n";            
			m_file << loadcmd;
            
            if (need_file_transfer_) 
            {
                if(!(file_transfer_->put_file(file_transfer_->local_file(input_nrrd_matfile_[p]),file_transfer_->remote_file(input_nrrd_matfile_[p]))))
                {
                    error("Matlab: Could not transfer file");
                    std::string err = "Error :" + file_transfer_->geterror();
                    error(err);
                    return(false);
                }
                
            }  
            input_nrrd_name_list_old_[p] = input_nrrd_name_list_[p];       
            input_nrrd_generation_old_[p] = handle->generation;       
		}
	}
	catch (matlabfile::could_not_open_file)
	{   // Could not open the temporary file
		error("Matlab: Could not open temporary matlab file");
		return(false);
	}
	catch (matlabfile::io_error)
	{   // IO error from ferror
		error("Matlab: IO error");
		return(false);		
	}
	catch (matlabfile::matfileerror) 
	{   // All other errors are classified as internal
		// matfileerrror is the base class on which all
		// other exceptions are based.
		error("Matlab: Internal error in writer");
		return(false);		
	}
	
	return(true);
}


bool Matlab::create_temp_directory()
{
	if (temp_directory_ == "")
	{
		return(tfmanager_.create_tempdir("matlab-engine.XXXXXX",temp_directory_));
	}
	return(true);
}


bool Matlab::delete_temp_directory()
{
    if(temp_directory_ != "") tfmanager_.delete_tempdir(temp_directory_);
	temp_directory_ = "";
	return(true);
}

std::string Matlab::totclstring(std::string &instring)
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

void Matlab::tcl_command(GuiArgs& args, void* userdata)
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
            error("Matlab: Could not close matlab engine");
            return;
        }

        update_status("Matlab engine not running\n");

        // Synchronise input: translate TCL lists into C++ STL lists
        if (!(synchronise_input()))
        {
            error("Matlab: Could not retreive GUI input");
            return;
        }

        // If we haven't created a temporary directory yet
        // we open one to store all temp files in
        if (!(create_temp_directory()))
        {
            error("Matlab: Could not create temporary directory");
            return;
        }

        if (!(open_matlab_engine()))
        {
            error("Matlab: Could not open matlab engine");
            return;
        }
        return;
    }

  }

  Module::tcl_command(args, userdata);
}

} // End namespace MatlabInterface

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#pragma reset woff 1209 
#endif
