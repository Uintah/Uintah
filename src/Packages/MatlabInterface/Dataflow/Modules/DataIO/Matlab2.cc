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
 *  Matlab2.cc:
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
#include <Core/ICom/IComSocket.h>
#include <iostream>
#include <fstream>
 
namespace MatlabIO {

using namespace SCIRun;

class Matlab2;
class MatlabEngineThreadInfo;
class MatlabEngineThread;

typedef LockingHandle<MatlabEngineThreadInfo> MatlabEngineThreadInfoHandle;

class MatlabEngineThread : public Runnable, public ServiceBase 
{
  public:
	MatlabEngineThread(ServiceClientHandle serv_handle,MatlabEngineThreadInfoHandle info_handle);
	void run();

  private:
	ServiceClientHandle serv_handle_;
	MatlabEngineThreadInfoHandle info_handle_;
};

class MatlabEngineThreadInfo 
{
  public:
	MatlabEngineThreadInfo();
	~MatlabEngineThreadInfo();

	void dolock();
	void unlock();
	
  public:
	Mutex				lock;
	int					ref_cnt;
	
	Matlab2*			module_;
	ConditionVariable   wait_code_done_;
	bool				code_done_;
	bool				code_success_;
	std::string			code_error_;
	
	ConditionVariable   wait_exit_;
	bool				exit_;
	bool				passed_test_;
};

class Matlab2 : public Module, public ServiceBase 
{
  
  public:
    // Constructor
	Matlab2(GuiContext* ctx);

    // Destructor
	virtual ~Matlab2();
	
	// Std functions for each module
	// execute():
	//   Execute the module and put data on the output port
	
	void execute();
	
	matlabarray::mitype			convertdataformat(std::string dataformat);
	std::vector<std::string>	converttcllist(std::string str);
	
	void	add_stdout_line(std::string line);
	void	add_stderr_line(std::string line);
	void	update_status(std::string text);
  private:
  
	std::string totclstring(std::string &instring);
  
    bool	open_matlab_engine();
	bool	close_matlab_engine();
	
	bool	create_temp_directory();
	bool	delete_temp_directory();

	bool	save_input_matrices();
	bool	load_output_matrices();
	
	bool	generate_matlab_code();
	bool	send_matlab_job();
	
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
	std::vector<std::string>   input_field_name_list_;
	std::vector<std::string>   input_nrrd_name_list_;
	std::vector<std::string>   input_matrix_type_list_;
	std::vector<std::string>   input_nrrd_type_list_;
	std::vector<std::string>   input_matrix_array_list_;
	std::vector<std::string>   input_field_array_list_;
	std::vector<std::string>   input_nrrd_array_list_;
	std::vector<std::string>   output_matrix_name_list_;
	std::vector<std::string>   output_field_name_list_;
	std::vector<std::string>   output_nrrd_name_list_;

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
	GuiString		matlab_var_;
	GuiString		matlab_add_output_;
	GuiString		matlab_update_status_;
	
	ServiceClientHandle				matlab_engine_;
	MatlabEngineThreadInfoHandle	thread_info_;
};


MatlabEngineThreadInfo::MatlabEngineThreadInfo() :
	lock("MatlabEngineInfo lock"),
	ref_cnt(0),
	module_(0),
	wait_code_done_("MatlabEngineInfo condition var code"),
	code_done_(false),
	code_success_(false),
	wait_exit_("MatlabEngineInfo condition var exit"),
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
			if (info_handle_->module_)
			{
				info_handle_->module_->error("Error receiving packet from matlab engine");
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
		
		switch (packet->gettag())
		{
			case TAG_STDO:
				if (packet->getparam1() < 0)
				{
					info_handle_->dolock();
					if (info_handle_->module_)
						info_handle_->module_->add_stdout_line("STDOUT-END");
					info_handle_->unlock();
					break;
				}
				info_handle_->dolock();
				if (info_handle_->module_)
					info_handle_->module_->add_stdout_line(packet->getstring());
				info_handle_->unlock();
				break;
			case TAG_STDE:
				if (packet->getparam1() < 0)
				{
					info_handle_->dolock();
					if (info_handle_->module_)
						info_handle_->module_->add_stderr_line("STDERR-END");
					info_handle_->unlock();
					break;
				}
				info_handle_->dolock();
				if (info_handle_->module_)
					info_handle_->module_->add_stderr_line(packet->getstring());
				info_handle_->unlock();		
			break;
			case TAG_END_:
			case TAG_EXIT:
				info_handle_->dolock();
				info_handle_->code_done_ = true;
				info_handle_->code_success_ = false;
				info_handle_->wait_code_done_.conditionBroadcast();
				info_handle_->exit_ = true;
				info_handle_->wait_exit_.conditionBroadcast();
				info_handle_->unlock();
				done = true;
			break;
			case TAG_MCODE_SUCCESS:
				info_handle_->dolock();
				info_handle_->code_done_ = true;
				info_handle_->code_success_ = true;
				info_handle_->wait_code_done_.conditionBroadcast();				
				info_handle_->unlock();
			break;
			case TAG_MCODE_ERROR:
				info_handle_->dolock();
				info_handle_->code_done_ = true;
				info_handle_->code_success_ = false;
				info_handle_->code_error_ = packet->getstring();
				info_handle_->wait_code_done_.conditionBroadcast();				
				info_handle_->unlock();
			break;
		}
	}
}


DECLARE_MAKER(Matlab2)

Matlab2::Matlab2(GuiContext *context) :
  Module("Matlab2", context, Filter, "DataIO", "MatlabInterface"), 
  input_matrix_name_(context->subVar("input-matrix-name")),
  input_matrix_type_(context->subVar("input-matrix-type")),
  input_matrix_array_(context->subVar("input-matrix-array")),
  output_matrix_name_(context->subVar("output-matrix-name")),
  input_field_name_(context->subVar("input-field-name")),
  input_field_array_(context->subVar("input-field-array")),
  output_field_name_(context->subVar("output-field-name")),
  input_nrrd_name_(context->subVar("input-nrrd-name")),
  input_nrrd_type_(context->subVar("input-nrrd-type")),
  input_nrrd_array_(context->subVar("input-nrrd-array")),
  output_nrrd_name_(context->subVar("output-nrrd-name")),
  inet_address_(context->subVar("inet-address")),
  inet_port_(context->subVar("inet-port")),
  inet_passwd_(context->subVar("inet-passwd")),
  inet_session_(context->subVar("inet-session")),
  matlab_code_(context->subVar("matlab-code")),
  matlab_add_output_(context->subVar("matlab-add-output")),
  matlab_update_status_(context->subVar("matlab-update-status")),
  matlab_var_(context->subVar("matlab-var"))
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

}


Matlab2::~Matlab2()
{
	close_matlab_engine();
	delete_temp_directory();
}

void	Matlab2::add_stdout_line(std::string line)
{
	gui->lock();
	std::string cmd = matlab_add_output_.get() + " \"" + totclstring(line) + "\""; 
	gui->execute(cmd); 
	gui->unlock();
}

void	Matlab2::add_stderr_line(std::string line)
{
	gui->lock();
	std::string cmd = matlab_add_output_.get() + " \"STDERROR:" + totclstring(line) + "\""; 
	gui->execute(cmd);
	gui->unlock();
}

void	Matlab2::update_status(std::string text)
{
	gui->lock();
	std::string cmd = matlab_update_status_.get() + " \"" + totclstring(text) + "\"";
	gui->execute(cmd);
	gui->unlock();
}

matlabarray::mitype Matlab2::convertdataformat(std::string dataformat)
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

std::vector<std::string> Matlab2::converttcllist(std::string str)
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


bool Matlab2::synchronise_input()
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

	matlab_code_list_ = matlab_code_.get(); 
	
	return(true);
}


void Matlab2::execute()
{
	// Synchronise input: translate TCL lists into C++ STL lists
	if (!(synchronise_input()))
	{
		error("Matlab2: Could not retreive GUI input");
		return;
	}

	// If we haven't created a temporary directory yet
	// we open one to store all temp files in
	if (!(create_temp_directory()))
	{
		error("Matlab2: Could not create temporary directory");
		return;
	}

	if (!(save_input_matrices()))
	{
		error("Matlab2: Could not create the input matrices");
		return;
	}

	if (!(generate_matlab_code()))
	{
		error("Matlab2: Could not create m-file code for matlabengine");
		return;
	}	
	
	if (!(open_matlab_engine()))
	{
		error("Matlab2: Could not open matlab engine");
		return;
	}
	
	if (!send_matlab_job())
	{
	   error("Matlab2: Matlab returned an error or Matlab could not be launched");
	   return;
	}
	
	if (!load_output_matrices())
	{
		error("Matlab2: Could not load matrices that matlab generated");
		return;
	}
}

bool Matlab2::send_matlab_job()
{
	IComPacketHandle packet = scinew IComPacket;
	
	if (packet.get_rep() == 0)
	{
		error("Matlab2: Could not create packet");
		return(false);
	}

	thread_info_->dolock();
	thread_info_->code_done_ = false;
	thread_info_->unlock();
	
	packet->settag(TAG_MCODE);
	packet->setstring(mfile_.substr(0,mfile_.size()-2)); // strip the .m
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
			error("Matlab2: the matlab engine crashed or did not start: "+ thread_info_->code_error_);
            error("Matlab2: possible causes are:");
			error("(1) matlab code failed in such a way that the engine was no able to catch the error (e.g. failure mex of code)");
			error("(2) matlab crashed and the matlab engine detected an end of communication of the matlab process");
			error("(3) temparory files could not be created or are corrupted");
			error("(4) improper matlab version, use matlab V5 or higher, currently matlab V5-V7 are supported");
		}
		else
		{
			error("Matlab2: matlab code failed: "+thread_info_->code_error_);
			error("Matlab2: Detected an error in the Matlab code, the matlab engine is still running and caught the exception");
			error("Matlab2: Please check the matlab code in the GUI and try again. The output window in the GUI should contain the reported error message generated by matlab");            
		}
		thread_info_->code_done_ = false;
		thread_info_->unlock();
		return(false);
	}
	thread_info_->code_done_ = false;
	thread_info_->unlock();
	
	return(success);
}

bool Matlab2::open_matlab_engine()
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
		add_stdout_line("Please wait while launching matlab, this may take a few minutes ....\n");
		add_stdout_line("\n");
		
		matlab_engine_ = scinew ServiceClient();
		if(!(matlab_engine_->open(address,"matlabengine",sessionnum,passwd)))
		{
			error(std::string("Matlab2: Could not open matlab engine (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("Matlab2: Make sure the matlab engine has not been disabled in $HOME/SCIRun/services/matlabengine.rc"));
			error(std::string("Matlab2: Check remote address information, or leave all fields except 'session' blank to connect to local matlab engine"));
			
			matlab_engine_ = 0;
			return(false);
        }
		
		IComPacketHandle packet;
		if(!(matlab_engine_->recv(packet)))
		{
            matlab_engine_->close();
			error(std::string("Matlab2: Could not get answer from matlab engine (error=") + matlab_engine_->geterror() + std::string(")"));
			error(std::string("Matlab2: This is an internal communication error, make sure that the portnumber is correct"));
			error(std::string("Matlab2: If address information is correct, this most probably points to a bug in the SCIRun software"));
			matlab_engine_ = 0;
			return(false);	
		}
		
		if (packet->gettag() == TAG_MERROR)
		{
			matlab_engine_->close();
			error(std::string("Matlab2: Matlab engine returned an error (error=") + packet->getstring() + std::string(")"));
			error(std::string("Matlab2: Please check whether '$HOME/SCIRun/services/matlabengine.rc' has been setup properly"));
			error(std::string("Matlab2: Edit the 'startmatlab=' line to start matlab properly"));
			error(std::string("Matlab2: If you running matlab remotely, this file must be edited on the machine running matlab"));

			matlab_engine_ = 0;
			return(false);					
		}

		thread_info_ = scinew MatlabEngineThreadInfo();
		if (thread_info_.get_rep() == 0)
		{
			matlab_engine_->close();
			error(std::string("Matlab2: Could not create thread information object"));
			matlab_engine_ = 0;
			return(false);		
		}
		
		thread_info_->module_ = this;
		
		// By cloning the object, it will have the same fields and sockets, but the socket
		// and error handling will be separate. As the thread will invoke its own instructions
		// it is better to have a separate copy. Besides, the socket obejct will point to the
		// same underlying socket. Hence only the error handling part will be duplicated
		ServiceClientHandle matlab_engine_copy = matlab_engine_->clone();
		MatlabEngineThread* enginethread = scinew MatlabEngineThread(matlab_engine_copy,thread_info_);
		if (enginethread == 0)
		{
			matlab_engine_->close();
			matlab_engine_ = 0;
			error(std::string("Matlab2: Could not create thread object"));
			return(false);
		}
		
		Thread* thread = scinew Thread(enginethread,"Matlab2 receive thread");
		if (thread == 0)
		{
			delete enginethread;
			matlab_engine_->close();
			matlab_engine_ = 0;
			error(std::string("Matlab2: Could not create thread"));
			return(false);	
		}
		thread->detach();
	
		int sessionn = packet->getparam1();
		matlab_engine_->setsession(sessionn);
        
            	
		std::string status = "Matlab engine running\n\nmatlabengine version: " + matlab_engine_->getversion() + "\nmatlabengine address: " +
			matlab_engine_->getremoteaddress() + "\nmatlabengine session:" + matlab_engine_->getsession() + "\n";
		update_status(status);
	}

	return(true);
}


bool Matlab2::close_matlab_engine()
{

	if (thread_info_.get_rep())
	{
		thread_info_->dolock();
		thread_info_->module_ = 0;
		thread_info_->exit_ = true;
		thread_info_->wait_exit_.conditionBroadcast();
		thread_info_->unlock();
		thread_info_ = 0;
	}
	
	if (matlab_engine_.get_rep()) 
	{
		matlab_engine_->close();
		matlab_engine_ = 0;
	}
	
	return(true);
}


bool Matlab2::load_output_matrices()
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
				mf.open(output_matrix_matfile_[p],"r");
				ma = mf.getmatlabarray(output_matrix_name_list_[p]);
				mf.close();
			}
			catch(...)
			{
				error("Matlab2: Could not read output matrix");
				continue;
			}
			
			if (ma.isempty())
			{
				error("Matlab2: Could not read output matrix");
				continue;
			}
			
			MatrixHandle handle;
			translate_.mlArrayTOsciMatrix(ma,handle,static_cast<SCIRun::Module *>(this));
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
				mf.open(output_field_matfile_[p],"r");
				ma = mf.getmatlabarray(output_field_name_list_[p]);
				mf.close();
			}
			catch(...)
			{
				error("Matlab2: Could not read output matrix");
				continue;
			}
			
			if (ma.isempty())
			{
				error("Matlab2: Could not read output matrix");
				continue;
			}
			
			FieldHandle handle;
			translate_.mlArrayTOsciField(ma,handle,static_cast<SCIRun::Module *>(this));
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
				mf.open(output_nrrd_matfile_[p],"r");
				ma = mf.getmatlabarray(output_nrrd_name_list_[p]);
				mf.close();
			}
			catch(...)
			{
				error("Matlab2: Could not read output matrix");
				continue;
			}
			
			if (ma.isempty())
			{
				error("Matlab2: Could not read output matrix");
				continue;
			}
			
			NrrdDataHandle handle;
			translate_.mlArrayTOsciNrrdData(ma,handle,static_cast<SCIRun::Module *>(this));
			nrrd_oport_[p]->send(handle);
		}

	}
	catch(...)
	{
		return(false);
	}
	return(true);
}



bool Matlab2::generate_matlab_code()
{
	try
	{
		std::ofstream m_file;
		
		mfile_ = temp_directory_ + std::string("scirun_code.m");
		m_file.open(mfile_.c_str(),std::ios::app);
		m_file << matlab_code_list_ << "\n";
		
		for (int p = 0; p < NUM_MATRIX_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (matrix_oport_[p] == 0) continue;
			if (output_matrix_name_list_[p] == "") continue;

			ostringstream oss;
			oss << "output_matrix" << p << ".mat";
			output_matrix_matfile_[p] = temp_directory_ +oss.str();
			std::string cmd;
			cmd = "if exist('" + output_matrix_name_list_[p] + "','var'), save " + output_matrix_matfile_[p] + " " + output_matrix_name_list_[p] + "; end\n";
			m_file << cmd;
		}

		for (int p = 0; p < NUM_FIELD_PORTS; p++)
		{
			// Test whether the matrix port exists
			if (field_oport_[p] == 0) continue;
			if (output_field_name_list_[p] == "") continue;
		
			ostringstream oss;
			oss << "output_field" << p << ".mat";
			output_field_matfile_[p] = temp_directory_ +oss.str();
			std::string cmd;
			cmd = "if exist('" + output_field_name_list_[p] + "','var'), save " + output_field_matfile_[p] + " " + output_field_name_list_[p] + "; end\n";
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
			cmd = "if exist('" + output_nrrd_name_list_[p] + "','var'), save " + output_nrrd_matfile_[p] + " " + output_nrrd_name_list_[p] + "; end\n";
			m_file << cmd;
		}

		m_file.close();
		
	}
	catch(...)
	{
		return(false);
	}

	return(true);
}



bool Matlab2::save_input_matrices()
{

	try
	{

		std::ofstream m_file; 
		std::string loadcmd;

		mfile_ = temp_directory_ + std::string("scirun_code.m");

		m_file.open(mfile_.c_str(),std::ios::out);

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
				if (input_matrix_matfile_[p].size() > 0) tfmanager_.delete_tempfile(input_matrix_matfile_[p]);
				input_matrix_matfile_[p].clear();
				continue;
			}
			// if the data as the same before
			// do nothing
			if (handle == matrix_handle_[p])
			{
				// this one was not created again
				// hence we do not need to translate it again
				// with big datasets this should improve performance
				loadcmd = "load " + input_matrix_matfile_[p] + ";\n";
				m_file << loadcmd;
				
				continue;
			}
			
			matrix_handle_[p] = handle;
			
			// Create a new filename for the input matrix
			ostringstream oss;
			oss << "input_matrix" << p << ".mat";
			input_matrix_matfile_[p] = temp_directory_ + oss.str();
			
			matlabfile mf;
			matlabarray ma;

			mf.open(input_matrix_matfile_[p],"w");
			mf.setheadertext("Matlab V5 compatible file generated by SCIRun [module Matlab2 version 1.0]");

			translate_.converttostructmatrix();
			if (input_matrix_array_list_[p] == "numeric array") translate_.converttonumericmatrix();
			translate_.setdatatype(convertdataformat(input_matrix_type_list_[p]));
			translate_.sciMatrixTOmlArray(handle,ma,static_cast<SCIRun::Module *>(this));

			mf.putmatlabarray(ma,input_matrix_name_list_[p]);
			mf.close();
			
			loadcmd = "load " + input_matrix_matfile_[p] + ";\n";
			m_file << loadcmd;
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
				if (input_field_matfile_[p].size() > 0) tfmanager_.delete_tempfile(input_field_matfile_[p]);
				input_field_matfile_[p].clear();
				continue;
			}
			// if the data as the same before
			// do nothing
			if (handle == field_handle_[p])
			{
				// this one was not created again
				// hence we do not need to translate it again
				// with big datasets this should improve performance
				loadcmd = "load " + input_field_matfile_[p] + ";\n";
				m_file << loadcmd;

				continue;
			}
			
			field_handle_[p] = handle;
			
			// Create a new filename for the input matrix
			ostringstream oss;
			oss << "input_field" << p << ".mat";
			input_field_matfile_[p] = temp_directory_ + oss.str();
			
			matlabfile mf;
			matlabarray ma;

			mf.open(input_field_matfile_[p],"w");
			mf.setheadertext("Matlab V5 compatible file generated by SCIRun [module Matlab2 version 1.0]");

			translate_.converttostructmatrix();
			if (input_field_array_list_[p] == "numeric array") translate_.converttonumericmatrix();
			translate_.sciFieldTOmlArray(handle,ma,static_cast<SCIRun::Module *>(this));
			
			mf.putmatlabarray(ma,input_field_name_list_[p]);
			mf.close();
			
			loadcmd = "load " + input_field_matfile_[p] + ";\n";
			m_file << loadcmd;
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
				if (input_nrrd_matfile_[p].size() > 0) tfmanager_.delete_tempfile(input_nrrd_matfile_[p]);
				input_nrrd_matfile_[p].clear();
				continue;
			}
			// if the data as the same before
			// do nothing
			if (handle == nrrd_handle_[p])
			{
				// this one was not created again
				// hence we do not need to translate it again
				// with big datasets this should improve performance
				loadcmd = "load " + input_nrrd_matfile_[p] + ";\n";
				m_file << loadcmd;
				
				continue;
			}
			
			nrrd_handle_[p] = handle;
			
			// Create a new filename for the input matrix
			ostringstream oss;
			oss << "input_nrrd" << p << ".mat";
			input_nrrd_matfile_[p] = temp_directory_ + oss.str();
			
			matlabfile mf;
			matlabarray ma;

			mf.open(input_nrrd_matfile_[p],"w");
			mf.setheadertext("Matlab V5 compatible file generated by SCIRun [module Matlab2 version 1.0]");
		
			translate_.converttostructmatrix();
			if (input_nrrd_array_list_[p] == "numeric array") translate_.converttonumericmatrix();
			translate_.setdatatype(convertdataformat(input_nrrd_type_list_[p]));	
			translate_.sciNrrdDataTOmlArray(handle,ma,static_cast<SCIRun::Module *>(this));
			mf.putmatlabarray(ma,input_nrrd_name_list_[p]);
			mf.close();

			loadcmd = "load " + input_nrrd_matfile_[p] + ";\n";
			m_file << loadcmd;
		}
	}
	catch (matlabfile::could_not_open_file)
	{   // Could not open the temporary file
		error("Matlab2: Could not open temporary matlab file");
		return(false);
	}
	catch (matlabfile::io_error)
	{   // IO error from ferror
		error("Matlab2: IO error");
		return(false);		
	}
	catch (matlabfile::matfileerror) 
	{   // All other errors are classified as internal
		// matfileerrror is the base class on which all
		// other exceptions are based.
		error("Matlab2: Internal error in writer");
		return(false);		
	}
	
	return(true);
}


bool Matlab2::create_temp_directory()
{
	if (temp_directory_ == "")
	{
		return(tfmanager_.create_tempdir("matlab-engine.XXXXXX",temp_directory_));
	}
	return(true);
}


bool Matlab2::delete_temp_directory()
{
	if(mfile_.size() > 0) tfmanager_.delete_tempfile(mfile_);

	for (int p = 0; p < NUM_MATRIX_PORTS; p++)
	{
		if (input_matrix_matfile_[p].size() > 0) tfmanager_.delete_tempfile(input_matrix_matfile_[p]);
		if (output_matrix_matfile_[p].size() > 0) tfmanager_.delete_tempfile(output_matrix_matfile_[p]);
	}
	
	for (int p = 0; p < NUM_FIELD_PORTS; p++)
	{
		if (input_field_matfile_[p].size() > 0) tfmanager_.delete_tempfile(input_field_matfile_[p]);
		if (output_field_matfile_[p].size() > 0) tfmanager_.delete_tempfile(output_field_matfile_[p]);
	}
	
	for (int p = 0; p < NUM_NRRD_PORTS; p++)
	{
		if (input_nrrd_matfile_[p].size() > 0) tfmanager_.delete_tempfile(input_nrrd_matfile_[p]);
		if (output_nrrd_matfile_[p].size() > 0) tfmanager_.delete_tempfile(output_nrrd_matfile_[p]);
	}

	tfmanager_.delete_tempdir(temp_directory_);
	temp_directory_ = "";
	return(true);
}

std::string Matlab2::totclstring(std::string &instring)
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

} // End namespace MatlabInterface
