#!/usr/bin/ruby

#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

#
# Update, build, and deliver scirun docs.
#

require 'net/smtp'
require 'getoptlong'


class CmdLine
  attr_reader :checkSyntax, :confFile
  
  class UsageError < RuntimeError
    def initialize(msg)
      msg = msg + "\nUsage: #{File.basename($0)} [-c] [config-file]"
      super(msg)
    end
  end

  def initialize
    @confFile = "publish.conf"
    @checkSyntax = false
    checkSyntaxOpt = "--check-syntax"
    opts = GetoptLong.new([checkSyntaxOpt, "-c", GetoptLong::NO_ARGUMENT])
    opts.quiet = true;
    opts.each do |opt, arg|
      case opt
      when checkSyntaxOpt
	@checkSyntax = true
      else
	raise(UsageError, "Unrecognized option")
      end
    end
    raise(UsageError, "Wrong number of arguments") if ARGV.length > 1
    @confFile = ARGV[0] if ARGV.length == 1
  end
end

class Log
  def initialize(obj=nil)
    @log = nil
    if obj == nil
      @log = $stderr
    elsif obj.instance_of?(String)
      @log = File.new(obj, "w")
    elsif obj.kind_of?(IO)
      @log = obj
    else
      raise("Log::initialize: obj must be of String or IO type")
    end
  end
  def write(*args)
    if @log != nil
      @log.write(args)
      @log.flush
    end
  end
  # Execute cmd and write its output to log.
  def command(cmd)
    IO.popen(cmd) do |p|
      l = p.gets("\n")
      while l != nil
	write(l)
	l = p.gets("\n")
      end
    end
  end
end

module TreeDir
  def TreeDir.docRootRelP()
    raise "Can't find root of doc tree" if Dir.pwd == "/"
    if (FileTest.directory?("doc") && FileTest.directory?("src"))
      ""
    else
      Dir.chdir("..")
      "../" + docRootRelP()
    end
  end

  def TreeDir.docRootRel()
    begin
      pwd = Dir.pwd()
      docRootRelP()
    ensure
      Dir.chdir(pwd)
    end
  end
  
  def TreeDir.docRootAbs()
    File.expand_path(docRootRel())
  end

  def TreeDir.doc()
    docRootAbs() + "/doc"
  end

  def TreeDir.src()
    docRootAbs() + "/src"
  end
end

class SrcTopLevel
  def initialize(srcRoot, packageList)
    @srcRoot = srcRoot
    @packageList = packageList
  end
  
  # Invoke block for each top-level source file
  def eachFile
    Dir.foreach(@srcRoot) do |m|
      yield(m) if File.stat(@srcRoot + "/" + m).file? and not m =~ /^\./
    end
  end

  # Invoke block for each top-level directory
  def eachDir
    Dir.foreach(@srcRoot) do |m|
      yield(m) if File::stat(@srcRoot + "/" + m).directory? and not m =~ /^(Packages|CVS)$|^\./
    end
    @packageList.each do |p|
      yield("Packages/" + p)
    end
  end

  # Invoke block for each top-level file and directory
  def each
    self.eachFile do |f|
      yield(f)
    end
    self.eachDir do |d|
      yield(d)
    end
  end
  
end

class ConfError < RuntimeError
  def initialize(msg)
    super("Configuration  error: " + msg)
  end
end

class ConfHash < Hash
  def ivinit()
    @iv = {}
  end
  
  # selfP() is used to retrieve conf value without triggering its
  # iv.
  alias selfP []

  # Invoke iv on key before returning value.
  def [](key)
    @iv[key].call
    selfP(key)
  end

  # Utilities for iv procs
  def confError(m)
    raise(ConfError, m)
  end

  def missing?(key)
    not has_key?(key)
  end

  def empty?(key)
    selfP(key).size() == 0
  end

  def errorIfMissing(key)
    confError("Missing \"#{key}\"") if missing?(key)
  end

  def boolean?(key)
    selfP(key).instance_of?(FalseClass) or selfP(key).instance_of?(TrueClass)
  end

  def string?(key)
    selfP(key).instance_of?(String)
  end

  def dest?(d)
    d.instance_of?(Dest)
  end

  def array?(key)
    selfP(key).instance_of?(Array)
  end

  def errorIfNotBoolean(key)
    confError("Not boolean \"#{key}\"") if not boolean?(key)
  end

  def errorIfNotString(key)
    confError("Not string \"#{key}\"") if not string?(key)
  end

  def errorIfEmpty(key)
    confError("Empty \"#{key}\"") if selfP(key).size() == 0
  end

  def errorIfNotDest(d)
    confError("Not a Dest \"#{d}\"") if not dest?(d)
  end

  def errorIfNotArray(key)
    confError("Not an array \"#{key}\"") if not array?(key)
  end

  def errorIfNotEnum(key, enum)
    enum.each do |v|
      return if v == selfP(key)
    end
    confError("\"#{selfP(key)}\" is an invalid enumeration value")
  end

  def errorIfNotStringArray(key)
    errorIfNotArray(key)
    selfP(key).each do |v|
      confError("Not a string \"#{v}\"") if not v.instance_of?(String)
    end
  end
end

class SCMCommand
  def initialize()
    @redirect = "2>&1"
  end

  def update(m)
    $log.write("Updating ", m, "\n")
    if FileTest.directory?(m)
      Dir.chdir(m)
      cmd = updateDirCmd() + " " + @redirect
      $log.write(cmd, "\n")
      $log.command(cmd)
    elsif FileTest.file?(m)
      Dir.chdir(File.dirname(m))
      cmd = updateFileCmd() + " " + File.basename(m) + " " + @redirect
      $log.write(cmd, "\n")
      $log.command(cmd)
    else
      $log.write(m, " doesn't exist - ignoring\n")
    end
  end
  def updateDirCmd()
    raise("Oops from class SCMCommand::updateDirCmd: I'm not implemented!")
  end
  def updateFileCmd()
    raise("Oops from class SCMCommand::updateFileCmd: I'm not implemented!")
  end
end

class CVSCommand < SCMCommand
  def initialize()
    super
    ENV["CVS_RSH"] = "ssh"
  end
  def updateDirCmd()
    "cvs update -P -d"
  end
  def updateFileCmd()
    "cvs update"
  end
end

class SVNCommand < SCMCommand
  def updateDirCmd()
    "svn update"
  end
  def updateFileCmd()
    "svn update"
  end
end

class Dest < ConfHash
  attr_reader :remote

  User = "user"
  Mach = "mach"
  Dir = "dir"
  Tar = "tar"

  def Dest.[](*args)
    d = super
    d.ivinit()
    d
  end

  def ivinit()
    super
    @iv[Mach] = proc {
      self[Mach] = "." if missing?(Mach)
      errorIfNotString(Mach)
    }
    @iv[Tar] = proc {
      self[Tar] = "/usr/local/bin/tar" if missing?(Tar)
      errorIfNotString(Tar)
    }
    @iv[User] = proc {
      errorIfMissing(User)
      errorIfNotString(User)
    }
    @iv[Dir] = proc {
      errorIfMissing(Dir)
      errorIfNotString(Dir)
    }
    @remote = self[Mach] != "."
  end

end

class Configuration < ConfHash

  NO_UPDATE,  LOCAL_UPDATE, REMOTE_UPDATE = 0, 1, 2
  UPDATE_ENUM = [NO_UPDATE, LOCAL_UPDATE, REMOTE_UPDATE]
  
  SCM_CVS, SCM_SVN = 0, 1
  SCM_ENUM = [SCM_CVS, SCM_SVN]

  LogFile = "logFile"
  BuildDir = "buildDir"
  Tree = "treeToPublish"
  Wait = "wait"
  CodeViews = "codeViews"
  Deliver = "deliver"
  Tarball = "tarball"
  Dests = "destinations"
  SSHAgentFile = "sshAgentFile"
  Build = "build"
  ToolsPath = "toolspath"
  ClassPath = "classpath"
  Stylesheet_XSL_HTML = "stylesheet_XSL_HTML"
  Stylesheet_XSL_Print = "stylesheet_XSL_Print"
  Stylesheet_DSSSL_Print = "stylesheet_DSSSL_Print"
  XML_DCL = "XML_DCL"
  Catalog = "catalog"
  Make = "make"
  Update = "update"
  PwdOnly = "pwdOnly"
  Clean = "clean"
  DB_DTD = "dbdtd"
  SendMailOnError = "sendMailOnError"
  SMTPServer = "smtpServer"
  FromAddr = "fromAddr"
  ToAddr = "toAddr"
  SCM = "scm"
  SCM_Command = "scmCommand"
  PackageList = "packageList"

  def ivinit()
    super
    @iv[BuildDir] = proc {
      self[BuildDir] = "." if missing?(BuildDir)
      errorIfNotString(BuildDir)
      self[BuildDir] = File.expand_path(selfP(BuildDir))
    }
    @iv[Wait] = proc {
      self[Wait] = false if missing?(Wait)
      errorIfNotBoolean(Wait)
    }
    @iv[CodeViews] = proc {
      self[CodeViews] = false if missing?(CodeViews)
      errorIfNotBoolean(CodeViews)
    }
    @iv[PwdOnly] = proc {
      self[PwdOnly] = false if missing?(PwdOnly)
      errorIfNotBoolean(PwdOnly)
    }
    @iv[Tree] = proc {
      self[Tree] = "SCIRunDocs" if missing?(Tree)
      errorIfNotString(Tree)
    }
    @iv[Dests] = proc {
      errorIfMissing(Dests)
      errorIfNotArray(Dests)
      selfP(Dests).each { |d| errorIfNotDest(d) }
      selfP(Dests).each { |d|
	if d.remote
	  @iv[SSHAgentFile].call
	  break;
	end
      }
    }
    @iv[Deliver] = proc {
      self[Deliver] = false if missing?(Deliver) or self[PwdOnly] == true
      errorIfNotBoolean(Deliver)
      @iv[Dests].call if selfP(Deliver) == true
    }
    @iv[Tarball] = proc {
      self[Tarball] = false if missing?(Tarball) or self[PwdOnly] == true
      errorIfNotBoolean(Tarball)
    }
    @iv[Build] = proc {
      self[Build] = true if missing?(Build)
      errorIfNotBoolean(Build)
    }
    @iv[ToolsPath] = proc {
      self[ToolsPath] = "" if missing?(ToolsPath)
      errorIfNotString(ToolsPath)
    }
    @iv[Make] = proc {
      self[Make] = "/usr/bin/gnumake" if missing?(Make)
      errorIfNotString(Make)
    }
    @iv[Update] = proc {
      self[Update] = REMOTE_UPDATE if missing?(Update)
      errorIfNotEnum(Update, UPDATE_ENUM)
    }
    @iv[LogFile] = proc {
      self[LogFile] = $stderr if missing?(LogFile)
    }
    @iv[Clean] = proc {
      self[Clean] = false if missing?(Clean)
      errorIfNotBoolean(Clean)
    }
    @iv[DB_DTD] = proc {
      self[DB_DTD] = "/usr/local/share/sgml/dtd/docbook/4.3/docbook.dtd" if missing?(DB_DTD)
      errorIfNotString(DB_DTD)
    }
    @iv[SendMailOnError] = proc {
      self[SendMailOnError] = false if missing?(SendMailOnError)
      errorIfNotBoolean(SendMailOnError)
      if selfP(SendMailOnError) == true
	@iv[SMTPServer].call
	@iv[FromAddr].call
      end
    }
    @iv[SMTPServer]  = proc {
      errorIfMissing(SMTPServer)
      errorIfNotString(SMTPServer)
    }
    @iv[FromAddr]  = proc {
      self[FromAddr] = self[ToAddr] if missing?(FromAddr)
      errorIfNotString(FromAddr)
    }
    @iv[ToAddr]  = proc {
      errorIfMissing(ToAddr)
      errorIfNotString(ToAddr)
    }
    @iv[SSHAgentFile] = proc {
      errorIfMissing(SSHAgentFile)
      errorIfNotString(SSHAgentFile)
    }
    @iv[ClassPath]  = proc {
      errorIfMissing(ClassPath)
      errorIfNotString(ClassPath)
    }
    @iv[Stylesheet_XSL_HTML]  = proc {
      errorIfMissing(Stylesheet_XSL_HTML)
      errorIfNotString(Stylesheet_XSL_HTML)
    }
    @iv[Stylesheet_XSL_Print]  = proc {
      errorIfMissing(Stylesheet_XSL_Print)
      errorIfNotString(Stylesheet_XSL_Print)
    }
    @iv[Stylesheet_DSSSL_Print]  = proc {
      errorIfMissing(XML_DCL)
      errorIfNotString(Stylesheet_DSSSL_Print)
    }
    @iv[XML_DCL]  = proc {
      errorIfMissing(XML_DCL)
      errorIfNotString(XML_DCL)
    }
    @iv[Catalog]  = proc {
      errorIfMissing(Catalog)
      errorIfNotString(Catalog)
    }
    @iv[SCM] = proc {
      self[SCM] = SCM_SVN if missing?(SCM)
      errorIfNotEnum(SCM, SCM_ENUM)
    }
    @iv[SCM_Command] = proc {
      case self[SCM]
      when SCM_CVS
	self[SCM_Command] = CVSCommand.new
      when SCM_SVN
	self[SCM_Command] = SVNCommand.new
      end
    }
    @iv[PackageList] = proc {
      errorIfMissing(PackageList)
      errorIfNotStringArray(PackageList)
    }
  end

  def initialize(file)
    ivinit()

    eval("{#{File.new(file, 'r').read}}").each do |key, value|
      self[key] = value;
    end

    @iv[SendMailOnError].call

    remoteDelivery = false
    if self[Deliver] == true
      self[Dests].each do |d|
	if d.remote
	  remoteDelivery = true
	  break
	end
      end
    end
    if remoteDelivery || self[Update] == REMOTE_UPDATE
      begin
	File.open(self[SSHAgentFile], "r") do |f|
	  s = f.read
	  ENV['SSH_AUTH_SOCK']=/SSH_AUTH_SOCK=(.*?);/.match(s)[1]
	  ENV['SSH_AGENT_PID']=/SSH_AGENT_PID=(\d+);/.match(s)[1]
	end
      rescue
	confError("Can't get ssh agent info from #{self[SSHAgentFile]}")
      end
    end
  end

end

class Docs

  def initialize(conf)
    @conf = conf
    @treeRoot = @conf[Configuration::BuildDir] + '/' + @conf[Configuration::Tree]
    @redirect = "2>&1"
  end

  def publish()
    trys = 0
    doclock = File.new("#{@conf[Configuration::BuildDir]}/.doclock", "w+")
    callcc {|$tryAgain|}
    trys += 1
    if doclock.flock(File::LOCK_EX|File::LOCK_NB) == 0
      begin
	$log = Log.new(@conf[Configuration::LogFile])
      rescue
	doclock.flock(File::LOCK_UN)
	doclock.close
	raise
      end
      begin
	tbeg = Time.now
	build() if @conf[Configuration::Build] == true
	deliver() if @conf[Configuration::Deliver] == true
	tend = Time.now
	$log.write("Elapsed time: ", tend - tbeg, "\n")
      rescue
	$log.write($!, "\n")
	if @conf[Configuration::SendMailOnError] == true
	  msg = "From " + @treeRoot + ": " + $!
	  sendMail(msg)
	end
      ensure
	doclock.flock(File::LOCK_UN)
      end
    elsif @conf[Configuration::Wait]
      $stderr.print( (trys > 1 ? "." : "Someone else is updating the docs.  Waiting...") )
      sleep(10)
      $tryAgain.call
    else
      $stderr.print("Someone else is updating the docs.  Quitting\n")
    end
    doclock.close
  end

  def build()
    ENV["PATH"] = ENV["PATH"] + ":" + @conf[Configuration::ToolsPath]
    ENV["CLASSPATH"] = @conf[Configuration::ClassPath]
    ENV["STYLESHEET_XSL_HTML"] = @conf[Configuration::Stylesheet_XSL_HTML]
    ENV["STYLESHEET_XSL_PRINT"] = @conf[Configuration::Stylesheet_XSL_Print]
    ENV["STYLESHEET_DSSSL_PRINT"] = @conf[Configuration::Stylesheet_DSSSL_Print]
    ENV["XML_DCL"] = @conf[Configuration::XML_DCL]
    ENV["CATALOG"] = @conf[Configuration::Catalog]
    ENV["DB_DTD"] = @conf[Configuration::DB_DTD]

    pwd = Dir.pwd
    if @conf[Configuration::Clean] == true
      if @conf[Configuration::PwdOnly] == true
	clean(Dir.pwd())
      else
	clean("#{@treeRoot}/doc/")
      end
    end
    if @conf[Configuration::Update] != Configuration::NO_UPDATE
      if @conf[Configuration::PwdOnly] == true
	updateOne(Dir.pwd())
      else
	update()
      end
    end
    if @conf[Configuration::PwdOnly] == true
      make(Dir.pwd())
    else
      make("#{@treeRoot}/doc/")
    end
    Dir.chdir(pwd)
  end

  def clean(dir)
    $log.write("Begin clean starting at ", dir, "\n")
    pwd = Dir.pwd()
    Dir.chdir(dir)
    $log.command("#{@conf[Configuration::Make]} veryclean #{@redirect}")
    Dir.chdir(pwd)
    $log.write("End clean\n")
  end

  def deliver()
    tarball = @treeRoot + "/doc/" + @conf[Configuration::Tree] + ".tar.gz"
    raise "No tarball to deliver" if !FileTest::exists?(tarball)
    pwd = Dir.pwd
    Dir.chdir(@conf[Configuration::BuildDir])
    @conf[Configuration::Dests].each do |d|
      deliverOne(tarball, d)
    end
    Dir.chdir(pwd)
  end

  # FIXME: Need some file locking on the destination side!
  def deliverOne(tarball, dest)
    installScript = <<INSTALL_SCRIPT
(cd #{dest[Dest::Dir]}
if #{dest[Dest::Tar]} zxf #{@conf[Configuration::Tree]}.tar.gz;
then 
  if test -d doc 
  then 
    if ! rm -rf doc
    then 
      echo failed to remove old doc
      exit 1
    fi 
  fi 
  if test -d src 
  then 
    if ! rm -rf src 
    then 
      echo failed to remove old src 
      exit 1 
    fi 
  fi 
  if ! mv #{@conf[Configuration::Tree]}/doc . 
  then 
    echo failed to install new doc 
    exit 1 
  fi 
  if ! mv #{@conf[Configuration::Tree]}/src . 
  then 
    echo failed to install new src 
    exit 1 
  fi 
  if ! rm -rf #{@conf[Configuration::Tree]} 
  then 
    echo failed to remove #{@conf[Configuration::Tree]} 
    exit 1 
  fi
  if ! mv #{@conf[Configuration::Tree]}.tar.gz doc
  then
    echo failed to move #{@conf[Configuration::Tree]}.tar.gz into doc
    exit 1
  fi
  exit 0   
else 
  echo tar failed 
  exit 1 
fi 
) #{@redirect}
INSTALL_SCRIPT

    $log.write("Delivering to ", dest[Dest::Mach], "\n")
    $log.write("Transfering #{tarball}...\n")
    if dest[Dest::Mach] == "."
      $log.command("cp #{tarball} #{dest[Dest::Dir]} #{@redirect}")
    else
      $log.command("scp -p -q #{tarball} #{dest[Dest::User]}@#{dest[Dest::Mach]}:#{dest[Dest::Dir]} #{@redirect}")
    end
    if $? != 0
      raise("Failed to transfer tarball.")
    else
      $log.write("Installing...\n")
      if dest[Dest::Mach] == "."
	$log.command("#{installScript}")
      else
	$log.command("ssh #{dest[Dest::User]}@#{dest[Dest::Mach]} '#{installScript}'")
      end
      if $? == 0
	$log.write("Finished this delivery.\n")
      else
	raise("Failed to install tarball.")
      end 
    end
  end

  def updateOne(m)
    pwd = Dir.pwd
    @conf[Configuration::SCM_Command].update(m);
    Dir.chdir(pwd)
  end

  def update()
    $log.write("Updating...\n")
    srcTop = @treeRoot + "/src"
    srcTopLevel = SrcTopLevel.new(srcTop, @conf[Configuration::PackageList])
    srcTopLevel.each do |m|
      updateOne(srcTop + "/" + m)
    end
    $log.write("Done Updating src directory\n")
    updateOne(@treeRoot + "/doc")
    $log.write("Done Updating doc directory\n")
  end

  def make(startDir)
    $log.write("Making docs...\n")
    pwd = Dir.pwd()
    Dir.chdir(startDir)
    cmd = "#{@conf[Configuration::Make]} WITH_CV=#{@conf[Configuration::CodeViews] ? 'true' : 'false'} #{@conf[Configuration::Tarball] ? 'tarball' : ''} #{@redirect}"
    $log.command(cmd)
    Dir.chdir(pwd)
    $log.write("Done making docs\n")
  end

  def sendMail(body)
    from_line = "From: publish.rb\n"
    subject_line = "Subject: Fatal Error\n"
    msg = [ from_line, subject_line, "\n", body ]
    Net::SMTP.start(@conf[Configuration::SMTPServer]) do |smtp|
      smtp.send_mail(msg, @conf[Configuration::FromAddr], @conf[Configuration::ToAddr]);
    end
  end

  private :sendMail, :updateOne, :update, :make, :deliverOne

end

def main
  
  begin
    cmdLine = CmdLine.new
  rescue => oops
    $stderr.print(oops.message, "\n")
    return 1
  end

  begin
    @conf = Configuration.new(cmdLine.confFile)
    $stdout.print("Syntax OK\n") if cmdLine.checkSyntax
  rescue => oops
    $stderr.print("Error reading configuration file: #{$!}\n")
    return 2
  end

  if !cmdLine.checkSyntax
    begin
      docs = Docs.new(@conf)
      docs.publish
    rescue => oops
      $stderr.print(oops.message, "\n")
      return 3
    end
  end

  return 0
end

main
