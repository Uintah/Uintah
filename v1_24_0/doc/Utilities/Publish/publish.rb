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

class ConfError < RuntimeError
  def initialize(msg)
    super("Configuration  error: " + msg)
  end
end

class ConfHash < Hash
  def ConfHash.[](*args)
    super
  end
  def confError(m)
    raise(ConfError, m)
  end
  def missing?(key)
    not has_key?(key)
  end
  def empty?(key)
    self[key].size() == 0
  end
  def errorIfMissing(key)
    confError("Missing \"#{key}\"") if missing?(key)
  end
  def boolean?(key)
    self[key].instance_of?(FalseClass) or self[key].instance_of?(TrueClass)
  end
  def string?(key)
    self[key].instance_of?(String)
  end
  def dest?(d)
    d.instance_of?(Dest)
  end
  def array?(key)
    self[key].instance_of?(Array)
  end
  def errorIfNotBoolean(key)
    confError("Not boolean \"#{key}\"") if not boolean?(key)
  end
  def errorIfNotString(key)
    confError("Not string \"#{key}\"") if not string?(key)
  end
  def errorIfEmpty(key)
    confError("Empty \"#{key}\"") if self[key].size() == 0
  end
  def errorIfNotDest(d)
    confError("Not a Dest \"#{d}\"") if not dest?(d)
  end
  def errorIfNotArray(key)
    confError("Not an array \"#{key}\"") if not array?(key)
  end
end

class Dest < ConfHash
  attr_reader :remote

  User = "user"
  Mach = "mach"
  Dir = "dir"
  Tar = "tar"

  def Dest.[](*args)
    super(*args).init()
  end

  def init()
    self[Mach] = "." if missing?(Mach)
    self[Tar] = "/usr/local/bin/tar" if missing?(Tar)
    self
  end

  def validate
    errorIfNotString(Mach)
    @remote = self[Mach] != "."
    if @remote == true
      errorIfMissing(User)
      errorIfNotString(User)
    end
    errorIfMissing(Dir)
    errorIfNotString(Dir)
    errorIfNotString(Tar)
  end
end

class Configuration < ConfHash
  attr_reader :groupDirs

  LogFile = "logFile"
  Group = "group"
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

  def Configuration.new(file)
    eval("Configuration[#{File.new(file, 'r').read}]").init()
  end

  def Configuration.[](*args)
    super
  end

  def init()
    self[Group] = "BioPSE" if missing?(Group)
    self[BuildDir] = "." if missing?(BuildDir)
    self[BuildDir] = File.expand_path(self[BuildDir])
    self[Wait] = false if missing?(Wait)
    self[CodeViews] = false if missing?(CodeViews)
    self[PwdOnly] = false if missing?(PwdOnly)
    self[Tree] = "SCIRunDocs" if missing?(Tree)
    self[Deliver] = false if missing?(Deliver) or self[PwdOnly] == true
    self[Tarball] = false if missing?(Tarball) or self[PwdOnly] == true
    self[Build] = true if missing?(Build)
    self[ToolsPath] = "" if missing?(ToolsPath)
    self[Make] = "/usr/bin/gnumake" if missing?(Make)
    self[Update] = true if missing?(Update)
    self[LogFile] = $stderr if missing?(LogFile)
    self[Clean] = false if missing?(Clean)
    self[DB_DTD] = "/usr/local/share/sgml/dtd/docbook/4.1/docbook.dtd" if missing?(DB_DTD)
    self[SendMailOnError] = false if missing?(SendMailOnError)
    
    validate()

    if self[PwdOnly] == false
      initializeGroupsDB()
      @groupDirs = @groupsDB[self[Group]]
    end

    if self[Deliver] == true
      self[Dests].each do |d|
	if d.remote
	  ENV["CVS_RSH"] = "ssh"
	  begin
	    File.open(self[SSHAgentFile], "r") do |f|
	      s = f.read
	      ENV['SSH_AUTH_SOCK']=/SSH_AUTH_SOCK=(.*?);/.match(s)[1]
	      ENV['SSH_AGENT_PID']=/SSH_AGENT_PID=(\d+);/.match(s)[1]
	    end
	  rescue
	    confError("Can't get ssh agent info from #{self[SSHAgentFile]}")
	  end
	  break;
	end
      end
    end
    self
  end

  def validate()
    errorIfNotString(Group)
    if not self[Group] =~ /^(BioPSE|SCIRun|Uintah)$/
      confError("\"#{Group}\" must be one of \"BioPSE\", \"SCIRun\", or \"Uintah\"")
    end
    errorIfNotString(BuildDir)
    errorIfEmpty(BuildDir)
    errorIfMissing(Tree)
    errorIfNotString(Tree)
    errorIfNotBoolean(Wait)
    errorIfNotBoolean(CodeViews)
    errorIfNotBoolean(Deliver)
    if self[Deliver] == true
      errorIfMissing(Dests)
      errorIfNotArray(Dests)
      needAgent = false
      self[Dests].each do |d|
	errorIfNotDest(d)
	d.validate()
	needAgent = true if d.remote
      end
      if needAgent
	errorIfMissing(SSHAgentFile)
	errorIfNotString(SSHAgentFile)
	errorIfEmpty(SSHAgentFile)
      end
    end
    errorIfNotBoolean(Tarball)
    errorIfNotBoolean(Build)
    errorIfNotString(ToolsPath)
    errorIfMissing(ClassPath)
    errorIfNotString(ClassPath)
    errorIfMissing(Stylesheet_XSL_HTML)
    errorIfNotString(Stylesheet_XSL_HTML)
    errorIfMissing(Stylesheet_DSSSL_Print)
    errorIfNotString(Stylesheet_DSSSL_Print)
    errorIfMissing(XML_DCL)
    errorIfNotString(XML_DCL)
    errorIfMissing(Catalog)
    errorIfNotString(Catalog)
    errorIfNotString(Make)
    errorIfEmpty(Make)
    errorIfNotBoolean(Update)
    errorIfNotBoolean(Clean)
    errorIfNotString(DB_DTD)
    if self[SendMailOnError] == true
      errorIfMissing(SMTPServer)
      errorIfNotString(SMTPServer)
      errorIfMissing(ToAddr)
      errorIfNotString(ToAddr)
      self[FromAddr] = self[ToAddr] if missing?(self[FromAddr])
      errorIfNotString(FromAddr)
    end
  end

  def initializeGroupsDB()
    @groupsDB = {}
    @groupsDB["SCIRun"] = ["doc"]
    srcRoot = self[BuildDir] + "/" + self[Tree] + "/src"
    Dir.foreach(srcRoot) do |m|
      @groupsDB["SCIRun"] << "src/#{m}" if not m =~ /^(\.|\.\.|Packages|CVS)$/
    end
    @groupsDB["BioPSE"] = @groupsDB["SCIRun"] + [ "src/Packages/BioPSE",
      "src/Packages/Teem", "src/Packages/MatlabInterface",
      "src/Packages/DataIO", "src/Packages/Fusion" ]
    @groupsDB["Uintah"] = @groupsDB["SCIRun"] + [ "src/Packages/Uintah" ]
  end

end

class Docs

  def initialize()
    file = nil
    case ARGV.length
    when 0
      file = "publish.conf"
    when 1
      file = ARGV[0]
    else
      raise("Usage: #{File.basename($0)} [config-file]")
    end
    begin
      @conf = Configuration.new(file)
    rescue => oops
      raise("Error reading configuration file: #{$!}\n")
    end
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
	sendMail($!) if @conf[Configuration::SendMailOnError] == true
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
    # Next one is for compatibility with 1.10.1 and earlier docs.
    ENV["STYLESHEET_PATH"] = ENV["STYLESHEET_XSL_HTML"]
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
    if @conf[Configuration::Update] == true
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
    $log.write(`#{@conf[Configuration::Make]} veryclean #{@redirect}`)
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
      $log.write(`cp #{tarball} #{dest[Dest::Dir]} #{@redirect}`)
    else
      $log.write(`scp -p -q #{tarball} #{dest[Dest::User]}@#{dest[Dest::Mach]}:#{dest[Dest::Dir]} #{@redirect}`)
    end
    if $? != 0
      raise("Failed to transfer tarball.")
    else
      $log.write("Installing...\n")
      if dest[Dest::Mach] == "."
	$log.write(`#{installScript}`)
      else
	$log.write(`ssh #{dest[Dest::User]}@#{dest[Dest::Mach]} '#{installScript}'`)
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
    $log.write("Updating ", m, "\n")
    if FileTest.directory?(m)
      Dir.chdir(m)
      $log.write(`cvs update -P -d #{@redirect}`, "\n")
    elsif FileTest.file?(m)
      Dir.chdir(File.dirname(m))
      $log.write(`cvs update #{File.basename(m)} #{@redirect}`, "\n")
    else
      $log.write(m, " doesn't exist - ignoring\n")
    end
    Dir.chdir(pwd)
  end

  def update()
    $log.write("Updating...\n")
    @conf.groupDirs.each do |m|
      updateOne(@treeRoot + "/" + m)
    end
    $log.write("Done Updating\n")
  end

  def make(startDir)
    $log.write("Making docs...\n")
    group = @conf[Configuration::Group].downcase()
    pwd = Dir.pwd()
    Dir.chdir(startDir)
    $log.write(`#{@conf[Configuration::Make]} SRGROUP=#{group} WITH_CV=#{@conf[Configuration::CodeViews] ? "true" : "false"} #{@conf[Configuration::Tarball] ? "tarball" : ""} #{@redirect}`)
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
    docs = Docs.new
    docs.publish
  rescue => oops
    $stderr.print(oops.message, "\n")
  end
end

main
