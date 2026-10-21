# -*- python -*-
# ex: set syntax=python:
#______________________________________________________________________
#
#  Email formatting for the Uintah buildbot mail notifiers.
#  Used by master.cfg:  from emailFormatters import CompileErrorFormatter

import re                # regular expressions

from buildbot.plugins               import reporters
from buildbot.process.results       import FAILURE
from twisted.internet               import defer
from twisted.python                 import log as twlog

#______________________________________________________________________
#  Compiler errors in the failure emails
#
#  The compile step's stdio log is read only when an email is being
#  written, and only if that step failed.  Each log line starts with a
#  stream character: 'h' header, 'o' stdout, 'e' stderr.

# a line that is a compiler or linker error
error_re      = re.compile( r": (fatal )?error:|undefined reference to" )

# a line that starts a new diagnostic, which ends the current error block
new_diag_re   = re.compile( r":[0-9]+:[0-9]+: (warning|error|fatal error):|^make(\[[0-9]+\])?: " )

# gcc/ld context printed before an error ("In file included from", "In member function",
# ld's "sus.o: in function `main':")
context_re    = re.compile( r"^In file included from |^\s+from |^\S+: In |^\S+: At global scope|: in function " )

# make reporting a failed target, e.g. make: *** [configVars.mk:318: ICE.o] Error 1
make_error_re = re.compile( r"^make(\[[0-9]+\])?: \*\*\* .*Error [0-9]+" )

#__________________________________
#  split a stdio log into (stream, text) pairs

def split_stdio_log( content ):

    lines = []
    for raw in content.splitlines():
      if raw == "":
        continue

      if raw[0] in "ohe":
        lines.append( (raw[0], raw[1:]) )
      else:
        lines.append( ("o", raw) )

    return lines

#__________________________________
#  return the error text for the email: the first error blocks, the make
#  targets that failed and, if no error lines were found, the end of the log

def extract_compile_errors( content,
                            max_lines ):

    lines      = split_stdio_log( content )
    blocks     = []
    seen       = set()
    n_lines    = 0
    n_skipped  = 0
    make_lines = []

    for i in range( len( lines ) ):
      stream, text = lines[i]

      if stream != "e":
        continue

      if make_error_re.search( text ):
        # 'make cleanreally' fails harmlessly in make.py
        if "cleanreally" not in text and text not in make_lines:
          make_lines.append( text )
        continue

      if not error_re.search( text ):
        continue

      # the same error is often reported by several parallel compiles
      if text in seen:
        continue
      seen.add( text )

      # gcc context lines just before the error
      start = i
      while start > 0 and i - start < 6:
        prev_stream, prev_text = lines[start - 1]
        if prev_stream == "e" and context_re.search( prev_text ):
          start = start - 1
        else:
          break

      # the error, its source line and caret, and any notes
      end = i + 1
      while end < len( lines ) and end - i < 12:
        next_stream, next_text = lines[end]
        if next_stream != "e" or new_diag_re.search( next_text ) or error_re.search( next_text ):
          break
        end = end + 1

      # drop trailing "In file included from" lines that belong to the next diagnostic
      while end - 1 > i and context_re.search( lines[end - 1][1] ) and ": note:" not in lines[end - 1][1]:
        end = end - 1

      block = []
      for j in range( start, end ):
        block.append( lines[j][1][:400] )

      if n_lines + len( block ) > max_lines:
        n_skipped = n_skipped + 1
        continue

      blocks.append( "\n".join( block ) )
      n_lines = n_lines + len( block ) + 1

    text = ""

    if len( blocks ) > 0:
      text = "\n\n".join( blocks )
      if n_skipped > 0:
        text = text + "\n\n... %d more error(s) not shown; see the full log." % n_skipped
    else:
      # no compiler error lines: show the end of the log instead
      tail = []
      for stream, line in lines:
        if stream != "h":
          tail.append( line[:400] )

      # nothing but buildbot's header lines (e.g. the command was killed)
      if len( tail ) == 0:
        for stream, line in lines:
          tail.append( line[:400] )

      text = "No compiler error lines were found.  Last lines of the log:\n\n" + "\n".join( tail[-20:] )

    if len( make_lines ) > 0:
      text = text + "\n\nmake reported:\n  " + "\n  ".join( make_lines[:10] )

    return text

#__________________________________
#  email body: Buildbot's default text plus the compiler errors

compile_error_template = u"""\
The Buildbot has detected a {{ status_detected }} on builder {{ buildername }} while building {{ projects }}.
Full details are available at:
    {{ build_url }}

Buildbot URL: {{ buildbot_url }}

Worker for this Build: {{ workername }}

Build Reason: {{ build['properties'].get('reason', ["<unknown>"])[0] }}
Blamelist: {{ ", ".join(blamelist) }}

{{ summary }}
{% if compile_errors %}
______________________________________________________________________
Compile errors:

{{ compile_errors }}
______________________________________________________________________
{% endif %}
Sincerely,
 -The Buildbot
"""

#__________________________________
#  MessageFormatter that adds compile_errors to the email when the
#  compile step failed.  Builds where it passed get the default text.

class CompileErrorFormatter(reporters.MessageFormatter):

    def __init__( self,
                  max_lines = 60 ):
        reporters.MessageFormatter.__init__( self, template = compile_error_template )
        self.max_lines = max_lines

    @defer.inlineCallbacks
    def buildAdditionalContext( self,
                                master,
                                ctx ):
        reporters.MessageFormatter.buildAdditionalContext( self, master, ctx )
        ctx["compile_errors"] = ""

        # never let a problem here stop the email from being sent
        try:
          steps = yield master.data.get( ("builds", ctx["build"]["buildid"], "steps") )

          for step in steps:
            if step["name"] != "compile" or step["results"] != FAILURE:
              continue

            logs = yield master.data.get( ("steps", step["stepid"], "logs") )

            for log in logs:
              if log["slug"] != "stdio":
                continue

              chunk = yield master.data.get( ("logs", log["logid"], "contents") )
              if chunk is not None:
                ctx["compile_errors"] = extract_compile_errors( chunk["content"], self.max_lines )

        except Exception as e:
          twlog.err( e, "CompileErrorFormatter: could not read the compile log" )
          ctx["compile_errors"] = "(could not read the compile log: %s)" % e
