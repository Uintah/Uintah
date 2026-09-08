#!/usr/bin/env python3
# Reformats .ups XML files to the house style used under StandAlone/inputs/ICE:
# 2-space indent, unspaced attributes, column-aligned BCType attributes per Face,
# column-aligned sibling values (via the "column" utility), compacted vector
# literals, and verbatim-preserved comments.

import argparse
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

#______________________________________________________________________
# Spacing options - edit these to change the formatter's output style.
#______________________________________________________________________
INDENT_WIDTH = 2      # spaces per XML nesting level
VALUE_PADDING = 3     # spaces on each side of a leaf element's value
EQUALS_PADDING = 1    # spaces on each side of "=" in an attribute (0 = name="value")
PRESERVE_BLANK_LINES = True  # keep a single blank line between children that had one in the source

INDENT_UNIT = ' ' * INDENT_WIDTH
VALUE_PAD = ' ' * VALUE_PADDING
EQUALS_PAD = ' ' * EQUALS_PADDING
VECTOR_RE = re.compile( r'^\[.*\]$', re.DOTALL )
WS_RE = re.compile( r'\s+' )

#______________________________________________________________________
#   Parse a .ups file into an ElementTree, keeping comments as Comment nodes.
def parse_ups( path ):
   builder = ET.TreeBuilder( insert_comments=True )
   parser = ET.XMLParser( target=builder )
   with open( path, 'r', encoding='iso-8859-1' ) as ups_file:
      parser.feed( ups_file.read() )
   return parser.close()

#______________________________________________________________________
#   Escape the characters that aren't safe to write back into XML text/attributes.
def escape_xml( text ):
   escaped = text.replace( '&', '&amp;' )
   escaped = escaped.replace( '<', '&lt;' )
   escaped = escaped.replace( '>', '&gt;' )
   escaped = escaped.replace( '"', '&quot;' )
   return escaped

#______________________________________________________________________
#   Render one XML attribute as name="value".
def render_attr( name, value ):
   return '%s%s=%s"%s"' % ( name, EQUALS_PAD, EQUALS_PAD, escape_xml( value ) )

#______________________________________________________________________
#   True if elem is an XML comment node rather than a real element.
def is_comment( elem ):
   return elem.tag is ET.Comment

#______________________________________________________________________
#   Strip the whitespace out of a "[ 1, 0, 0 ]"-style vector literal.
def compact_vector_text( text ):
   inner = text.strip()[ 1:-1 ]
   parts = inner.split( ',' )
   compacted_parts = []
   
   for part in parts:
      compacted_parts.append( part.strip() )
   return '[' + ','.join( compacted_parts ) + ']'

#______________________________________________________________________
#   Normalize a leaf element's text: collapse whitespace, compact vector literals.
def leaf_text( elem ):

   if elem.text is None:
      return ''
   text = WS_RE.sub( ' ', elem.text.strip() )
   
   if VECTOR_RE.match( text ):
      text = compact_vector_text( text )
   return text

#______________________________________________________________________
#   For each Face, pad sibling BCType attributes so they line up in columns;
#   fills padded_attrs (keyed by id(elem)) for render_attrs to use.
def compute_bctype_padding( root, padded_attrs ):
   for face in root.iter( 'Face' ):
      bctypes = list( face.findall( 'BCType' ) )
      if len( bctypes ) < 2:
         continue

      max_width = {}
      for bctype in bctypes:
         for name, value in bctype.attrib.items():
            rendered = render_attr( name, value )
            if name not in max_width or len( rendered ) > max_width[ name ]:
               max_width[ name ] = len( rendered )

      for bctype in bctypes:
         attrs = []
         for name, value in bctype.attrib.items():
            attrs.append( render_attr( name, value ).ljust( max_width[ name ] ) )
         padded_attrs[ id( bctype ) ] = attrs

#______________________________________________________________________
#   Render elem's attributes, using BCType column padding if one was computed.
def render_attrs( elem, padded_attrs ):
   if id( elem ) in padded_attrs:
      return padded_attrs[ id( elem ) ]
   attrs = []
   for name, value in elem.attrib.items():
      attrs.append( render_attr( name, value ) )
   return attrs

#______________________________________________________________________
#   Build an opening tag, e.g. "<tag attr="val">" or "<tag attr="val"/>".
def build_tag( tag, attrs, self_close ):
   closing = '>'
   if self_close:
      closing = '/>'
   if len( attrs ) == 0:
      return '<%s%s' % ( tag, closing )
   return '<%s %s%s' % ( tag, ' '.join( attrs ).rstrip(), closing )

#______________________________________________________________________
#   Render a childless element as one line: self-closing if empty, else inline text.
def render_leaf( elem, indent, padded_attrs ):
   attrs = render_attrs( elem, padded_attrs )
   text = leaf_text( elem )

   if len( text ) == 0:
      return '%s%s' % ( indent, build_tag( elem.tag, attrs, self_close=True ) )

   open_tag = build_tag( elem.tag, attrs, self_close=False )
   return '%s%s%s%s%s</%s>' % ( indent, open_tag, VALUE_PAD, escape_xml( text ), VALUE_PAD, elem.tag )

#______________________________________________________________________
# True if elem can safely be one row in a column-aligned block: no attributes
# (attributes are aligned separately, see compute_bctype_padding) and a
# single-token value, so "column -t" can't mistake its value for two fields.
def is_batchable_leaf( elem ):
   if is_comment( elem ):
      return False
   if len( list( elem ) ) > 0:
      return False
   if len( elem.attrib ) > 0:
      return False
   if ' ' in leaf_text( elem ):
      return False
   return True

#______________________________________________________________________
# Vertically align a block of sibling one-line elements using the "column"
# utility, so values and closing tags line up (e.g. <lower>/<upper>/<patches>
# under a Box). Falls back to the unaligned lines if "column" isn't available.
def align_lines( lines, indent ):
   if len( lines ) < 2:
      return lines
   unindented_lines = []
   for line in lines:
      unindented_lines.append( line[ len( indent ): ] )
   input_text = '\n'.join( unindented_lines ) + '\n'
   try:
      result = subprocess.run( [ 'column', '-t' ], input=input_text, capture_output=True, text=True, check=True )
   except ( FileNotFoundError, subprocess.CalledProcessError ):
      return lines
   aligned_lines = []
   for aligned_line in result.stdout.splitlines():
      aligned_lines.append( indent + aligned_line )
   return aligned_lines

#______________________________________________________________________
# True if a blank line separated elem from its previous sibling (or from the
# parent's opening tag, if elem is the first child) in the original source.
def has_blank_line_before( elem, prev_sibling, parent ):
   if prev_sibling is not None:
      gap_text = prev_sibling.tail
   else:
      gap_text = parent.text
   if gap_text is None:
      return False
   return gap_text.count( '\n' ) >= 2

#______________________________________________________________________
# Recursively append elem's formatted lines (comment, leaf, or container) to out_lines.
def format_element( elem, depth, out_lines, padded_attrs ):
   indent = INDENT_UNIT * depth

   if is_comment( elem ):
      out_lines.append( '%s<!--%s-->' % ( indent, elem.text ) )
      return

   children = list( elem )
   if len( children ) == 0:
      out_lines.append( render_leaf( elem, indent, padded_attrs ) )
      return

   attrs = render_attrs( elem, padded_attrs )
   out_lines.append( '%s%s' % ( indent, build_tag( elem.tag, attrs, self_close=False ) ) )

   child_indent = INDENT_UNIT * ( depth + 1 )
   pending_batch = []
   prev_child = None

   for child in children:
      if PRESERVE_BLANK_LINES and has_blank_line_before( child, prev_child, elem ):
         if len( pending_batch ) > 0:
            out_lines.extend( align_lines( pending_batch, child_indent ) )
            pending_batch = []
         out_lines.append( '' )

      if is_batchable_leaf( child ):
         pending_batch.append( render_leaf( child, child_indent, padded_attrs ) )
         prev_child = child
         continue

      if len( pending_batch ) > 0:
         out_lines.extend( align_lines( pending_batch, child_indent ) )
         pending_batch = []
      format_element( child, depth + 1, out_lines, padded_attrs )
      prev_child = child
   if len( pending_batch ) > 0:
      out_lines.extend( align_lines( pending_batch, child_indent ) )

   out_lines.append( '%s</%s>' % ( indent, elem.tag ) )

#______________________________________________________________________
#   Group root's direct children into sections: leading comments plus the
#   element they annotate, so a blank line is inserted between sections but
#   never between a banner comment and the element it precedes.
def build_top_level_sections( top_children ):
   sections = []
   current = []
   
   for child in top_children:
      current.append( child )
      if not is_comment( child ):
         sections.append( current )
         current = []
   if len( current ) > 0:
      sections.append( current )
   return sections

#______________________________________________________________________
#   Build the complete formatted .ups file text from a parsed tree.
def format_ups_text( root ):
   padded_attrs = {}
   compute_bctype_padding( root, padded_attrs )

   out_lines = [ '<?xml version="1.0" encoding="iso-8859-1"?>', '' ]
   root_tag = build_tag( root.tag, render_attrs( root, padded_attrs ), self_close=False )
   out_lines.append( root_tag )

   sections = build_top_level_sections( list( root ) )
   last_index = len( sections ) - 1
   for index, section in enumerate( sections ):
      for child in section:
         format_element( child, 1, out_lines, padded_attrs )
         
      if index != last_index:
         out_lines.append( '' )
         
   out_lines.append( '</%s>' % root.tag )
   return '\n'.join( out_lines ) + '\n'

#______________________________________________________________________
#   Build a whitespace-insensitive (tag, attrs, text) signature of every node,
#   in document order, for comparing content before/after reformatting.
def extract_signature( root ):
   signature = []
   for elem in root.iter():
      if is_comment( elem ):
         signature.append( ( 'COMMENT', elem.text ) )
         continue
         
      signature.append( ( elem.tag, dict( elem.attrib ), leaf_text( elem ) ) )
   return signature

#______________________________________________________________________
#   Re-parse the reformatted file and confirm its content signature matches the original.
def verify_round_trip( path, original_root ):
   try:
      reformatted_root = parse_ups( path )
   except ET.ParseError as parse_error:
      sys.stderr.write( 'ERROR: reformatted %s is not well-formed XML: %s\n' % ( path, parse_error ) )
      return False

   if extract_signature( original_root ) != extract_signature( reformatted_root ):
      sys.stderr.write( 'ERROR: reformatting %s changed its content, restoring original.\n' % path )
      return False
   return True

#______________________________________________________________________
#   Reformat one .ups file in place, backing it up first and rolling back on failure.
def format_ups_file( path ):
   original_root = parse_ups( path )
   formatted_text = format_ups_text( original_root )

   backup_path = path + '.bak'
   shutil.copyfile( path, backup_path )
   with open( path, 'w', encoding='iso-8859-1' ) as ups_file:
      ups_file.write( formatted_text )

   if not verify_round_trip( path, original_root ):
      shutil.copyfile( backup_path, path )
      return False
   return True

#______________________________________________________________________
#   CLI entry point: reformat every .ups file given on the command line.
def main():
   arg_parser = argparse.ArgumentParser( description='Reformat .ups files to the ICE house style, in place.' )
   arg_parser.add_argument( 'files', nargs='+', help='.ups files to reformat' )
   args = arg_parser.parse_args()

   exit_code = 0
   for path in args.files:
      ok = format_ups_file( path )
      if ok:
         print( 'formatted: %s (backup: %s.bak)' % ( path, path ) )
      else:
         exit_code = 1
   sys.exit( exit_code )

#______________________________________________________________________
if __name__ == '__main__':
   main()
