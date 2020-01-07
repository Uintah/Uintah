#!/usr/bin/env python3


import subprocess
import sys
import re


'''
    upsDiff version 1.0 written by Adam Gaia on 2/15/19
    This script compares two .ups files for differences based on XML paths. Input files are assumed to have valid XML syntax.
    The XML paths of unique leaf nodes not found in the opposing ups file are output. Nodes marked with '*' have siblings of the same name. 
    Multiple asterisks indicate multiple similarly named siblings. Please manually compare similar siblings of the same file to determine 
    the best matching node in the opposing input file.
    Note:  This script may fail if elements or attributes contain any of these characters:  <, >, /, *
    
    Usage: upsDiff.py file1.ups file2.ups <optional args>
    
    Optional arguments:")
        --no-output          |Do not display output. Does not apply to error messages. Overrules all other options.
        --ellipsis <int>     |Truncate path differences after <int> number of different elements. Default is 1. Setting 0 for <int> will turn truncation off.
    
'''

# ____________________________________________________________
# Codes returned by this script
returnCodeNoDifference = 0
returnCodeDifferencesFound = 1
returnCodeArgumentError = 2
returnCodeFileNotFound = 3



# ______________________________________________________________________
# Classes

# ____________________________________________________________
# Tree Class used to store XML hierarchy
class Tree(object):


  # ____________________________________________________________
  # Tree constructor
  def __init__(self, _attribute, _element):
    # Tree info
    self.numChildren = 0
    self.children = []
    self.parent = None

    # XML info
    self.attribute = _attribute
    self.element = _element
  # end init()


  # ____________________________________________________________
  # Add a child tree to a node of this.tree
  def addChildTree(self, newChildTree):
    # New child's parent becomes this tree
    newChildTree.parent = self

    # add child to next available index in the parent tree
    self.children.append(newChildTree)
    self.children = sorted(self.children, key=lambda tree: tree.element)

    # increment num children
    self.numChildren = self.numChildren + 1
  # end addChildTree()

  # ____________________________________________________________
  # Check if two tree nodes are equal by comparing elements and attribute
  def equalsNode(self, otherNode):
    if self.element == otherNode.element and self.attribute == otherNode.attribute:
      return True
    return False
  # end equalsNode()


  # ____________________________________________________________
  # Generates a string representation of a node. Used in Tree.getPath()
  def nodeToString(self):

    returnString = ""

    # Check if this node has siblings of the same name
    if self.parent is not None:
      for sibling in self.parent.children:
        if sibling is not self and self.equalsNode(sibling):
          returnString = returnString + "*"


    if self.attribute is None:
      returnString = returnString + self.element
    else:
      returnString = returnString + self.element + "/" + self.attribute

    return returnString
  # end nodeToString()


  # ____________________________________________________________
  # Generate the XML path of a node
  def getPath(self):
    path = self.nodeToString()
    nextParent = self.parent
    while nextParent.parent is not None:
      path = nextParent.nodeToString() + "/" + path
      nextParent = nextParent.parent
    return path
  # end getPath()


# End Tree Class


# ______________________________________________________________________
# Functions

# ____________________________________________________________
# Read in file, save XML hierarchy as a tree
def generateXMLTree(path):
  # Create output tree
  xmlTree = Tree(None, path)
  currentNode = xmlTree

  # Open and read file
  with open(path, 'r') as file:
    fileData = file.read()

  # Split the data to create a collection of <token>
  tokens = re.split('(<[^>]*>)', fileData)

  # First token (tokens[0]) is always empty, therefor the true
  # start index is always > 0.
  # The next token is usually the XML encoding.
  # Something like <?xml version="1.0" encoding="iso-8859-1"?>
  # If so, skip that token too
  if tokens[1][0:2] == "<?" and tokens[1][-2:] == "?>":
    startIdx = 2
  else:
    startIdx = 1

  # Loop through each token to process
  currentlyInComment = False
  for idx in range(startIdx, len(tokens)):

    # Set current token while removing leading/trailing whitespace
    token = tokens[idx].replace('\n', '').strip()

    # Skip empty tokens
    if token == "":
      continue

    # If token start a comment, turn on currentlyInComment bool
    if (not currentlyInComment) and token[0:4] == "<!--":
      currentlyInComment = True

    # Keep continuing, until the end of comment is found, then turn off currentlyInComment bool
    if currentlyInComment:
      if token[-3:] == "-->":
        currentlyInComment = False
        continue  # Skip current token because it is part of a comment
      else:
        continue  # Skip current token because it is the end of a comment ("-->")

    # Now that we are sure the current token isn't garbage, determine what kind of token.

    # 1. Normal close:   </element>
    if len(token) > 3 and token[1] == "/":

      currentNode = currentNode.parent

    # 2. Shorthand open and close:  <token attribute />
    elif len(token) > 3 and token[-2] == "/":

      node = Tree(None, token)
      currentNode.addChildTree(node)

    # 3. Normal open:    <element> or <element attribute>
    elif len(token) > 2 and token[0] == "<" and token[-1] == ">":

      attribute = None
      element = token

      node = Tree(attribute, element)
      currentNode.addChildTree(node)
      currentNode = node

    # 4. Token is just <element> attribute </element>
    else:
      currentNode.attribute = token

    # } end token type if branches

  # } end for loop through tokens

  return xmlTree
# end generateXMLPaths()


# ____________________________________________________________
# Generates and returns a list of leaf-nodes xml paths from a tree. Driver for recursive implementation
def generateTreePaths(tree):
  paths = []
  generateTreePathsRecursive(tree, paths)
  return paths
# end generateTreePaths()


# ____________________________________________________________
# Recursive method to traverse a tree down to the leaf-nodes. Compiles a list of leaf-node XML paths
def generateTreePathsRecursive(tree, paths):
  for child in tree.children:

    if len(child.children) == 0:
      paths.append(child.getPath())

    generateTreePathsRecursive(child, paths)
# end generateTreePathsRecursive()


# ____________________________________________________________
# Compares two lists of paths for differences. No output is displayed
def processDifferencesNoOutput(list1, list2):
  # Call helper method to generate differences
  [uniqueToFirst, uniqueToSecond] = setDifferenceHelper(list1, list2)

  # If length of inputs is 0, no differences
  length1 = len(uniqueToFirst)
  length2 = len(uniqueToSecond)
  if length1 == 0 and length2 == 0:
    return returnCodeNoDifference
  else:
    return returnCodeDifferencesFound
# end processDiffs()


# ____________________________________________________________
# Compares two lists of paths for differences.
def processDifferences(list1, list2, tree1Name, tree2Name, tree1, tree2):
  # Call helper method to generate differences
  [uniqueToFirst, uniqueToSecond] = setDifferenceHelper(list1, list2)

  # If length of inputs is 0, no differences
  length1 = len(uniqueToFirst)
  length2 = len(uniqueToSecond)
  
  if length1 == 0 and length2 == 0:
    print("No differences")
    return returnCodeNoDifference
  else:
    print("Differences detected:")

  # Otherwise, print differences then return
  if length1 > 0:
    printPaths(uniqueToFirst, tree2, tree1Name, "<")

  print("\n---")

  if length2 > 0:
    printPaths(uniqueToSecond, tree1, tree2Name, ">")

  print()

  return returnCodeDifferencesFound
# end processDiffs()


# ____________________________________________________________
# Converts two lists to sets, generates the difference of A-B and B-A
# Helper method to eliminate redundant operations
def setDifferenceHelper(list1, list2):
  # Convert lists to sets for easy comparison
  A = set(list1)
  B = set(list2)
  uniqueToFirst = A - B
  uniqueToSecond = B - A
  return [uniqueToFirst, uniqueToSecond]
# end setHelper()


# ____________________________________________________________
# Prints every path from a list of path, in sorted order.
# Each path is compared to the other tree.
# If the non-matching part has a long path, print the first non-matching elements + "/..."
def printPaths(nodes, otherTree, thisTreeFileName, angleBracket):

  setOfPaths = set()
  lineNumbers = []

  for node in sorted(nodes):

    diffIdx = findEquivalentPartialPath(node, otherTree)

    splitByPathParts = node.split("/")

    printPath = ""

    # Add matching part of path
    for partOfPath in splitByPathParts[0:diffIdx]:
      printPath = printPath + partOfPath + "/"

    addLastBit = True

    # Add non-matching part of path. Previously added this part with a different color.
    # This block of code could most likely be consolidated with the previous block. All we really need now is the diffIdx
    for i in range(0, len(splitByPathParts[diffIdx:-1])):
      partOfPath = splitByPathParts[diffIdx:-1][i]

      # if the non-matching part has a long path
      if ellipsisOn:
        if len(splitByPathParts) - diffIdx > 1 and i > ellipsisLevel:
          printPath = printPath + "..."
          addLastBit = False
          break

      # else
      printPath = printPath + partOfPath + "/"

    # Add last element separate to exclude the "/"
    if addLastBit:
      printPath = printPath + splitByPathParts[-1]

    if not setOfPaths.__contains__(printPath):
      setOfPaths.add(printPath)
      lineNumb = getLineNumber(printPath, thisTreeFileName, diffIdx)
      lineNumbers.append(lineNumb)


  #Print
  i = 0
  for path in sorted(setOfPaths):
    print(angleBracket, "    ", path, end="    Line ")
    print(lineNumbers[i])
    i += 1

    if "*" in path:
      print("   Element with difference has at least one sibling of the same name. Please manually compare both/all similar siblings." )

# end printPaths()


# ____________________________________________________________
# Find the line where the difference occurs
def getLineNumber(path, inputFileName, divergeIdx):

  pathParts = path.split("/")
  divergePart = pathParts[divergeIdx]

  line1 = runGrep(divergePart, inputFileName)

  # If grep returned multiple instances of the search term, check the previous term.
  # The true line number should be the closest number >= the previous term's line number
  if len(pathParts) >= divergeIdx and len(line1) > 1:
    beforeDivertPart = pathParts[divergeIdx - 1]
    line2 = runGrep(beforeDivertPart, inputFileName)

    for line in line1:
      if line < line2[0]:
        continue
      else:
        line1 = line
        break


  return line1
# end getLineNumber()


# ____________________________________________________________
# Use subprocess to run grep to find where an element occurs in the input file
def runGrep(input, inputFileName):

  if "*" in input:
    input = input.replace("*", "")

  commandArgs1 = ["grep", "-n", input, inputFileName]
  out = subprocess.run(commandArgs1, stdout=subprocess.PIPE, check=True)
  output = str(out.stdout, 'utf-8')

  possibleLineNumbers = []
  for line in output.split("\n")[0:-1]:
    match = re.search("^[0-9]+", line)
    possibleLineNumbers.append(int(match.group(0)))

  return possibleLineNumbers
# end runGrep


# ____________________________________________________________
# Returns the index representing where a path diverges from the other tree
#
# In this example Path1 is from tree 1, while Path2 and Path3 are from the other tree.
# Path 3 branches from Path1 after index 0, while Path2 branches after index 2. Therefore, index 2 is returned.
# Path1:    root/example/path/leaf
# Path2:    root/example/path/diff/leaf
# Path2:    root/exam___/path/diff/leaf
# Index:      0 /     1 /  2 /  3 /  4
#
# This function is a driver for recursive implementation
def findEquivalentPartialPath(path, otherTree):
  pathParts = path.split("/")
  startIdx = 0
  return findEquivalentPartialPathRecursive(startIdx, otherTree, pathParts)
# end findEquivalentPartialPath()


# ____________________________________________________________
# Recursive helper for findEquivalentPartialPath()
def findEquivalentPartialPathRecursive(currentIdx, otherTree, pathParts):
  currentPathPart = pathParts[currentIdx]

  possibleMatches = []
  for child in otherTree.children:
    if child.element == currentPathPart.split("/")[0]:
      possibleMatches.append(child)

  if len(possibleMatches) > 0:
    currentIdx = currentIdx + 1

  greatestIdxSoFar = currentIdx
  for possibility in possibleMatches:
    nextIdx = findEquivalentPartialPathRecursive(currentIdx, possibility, pathParts)
    if nextIdx > greatestIdxSoFar:
      greatestIdxSoFar = nextIdx

  return greatestIdxSoFar
# end findEquivalentPartialPathRecursive()


# ____________________________________________________________
# Display usage and info message
def displayHelpMessage():
  print()
  print(
    "This script compares two .ups files for differences based on XML paths. Input files are assumed to have valid XML syntax.")
  print("The XML paths of unique leaf nodes not found in the opposing ups file are output. Nodes marked with '*' have siblings of the same name. Multiple asterisks indicate multiple similarly named siblings. Please manually compare similar siblings of the same file to determine the best matching node in the opposing input file.")
  print("Note:  This script may fail if elements or attributes contain any of these characters:  <, >, /, *")
  print()
  print("Usage: upsDiff.py file1.ups file2.ups <optional args>")
  print()
  print("Optional arguments:")
  print("    --no-output        |Do not display output. Does not apply to error messages.")
  print("    --ellipsis <int>   |Truncate path differences after <int> number of different elements.")
  print("                       |    Default is 1. Setting 0 for <int> will turn truncation off.")
  print()
# end displayHelpMessage()


# ______________________________________________________________________
# Main

numArgs = len(sys.argv)

# Display help message if user asks for help
if numArgs > 1 and (sys.argv[1] == "help" or sys.argv[1] == "-help" or sys.argv[1] == "--help" or sys.argv[1] == "-h"):
  displayHelpMessage()
  sys.exit(returnCodeArgumentError)


# Otherwise, check args
if numArgs < 3:  # Two few args
  print("\nError:  Must enter two input files.")
  displayHelpMessage()
  sys.exit(returnCodeArgumentError)


# Default optional parameter values
printOutput = True
ellipsisOn = True
ellipsisLevel = 0 # Default is 1, -1 to fix offset

# Skip first argument -- its just the name of this python file
inputFiles = []
skipNext = False
for i in range(1, numArgs):

  if skipNext:
    skipNext = False
    continue

  arg = sys.argv[i]

  if arg[0: 2] == "--":  # Arg is an optional parameter

    if arg == "--no-output":
      printOutput = False

    elif arg == "--ellipsis":
      try:
        ellipsisLevelString = sys.argv[i+1]
      except IndexError:
        print("\nError:  Must provide an integer for the ellipsis level.")
        displayHelpMessage()
        sys.exit(returnCodeArgumentError)
      try:
        ellipsisLevel = int(ellipsisLevelString) - 1 # -1 to fix offset
        if ellipsisLevel < 0:
          ellipsisOn = False
        skipNext = True # Skip next arg so we do not attempt to parse <int> as a stand alone option
      except ValueError:
        print("\nError:  Must provide an integer for the ellipsis level.")
        displayHelpMessage()
        sys.exit(returnCodeArgumentError)

    else:
      print("\nError:  Invalid argument ", arg, ".")
      displayHelpMessage()
      sys.exit(returnCodeArgumentError)

  else:  # Arg is an input file

    inputFiles.append(arg)

    if len(inputFiles) > 2:  # Too many input files
      print("\nInvalid argument(s)")
      displayHelpMessage()
      sys.exit(returnCodeArgumentError)

path1 = inputFiles[0]
path2 = inputFiles[1]


# Read in input files to Tree structure.
try:
  xmlTree1 = generateXMLTree(path1)
except FileNotFoundError as e:
  print("\nError:  File \'" + path1 + "\' does not exist.\n")
  sys.exit(returnCodeFileNotFound)

try:
  xmlTree2 = generateXMLTree(path2)
except FileNotFoundError as e:
  print("\nError:  File \'" + path2 + "\' does not exist.\n")
  sys.exit(returnCodeFileNotFound)


# Format for easy comparision
formattedListOfPaths1 = generateTreePaths(xmlTree1)
formattedListOfPaths2 = generateTreePaths(xmlTree2)


# Compare and print differences
if printOutput:

  outCode = processDifferences(formattedListOfPaths1, formattedListOfPaths2, path1, path2, xmlTree1, xmlTree2)

else:
  outCode = processDifferencesNoOutput(formattedListOfPaths1, formattedListOfPaths2)


sys.exit(outCode)
