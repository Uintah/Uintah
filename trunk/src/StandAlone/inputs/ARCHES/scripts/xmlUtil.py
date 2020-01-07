import lxml.etree as ET
from xml.dom import minidom as MD

from lxml import etree

class UPSNodeObject:

    def __init__(self, subelement, value=None, attributes=None, att_values=None, children=None, children_values=None):
        self.subelement = subelement
        self.value = value
        self.attributes = attributes
        self.att_values = att_values
        self.children = children
        self.children_values = children_values

class UPSxmlHelper: 
    """Helper utility for editing UPS files"""
    def __init__(self, file_name): 
        """
        Parses the UPS file and stores it as an etree object
        """
        parser = etree.XMLParser(remove_blank_text=True)
        self.root = ET.parse(file_name, parser)

        return 

    def rm_node( self, node ): 
        """
        Remove a node from the tree
        """
        delete_me = self.root.find(node)
        parent = delete_me.getparent()
        parent.remove(delete_me)

    def get_node( self, node ): 
        """
        Return a node given a tag
        """
        the_node = self.root.find(node)
        return the_node

    def get_node_subnodes( self, node, subnode, atts=[], att_values=[] ): 
        """
        Return the subnodes of a node (maybe multiple subnodes with same tag)
        If att and value are supplied then it returns only those subnodes
        who match with the exact same atts and values

        If att and values are not supplied, the return value if simply the 
        last subnode found (if multiple are present)
        """
        the_node = self.root.find(node)

        if the_node is not None: 
          return_node = None
          total_atts = len(atts)

          for elem in the_node.iter(tag=subnode): 

            this_att = elem.attrib
            #When atts and their values aren't supplied, just return the last 
            #one found. 
            return_node = elem 

            counter = 0

            for i, a in enumerate(atts):
              if this_att[a] is not None: 
                if att_values[i] == this_att[a]: 
                  counter += 1

            if counter == total_atts: 
              #all attributes found and the values match
              #returns the elem that first matches the criteria
              return elem
          if total_atts == 0:     
            return return_node
          else: 
            return None
        else: 
          return None

    def add_node( self, parent_node, new_node, xml_node=None ):
        """
        Add a new node to the root at the parent_node location
        """
        if xml_node == None: 
            node = self.root.find(parent_node)
        else: 
            node = xml_node
        
        added_node = ET.SubElement(node, new_node.subelement)

        #now check for non-required information: 
        if new_node.attributes is not None: 
            for i, attr in enumerate(new_node.attributes): 
                added_node.set(attr, new_node.att_values[i])

        if new_node.children is not None: 
            for i, child in enumerate(new_node.children): 
                new_child = ET.SubElement(added_node, child)
                new_child.text = new_node.children_values[i]

        if new_node.value is not None: 
            added_node.text = new_node.value

        return

    def change_node_value( self, parent_node, new_value, xml_node=None ): 
        """
        Change the value of an existing node.
        """
        if xml_node == None: 
            node = self.root.find(parent_node)
        else:
            node = xml_node

        if node is not None: 
            node.text = new_value
        else: 
            print('Warning: node not found: '+parent_node)

        return

    def change_attribute_value( self, parent_node, attributes, values, xml_node=None ): 
        """
        Change the value of an existing attribute.
        """
        if xml_node == None: 
            node = self.root.find(parent_node)
        else: 
            node = xml_node

        if node is not None: 
            for i, new_att in enumerate(attributes): 
                node.set(new_att,values[i])
        else: 
            print('Warning: node not found: '+parent_node)

        return 

    def save_ups( self, name ): 
        self.root.write(name, pretty_print=True, xml_declaration=True, encoding='UTF-8')

        return 



