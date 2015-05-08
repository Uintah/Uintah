#!/bin/python
import os
import re

class Profiler(object):
	def __init__(self):
		self.fin = open('result.0', 'r')
		self.fout = open('id2func.txt', 'w')
		self.count = {}
		self.edge = {}
		self.stack_id_map = {}
		self.id_list = []
		self.func_map = {}
		self.func_list = []
		self.nodeNum = -1
	def get_key(self, stack):
		key = 0
		for f in stack[::-1]: key = key * 1024 + f
		return key
	def stack_to_id(self, stack):
		key = self.get_key(stack)
		if key not in self.stack_id_map: 
			self.stack_id_map[key] = len(self.id_list)
			self.id_list.append(key)
		return self.stack_id_map[key]
	def func_to_id(self, func): 
		key = func
		if key not in self.func_map: 
			self.func_map[key] = len(self.func_list)
			self.func_list.append(key)
		return self.func_map[key]
	def print_stack(self, id):
		stack = self.id_list[id]
		while stack > 0:
			func = stack % 1024
			print self.func_list[func]
			stack /= 1024
	def processing(self):
		stack = []
		for line in self.fin:
			line = line.strip()
			if line == "===":
				for depth in range(1, len(stack) + 1):
					cur_stack = stack[-depth:]
					cur_key = self.stack_to_id(cur_stack)
					prv_key = self.stack_to_id(cur_stack[1:])
					self.count[cur_key] = self.count.get(cur_key, 0) + 1
					if prv_key not in self.edge: self.edge[prv_key] = []
					self.edge[prv_key].append(cur_key)
				stack = []
			else:
				stack.append(self.func_to_id(line))
		printed = set()
		print "digraph N{"
		for node in self.count:
			if self.count[node] < 100: continue
			nodeid = self.id_list[node]%1024
			if nodeid not in printed:
				print "Node%d [label=\"%d\"]"%(nodeid, nodeid)
				printed.add(nodeid)
		printed = {}
		for node in self.edge:
			if node not in self.count or self.count[node] < 100: continue
			nodeid = self.id_list[node]%1024
			for target in self.edge[node]:
				if target not in self.count or self.count[target] < 100: continue
				tid = self.id_list[target]%1024
				if (nodeid, tid) not in printed: printed[(nodeid, tid)] = (0,0)
				v = printed[(nodeid, tid)]
				n = (v[0] + self.count[target], v[1] + self.count[node])
				printed[(nodeid, tid)] = n
				#if (nodeid, tid) not in printed:
				#print "Node%d -> Node%d"%(nodeid, tid)
				#printed.add((nodeid, tid))
		for a,b in printed:
			print "Node%d -> Node%d [label=\"%.2f\"]"%(a, b, 100 * printed[(a,b)][0] / float(printed[(a,b)][1]))
		print "}"
		self.fout.write("ID -> Function Name\n")
		for i in range(0, len(self.func_list)):
			self.fout.write("%d -> %s\n"%(i, self.func_list[i]))

	
	def close(self):
		self.fin.close()
		self.fout.close()

if __name__ == "__main__":
	os.system('''awk -F'[\(\)\+]' 'NF>2{if($2 != "")print $2; else print $3} NF==1{print $1}'   tm_0  > temp.0''')
	os.system('''c++filt < temp.0 > result.0 ''')
	p = Profiler()
	p.processing()
	p.close()
