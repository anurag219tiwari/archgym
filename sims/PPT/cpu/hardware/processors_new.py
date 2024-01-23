# Copyright (c) 2017, Los Alamos National Security, LLC
# All rights reserved.
# Copyright 2017. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
#
# Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""
*********** Performance Prediction Toolkit PPT *********

File: processors.py
Description main library of hardware core definitions.

Comments:
 2016-04-26: included into repository, contains
  1. ThreadedProcessor
  3. CieloCore
  4. MLIntelCore
  5. MustangCore
 2016-12-15: contains
  1. ThreadedProcessor
  2. CieloCore
  3. MLIntelCore
  4. EdisonCore
  5. TTNCore (Titan)
  6. MLIntelPlusGPUCore
  7. MustangCore
  6. GPUCore
"""

import math

from simian import Simian, Entity


class ThreadedProcessor(object):
  """
  Defined as base class as other bases seem useless in SNAPSim context. A
  Simian resource that represents a computer CPU (or CPU core) with hardware
  threads.
  """

  def __init__(self, node):
    super(ThreadedProcessor, self).__init__()
    self.activethreads = 0
    self.maxthreads = 100000
    self.node = node  # needed so processor can access node memory
                                   #   parameters
    self.waiting_processes = []  # list of processes waiting to be executed
                                   #   (only non-empty if maxthreads is
                                   #   exceeded)

class GenericProcessor(ThreadedProcessor):
    """
    A generic processor that is defined as the base class, which contains
    all the methods (time_compute, time_compute2) used for predicting the execution time.
    """
    def __init__(self, node):
        """
        Initialize machine-specific values.
        """
        super(GenericProcessor, self).__init__(node)
        # Units
        self.ns = 1.0 * 10 ** (-9)  # nanoseconds
        self.kb = 1024.0  # Kilobytes
        self.mb = 1024.0 ** 2  # Megabytes
        self.isize = 4.0  # bytes per word
        self.fsize = 8.0  # bytes per word

        # Threads and vectors
        self.maxthreads = 16  # upper bound on number active threads
        self.clockspeed = 2.40 * 10 ** 9  # Hertz
        self.hwthreads = 1  # number of hardware threads
        self.vector_width = 16  # width of vector unit in bytes, 128 bit?

        # Registers
        self.num_registers = 16.0  # number of registers [bytes]
        self.register_cycles = 1  # cycles per register access

        # Cache details
        self.cache_levels = 3
        self.cache_sizes = [64.0 * self.kb, 512.0 * self.kb, 1.5 * self.mb]  # bytes
        self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
        self.cache_latency = [1.0 * self.ns, 5.0 * self.ns, 25.0 * self.ns]  # seconds
        self.cache_cycles = [self.cache_latency[0] * self.clockspeed,
                             self.cache_latency[1] * self.clockspeed,
                             self.cache_latency[2] * self.clockspeed]

        # Main memory
        self.ram_page_size = 4096.0  # bytes
        self.ram_latency = 60.0 * self.ns  # seconds
        self.ram_cycles = self.ram_latency * self.clockspeed

        # Operations
        # Each operation takes 1 cycle but can execute more than 1 instruction per
        #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
        #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
        #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
        #   assumptions about not perfect efficiency.
        self.cycles_per_CPU_ops = 1.0
        self.cycles_per_iALU = 0.5
        self.cycles_per_int_vec = 0.1
        self.cycles_per_fALU = 1.0
        self.cycles_per_vector_ops = 0.5
        self.cycles_per_division = 10.0

    def time_compute(self, tasklist, statsFlag=False):
        """
        Computes the cycles that the items in the tasklist (CPU ops, data access,
        vector ops, memory alloc) take.
        """

        # Initialize
        cycles = 0.0
        time = 0.0
        stats = {}
        stats['L1_float_hits'] = 0
        stats['L2_float_hits'] = 0
        stats['L1_int_hits'] = 0
        stats['L2_int_hits'] = 0
        stats['L1_int_misses'] = 0
        stats['L2_int_misses'] = 0
        stats['L1_float_misses'] = 0
        stats['L2_float_misses'] = 0
        stats['RAM accesses'] = 0
        stats['L1 cycles'] = 0
        stats['L2 cycles'] = 0
        stats['RAM cycles'] = 0
        stats['CPU cycles'] = 0
        stats['iALU cycles'] = 0
        stats['fALU cycles'] = 0
        stats['fDIV cycles'] = 0
        stats['INTVEC ops'] = 0
        stats['INTVEC cycles'] = 0
        stats['VECTOR ops'] = 0
        stats['VECTOR cycles'] = 0
        stats['internode comm time'] = 0
        stats['intranode comm time'] = 0

        #X#print ("cycles = ", cycles)

        for item in tasklist:
          # TASK: Cycles associated with moving data through memory and cache
          if item[0] == 'MEM_ACCESS':

            num_index_vars = item[1]
            num_float_vars = item[2]
            avg_dist = item[3]
            avg_reuse_dist = item[4]
            stdev_reuse_dist = item[5]
            index_loads = item[6]
            float_loads = item[7]
            init_flag = item[8]

            # Insert formula that turns these variables into actual L1, L2, ram
            #   accesses. This is a V0.1 model for realistic caching behavior. The
            #   arguments to MEM_ACCESS are architecture independent. We use index
            #   vars and integers vars interchageably, ie all int variables are
            #   treated as array indices, all floats are array elements.

            # Assume index variables are in registers as much as possible. Compute
            #   number of index loads that are not handled by registers
            avg_index_loads = index_loads / num_index_vars
            num_reg_accesses = int(self.num_registers * avg_index_loads)
            nonreg_index_loads = max(0, index_loads - num_reg_accesses)

            # Compute number of variables on a typical page in memory
            num_vars_per_page = self.ram_page_size / avg_dist

        # Check memory layout to begin
            if init_flag:
              # New function call (true); floats need loaded from main memory and
              #   fewer float_loads from cache.
              initial_ram_pages = num_float_vars / num_vars_per_page
              float_loads -= num_float_vars
            else:
              # float data already in cache
              initial_ram_pages = 0.0

            # Main memory: initial_ram_pages moving through the cache
            cycles += self.ram_cycles * initial_ram_pages
            # cycles += self.cache_cycles[2]*initial_ram_pages*self.ram_page_size/ \
            #          self.cache_line_sizes[2]
            # cycles += self.cache_cycles[1]*initial_ram_pages*self.ram_page_size/ \
            #          self.cache_line_sizes[1]
            # cycles += self.cache_cycles[0]*initial_ram_pages*self.ram_page_size/ \
            #          self.cache_line_sizes[1]

            #X#print ("inital_ram_pages, cycles=", cycles, initial_ram_pages)

            # Compute probability that reload is required, assume normal
            #   distribution with reuse dist larger than cache page size

            # L1
            nlines = self.cache_sizes[0] / self.cache_line_sizes[0]
            nvars = avg_reuse_dist / self.cache_line_sizes[0]
            vvars = math.sqrt(2.0 * (stdev_reuse_dist / self.cache_line_sizes[0]) ** 2)
            L1_hitrate = 0.5 * (1.0 + math.erf((nlines - nvars) / vvars))

            L1_int_hits = int(L1_hitrate * nonreg_index_loads)
            L1_float_hits = int(L1_hitrate * float_loads)

            L1_int_misses = nonreg_index_loads - L1_int_hits
            L1_float_misses = float_loads - L1_float_hits

            #X#print ("l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses, L1_float_misses)

            # L2
            nlines = self.cache_sizes[1] / self.cache_line_sizes[1]
            nvars = avg_reuse_dist / self.cache_line_sizes[1]
            vvars = math.sqrt(2.0 * (stdev_reuse_dist / self.cache_line_sizes[1]) ** 2)
            L2_hitrate = 0.5 * (1.0 + math.erf((nlines - nvars) / vvars))

            L2_int_hits = int(L2_hitrate * L1_int_misses)
            L2_float_hits = int(L2_hitrate * L1_float_misses)

            L2_int_misses = L1_int_misses - L2_int_hits
            L2_float_misses = L1_float_misses - L2_float_hits

            #X#print ("l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses, L2_float_misses)

            # L3
            nlines = self.cache_sizes[2] / self.cache_line_sizes[2]
            nvars = avg_reuse_dist / self.cache_line_sizes[2]
            vvars = math.sqrt(2.0 * (stdev_reuse_dist / self.cache_line_sizes[2]) ** 2)
            L3_hitrate = 0.5 * (1.0 + math.erf((nlines - nvars) / vvars))

            L3_int_hits = int(L3_hitrate * L2_int_misses)
            L3_float_hits = int(L3_hitrate * L2_float_misses)

            L3_int_misses = L2_int_misses - L3_int_hits
            L3_float_misses = L2_float_misses - L3_float_hits

            #X#print ("l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses, L3_float_misses)

            # Update the cycles count for cache hits
            # Registers
            cycles += num_reg_accesses * self.register_cycles
            #X#print ("registers, cycles= ", cycles)

            # L1, data moves together in cache lines
            cycles += self.cache_cycles[0] * \
                      (L1_float_hits * self.fsize + L1_int_hits * self.isize) / \
                      self.cache_line_sizes[0]
            #X#print ("l1 ", cycles)

            # L2
            cycles += self.cache_cycles[1] * \
                      (L2_float_hits * self.fsize + L2_int_hits * self.isize) / \
                      self.cache_line_sizes[1]
            #X#print ("l2 ", cycles)

            # L3
            cycles += self.cache_cycles[2] * \
                      (L3_float_hits * self.fsize + L3_int_hits * self.isize) / \
                      self.cache_line_sizes[2]
            #X#print ("l3 ", cycles)

            # L3 misses accumulate time from main memory
            cycles += self.ram_cycles * \
                      (L3_float_misses + L3_int_misses) / num_vars_per_page
            #X#print ("l3 misses", cycles)

            # L2 misses accumulate time from L3
            # cycles += self.cache_cycles[2] * \
            #          (L2_float_misses*self.fsize + L2_int_misses*self.isize) / \
            #          self.cache_line_sizes[2]
            # print "l2 misses", cycles

            # L1 misses accumulate time from L2
            # cycles += self.cache_cycles[1] * \
            #          (L1_float_misses*self.fsize + L1_int_misses*self.isize) / \
            #          self.cache_line_sizes[2]
            # print "l1 misses", cycles

            #X#print ("memaccess cycles= ", cycles)

            stats['L1_float_hits'] += L1_float_hits
            stats['L2_float_hits'] += L2_float_hits
            stats['L1_int_hits'] += L1_int_hits
            stats['L2_int_hits'] += L2_int_hits
            stats['L1_int_misses'] += L1_int_misses
            stats['L2_int_misses'] += L2_int_misses
            stats['L1_float_misses'] += L1_float_misses
            stats['L2_float_misses'] += L2_float_misses
            stats['RAM accesses'] += (2 * L2_float_misses + L2_int_misses) / \
                                        num_vars_per_page
            stats['L1 cycles'] += self.cache_cycles[0] * \
                                        (2 * L1_float_hits + L1_int_hits)
            stats['L2 cycles'] += self.cache_cycles[1] * \
                                        (2 * L2_float_hits + L2_int_hits)
            stats['RAM cycles'] += self.ram_cycles * \
                                        (2 * L2_float_misses + L2_int_misses) / \
                                        num_vars_per_page

          # TASK: Direct input of cache level hitrates
          elif item[0] == 'HITRATES':

            L1_hitrate = item[1]
            L2_hitrate = item[2]
            L3_hitrate = item[3]
            num_index_vars = item[4]
            num_float_vars = item[5]
            avg_dist = item[6]
            index_loads = item[7]
            float_loads = item[8]
            init_flag = item[9]

            # Insert formula that turns these variables into actual L1, L2, ram
            #   accesses. This is a V0.1 model for realistic caching behavior. The
            #   arguments to MEM_ACCESS are architecture independent. We use index
            #   vars and integers vars interchageably, ie all int variables are
            #   treated as array indices, all floats are array elements.

            # Assume index variables are in registers as much as possible. Compute
            #   number of index loads that are not handled by registers
            avg_index_loads = index_loads / num_index_vars
            num_reg_accesses = int(self.num_registers * avg_index_loads)
            nonreg_index_loads = max(0, index_loads - num_reg_accesses)

            # Compute number of variables on a typical page in memory
            num_vars_per_page = self.ram_page_size / avg_dist

        # Check memory layout to begin
            if init_flag:
              # New function call (true); floats need loaded from main memory and
              #   fewer float_loads from cache.
              initial_ram_pages = num_float_vars / num_vars_per_page
              float_loads -= num_float_vars
            else:
              # float data already in cache
              initial_ram_pages = 0.0

            # Main memory: initial_ram_pages moving through the cache
            cycles += self.ram_cycles * initial_ram_pages

            #X#print ("inital_ram_pages, cycles=", cycles, initial_ram_pages)

            # Compute probability that reload is required, assume normal
            #   distribution with reuse dist larger than cache page size

            # L1
            L1_int_hits = int(L1_hitrate * nonreg_index_loads)
            L1_float_hits = int(L1_hitrate * float_loads)

            L1_int_misses = nonreg_index_loads - L1_int_hits
            L1_float_misses = float_loads - L1_float_hits

            #X#print ("l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses, L1_float_misses)

            # L2
            L2_int_hits = int(L2_hitrate * L1_int_misses)
            L2_float_hits = int(L2_hitrate * L1_float_misses)

            L2_int_misses = L1_int_misses - L2_int_hits
            L2_float_misses = L1_float_misses - L2_float_hits

            #X#print ("l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses, L2_float_misses)

            # L3
            L3_int_hits = int(L3_hitrate * L2_int_misses)
            L3_float_hits = int(L3_hitrate * L2_float_misses)

            L3_int_misses = L2_int_misses - L3_int_hits
            L3_float_misses = L2_float_misses - L3_float_hits

            #X#print ("l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses, L3_float_misses)

            # Update the cycles count for cache hits
            # Registers
            cycles += num_reg_accesses * self.register_cycles
            #X#print ("registers, cycles= ", cycles)

            # L1, data moves together in cache lines
            cycles += self.cache_cycles[0] * \
                      (L1_float_hits * self.fsize + L1_int_hits * self.isize) / \
                      self.cache_line_sizes[0]
            #X#print ("l1 ", cycles)

            # L2
            cycles += self.cache_cycles[1] * \
                      (L2_float_hits * self.fsize + L2_int_hits * self.isize) / \
                      self.cache_line_sizes[1]
            #X#print ("l2 ", cycles)

            # L3
            cycles += self.cache_cycles[2] * \
                      (L3_float_hits * self.fsize + L3_int_hits * self.isize) / \
                      self.cache_line_sizes[2]
            #X#print ("l3 ", cycles)

            # L3 misses accumulate time from main memory
            cycles += self.ram_cycles * \
                      (L3_float_misses + L3_int_misses) / num_vars_per_page
            #X#print ("l3 misses", cycles)

            #X#print ("memaccess cycles= ", cycles)

            stats['L1_float_hits'] += L1_float_hits
            stats['L2_float_hits'] += L2_float_hits
            stats['L1_int_hits'] += L1_int_hits
            stats['L2_int_hits'] += L2_int_hits
            stats['L1_int_misses'] += L1_int_misses
            stats['L2_int_misses'] += L2_int_misses
            stats['L1_float_misses'] += L1_float_misses
            stats['L2_float_misses'] += L2_float_misses
            stats['RAM accesses'] += (2 * L2_float_misses + L2_int_misses) / \
                                        num_vars_per_page
            stats['L1 cycles'] += self.cache_cycles[0] * \
                                        (2 * L1_float_hits + L1_int_hits)
            stats['L2 cycles'] += self.cache_cycles[1] * \
                                        (2 * L2_float_hits + L2_int_hits)
            stats['RAM cycles'] += self.ram_cycles * \
                                        num_vars_per_page

          # TASK: Direct input of L1 accesses
          elif item[0] == 'L1':

            num_accesses = item[1]
            cycles += num_accesses * self.cache_cycles[0]
            #print ("l1 ", cycles)

          # TASK: Direct input of L2 accesses
          elif item[0] == 'L2':

            num_accesses = item[1]

            cycles += num_accesses * self.cache_cycles[1]
            #print ("l2 ", cycles)

            cycles += num_accesses * self.cache_cycles[1]
            print ("l2 ", cycles)

          # TASK: Direct input of higher cache and memory accesses
          elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:

            num_accesses = item[1]
            cycles += num_accesses * self.ram_cycles
            #print ("l3 ", cycles)

          # TASK: CPU ops
          elif item[0] == 'CPU':

            num_ops = item[1]
            cycles += num_ops * self.cycles_per_CPU_ops
            stats['CPU cycles'] += num_ops * self.cycles_per_CPU_ops
            #X#print ("cpu ", cycles)

          # TASK: Integer operations
          elif item[0] == 'iALU':

            num_ops = item[1]
            cycles += num_ops * self.cycles_per_iALU
            stats['iALU cycles'] += num_ops * self.cycles_per_iALU
            #X#print ("ialu ", cycles)

          # TASK: Floating point operations (add/multiply)
          elif item[0] == 'fALU':

            num_ops = item[1]
            cycles += num_ops * self.cycles_per_fALU
            stats['fALU cycles'] += num_ops * self.cycles_per_fALU
            #X#print ("falu ", cycles, num_ops)

          # TASK: Floating point divisions
          elif item[0] == 'fDIV':

            num_ops = item[1]
            cycles += num_ops * self.cycles_per_division
            stats['fDIV cycles'] += num_ops * self.cycles_per_division
            #X#print ("fDIV ", cycles, num_ops)

          # TASK: Integer vector operations
          elif item[0] == 'INTVEC':

            num_ops = item[1]
            vec_width = item[2]

            # vec_ops is the actual number of operations necessary, even if
            #   required width was too high. The // operator is floor integer
            #   division.

            vec_ops = (1 + vec_width // self.vector_width) * num_ops
            cycles += vec_ops * self.cycles_per_int_vec
            stats['INTVEC ops'] += vec_ops
            stats['INTVEC cycles'] += vec_ops * self.cycles_per_int_vec
            #X#print ("intvect ", cycles, num_ops, vec_ops)

          # TASK: Vector operations
          elif item[0] == 'VECTOR':

            num_ops = item[1]
            vec_width = item[2]

            # vec_ops is the actual number of operations necessary, even if
            #   required width was too high. The // operator is floor integer
            #   division.

            vec_ops = (1 + vec_width // self.vector_width) * num_ops
            cycles += vec_ops * self.cycles_per_vector_ops
            stats['VECTOR ops'] += vec_ops
            stats['VECTOR cycles'] += vec_ops * self.cycles_per_vector_ops
            #X#print ("vector ", cycles, num_ops, vec_ops)

          # TASK: communication across nodes, update time directly instead of cycles
          elif item[0] == 'internode':

            msg_size = item[1]
            tmp = msg_size / self.node.interconnect_bandwidth + \
                   self.node.interconnect_latency
            time += tmp
            stats['internode comm time'] += tmp
            #X#print ("inter ", cycles)

          # TASK: communication within a node treated as memory access
          elif item[0] == 'intranode':

            num_accesses = float(item[1]) / self.ram_page_size
            cycles += num_accesses * self.ram_cycles
            stats['intranode comm time'] += num_accesses * self.ram_cycles
            #X#print ("intra ", cycles)

          # TASK: memory management allocation
          elif item[0] == 'alloc':

            mem_size = item[1]
            if mem_size < 0:
              mem_size = -mem_size
            mem_alloc_success = self.node.mem_alloc(mem_size)
            if mem_alloc_success:
               # Count this as a single memory access for timing purposes
               cycles += self.ram_cycles
            else:
              # File system access or just an exception, add to time not cycles
              time += self.node.filesystem_access_time

            #X#print ("alloc ", cycles)

          # TASK: memory management deallocation; changes memory footprint, not time
          elif item[0] == 'unalloc':

            mem_size = item[1]
            if mem_size > 0:
              mem_size = -mem_size
            mem_alloc_success = self.node.mem_alloc(mem_size)
            #X#print ("unalloc ", cycles)

          # Invalid task
          else:

            print ('Warning: task list item', item, ' cannot be parsed, ignoring it')

        # Divide cumulative cycles by clockspeed to get time; apply thread eff
        time += cycles / self.clockspeed * self.thread_efficiency()
        stats['Thread Efficiency'] = self.thread_efficiency()

        if statsFlag:
          return time, stats
        else:
          return time

    def time_compute_amm(self, tasklist, statsFlag=False):
      """
      Computes the cycles that the items in the tasklist (CPU ops, data access,
      vector ops, memory alloc) take.
      """
      # Initialize
      cycles = 0.0
      time = 0.0
      stats = {}
      stats['L1_hit_rate'] = 0
      stats['L2_hit_rate'] = 0
      stats['L3_hit_rate'] = 0
      stats['Effective Latency'] = 0
      stats['Effective Throughput'] = 0
      stats['RAM cycles'] = 0
      stats['CPU cycles'] = 0
      stats['iALU cycles'] = 0
      stats['fALU cycles'] = 0
      stats['fDIV cycles'] = 0

      stats['T_eff_cycles'] = 0

      ##X#print ("cycles = ", cycles)

      for item in tasklist:
        # TASK: Cycles associated with moving data through memory and cache
        if item[0] == 'MEM_ACCESS':
          '''
          num_index_vars = item[1]
          num_float_vars = item[2]
          avg_dist = item[3]
          avg_reuse_dist = item[4]
          stdev_reuse_dist = item[5]
          index_loads = item[6]
          float_loads = item[7]
          init_flag = item[8]
          '''
          sd = item[1] #stack distance
          psd = item[2] #probability p(sd)
          block_size = item[3] #blocksize
          total_bytes = item[4]
          data_bus_width = item[5]

          # Rather measuring the hitrate on the average of the stack distances,
          # we measure the hitrates on the entire distribution of stack distances

          # L1
          #L1 probabilities for all the stack distances
          L1_phits_d = self.phit_sd(sd, self.associativity[0], self.cache_sizes[0], self.cache_line_sizes[0])
          #L1_phit = p(d) * p(h/d) i.e; psd * phits_d
          L1_phit = self.phit(psd, L1_phits_d)
          #print ("l1_hit ", L1_phit)

          # L2
          #L2 probabilities for all the stack distances
          L2_phits_d = self.phit_sd(sd, self.associativity[1], self.cache_sizes[1], self.cache_line_sizes[1])
          #L2_phit = p(d) * p(h/d) i.e; psd * phits_d
          L2_phit = self.phit(psd, L2_phits_d)
          #print ("l2_hit ", L2_phit)

          # L3
          #L3 probabilities for all the stack distances
          L3_phits_d = self.phit_sd(sd, self.associativity[2], self.cache_sizes[2], self.cache_line_sizes[2])
          #L3_phit = p(d) * p(h/d) i.e; psd * phits_d
          L3_phit = self.phit(psd, L3_phits_d)
          #print ("l3_hit ", L3_phit)

          # Measure the effective latency ( L_eff) in cycles
          L_eff = self.effective_cycles(L1_phit,L2_phit,L3_phit,self.cache_cycles, self.ram_cycles)

          # Measure effective bandwidth (B_eff) in cycles -- We need to fix the input for this later
          B_eff = self.effective_cycles(L1_phit,L2_phit,L3_phit,self.cache_bandwidth_cycles,self.bw_ram_miss_penality)

          # Effective access time (in cycles)
          #cycles_per_byte = ((L_eff + (block_size-1)*B_eff)/block_size)/data_bus_width # Data bus width is 1 cache-line = 64 bytes

          cycles_per_byte = ((L_eff + (block_size-1)*B_eff)/block_size)
          #print(L_eff, B_eff, cycles_per_byte)

          cycles_per_byte = ((L_eff + (block_size-1)*B_eff)/block_size)
          print(L_eff, B_eff, cycles_per_byte)

          #total_bytes /= block_size
          #T_eff_cycles = cycles_per_byte * total_bytes
          T_eff_cycles = cycles_per_byte * total_bytes/8.0 # Divide by 8.0 because 'cycles_per_byte' is cycles_per_mem_access
          # In case if we need in secs
          T_eff_secs = T_eff_cycles / self.clockspeed * self.thread_efficiency()

          #Number of cycles for effective access time
          cycles += T_eff_cycles

          print ("T_eff_cycles= ", T_eff_cycles)
          print ("Partial cycles= ", cycles)

          stats['L1_hit_rate'] += L1_phit
          stats['L2_hit_rate'] += L2_phit
          stats['L3_hit_rate'] += L3_phit
          stats['Effective Latency'] += L_eff
          stats['Effective Throughput'] += B_eff

          stats['T_eff_cycles'] += cycles

        # TASK: CPU ops
        elif item[0] == 'CPU':

          num_ops = item[1]
          cycles += num_ops * self.cycles_per_CPU_ops
          stats['CPU cycles'] += num_ops * self.cycles_per_CPU_ops
          #X#print ("cpu ", cycles)

        # TASK: Integer operations
        elif item[0] == 'iALU':

          num_ops = item[1]
          cycles += num_ops * self.cycles_per_iALU
          stats['iALU cycles'] += num_ops * self.cycles_per_iALU
          #X#print ("ialu ", cycles)

        # TASK: Floating point operations
        elif item[0] == 'fALU':

          num_ops = item[1]
          cycles += num_ops * self.cycles_per_fALU
          stats['fALU cycles'] += num_ops * self.cycles_per_fALU
          #X#print ("falu ", cycles, num_ops)

        # TASK: Floating point divisions
        elif item[0] == 'fDIV':

          num_ops = item[1]
          cycles += num_ops * self.cycles_per_division
          stats['fDIV cycles'] += num_ops * self.cycles_per_division
          #print ("fDIV ", cycles, num_ops)

        # Invalid task
        else:
          print ('Warning: task list item', item, ' cannot be parsed, ignoring it')

      # Divide cumulative cycles by clockspeed to get time; apply thread eff
      time += cycles / self.clockspeed * self.thread_efficiency()
      stats['Thread Efficiency'] = self.thread_efficiency()

      if statsFlag:
        return time, stats
      else:
        return time

    def ncr(self, n, m):
          """
          n choose m
          """
          if(m>n): return 0
          r = 1
          for j in xrange(1,m+1):
              r *= (n-m+j)/float(j)
          return r

    #   def phit_D(self, D=9.0, A, cs, ls): #We can keep 9.0 as the default Stack distance if Associativity=8
    def phit_D(self, D, A, cs, ls):
      """
      Calculate the probability of hit (given stack distance, D) for a give cache level
      Output: Gives the probability of a hit given D -- P(h/D)
      """
      #D = 4   stack distance (need to take either from tasklist or use grammatical function)
      # A (Associativity)
      phit_D = 0.0 #To compute probability of a hit given D
      B = (1.0 * cs)/ls  # B = Block size (cache_size/line_size)

      if (D <= A):
          if (D == -1):   D = self.cache_sizes[2]
          elif (D == 0):  phit_D = 1.0
          else:    phit_D = math.pow((1 - (1/B)), D)
      # Don't be too creative to change the follow condition to just 'else:'
      # I am changing the value of D in the previous condition.
      if(D > A):
        for a in xrange(int(A)):
          term_1 = self.ncr(D,a)
          #term_1 = math.gamma(D + 1) / (1.0 * math.gamma(D - a + 1) * math.gamma(a + 1))
          term_2 = math.pow((A/B), a)
          term_3 = math.pow((1 - (A/B)), (D - a))
          phit_D += (term_1 * term_2 * term_3)

      return phit_D

    def phit_sd(self, stack_dist, assoc, c_size, l_size):
        """
        Calculate probability of hits for all the stack distances
        """
        phit_sd = [self.phit_D(d, assoc, c_size, l_size) for d in stack_dist]
        return phit_sd

    def phit(self, Pd, Phd):
      """
      Calculate probability of hit (given P(D), P(h/D) for a given cache level
      Output: Gives the probability of a hit -- P(h) = P(D)*P(h/D)
      """
      phit = 0.0 #Sum (probability of stack distance * probability of hit given D)
      Ph = map(lambda pd,phd:pd*phd,Pd,Phd)
      phit = sum(Ph)
      return phit

    def effective_cycles(self, phit_L1, phit_L2, phit_L3, cycles, ram_penality):
      """
      Calculate effective clock cycles for the given arguments
      """
      eff_clocks = 0.0
      #X#print "Latencies/ReciprocalThroughput(1/BW):", cycles
      eff_clocks=(cycles[0]*phit_L1+(1.0-phit_L1)* (cycles[1]*phit_L2 + (1.0-phit_L2)* \
                (cycles[2]*phit_L3+(1.0-phit_L3)* ram_penality)))
      return eff_clocks

    def thread_efficiency(self):
        """
        Gives the efficiency back as a function of the number of active threads.
        Function chosen as inverse of active threads. This is a cheap way of
        mimicing time slicing.
        """
        efficiency = 0.0
        if self.activethreads <= self.hwthreads:
          efficiency = 1.0
        else:
          efficiency = float(self.hwthreads) / float(self.activethreads)

        return efficiency

###########################################
class GenericPipelineProcessorParameters(ThreadedProcessor):
  """
  Parameters to be passed around. Using Mustang values for now
  """

  def __init__(self):
    #super(GenericPipelineProcessorParameters, self).__init__()

	# Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 32  # upper bound on number active threads
    self.clockspeed = 2.3 * 10 ** 9  # Hertz
    self.hwthreads = 32  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [64.0 * self.kb, 512.0 * self.kb, 12.0 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    #self.cache_cycles = [3, 26, 100]  # cycles (Original settings: not correct)
    self.cache_cycles = [4, 10, 65]  # cycles (Intel forums for i7-Xeon cpu)
    #normalizing these latencies as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    # self.cache_cycles = [4/8.0, 10/8.0, 65/8.0]  # cycles (Intel forums for i7-Xeon cpu)
    # self.cache_cycles = [5, 12, 40]  # cycles (From 7-cpu)
    self.cache_latency = [self.cache_cycles[0] / self.clockspeed,
                          self.cache_cycles[1] / self.clockspeed,
                          self.cache_cycles[2] / self.clockspeed]  # seconds
    self.associativity = [8.0, 8.0, 20.0]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 10.8 * self.ns  # Calculated from Nandu's measured values (from Intel MLC tool)
    #self.ram_latency = 36.0 / self.clockspeed + 57 * self.ns #from 7-cpu.com -- (36 cycles + 57 ns)
    #self.ram_latency = 60 * self.ns #from Intel forums
    self.ram_cycles = self.ram_latency * self.clockspeed


    self.num_cache_lines = [self.cache_sizes[0] / self.cache_line_sizes[0],
                            self.cache_sizes[1] / self.cache_line_sizes[1],
                            self.cache_sizes[2] / self.cache_line_sizes[2]] # #_of_cache_lines = cache_size / cache_line_size
    #self.cache_bandwidth_cycles = [0.5/64, (2.3+6.1)/2/64, (5.0+8.4)/2/64] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)

    self.cache_bandwidth_cycles = [4/8.0, ((2.2+2.3)/2.0+6.1)/2.0/8.0, ((4.7+5.0)/2.0+8.4)/2.0/8.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    # However, normalizing these value as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    # self.cache_bandwidth_cycles = [0.5, 6.1/64.0, 8.4/64.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)

    #self.bandwidth = [128/(64.0+32.0), 128/64.0, 128/32] #It is 1/bandwidth (cycle/bytes) values read from the spec sheet
    #self.ram_bandwidth = 128/16  #which is also 1/bandwidth (cycle/bytes) wrong values
    self.bw_ram = 20222.0937 * self.mb # mega bytes/sec (calculated from Nandu's measured values)
    # self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed # cycles/byte
    self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed/8.0 # cycles/mem access (word)
    #self.bw_ram = (17500 + 11000)/2 * self.mb # (bytes/sec) from 7-cpu.com
    #self.bw_ram_miss_penality = 1/self.bw_ram * self.clockspeed # cycles/bytes

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.1
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.1
    self.cycles_per_vector_ops = 0.075 * 2
    self.cycles_per_division = 1.0

    # Pipelines
    self.pipelinetypes = ['iadd', 'fadd', 'idiv', 'fdiv', 'imul', 'fmul', \
		'alu', 'br', 'load', 'store', 'unknown']
    self.pipelinecounts = {'iadd' : 2, 'fadd': 2, 'idiv': 2, 'fdiv': 2, 'imul': 2, \
        'fmul': 2, 'load': 2, 'store': 2, 'alu' : 2, 'br': 1, 'unknown' : 1}
    self.pipelinelatencies = { # in seconds
		'iadd' : 3e-8, 'fadd': 3e-8, 'idiv': 1e-8, 'fdiv': 32e-8, 'imul': 2e-8,\
		'fmul': 2e-8,'load': 2e-8, 'store': 2e-8, 'alu' : 3e-8, 'br': 1e-9, 'unknown' : 1e-8}
    self.pipelinethroughputs = { # in seconds
		'iadd' : 3e-9, 'fadd': 3e-9, 'idiv': 1e-9, 'fdiv': 32e-9, 'imul': 2e-9,
		'fmul': 2e-9,'load': 2e-9, 'store': 2e-9, 'alu' : 3e-9, 'br': 1e-9, 'unknown' : 1e-9}
	# TO DO: How do we model vector units?

###############################################################################
# DropIn entity, defined under dropEngine, not simianEngine
###############################################################################

class dropCore(Entity):
    def __init__(self, baseInfo, hwparams, taskgraph, *args):
        super(dropCore, self).__init__(baseInfo)
        self.hw = hwparams # passed in from main PPT processor class
        # Create and start a process for each pipeline
        self.q, self.pipe_active = {}, {}
        for pipeline_type in self.hw.pipelinecounts:
            self.q[pipeline_type], self.pipe_active[pipeline_type] = {}, {}
            for i in range(self.hw.pipelinecounts[pipeline_type]):
                self.q[pipeline_type][i] = [] # q for this pipeline
                self.pipe_active[pipeline_type][i] = False
                self.createProcess(pipeline_type+str(i), pipeline_process)
                self.startProcess(pipeline_type+str(i), pipeline_type, i, \
                    self.hw.pipelinelatencies[pipeline_type], \
                    self.hw.pipelinethroughputs[pipeline_type])

		# Create master process, e.g, the drop in process
        self.done_Q = []
        self.createProcess("dropin_process", dropin_process)
        self.startProcess("dropin_process", taskgraph)

###########################################################
def dropin_process(this, taskgraph):
    """
    A process that executes the taskgraph in the dropin engine
    """
    # Taskgraph is a list of instruction vertices each being a dictionary
    # with entries  'children', 'instr' with dictionary entries lists

    # Initialization
    #print this
    core = this.entity
    g = taskgraph
    #print g, len(g)
    # Iterate over each item (an item is a task-graphlet/BB-graph) in the list
    # For each item gt the key values, key is the vertex and values are
    '''
    for item in g:
        for vert, value in item.iteritems():
            print vert, value['children']
    '''

    v_ready = [] # List of vertices ready to be executed
    num_outstanding = len(g) # number of nodes still to be processed
    print num_outstanding
    core.active = True
	# Count number of parents for each vertex
    for v in g:
        #print 'Before assignment : ', g[v]
        g[v]['num_parents_outstanding'] = 0
        #print 'After assignment : ', g[v]
        #v['num_parents_outstanding'] = 0
    for v, val in g.iteritems():
    #for v in g:
        #print 'Children at each vertex : ', val['children']
        for w in val['children']:
        #for w in v['children']:
            g[w]['num_parents_outstanding'] += 1
            #print g[w]['num_parents_outstanding']
    for v in g:
        g[v]['done'] = False
        g[v]['in_pipeline'] = False
        if g[v]['num_parents_outstanding'] == 0:
            v_ready.append(v)
	# Main loop
    while num_outstanding > 0: # Not all instruction vertex completed yet
		# 1: Check for new messages
        print v_ready, core.done_Q
        while core.done_Q: # A pipeline has completed an instruction
            v_done = core.done_Q.pop()
            num_outstanding -= 1
            # 'v_done : ', v_done, num_outstanding, core.done_Q
            g[v_done]['time_done'] = core.engine.now
            for w in g[v_done]['children']:
                g[w]['num_parents_outstanding'] -= 1
                if g[w]['num_parents_outstanding'] == 0:
                    v_ready.append(w)
        # 2: Get vertices without outstanding parents sent to pipelines
        for v in v_ready: #
            instr_type = g[v]['inst']
            print instr_type
            if not instr_type in core.hw.pipelinetypes: # "unknown instruction"
                instr_type = 'unknown'
            n_pipes = core.hw.pipelinecounts[instr_type]
            dispatched = False
            for i in range(n_pipes): # send to first sleeping queue
                if not core.pipe_active[instr_type][i]:
                    dispatched = True
                    core.q[instr_type][i].insert(0, v)
                    core.wakeProcess(instr_type+str(i))
            if not dispatched: # all queues are active, insert in shortest one
                min_q_length, min_i = 1000000, 0 # what is MAXINT in Python?
                for i in range(n_pipes):
                    if len(core.q[instr_type][i]) < min_q_length:
                        min_q_length = len(core.q[instr_type][i])
                        min_i = i
                core.q[instr_type][min_i].insert(0, v)
        print 'V_ready : ',v_ready
        v_ready = []
        core.active = False
        g["totalBBTime"] = core.engine.now
        #print "Appended: ", core.engine.now
        #print "Modified TaskGraph: ", g
        this.hibernate()
        core.active = True
        print num_outstanding
    #print 'End of loop Main',g
    # We are done, let's report the time this took
	# We do this by appending an element to the task graph list at the end.

###########################################################
def pipeline_process(this, type, id, latency, throughput_time):
	"""
	The pipeline process mimicking a single resource, such as an ADD pipeline
	"""
	core = this.entity
	in_q = core.q[type][id] # incoming instructions are in this queue
	depth = int(latency/float(throughput_time))

	print core.engine.now, " DropIn Pipeline Process ", id, " with depth ", depth, " started"


	pipe = {} # the actual pipeline that processes instructions
			  # instructions enter at pipe[depth] and work their way to
			  # pipe[0] position, then get put back into dropIn proc queue
	for i in range(depth):
		pipe[i] = None
	i = 0 # counter
	last = 0
	core.pipe_active[type][id] = False
	this.hibernate() # wait to be woken up with an instruction to process
	core.pipe_active[type][id] = True
	while core.pipe_active[type][id]:
		this.sleep(throughput_time)
		if pipe[0] <> None: # an instruction is done
			core.done_Q.append(pipe[0])
			if not core.active: # main process is hibernating, so let's wake it up
				core.wakeProcess('dropin_process')
			print core.engine.now, " DropIn Pipeline Process ", id, " completed instr ", pipe[0]
		for i in range(depth-1):  # advance the pipeline
			pipe[i] = pipe[i+1]
		if in_q: # another instruction in the q
			pipe[depth-1] = in_q.pop()
			last = 0
		else:
			pipe[depth] = None
			last += 1
			if last == depth: # the entire pipeline is empty
			    core.pipe_active[type][id] = False
			    this.hibernate() # sleep until woken up again
			    core.pipe_active[type][id] = True

###########################################################
class GenericCore(ThreadedProcessor):
  """
  A PPT Generic Core, initially modeled after mustamg
  """
  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(GenericCore, self).__init__(node)
    self.hw = GenericPipelineProcessorParameters()

  def time_compute_taskgraph(self, taskGraphDict, mem_access_params):
    """
    Computes the cycles that the items in the taskl graph (CPU ops, data access,
    vector ops, memory alloc) take.
    """
    ########################
	# 0. Drop-in Simian Initialization
    simName, startTime, endTime, minDelay, useMPI = \
  		"snapsim", 0.0, 100000000000.0, 0.1, False
    dropEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

    # Set pipeline for mem_access_time correct (blocksize ignored this way . . .)
    mem_latency, mem_bw = self.mem_access_time(mem_access_params)
    self.hw.pipelinelatencies['load'] = mem_latency
    self.hw.pipelinethroughputs['load'] = mem_bw
    #print taskGraphDict
    dropTimeDict = {}
    for bbName, taskgraph in taskGraphDict.iteritems():
        dropEngine.addEntity('dropCore', dropCore, 0, self.hw, taskgraph)
        dropEngine.run()
        dropEngine.exit()
        # By convention, dropEngine added an element to the taskgraph with the time
        # it took to complete it. Every instruction vertex in the taskgraph also has an
        # attribute 'time_done' that lists when that instruction was completed.
        #print taskgraph
        dropTime = taskgraph["totalBBTime"]

        #print "!!!!!!!!!!!!! DROP ENGINE DONE !!!!!!!!!!!"
        #print 'The ', bbName, " BB dropTime: ", dropTime
        dropTimeDict[bbName] = dropTime

    #print 'dropTimeDict : ', dropTimeDict
    return dropTimeDict

  #############################
  def mem_access_time(self, mem_access_params):
    """
    Computes the cycles that the items in the taskl graph (CPU ops, data access,
    vector ops, memory alloc) take. Arguments are: stack distance, Probability p(sd),
    blocksize
    """
    mem_acc = 'MEM_ACCESS'
    mem_acc_idx = [[i, mem.index(mem_acc)] for i, mem in enumerate(mem_access_params) if mem_acc in mem]
    #print mem_acc_idx[0][0]
    mem_access_params = mem_access_params[mem_acc_idx[0][0]]
    #print mem_access_params
    sd = mem_access_params[1] #stack distance
    psd = mem_access_params[2] #probability p(sd)
    block_size = mem_access_params[3] #blocksize
    total_bytes = mem_access_params[4]
    data_bus_width = mem_access_params[5]
	# we measure the hitrates on the entire distribution of stack distances

    #L1
	#L1 probabilities for all the stack distances
    L1_phits_d = self.phit_sd(sd, self.hw.associativity[0], self.hw.cache_sizes[0], self.hw.cache_line_sizes[0])
	#L1_phit = p(d) * p(h/d) i.e; psd * phits_d
    L1_phit = self.phit(psd, L1_phits_d)

	#print ("l1_hit ", L1_phit)
	# L2

	#L2 probabilities for all the stack distances
    L2_phits_d = self.phit_sd(sd, self.hw.associativity[1], self.hw.cache_sizes[1], self.hw.cache_line_sizes[1])
	#L2_phit = p(d) * p(h/d) i.e; psd * phits_d
    L2_phit = self.phit(psd, L2_phits_d)

	#print ("l2_hit ", L2_phit)
	# L3

	#L3 probabilities for all the stack distances
    L3_phits_d = self.phit_sd(sd, self.hw.associativity[2], self.hw.cache_sizes[2], self.hw.cache_line_sizes[2])
	#L3_phit = p(d) * p(h/d) i.e; psd * phits_d
    L3_phit = self.phit(psd, L3_phits_d)

	#print ("l3_hit ", L3_phit)

	# Measure the effective latency ( L_eff) in cycles
    L_eff = self.effective_cycles(L1_phit,L2_phit,L3_phit,self.hw.cache_cycles, self.hw.ram_cycles)

	# Measure effective bandwidth (B_eff) in cycles -- We need to fix the input for this later
    B_eff = self.effective_cycles(L1_phit,L2_phit,L3_phit,self.hw.cache_bandwidth_cycles,self.hw.bw_ram_miss_penality)

	# Effective access time (in cycles)
	#cycles_per_byte = ((L_eff + (block_size-1)*B_eff)/block_size)/data_bus_width # Data bus width is 1 cache-line = 64 bytes
    cycles_per_memaccess = ((L_eff + (block_size-1)*B_eff)/block_size)
    time_per_memaccess = cycles_per_memaccess / self.hw.clockspeed

    B_eff_time = B_eff/ self.hw.clockspeed
    L_eff_time = L_eff/ self.hw.clockspeed

    return L_eff_time, B_eff_time

  ################
  ###### Helper Functions for Hitrate computation
  ################
  def ncr(self, n, m):
        """
        n choose m
        """
        if(m>n): return 0
        r = 1
        for j in xrange(1,m+1):
            r *= (n-m+j)/float(j)
        return r

  # def phit_D(self, D=9.0, A, cs, ls): #We can keep 9.0 as the default Stack distance if Associativity=8
  def phit_D(self, D, A, cs, ls):
    """
    Calculate the probability of hit (given stack distance, D) for a give cache level
    Output: Gives the probability of a hit given D -- P(h/D)
    """
    #D = 4   stack distance (need to take either from tasklist or use grammatical function)
    # A (Associativity)
    phit_D = 0.0 #To compute probability of a hit given D
    B = (1.0 * cs)/ls  # B = Block size (cache_size/line_size)

    if (D <= A):
        if (D == -1):   D = self.hw.cache_sizes[2]
        elif (D == 0):  phit_D = 1.0
        else:    phit_D = math.pow((1 - (1/B)), D)
    # Don't be too creative to change the follow condition to just 'else:'
    # I am changing the value of D in the previous condition.
    if(D > A):
      for a in xrange(int(A)):
        term_1 = self.ncr(D,a)
        #term_1 = math.gamma(D + 1) / (1.0 * math.gamma(D - a + 1) * math.gamma(a + 1))
        term_2 = math.pow((A/B), a)
        term_3 = math.pow((1 - (A/B)), (D - a))
        phit_D += (term_1 * term_2 * term_3)

    return phit_D

  def phit_sd(self, stack_dist, assoc, c_size, l_size):
      """
      Calculate probability of hits for all the stack distances
      """
      phit_sd = [self.phit_D(d, assoc, c_size, l_size) for d in stack_dist]
      return phit_sd

  def phit(self, Pd, Phd):
    """
    Calculate probability of hit (given P(D), P(h/D) for a given cache level
    Output: Gives the probability of a hit -- P(h) = P(D)*P(h/D)
    """
    phit = 0.0 #Sum (probability of stack distance * probability of hit given D)
    Ph = map(lambda pd,phd:pd*phd,Pd,Phd)
    phit = sum(Ph)
    return phit

  def effective_cycles(self, phit_L1, phit_L2, phit_L3, cycles, ram_penality):
    """
    Calculate effective clock cycles for the given arguments
    """
    eff_clocks = 0.0
    print "Latencies/ReciprocalThroughput(1/BW):", cycles
    eff_clocks=(cycles[0]*phit_L1+(1.0-phit_L1)* \
                 (cycles[1]*phit_L2+ (1.0-phit_L2)* \
                 (cycles[2]*phit_L3+(1.0-phit_L3)* \
                  ram_penality)))
    return eff_clocks

###########################################################
###########################################################
###########################################################
class CieloCore(GenericProcessor):
  """
  A Simian resource that represents Cielo core. It has 3 cache levels and
  vector units.
  """
  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(CieloCore, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 16  # upper bound on number active threads
    self.clockspeed = 2.40 * 10 ** 9  # Hertz
    self.hwthreads = 1  # number of hardware threads
    self.vector_width = 16  # width of vector unit in bytes, 128 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [64.0 * self.kb, 512.0 * self.kb, 1.5 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.cache_latency = [1.0 * self.ns, 5.0 * self.ns, 25.0 * self.ns]  # seconds
    self.cache_cycles = [self.cache_latency[0] * self.clockspeed,
                         self.cache_latency[1] * self.clockspeed,
                         self.cache_latency[2] * self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 60.0 * self.ns  # seconds
    self.ram_cycles = self.ram_latency * self.clockspeed

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.5
    self.cycles_per_int_vec = 0.1
    self.cycles_per_fALU = 1.0
    self.cycles_per_vector_ops = 0.5
    self.cycles_per_division = 10.0

###########################################################
class MLIntelCore(GenericProcessor):
  """
  A Simian resource that represents ML Intel Sandy Bridge core. It has
  3 cache levels and vector units.
  """
  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(MLIntelCore, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 32  # upper bound on number active threads
    self.clockspeed = 2.60 * 10 ** 9  # Hertz
    self.hwthreads = 32  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [32.0 * self.kb, 256.0 * self.kb, 2.5 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.cache_latency = [0.3 * self.ns, 4.0 * self.ns, 16.0 * self.ns]  # seconds
    self.cache_cycles = [self.cache_latency[0] * self.clockspeed,
                         self.cache_latency[1] * self.clockspeed,
                         self.cache_latency[2] * self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 50.0 * self.ns  # seconds
    self.ram_cycles = self.ram_latency * self.clockspeed

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.3
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.3
    self.cycles_per_vector_ops = 0.075
    self.cycles_per_division = 3.0

###########################################################
class EdisonCore(GenericProcessor):
  """
  A Simian resource that represents Edison Intel Ivy Bridge core. It has
  3 cache levels and vector units.
  """

  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(EdisonCore, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 48  # upper bound on number active threads
    self.clockspeed = 2.40 * 10 ** 9  # Hertz
    self.hwthreads = 48  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [32.0 * self.kb, 256.0 * self.kb, 2.5 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.cache_latency = [0.3 * self.ns, 4.0 * self.ns, 16.0 * self.ns]  # seconds
    self.cache_cycles = [self.cache_latency[0] * self.clockspeed,
                         self.cache_latency[1] * self.clockspeed,
                         self.cache_latency[2] * self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 50.0 * self.ns  # seconds
    self.ram_cycles = self.ram_latency * self.clockspeed

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.3
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.3
    self.cycles_per_vector_ops = 0.075
    self.cycles_per_division = 3.0

###########################################################
class TTNCore(GenericProcessor):
	"""
 	A SimX resource	 that represents Titan core
	 	It has 2 cache levels and a vector unit
 	"""
	def __init__(self, node):
		super(TTNCore, self).__init__(node)
		#
		#  PARAMETERS
		#
		self.maxthreads = 16			# upper bound on number active threads
		self.hwthreads = 16			# number of hardware threads
		self.clockspeed = 2.2*10**9		# Hertz
		self.cycles_per_CPU_ops = 5		# Number of clock cycles per CPU operation (avg)
		self.cycles_per_iALU = .5		# Number of clock cycles per integer ALU operation
		self.cycles_per_fALU = .5		# Number of clock cycles per float ALU operation
		self.cycles_per_vector_ops = 5 		# Number of clock cycles per vector operation
		self.vector_width = 64			# width of vector unit in bytes
		self.cache_levels = 3			# number of cache levels
		self.cache_sizes = [48*10**3, 1*10**6, 16*10**6]			  #	  list of cache sizes
 		self.cache_page_sizes = [16, 16, ] 	# list of page sizes for the different cache levels[bytes]
 		self.num_registers = 64			# number of registers (single precision, holds an int) [4 bytes]
 		self.register_cycles = 3		# cycles per register access
		self.cache_cycles = [3.5, 17.8, 53.9]	# list of cache access cycles per level
 		self.ram_cycles = 150			# number of cycles to load from main memory
 		self.ram_page_size = 1024		# page size for main memory access [bytes]
 		self.end_device_comps = []		# Contains the number of CPU cycles after which each device will finish
 							# computing or 0 if no computation is being run the device
 		for i in range(self.node.num_accelerators):
 			self.end_device_comps.append(0)
	def time_compute(self, tasklist, start, statsFlag=False):
		"""
 		Computes the cycles that
 		the items in the tasklist take
		"""
		cycles = 0.0
 		time = 0.0
 		stats = {}
 		stats['L1_float_hits'] =0
 		stats['L2_float_hits'] =0
 		stats['L1_int_hits'] =0
 		stats['L2_int_hits'] =0
 		stats['L1_int_misses'] =0
 		stats['L2_int_misses'] =0
 		stats['L1_float_misses'] =0
 		stats['L2_float_misses'] =0
 		stats['RAM accesses'] =0
 		stats['L1 cycles'] =0
 		stats['L2 cycles'] =0
 		stats['RAM cycles'] =0
 		stats['CPU cycles'] =0
 		stats['iALU cycles'] =0
 		stats['fALU cycles'] =0
 		stats['VECTOR ops'] =0
 		stats['VECTOR cycles'] =0
 		stats['internode comm time'] =0
 		stats['intranode comm time'] =0
 		for item in tasklist:
 			#####  Memory Access #########
 			if item[0] == 'MEM_ACCESS':
	 				# memory signature access, followed by
 				num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[1], item[2], item[3], item[4], item[5]
 				index_loads, float_loads, init_flag = item[6], item[7], item[8]
 				#print "Task list received", item
 								#TODO: Insert formula that turns these variables into actual L1, L2, ram accesses
 				# This is a V0.1 model for realistic caching behavior
 				# The arguments to MEM_ACCESS are architecture independent
 				#
 				# We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
 				# Assume index variables are in registers as much as possible
 				avg_index_loads = index_loads / num_index_vars
 				num_reg_accesses = int(self.num_registers * avg_index_loads)
 				nonreg_index_loads =  max(0, index_loads - num_reg_accesses) # number of index loads that are not handled by register
 				num_vars_per_page = self.ram_page_size / avg_dist # avg number of variables per ram page
 				if init_flag: # We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
 					initial_ram_pages = num_float_vars / num_vars_per_page # number of ram pages to be loaded at least once
		 			float_loads -= num_float_vars # we treat the first time that a float is loaded separately
 				else: # No init_flag false means that the float data may be already used, so no special treatment
 					pass
 				# Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
 				L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
 					(avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
 				L1_float_hits = int(L1_hitrate*float_loads)
 				L1_int_hits = int(L1_hitrate*nonreg_index_loads)
 				L1_int_misses = nonreg_index_loads - L1_int_hits
 				L1_float_misses = float_loads - L1_float_hits
 				L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[1]/float(self.cache_page_sizes[1]) - \
 					(avg_reuse_dist/float(self.cache_page_sizes[1])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[1]))**2)))
 				L2_float_hits = int(L2_hitrate*L1_float_misses)
 				L2_int_hits = int(L2_hitrate*L1_int_misses)
		 		L2_int_misses = L1_int_misses - L2_int_hits
 				L2_float_misses = L1_float_misses - L2_float_hits

				# Update the cycles number
 				cycles += num_reg_accesses*self.register_cycles
 				cycles += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits) # float accesses are twice as expensive
 				cycles += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits) # float accesses are twice as expensive
 				#cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
 				cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) # forget initial_ram_pages for now. 1.6.15
 				stats['L1_float_hits'] += L1_float_hits
 				stats['L2_float_hits'] += L2_float_hits
 				stats['L1_int_hits'] += L1_int_hits
 				stats['L2_int_hits'] += L2_int_hits
 				stats['L1_int_misses'] += L1_int_misses
 				stats['L2_int_misses'] += L2_int_misses
 				stats['L1_float_misses'] += L1_float_misses
 				stats['L2_float_misses'] += L2_float_misses
 				stats['RAM accesses'] += ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
 				stats['L1 cycles'] += self.cache_cycles[0] * (2*L1_float_hits + L1_int_hits)
 				stats['L2 cycles'] += self.cache_cycles[1] * (2*L2_float_hits + L2_int_hits)
 				stats['RAM cycles'] += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
 			elif item[0] == 'L1':  # L1 accesses, followed by number
 				num_accesses = item[1]
 				cycles += num_accesses*self.cache_cycles[0]
 			elif item[0] == 'L2':  # L2 accesses, followed by number
 				num_accesses = item[1]
 				cycles += num_accesses*self.cache_cycles[1]
			elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:  # Higher cache access defaults to memory
 				num_accesses = item[1]
 				cycles += num_accesses*self.ram_cycles
 							##### CPU access  ###############
 			elif item[0] == 'CPU':	 # CPU ops
 				num_ops = item[1]
 				cycles += num_ops*self.cycles_per_CPU_ops
 				stats['CPU cycles'] += num_ops*self.cycles_per_CPU_ops
 			elif item[0] == 'iALU':	  # Integer additions
 				num_ops = item[1]
 				cycles += num_ops*self.cycles_per_iALU
 				stats['iALU cycles'] +=	 num_ops*self.cycles_per_iALU
 			elif item[0] == 'fALU':	  # Integer additions
 				num_ops = item[1]
 				cycles += num_ops*self.cycles_per_fALU
				stats['fALU cycles'] +=	 num_ops*self.cycles_per_fALU
 			elif item[0] == 'VECTOR':  # ['vector', n_ops, width ]
 				num_ops = item[1]
 				vec_width = item[2]
 				# vec_ops is the actual number of operations necessary,
	 			# even if required width was too high.
 				# the // operator is floor integer division
 				vec_ops = (1+vec_width//self.vector_width)*num_ops
 				cycles += vec_ops*self.cycles_per_vector_ops
 				stats['VECTOR ops'] +=	vec_ops
 				stats['VECTOR cycles'] += vec_ops*self.cycles_per_vector_ops
 			#####  Inter-process communication #########
 			elif item[0] == 'internode':	 # communication across node
 				msg_size = item[1]
			# number of bytes to be sent
 				time += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
 				stats['internode comm time'] += msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
 			elif item[0] == 'intranode':	# communication across cores on same node
 				num_accesses = item[1]		# number of bytes to be sent
 								# We treat this case  as memory access
 				cycles += num_accesses*self.ram_cycles
 				stats['intranode comm time'] += num_accesses*self.ram_cycles
 			#####  Memory Management #########
 			elif item[0] == 'alloc':  # ['alloc', n_bytes]
 				mem_size = item[1]
 				if mem_size < 0:
 					mem_size = - mem_size
 				mem_alloc_success = self.node.mem_alloc(mem_size)
 				if mem_alloc_success:
 					 # we will count this as a single memory access for timing purposes
 					 # TODO: check validity of above assumption with Nandu
 					 cycles += self.ram_cycles
 				else:
 					# well, that would be a file system access then or just an exception
 					# we add to time, not cycles
 					time += self.node.filesystem_access_time
 					#print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
 					#	 " attempted allocated memory that is not available, thus causing ", \
 					#	 " a filesystem action. You are in swapping mode now. "
 			elif item[0] == 'unalloc':	# ['unalloc', n_bytes]
 				mem_size = item[1]
 				if mem_size > 0:
 					mem_size = - mem_size
 				mem_alloc_success = self.node.mem_alloc(mem_size)
 				# Unalloc just changes the memory footprint, has no impact on time
	 		elif item[0] == 'DEVICE_ALLOC':
 				if item[1] >= self.node.num_accelerators:
 					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
 						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
 				else:
 					self.node.accelerators[item[1]].mem_alloc(item[2])
 			elif item[0] == 'DEVICE_TRANSFER':
 				if item[1] >= self.node.num_accelerators:
 					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
 						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
 				else:
 					cycles += self.node.accelerators[item[1]].transfer(item[2])*self.clockspeed
 			elif item[0] == 'KERNEL_CALL':
 				if item[1] >= self.node.num_accelerators:
 					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
 						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
 				else:
 					# Compute the number of *CPU* cycles during which the device will be busy
 					# This number will be added to the total number of cycles when synchronizing with the GPU
 					# under the condition that it is higher than the current number of cycles (concurrency between CPU and device)
 					self.end_device_comps[item[1]] = cycles+self.node.accelerators[item[1]].kernel_call(item[2], item[3], item[4], item[5], item[6], item[7], start+cycles*1/self.clockspeed)*self.clockspeed
 			elif item[0] == 'DEVICE_SYNC':
 				if item[1] >= self.node.num_accelerators:
 					print "Warning: TTNCore", id(self), " on TTNNode ", id(self.node), \
 						" attempted to communicate with a non-existing device numbered ", str(item[1]),"."
 				else:
 					cycles = max(self.end_device_comps[item[1]], cycles)
 												################
 			else:
 				print 'Warning: task list item', item,' cannot be parsed, ignoring it'
 		time += cycles * 1/self.clockspeed * self.thread_efficiency()
 		stats['Thread Efficiency'] = self.thread_efficiency()
 		if statsFlag:
	 		return time, stats
 		else:
 			return time

###########################################################
class MLIntelPlusGPUCore(GenericProcessor):
  """
  A Simian resource that represents ML Intel Sandy Bridge core. It has
  3 cache levels and vector units and NVIDIA M2090 GPU accelerator(s).
  """

  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(MLIntelPlusGPUCore, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 32  # upper bound on number active threads
    self.clockspeed = 2.60 * 10 ** 9  # Hertz
    self.hwthreads = 32  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [32.0 * self.kb, 256.0 * self.kb, 2.5 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.cache_latency = [0.3 * self.ns, 4.0 * self.ns, 16.0 * self.ns]  # seconds
    self.cache_cycles = [self.cache_latency[0] * self.clockspeed,
                         self.cache_latency[1] * self.clockspeed,
                         self.cache_latency[2] * self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 50.0 * self.ns  # seconds
    self.ram_cycles = self.ram_latency * self.clockspeed

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.3
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.3
    self.cycles_per_vector_ops = 0.075
    self.cycles_per_division = 3.0

    # Device specific
    self.end_device_comps = []              # Contains the number of CPU cycles after which each device will finish
                                            # computing or 0 if no computation is being run on the device
    for i in range(self.node.num_accelerators):
      self.end_device_comps.append(0)

  def time_compute(self, tasklist, start, statsFlag=False):
    """
    Computes the cycles that the items in the tasklist (CPU ops, data access,
    vector ops, memory alloc, GPU ops) take.
    """
    
    # Initialize
    cycles = 0.0
    time = 0.0
    stats = {}
    stats['L1_float_hits'] = 0
    stats['L2_float_hits'] = 0
    stats['L1_int_hits'] = 0
    stats['L2_int_hits'] = 0
    stats['L1_int_misses'] = 0
    stats['L2_int_misses'] = 0
    stats['L1_float_misses'] = 0
    stats['L2_float_misses'] = 0
    stats['RAM accesses'] = 0
    stats['L1 cycles'] = 0
    stats['L2 cycles'] = 0
    stats['RAM cycles'] = 0
    stats['CPU cycles'] = 0
    stats['iALU cycles'] = 0
    stats['fALU cycles'] = 0
    stats['fDIV cycles'] = 0
    stats['INTVEC ops'] = 0
    stats['INTVEC cycles'] = 0
    stats['VECTOR ops'] = 0
    stats['VECTOR cycles'] = 0
    stats['internode comm time'] = 0
    stats['intranode comm time'] = 0

    #X#print ("cycles = ", cycles)

    for item in tasklist:

      # TASK: Cycles associated with moving data through memory and cache
      if item[0] == 'MEM_ACCESS':
       
        num_index_vars = item[1]
        num_float_vars = item[2]
        avg_dist = item[3]
        avg_reuse_dist = item[4]
        stdev_reuse_dist = item[5]
        index_loads = item[6]
        float_loads = item[7]
        init_flag = item[8]

        # Insert formula that turns these variables into actual L1, L2, ram
        #   accesses. This is a V0.1 model for realistic caching behavior. The
        #   arguments to MEM_ACCESS are architecture independent. We use index
        #   vars and integers vars interchageably, ie all int variables are
        #   treated as array indices, all floats are array elements.

        # Assume index variables are in registers as much as possible. Compute
        #   number of index loads that are not handled by registers
        avg_index_loads = index_loads / num_index_vars
        num_reg_accesses = int(self.num_registers * avg_index_loads)
        nonreg_index_loads =  max(0, index_loads - num_reg_accesses)

        # Compute number of variables on a typical page in memory
        num_vars_per_page = self.ram_page_size / avg_dist

        # Check memory layout to begin
        if init_flag:
          # New function call (true); floats need loaded from main memory and
          #   fewer float_loads from cache.
          initial_ram_pages = num_float_vars / num_vars_per_page
          float_loads -= num_float_vars
        else:
          # float data already in cache
          initial_ram_pages = 0.0

        # Main memory: initial_ram_pages moving through the cache
        cycles += self.ram_cycles * initial_ram_pages
        # cycles += self.cache_cycles[2]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[2]
        # cycles += self.cache_cycles[1]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[1]
        # cycles += self.cache_cycles[0]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[1]
        #X#print ("inital_ram_pages, cycles=", cycles, initial_ram_pages)

        # Compute probability that reload is required, assume normal
        #   distribution with reuse dist larger than cache page size

        # L1
        nlines = self.cache_sizes[0] / self.cache_line_sizes[0]
        nvars = avg_reuse_dist / self.cache_line_sizes[0]
        vvars = math.sqrt(2.0 * (stdev_reuse_dist / self.cache_line_sizes[0]) ** 2)
        L1_hitrate = 0.5 * (1.0 + math.erf((nlines - nvars) / vvars))

        L1_int_hits = int(L1_hitrate * nonreg_index_loads)
        L1_float_hits = int(L1_hitrate * float_loads)

        L1_int_misses = nonreg_index_loads - L1_int_hits
        L1_float_misses = float_loads - L1_float_hits

        #X#print ("l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses, \
        #X#                L1_float_misses)

        # L2
        nlines = self.cache_sizes[1] / self.cache_line_sizes[1]
        nvars = avg_reuse_dist / self.cache_line_sizes[1]
        vvars = math.sqrt(2.0 * (stdev_reuse_dist / self.cache_line_sizes[1]) ** 2)
        L2_hitrate = 0.5 * (1.0 + math.erf((nlines - nvars) / vvars))

        L2_int_hits = int(L2_hitrate * L1_int_misses)
        L2_float_hits = int(L2_hitrate * L1_float_misses)

        L2_int_misses = L1_int_misses - L2_int_hits
        L2_float_misses = L1_float_misses - L2_float_hits

        #X#print ("l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses, \
        #X#                L2_float_misses)

        # L3
        nlines = self.cache_sizes[2] / self.cache_line_sizes[2]
        nvars = avg_reuse_dist / self.cache_line_sizes[2]
        vvars = math.sqrt(2.0 * (stdev_reuse_dist / self.cache_line_sizes[2]) ** 2)
        L3_hitrate = 0.5 * (1.0 + math.erf((nlines - nvars) / vvars))

        L3_int_hits = int(L3_hitrate * L2_int_misses)
        L3_float_hits = int(L3_hitrate * L2_float_misses)

        L3_int_misses = L2_int_misses - L3_int_hits
        L3_float_misses = L2_float_misses - L3_float_hits

        #X#print ("l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses, \
        #X#                L3_float_misses)

        # Update the cycles count for cache hits
        # Registers
        cycles += num_reg_accesses * self.register_cycles
        #X#print ("registers, cycles= ", cycles)

        # L1, data moves together in cache lines
        cycles += self.cache_cycles[0] * \
                  (L1_float_hits * self.fsize + L1_int_hits * self.isize) / \
                  self.cache_line_sizes[0]
        #X#print ("l1 ", cycles)

        # L2
        cycles += self.cache_cycles[1] * \
                  (L2_float_hits * self.fsize + L2_int_hits * self.isize) / \
                  self.cache_line_sizes[1]
        #X#print ("l2 ", cycles)

        # L3
        cycles += self.cache_cycles[2] * \
                  (L3_float_hits * self.fsize + L3_int_hits * self.isize) / \
                  self.cache_line_sizes[2]
        #X#print ("l3 ", cycles)

        # L3 misses accumulate time from main memory
        cycles += self.ram_cycles * \
                  (L3_float_misses + L3_int_misses) / num_vars_per_page
        #X#print ("l3 misses", cycles)

        # L2 misses accumulate time from L3
        # cycles += self.cache_cycles[2] * \
        #          (L2_float_misses*self.fsize + L2_int_misses*self.isize) / \
        #          self.cache_line_sizes[2]
        # print "l2 misses", cycles

        # L1 misses accumulate time from L2
        # cycles += self.cache_cycles[1] * \
        #          (L1_float_misses*self.fsize + L1_int_misses*self.isize) / \
        #          self.cache_line_sizes[2]
        # print "l1 misses", cycles

        #X#print ("memaccess cycles= ", cycles)

        stats['L1_float_hits'] += L1_float_hits
        stats['L2_float_hits'] += L2_float_hits
        stats['L1_int_hits'] += L1_int_hits
        stats['L2_int_hits'] += L2_int_hits
        stats['L1_int_misses'] += L1_int_misses
        stats['L2_int_misses'] += L2_int_misses
        stats['L1_float_misses'] += L1_float_misses
        stats['L2_float_misses'] += L2_float_misses
        stats['RAM accesses'] += (2 * L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page
        stats['L1 cycles'] += self.cache_cycles[0] * \
                                    (2 * L1_float_hits + L1_int_hits)
        stats['L2 cycles'] += self.cache_cycles[1] * \
                                    (2 * L2_float_hits + L2_int_hits)
        stats['RAM cycles'] += self.ram_cycles * \
                                    (2 * L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page

      # TASK: Direct input of cache level hitrates
      elif item[0] == 'HITRATES':

        L1_hitrate = item[1]
        L2_hitrate = item[2]
        L3_hitrate = item[3]
        num_index_vars = item[4]
        num_float_vars = item[5]
        avg_dist = item[6]
        index_loads = item[7]
        float_loads = item[8]
        init_flag = item[9]

        # Insert formula that turns these variables into actual L1, L2, ram
        #   accesses. This is a V0.1 model for realistic caching behavior. The
        #   arguments to MEM_ACCESS are architecture independent. We use index
        #   vars and integers vars interchageably, ie all int variables are
        #   treated as array indices, all floats are array elements.

        # Assume index variables are in registers as much as possible. Compute
        #   number of index loads that are not handled by registers
        avg_index_loads = index_loads / num_index_vars
        num_reg_accesses = int(self.num_registers * avg_index_loads)
        nonreg_index_loads = max(0, index_loads - num_reg_accesses)

        # Compute number of variables on a typical page in memory
        num_vars_per_page = self.ram_page_size / avg_dist

        # Check memory layout to begin
        if init_flag:
          # New function call (true); floats need loaded from main memory and
          #   fewer float_loads from cache.
          initial_ram_pages = num_float_vars / num_vars_per_page
          float_loads -= num_float_vars
        else:
          # float data already in cache
          initial_ram_pages = 0.0

        # Main memory: initial_ram_pages moving through the cache
        cycles += self.ram_cycles * initial_ram_pages

        #X#print ("inital_ram_pages, cycles=", cycles, initial_ram_pages)

        # Compute probability that reload is required, assume normal
        #   distribution with reuse dist larger than cache page size

        # L1
        L1_int_hits = int(L1_hitrate * nonreg_index_loads)
        L1_float_hits = int(L1_hitrate * float_loads)

        L1_int_misses = nonreg_index_loads - L1_int_hits
        L1_float_misses = float_loads - L1_float_hits

        #X#print ("l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses, \
        #X#                L1_float_misses)

        # L2
        L2_int_hits = int(L2_hitrate * L1_int_misses)
        L2_float_hits = int(L2_hitrate * L1_float_misses)

        L2_int_misses = L1_int_misses - L2_int_hits
        L2_float_misses = L1_float_misses - L2_float_hits

        #X#print ("l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses, \
        #X#                L2_float_misses)

        # L3
        L3_int_hits = int(L3_hitrate * L2_int_misses)
        L3_float_hits = int(L3_hitrate * L2_float_misses)

        L3_int_misses = L2_int_misses - L3_int_hits
        L3_float_misses = L2_float_misses - L3_float_hits

        #X#print ("l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses, \
        #X#                L3_float_misses)

        # Update the cycles count for cache hits
        # Registers
        cycles += num_reg_accesses * self.register_cycles
        #X#print ("registers, cycles= ", cycles)

        # L1, data moves together in cache lines
        cycles += self.cache_cycles[0] * \
                  (L1_float_hits * self.fsize + L1_int_hits * self.isize) / \
                  self.cache_line_sizes[0]
        #X#print ("l1 ", cycles)

        # L2
        cycles += self.cache_cycles[1] * \
                  (L2_float_hits * self.fsize + L2_int_hits * self.isize) / \
                  self.cache_line_sizes[1]
        #X#print ("l2 ", cycles)

        # L3
        cycles += self.cache_cycles[2] * \
                  (L3_float_hits * self.fsize + L3_int_hits * self.isize) / \
                  self.cache_line_sizes[2]
        #X#print ("l3 ", cycles)

        # L3 misses accumulate time from main memory
        cycles += self.ram_cycles * \
                  (L3_float_misses + L3_int_misses)
        #X#print ("l3 misses", cycles)

        #X#print ("memaccess cycles= ", cycles)

        stats['L1_float_hits'] += L1_float_hits
        stats['L2_float_hits'] += L2_float_hits
        stats['L1_int_hits'] += L1_int_hits
        stats['L2_int_hits'] += L2_int_hits
        stats['L1_int_misses'] += L1_int_misses
        stats['L2_int_misses'] += L2_int_misses
        stats['L1_float_misses'] += L1_float_misses
        stats['L2_float_misses'] += L2_float_misses
        stats['RAM accesses'] += (2 * L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page
        stats['L1 cycles'] += self.cache_cycles[0] * \
                                    (2 * L1_float_hits + L1_int_hits)
        stats['L2 cycles'] += self.cache_cycles[1] * \
                                    (2 * L2_float_hits + L2_int_hits)
        stats['RAM cycles'] += self.ram_cycles * \
                                    num_vars_per_page

      # TASK: Direct input of L1 accesses
      elif item[0] == 'L1':
        
        num_accesses = item[1]
        cycles += num_accesses * self.cache_cycles[0]
        #X#print ("l1 ", cycles)

      # TASK: Direct input of L2 accesses
      elif item[0] == 'L2':

        num_accesses = item[1]
        cycles += num_accesses * self.cache_cycles[1]
        #X#print ("l2 ", cycles)

      # TASK: Direct input of higher cache and memory accesses
      elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:

        num_accesses = item[1]
        cycles += num_accesses * self.ram_cycles
        #X#print ("l3 ", cycles)

      # TASK: CPU ops
      elif item[0] == 'CPU':

        num_ops = item[1]
        cycles += num_ops * self.cycles_per_CPU_ops
        stats['CPU cycles'] += num_ops * self.cycles_per_CPU_ops
        #X#print ("cpu ", cycles)

      # TASK: Integer operations
      elif item[0] == 'iALU':
        
        num_ops = item[1]
        cycles += num_ops * self.cycles_per_iALU
        stats['iALU cycles'] += num_ops * self.cycles_per_iALU
        #X#print ("ialu ", cycles)

      # TASK: Floating point operations
      elif item[0] == 'fALU':
        
        num_ops = item[1]
        cycles += num_ops * self.cycles_per_fALU
        stats['fALU cycles'] += num_ops * self.cycles_per_fALU
        #X#print ("falu ", cycles, num_ops)

      # TASK: Floating point divisions
      elif item[0] == 'fDIV':

        num_ops = item[1]
        cycles += num_ops * self.cycles_per_division
        stats['fDIV cycles'] += num_ops * self.cycles_per_division
        #X#print ("fDIV ", cycles, num_ops)

      # TASK: Integer vector operations
      elif item[0] == 'INTVEC':

        num_ops = item[1]
        vec_width = item[2]

        # vec_ops is the actual number of operations necessary, even if
        #   required width was too high. The // operator is floor integer
        #   division.

        vec_ops = (1 + vec_width // self.vector_width) * num_ops
        cycles += vec_ops * self.cycles_per_int_vec
        stats['INTVEC ops'] += vec_ops
        stats['INTVEC cycles'] += vec_ops * self.cycles_per_int_vec
        #X#print ("intvect ", cycles, num_ops, vec_ops)

      # TASK: Vector operations
      elif item[0] == 'VECTOR':

        num_ops = item[1]
        vec_width = item[2]

        # vec_ops is the actual number of operations necessary, even if
        #   required width was too high. The // operator is floor integer
        #   division.

        vec_ops = (1 + vec_width // self.vector_width) * num_ops
        cycles += vec_ops * self.cycles_per_vector_ops
        stats['VECTOR ops'] += vec_ops
        stats['VECTOR cycles'] += vec_ops * self.cycles_per_vector_ops
        #X#print ("vector ", cycles, num_ops, vec_ops)

      # TASK: communication across nodes, update time directly instead of cycles
      elif item[0] == 'internode':

        msg_size = item[1]
        tmp = msg_size / self.node.interconnect_bandwidth + \
               self.node.interconnect_latency
        time += tmp
        stats['internode comm time'] += tmp
        #X#print ("inter ", cycles)

      # TASK: communication within a node treated as memory access
      elif item[0] == 'intranode':

        num_accesses = float(item[1]) / self.ram_page_size
        cycles += num_accesses * self.ram_cycles
        stats['intranode comm time'] += num_accesses * self.ram_cycles
        #X#print ("intra ", cycles)

      # TASK: memory management allocation
      elif item[0] == 'alloc':

        mem_size = item[1]
        if mem_size < 0:
          mem_size = -mem_size
        mem_alloc_success = self.node.mem_alloc(mem_size)
        if mem_alloc_success:
           # Count this as a single memory access for timing purposes
           cycles += self.ram_cycles
        else:
          # File system access or just an exception, add to time not cycles
          time += self.node.filesystem_access_time

        #X#print ("alloc ", cycles)

      # TASK: memory management deallocation; changes memory footprint, not time
      elif item[0] == 'unalloc':

        mem_size = item[1]
        if mem_size > 0:
          mem_size = -mem_size
        mem_alloc_success = self.node.mem_alloc(mem_size)
        #X#print ("unalloc ", cycles)

      # TASK: memory management on GPU
      elif item[0] == 'DEVICE_ALLOC':
      
        if item[1] >= self.node.num_accelerators:
          print "Warning: MLIntelPlusGPUCore", id(self), " on MLIntelPlusGPUNode ", id(self.node), \
                  " attempted to communicate with a non-existing device numbered ", str(item[1]),"."
        else:
          self.node.accelerators[item[1]].allocate_device_mem(item[2])

      # TASK: data transfer with GPU
      elif item[0] == 'DEVICE_TRANSFER':
        
        if item[1] >= self.node.num_accelerators:
          print "Warning: MLIntelPlusGPUCore", id(self), " on MLIntelPlusGPUNode ", id(self.node), \
                  " attempted to communicate with a non-existing device numbered ", str(item[1]),"."
        else:
          cycles += self.node.accelerators[item[1]].transfer_to_device(item[2])*self.clockspeed

      # TASK: kernel launch on GPU
      elif item[0] == 'KERNEL_CALL':

        if item[1] >= self.node.num_accelerators:
          print "Warning: MLIntelPlusGPUCore", id(self), " on MLIntelPlusGPUNode ", id(self.node), \
                  " attempted to communicate with a non-existing device numbered ", str(item[1]),"."
        else:
          # Compute the number of *CPU* cycles during which the device will be busy
          # This number will be added to the total number of cycles when synchronizing with the GPU
          # under the condition that it is higher than the current number of cycles (concurrency between CPU and device)
          self.end_device_comps[item[1]] = cycles+self.node.accelerators[item[1]].kernel_call(item[2], item[3], item[4], item[5], item[6], item[7], start+cycles*1/self.clockspeed)*self.clockspeed

      # TASK: synchronization of GPU
      elif item[0] == 'DEVICE_SYNC':

        if item[1] >= self.node.num_accelerators:
          print "Warning: MLIntelPlusGPUCore", id(self), " on MLIntelPlusGPUNode ", id(self.node), \
                  " attempted to communicate with a non-existing device numbered ", str(item[1]),"."
        else:
          cycles = max(self.end_device_comps[item[1]], cycles)

      # Invalid task
      else:

        print ('Warning: task list item', item, ' cannot be parsed, ignoring it')

    # Divide cumulative cycles by clockspeed to get time; apply thread eff
    time += cycles / self.clockspeed * self.thread_efficiency()
    stats['Thread Efficiency'] = self.thread_efficiency()
    if statsFlag:
      return time, stats
    else:
      return time

###########################################################
class MustangCore(GenericProcessor):
  """
  A Simian resource that represents Mustang AMD Magny-Cours core. It has
  3 cache levels and vector units.
  """

  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(MustangCore, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 32  # upper bound on number active threads
    self.clockspeed = 2.3 * 10 ** 9  # Hertz
    self.hwthreads = 32  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [64.0 * self.kb, 512.0 * self.kb, 12.0 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    #self.cache_cycles = [3, 26, 100]  # cycles (Original settings: not correct)
    self.cache_cycles = [4, 10, 65]  # cycles (Intel forums for i7-Xeon cpu)
    #normalizing these latencies as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    # self.cache_cycles = [4/8.0, 10/8.0, 65/8.0]  # cycles (Intel forums for i7-Xeon cpu)
    # self.cache_cycles = [5, 12, 40]  # cycles (From 7-cpu)
    self.cache_latency = [self.cache_cycles[0] / self.clockspeed,
                          self.cache_cycles[1] / self.clockspeed,
                          self.cache_cycles[2] / self.clockspeed]  # seconds
    self.associativity = [8.0, 8.0, 20.0]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 10.8 * self.ns  # Calculated from Nandu's measured values (from Intel MLC tool)
    #self.ram_latency = 36.0 / self.clockspeed + 57 * self.ns #from 7-cpu.com -- (36 cycles + 57 ns)
    #self.ram_latency = 60 * self.ns #from Intel forums
    self.ram_cycles = self.ram_latency * self.clockspeed


    self.num_cache_lines = [self.cache_sizes[0] / self.cache_line_sizes[0],
                            self.cache_sizes[1] / self.cache_line_sizes[1],
                            self.cache_sizes[2] / self.cache_line_sizes[2]] # #_of_cache_lines = cache_size / cache_line_size
    #self.cache_bandwidth_cycles = [0.5/64, (2.3+6.1)/2/64, (5.0+8.4)/2/64] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)

    self.cache_bandwidth_cycles = [4/8.0, ((2.2+2.3)/2.0+6.1)/2.0/8.0, ((4.7+5.0)/2.0+8.4)/2.0/8.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    # However, normalizing these value as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    # self.cache_bandwidth_cycles = [0.5, 6.1/64.0, 8.4/64.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)

    #self.bandwidth = [128/(64.0+32.0), 128/64.0, 128/32] #It is 1/bandwidth (cycle/bytes) values read from the spec sheet
    #self.ram_bandwidth = 128/16  #which is also 1/bandwidth (cycle/bytes) wrong values
    self.bw_ram = 20222.0937 * self.mb # mega bytes/sec (calculated from Nandu's measured values)
    # self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed # cycles/byte
    self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed/8.0 # cycles/mem access (word)
    #self.bw_ram = (17500 + 11000)/2 * self.mb # (bytes/sec) from 7-cpu.com
    #self.bw_ram_miss_penality = 1/self.bw_ram * self.clockspeed # cycles/bytes

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.1
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.1
    self.cycles_per_vector_ops = 0.075 * 2
    self.cycles_per_division = 1.0

###########################################################
class I7Core(GenericProcessor):
  """
  A Simian resource that represents Grizly Intel Xeon CPU core. It has
  3 cache levels with L3 being a SmartCache. Exact prcessor name: i7-4770HQ CPU
  """

  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(I7Core, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 36  # upper bound on number active threads
    self.clockspeed = 2.2 * 10 ** 9  # Hertz (Turbo frequency 3.3 GHz, that's all I have at the moment)
    self.hwthreads = 36  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit? (I don't know this value, how can get this)

    # Registers (we do not need them for reuse distance based model)
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details (retrieved using 'lscpu')
    self.cache_levels = 3
    self.cache_sizes = [256.0 * self.kb, 1024.0 * self.kb, 6.0 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.associativity = [8.0, 8.0, 20.0]
    self.num_cache_lines = [self.cache_sizes[0] / self.cache_line_sizes[0],
                            self.cache_sizes[1] / self.cache_line_sizes[1],
                            self.cache_sizes[2] / self.cache_line_sizes[2]] # #_of_cache_lines = cache_size / cache_line_size

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    '''
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.1
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.1
    self.cycles_per_vector_ops = 0.075 * 2
    self.cycles_per_division = 1.0
    '''
    self.cycles_per_CPU_ops = 4.5
    self.cycles_per_iALU = 0.45
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.45
    self.cycles_per_vector_ops = 0.075 * 2
    self.cycles_per_division = 52.5

    # Following parameters are specific to Reuse based hardware model

    #self.cache_cycles = [3, 26, 100]  # cycles (Original settings: not correct)
    self.cache_cycles = [4, 12, 36]  # cycles (Intel forums for i7-Xeon cpu)
    #normalizing these latencies as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    #self.cache_cycles = [4/8.0, 10/8.0, 65/8.0]  # cycles (Intel forums for i7-Xeon cpu)
    #self.cache_cycles = [5, 12, 40]  # cycles (From 7-cpu)
    self.cache_latency = [self.cache_cycles[0] / self.clockspeed,
                          self.cache_cycles[1] / self.clockspeed,
                          self.cache_cycles[2] / self.clockspeed]  # seconds

    #self.cache_bandwidth_cycles = [0.5/64, (2.3+6.1)/2/64, (5.0+8.4)/2/64] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    self.cache_bandwidth_cycles = [4/8.0, ((2.2+2.3)/2.0+6.1)/2.0/8.0, ((4.7+5.0)/2.0+8.4)/2.0/8.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    # However, normalizing these value as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    #self.cache_bandwidth_cycles = [0.5, 6.1/64.0, 8.4/64.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    #self.bandwidth = [128/(64.0+32.0), 128/64.0, 128/32] #It is 1/bandwidth (cycle/bytes) values read from the spec sheet
    #self.ram_bandwidth = 128/16  #which is also 1/bandwidth (cycle/bytes) wrong values
    self.bw_ram = 20222.0937 * self.mb # mega bytes/sec (calculated from Nandu's measured values)
    #self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed # cycles/byte
    self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed/8.0 # cycles/mem access (word)
    #self.bw_ram = (17500 + 11000)/2 * self.mb # (bytes/sec) from 7-cpu.com
    #self.bw_ram_miss_penality = 1/self.bw_ram * self.clockspeed # cycles/bytes

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 10.8 * self.ns  # Calculated from Nandu's measured values (from Intel MLC tool)
    #self.ram_latency = 36.0 / self.clockspeed + 57 * self.ns #from 7-cpu.com -- (36 cycles + 57 ns)
    #self.ram_latency = 60 * self.ns #from Intel forums
    self.ram_cycles = self.ram_latency * self.clockspeed

###########################################################
class GrizzlyCore(GenericProcessor):
  """
  A Simian resource that represents Grizly Intel Xeon CPU core. It has
  3 cache levels with L3 being a SmartCache.
  """

  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(GrizzlyCore, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 36  # upper bound on number active threads
    self.clockspeed = 2.1 * 10 ** 9  # Hertz (Turbo frequency 3.3 GHz, that's all I have at the moment)
    self.hwthreads = 36  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit? (I don't know this value, how can get this)

    # Registers (we do not need them for reuse distance based model)
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details (retrieved using 'lscpu')
    self.cache_levels = 3
    self.cache_sizes = [64.0 * self.kb, 256.0 * self.kb, 45.0 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.associativity = [8.0, 8.0, 20.0]
    self.num_cache_lines = [self.cache_sizes[0] / self.cache_line_sizes[0],
                            self.cache_sizes[1] / self.cache_line_sizes[1],
                            self.cache_sizes[2] / self.cache_line_sizes[2]] # #_of_cache_lines = cache_size / cache_line_size

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.
    '''
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.1
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.1
    self.cycles_per_vector_ops = 0.075 * 2
    self.cycles_per_division = 1.0
    '''
    self.cycles_per_CPU_ops = 4.5
    self.cycles_per_iALU = 0.45
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.40
    self.cycles_per_vector_ops = 0.075 * 2
    self.cycles_per_division = 55.0

    # Following parameters are specific to Reuse based hardware model

    #self.cache_cycles = [3, 26, 100]  # cycles (Original settings: not correct)
    self.cache_cycles = [4, 10, 65]  # cycles (Intel forums for i7-Xeon cpu)
    #normalizing these latencies as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    #self.cache_cycles = [4/8.0, 10/8.0, 65/8.0]  # cycles (Intel forums for i7-Xeon cpu)
    #self.cache_cycles = [5, 12, 40]  # cycles (From 7-cpu)
    self.cache_latency = [self.cache_cycles[0] / self.clockspeed,
                          self.cache_cycles[1] / self.clockspeed,
                          self.cache_cycles[2] / self.clockspeed]  # seconds

    #self.cache_bandwidth_cycles = [0.5/64, (2.3+6.1)/2/64, (5.0+8.4)/2/64] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    self.cache_bandwidth_cycles = [4/8.0, ((2.2+2.3)/2.0+6.1)/2.0/8.0, ((4.7+5.0)/2.0+8.4)/2.0/8.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    # However, normalizing these value as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor)
    #self.cache_bandwidth_cycles = [0.5, 6.1/64.0, 8.4/64.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes)
    #self.bandwidth = [128/(64.0+32.0), 128/64.0, 128/32] #It is 1/bandwidth (cycle/bytes) values read from the spec sheet
    #self.ram_bandwidth = 128/16  #which is also 1/bandwidth (cycle/bytes) wrong values
    self.bw_ram = 20222.0937 * self.mb # mega bytes/sec (calculated from Nandu's measured values)
    #self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed # cycles/byte
    self.bw_ram_miss_penality = 1/(self.bw_ram) * self.clockspeed/8.0 # cycles/mem access (word)
    #self.bw_ram = (17500 + 11000)/2 * self.mb # (bytes/sec) from 7-cpu.com
    #self.bw_ram_miss_penality = 1/self.bw_ram * self.clockspeed # cycles/bytes

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 10.8 * self.ns  # Calculated from Nandu's measured values (from Intel MLC tool)
    #self.ram_latency = 36.0 / self.clockspeed + 57 * self.ns #from 7-cpu.com -- (36 cycles + 57 ns)
    #self.ram_latency = 60 * self.ns #from Intel forums
    self.ram_cycles = self.ram_latency * self.clockspeed

###########################################################
class GPUCore(GenericProcessor):
  """
  A Simian resource that represents dummy host which has GPU accelrator(s). It has
  3 cache levels and vector units and a GPU accelerator.
  """

  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(GPUCore, self).__init__(node)

    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 32  # upper bound on number active threads
    self.clockspeed = 2.60 * 10 ** 9  # Hertz
    self.hwthreads = 32  # number of hardware threads
    self.vector_width = 32  # width of vector unit in bytes, 256 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1.0  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [32.0 * self.kb, 256.0 * self.kb, 2.5 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.cache_latency = [0.3 * self.ns, 4.0 * self.ns, 16.0 * self.ns]  # seconds
    self.cache_cycles = [self.cache_latency[0] * self.clockspeed,
                         self.cache_latency[1] * self.clockspeed,
                         self.cache_latency[2] * self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 50.0 * self.ns  # seconds
    self.ram_cycles = self.ram_latency * self.clockspeed
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.3
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.3
    self.cycles_per_vector_ops = 0.075
    self.cycles_per_division = 3.0

  def time_compute(self, tasklist, start, statsFlag=False):
    
    for item in tasklist:
      # TASK: kernel launch on GPU
      if item[0] == 'KERNEL_CALL':
        if item[1] >= self.node.num_accelerators:
          print "Warning: GPU Accelrator ", id(self), " on GPUNode ", id(self.node), " attempted to communicate with a non-existing device numbered ", str(item[1]),"."
        else:
          self.node.accelerators[item[1]].kernel_call(item[2], item[3], item[4], item[5], item[6], item[7], start)
      else:
        print ('Warning: task list item', item, ' cannot be parsed, ignoring it')