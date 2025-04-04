from __future__ import annotations
import logging
import subprocess
import threading
import json
import asyncio
import os
import time

import requests

from sglang.srt.bootstrap.app import start_bootstrap_server


import uuid
from typing import Dict, Optional
import numpy as np
from sglang.srt.bootstrap.rdma_utils import RdmaServer, RdmaClient

logger = logging.getLogger(__name__)

from sglang.srt.utils import get_open_port



class KVBootstrapServer:
    def __init__(self, port: int):
        self.bootstrap_server_port = port
        self.start_server()

    def start_server(self):
        server = start_bootstrap_server("0.0.0.0", self.bootstrap_server_port)
        logger.info(" bootstrap server started")


class KVArgs:
    """Arguments for KV cache management"""
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str = "all"


class KVManager:
    def __init__(self, args: KVArgs, bootstrap_server: KVBootstrapServer = None):
        self.args = args
        self.engine_rank = args.engine_rank
        self.kv_data_ptrs = args.kv_data_ptrs
        self.kv_data_lens = args.kv_data_lens
        self.kv_item_lens = args.kv_item_lens
        self.aux_data_ptrs = args.aux_data_ptrs
        self.aux_data_lens = args.aux_data_lens
        self.aux_item_lens = args.aux_item_lens

        self.bootstrap_server = bootstrap_server

    def set_bootstrap_server(self, bootstrap_server):
        self.bootstrap_server = bootstrap_server
    
    def calculate_all_token_kv_addresses(self, token_indices: list[int]):
        # Initialize result containers
        addresses_and_len = []
        # Process each layer
        for layer_id in range(len(self.args.kv_data_ptrs)):
            address = self.args.kv_data_ptrs[layer_id]
            length = self.args.kv_item_lens[layer_id] * len(token_indices)
            addresses_and_len.append((address, length))
        return addresses_and_len

class KVPoll:
    """Status codes for KV operations"""
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.state = KVPoll.Bootstrapping

        logger.info(f"Sender registered with room {self.bootstrap_room}")

        # Network configuration
        self.target_ip = None

        # todo get dynamic ip
        self.ip_address = "10.246.59.104"

        # Memory management
        self.mrs_to_send = []  # Memory regions for data segments to be sent
        self.meta_has_sent = False  # Flag indicating if metadata has been sent

    def handshake(self):
        """Establish connection with the receiver through bootstrap server"""
        resp = requests.get(f"http://{self.bootstrap_addr}/get_room_info/{self.bootstrap_room}")

        if resp.status_code == 200:
            data = resp.json()
            return data
        return None

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """Initialize sender with metadata only

        Args:
            num_tokens: Number of tokens to transfer
            aux_idx: Index for auxiliary data

        Returns:
            bool: True if metadata sent successfully
        """
        metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        metadata_ptr_length = self.mgr.aux_item_lens[0]

        try:
            self.qp = RdmaClient(host_ip=self.target_ip, socket_port=self.target_port)
            if self.qp.init(metadata_ptr, metadata_ptr_length):
                logger.debug("Transferring...")
                self.state = KVPoll.Transferring
        except Exception as e:
            print(e)
            self.state = KVPoll.Bootstrapping

    def poll(self) -> KVPoll:
        """Poll transfer status and handle state transitions"""
        if self.state == KVPoll.Bootstrapping:
            data = self.handshake()
            if not data:
                self.state = KVPoll.Bootstrapping
            else:
                logger.debug(data)
                self.target_ip = data.get(str(self.mgr.engine_rank))['ip']
                self.target_port = data.get(str(self.mgr.engine_rank))['port']
                self.state = KVPoll.WaitingForInput
        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        if self.state == KVPoll.Transferring:
            self.qp.check_send_complete()

            # Check completion of all work requests including metadata
            if self.qp.completed_wrs == len(self.mrs_to_send) + 1 and self.meta_has_sent:
                print("Transferring complete")
                # Write remote metadata //todo
                self.state = KVPoll.Success
            elif self.qp.completed_wrs == len(self.mrs_to_send) and not self.meta_has_sent:
                self.qp.send_metadata_wrs()
                self.meta_has_sent = True

        return self.state

    def send(self, kv_indices: np.ndarray[np.int32]):
        """Send actual data synchronously

        Args:
            kv_indices: Array of KV indices to send
        """
        # Calculate addresses and prepare memory regions for transfer
        addresses_and_len = self.mgr.calculate_all_token_kv_addresses(kv_indices)
        mrs_info = []
        for (address, length) in addresses_and_len:
            mr = self.qp.create_mr(address, length)
            self.mrs_to_send.append(mr)
            mrs_info.append({
                "address": address,
                "length": length,
                "rkey": mr.rkey
            })

        self.qp.send_wrs(self.mrs_to_send, mrs_info)


class KVReceiver:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.ep = None
        self.state = KVPoll.Bootstrapping
        self.num_tokens = 0
        self.aux_idx = -1
        self.kv_indices = None

        # Network setup
        self.rdma_port = get_open_port()

        # For metrics
        self.start_time = time.time()

        # todo get dynamic ip
        self.ip_address = "10.246.59.104"
        self.qp = RdmaServer(socket_port=self.rdma_port)

        # Initialize connection
        # todo can use other rapid method...
        self.handshake()
        self.mrs_to_receive = []  # Memory regions for receiving data segments

    def handshake(self):
        """Establish connection with the bootstrap server"""
        post_data = {
            "room_id": self.bootstrap_room,
            "session_id": self.session_id,
            "engine_rank": self.mgr.args.engine_rank,
            "ib_device": self.mgr.args.ib_device,
            "ip_addr": {
                "ip": self.ip_address,
                "port": self.rdma_port
            }
        }
        http_start = time.time()
        resp = requests.post(f"http://{self.bootstrap_addr}/handshake", json=post_data)
        http_end = time.time()
        print(f"HD Request time: {http_end - http_start}")
        if resp.status_code != 200:
            self.state = KVPoll.Failed
            print(resp.status_code)
        else:
            self.state = KVPoll.WaitingForInput
            self.initialized = True
            print("boostraped success..")

    def init(self, kv_indices: np.ndarray[np.int32], aux_index: Optional[int] = None):
        """Initialize receiver with KV indices and auxiliary data index

        Args:
            kv_indices: Array of KV indices to receive
            aux_index: Optional index for auxiliary data

        Returns:
            bool: True if initialization successful
        """
        metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        metadata_length = self.mgr.aux_item_lens[0]

        # Initialize RDMA server and register memory regions
        addresses_and_len = self.mgr.calculate_all_token_kv_addresses(kv_indices)
        mrs_info = []
        for (address, length) in addresses_and_len:
            mr = self.qp.create_mr(address, length)
            self.mrs_to_receive.append(mr)
            mrs_info.append({
                "address": address,
                "length": length,
                "rkey": mr.rkey
            })

        try:
            self.qp.init(self.mrs_to_receive, mrs_info, metadata_ptr, metadata_length)
            self.state = KVPoll.Transferring
            self.qp.recv_metadata_mr()
        except Exception as e:
            self.state = KVPoll.Bootstrapping
            return

    def poll(self) -> KVPoll:
        """Poll receive status and handle state transitions"""
        if not self.initialized:
            return KVPoll.Bootstrapping

        if self.state == KVPoll.Transferring:
            self.qp.check_complete()
            # Check if metadata transfer is complete
            if self.qp.metadata_mr_complete_num == 1:
                logger.debug("Decode Transferring complete...")
                return KVPoll.Success

        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        return self.state

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'loop') and self.loop:
            self.loop.close()
