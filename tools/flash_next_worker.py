#!/usr/bin/env python3
"""Stage a Flash Next suffix in a worker's resident Vulkan cache over SSH/TCP."""
import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gguf', ROOT/'gguf-tools/embed_mtp.py')
gguf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gguf)
BLOCKS = {0:(1,4), 1:(1,2), 8:(32,34), 10:(256,84), 11:(256,110),
          12:(256,144), 13:(256,176), 14:(256,210), 16:(256,66),
          20:(32,18), 22:(256,82), 23:(256,136), 30:(1,2), 42:(64,18)}


def tensor_bytes(t):
    block, size = BLOCKS[t['kind']]
    elements = math.prod(t['dims'])
    if elements % block:
        raise ValueError('partial quantization block: '+t['name'])
    return elements//block*size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ssh', required=True, help='SSH destination, such as root@worker')
    parser.add_argument('--control-path')
    parser.add_argument('--ssh-option', action='append', default=[])
    parser.add_argument('--remote-dir', default='/root/q36-flash-next')
    parser.add_argument('--metadata', default='/dev/shm/q36-flash-next.q36-resident')
    parser.add_argument('--layer-start', type=int, help='worker first layer; default splits evenly, odd layer on coordinator')
    parser.add_argument('--coordinator', required=True)
    parser.add_argument('--port', type=int, default=9911)
    parser.add_argument('--ctx', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--env', action='append', default=[], help='worker Q36_NAME=value override')
    parser.add_argument('--bulk-host', help='Use the trusted direct link for the weight payload')
    parser.add_argument('--bulk-port', type=int, default=9912)
    args = parser.parse_args()
    for value in args.env:
        name, sep, _ = value.partition('=')
        if not sep or not name.startswith('Q36_') or not name.replace('_','').isalnum():
            parser.error('--env requires Q36_NAME=value')
    model = gguf.GGUF(args.model)
    if model.kv.get('general.architecture', (None,None))[1] != 'qwen4exp':
        parser.error('expected a Flash Next qwen4exp weight shard')
    layers = model.kv.get('qwen4exp.block_count', (None,0))[1]
    if type(layers) is not int or layers < 3:
        parser.error('invalid Flash Next layer count')
    if args.layer_start is None:
        args.layer_start = (layers+1)//2
    if not 2 <= args.layer_start < layers:
        parser.error('the coordinator must own PLE layer 1 and the worker must have a layer')
    print('Pipeline split: coordinator --layers 0:'+str(args.layer_start-1)+
          '; worker --layers '+str(args.layer_start)+':output', file=sys.stderr, flush=True)
    tensors = model.tensors
    for t in tensors:
        t['bytes'] = tensor_bytes(t)
    ssh = ['ssh']
    if args.control_path:
        ssh += ['-S', args.control_path]
    for option in args.ssh_option:
        ssh += ['-o', option]
    ssh += [args.ssh]
    selected = [t for t in tensors if
        (t['name'].startswith('blk.') and int(t['name'].split('.')[1]) >= args.layer_start) or
        t['name'].startswith('output_hc_')]
    metadata = args.metadata
    # The sparse descriptor contains only the original catalogue and bounded
    # fingerprint samples. The engine rejects it without the resident stream.
    receiver = '''import os,struct,sys
def read(n):
 data=bytearray()
 while len(data)<n:
  part=sys.stdin.buffer.read(n-len(data))
  if not part: raise SystemExit('incomplete worker metadata')
  data.extend(part)
 return data
size,catalogue,count=struct.unpack('>QQQ',read(24))
with open(sys.argv[1],'wb') as f:
 f.truncate(size); f.write(read(catalogue))
 for i in range(count):
  offset,n=struct.unpack('>QQ',read(16))
  if offset>size or n>size-offset: raise SystemExit('invalid sample range')
  f.seek(offset); f.write(read(n))
'''
    command = 'python3 -c '+shlex.quote(receiver)+' '+shlex.quote(metadata)
    process = subprocess.Popen(ssh+[command],stdin=subprocess.PIPE)
    with args.model.open('rb') as source:
        samples = [t for t in tensors if t['bytes'] >= 32]
        process.stdin.write(struct.pack('>QQQ',model.size,model.start,len(samples)*2))
        process.stdin.write(source.read(model.start))
        for t in samples:
            for offset in (t['absolute'],t['absolute']+t['bytes']-32):
                source.seek(offset)
                process.stdin.write(struct.pack('>QQ',offset,32))
                process.stdin.write(source.read(32))
    process.stdin.close()
    if process.wait():
        raise SystemExit('worker metadata deployment failed')
    worker = ['./q36','-m',metadata,'--vulkan','--ctx',str(args.ctx),'-t',str(args.threads),
              '--role','worker','--layers',str(args.layer_start)+':output',
              '--coordinator',args.coordinator,str(args.port)]
    env = ['env','Q36_RESIDENT_WEIGHT_FD=0','Q36_THREADS='+str(args.threads),
           'Q36_SSD_ROUTE_PREFETCH=0']+args.env
    if args.bulk_host:
        listener = '''import os,socket,sys
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind((sys.argv[1],int(sys.argv[2]))); s.listen(1)
c,peer=s.accept(); s.close()
if peer[0]!=socket.gethostbyname(sys.argv[3]): raise SystemExit('unexpected staging peer')
c.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024)
os.dup2(c.fileno(),0)
os.execvp(sys.argv[4],sys.argv[4:])
'''
        command = ['python3','-c',listener,args.bulk_host,str(args.bulk_port),args.coordinator]+env+worker
    else:
        command = env+worker
    remote = 'cd '+shlex.quote(args.remote_dir)+' && echo $$ > worker.pid && exec '+shlex.join(command)
    process = subprocess.Popen(ssh+[remote],stdin=subprocess.DEVNULL if args.bulk_host else subprocess.PIPE)
    connection = None
    try:
        if args.bulk_host:
            deadline = time.monotonic()+60
            while True:
                try:
                    connection = socket.create_connection((args.bulk_host,args.bulk_port),timeout=5)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise
                    time.sleep(.2)
            connection.settimeout(None)
            connection.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4*1024*1024)
            write = connection.sendall
        else:
            write = process.stdin.write
        digest = hashlib.sha256()
        total = sum(t['bytes'] for t in selected)
        sent = 0
        start = last = time.monotonic()
        with args.model.open('rb') as source:
            for t in selected:
                write(struct.pack('>QQ',t['absolute'],t['bytes']))
                source.seek(t['absolute'])
                remaining = t['bytes']
                while remaining:
                    data = source.read(min(4*1024*1024,remaining))
                    if not data:
                        raise EOFError(t['name'])
                    write(data)
                    digest.update(data)
                    sent += len(data)
                    remaining -= len(data)
                    now = time.monotonic()
                    if now-last >= 30:
                        print(f'Worker weights: {sent/2**30:.2f}/{total/2**30:.2f} GiB, {sent/(now-start)/2**20:.1f} MiB/s',flush=True)
                        last = now
        print(f'Worker resident stream complete: {sent} bytes, SHA256 {digest.hexdigest()}',flush=True)
        if connection:
            connection.close()
            connection = None
        else:
            process.stdin.close()
        raise SystemExit(process.wait())
    except BaseException:
        if connection:
            connection.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise


if __name__ == '__main__':
    main()
